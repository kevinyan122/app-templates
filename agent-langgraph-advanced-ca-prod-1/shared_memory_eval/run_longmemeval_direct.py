"""Run all LongMemEval-S pilot questions against direct managed-memory search."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import mlflow  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402

try:
    from . import longmemeval_adapter, longmemeval_eval, memory_eval
    from .run_longmemeval_pilot import DEFAULT_APP_URL, DEFAULT_MEMORY_STORE
except ImportError:  # pragma: no cover
    import longmemeval_adapter  # type: ignore[no-redef]
    import longmemeval_eval  # type: ignore[no-redef]
    import memory_eval  # type: ignore[no-redef]
    from run_longmemeval_pilot import (  # type: ignore[no-redef]
        DEFAULT_APP_URL,
        DEFAULT_MEMORY_STORE,
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--memory-store", default=DEFAULT_MEMORY_STORE)
    parser.add_argument("--dataset-name", default=longmemeval_eval.QUESTION_DATASET_NAME)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value or {}) if isinstance(value, dict) else {}


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    if args.top_k <= 0 or args.workers <= 0:
        raise ValueError("--top-k and --workers must be positive")

    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(args.workers)
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    print(json.dumps({"phase": "connecting", "profile": args.profile}), flush=True)
    client = WorkspaceClient(profile=args.profile)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    memory_eval.configure(
        app_url=DEFAULT_APP_URL,
        memory_store=args.memory_store,
        workspace_client=client,
    )

    print(
        json.dumps({"phase": "loading_dataset", "dataset": args.dataset_name}),
        flush=True,
    )
    records = longmemeval_eval.load_question_records(
        args.dataset_name, top_k=args.top_k
    )
    selected_ids = set(args.question_id)
    selected = [
        row
        for row in records
        if not selected_ids or row["inputs"]["question_id"] in selected_ids
    ]
    if not selected:
        raise ValueError(f"No cases matched question ids: {sorted(selected_ids)}")

    print(
        json.dumps({"phase": "validating_corpora", "cases": len(selected)}),
        flush=True,
    )
    validation = {}
    invalid = []
    for index, row in enumerate(selected, start=1):
        inputs = row["inputs"]
        status = longmemeval_eval.inspect_persistent_scope(
            inputs["scope_selector"],
            inputs["expected_memory_count"],
            inputs["expected_corpus_fingerprint"],
        )
        validation[inputs["case_id"]] = {
            "question_id": inputs["question_id"],
            "question_type": inputs["question_type"],
            "scope": status["full_scope"],
            "expected_count": status["expected_count"],
            "actual_count": status["actual_count"],
            "expected_fingerprint": status["expected_fingerprint"],
            "actual_fingerprint": status["actual_fingerprint"],
            "exact": status["exact"],
        }
        if not status["exact"]:
            invalid.append(inputs["question_id"])
        print(
            json.dumps(
                {
                    "phase": "validated_case",
                    "index": index,
                    "total": len(selected),
                    "question_id": inputs["question_id"],
                    "entries": status["actual_count"],
                    "exact": status["exact"],
                }
            ),
            flush=True,
        )
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} LongMemEval scopes are missing or drifted: {invalid}"
        )

    print(
        json.dumps(
            {
                "phase": "evaluating_direct_search",
                "cases": len(selected),
                "scorers": [scorer.name for scorer in longmemeval_eval.DIRECT_SEARCH_SCORERS],
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    with mlflow.start_run(
        run_name=f"memory-eval/longmemeval-s/raw/direct-search/{len(selected)}q",
        tags={
            "suite": longmemeval_adapter.SUITE_NAME,
            "adapter_version": longmemeval_adapter.ADAPTER_VERSION,
            "source_commit": longmemeval_adapter.SOURCE_COMMIT,
            "bundle_fingerprint": longmemeval_adapter.EXPECTED_PILOT_BUNDLE_FINGERPRINT,
            "runner": "local-user-direct-memory-store",
            "search_mode": "direct-store-endpoint",
            "agent_invoked": "false",
            "judge_enabled": "false",
            "question_count": str(len(selected)),
            "top_k": str(args.top_k),
            "question_dataset": args.dataset_name,
        },
    ):
        mlflow.log_dict(
            {
                "bundle_fingerprint": longmemeval_adapter.EXPECTED_PILOT_BUNDLE_FINGERPRINT,
                "cases": validation,
            },
            "longmemeval/direct_search_corpus_validation.json",
        )
        result = mlflow.genai.evaluate(
            data=selected,
            predict_fn=longmemeval_eval.direct_predict_fn,
            scorers=longmemeval_eval.DIRECT_SEARCH_SCORERS,
        )

    score_names = [scorer.name for scorer in longmemeval_eval.DIRECT_SEARCH_SCORERS]
    category_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failures = []
    latencies = []
    for _, row in result.result_df.iterrows():
        request = _object(row.get("request"))
        response = _object(row.get("response"))
        category = str(request.get("question_type", "unknown"))
        values = {}
        for name in score_names:
            value = row.get(f"{name}/value")
            if value is not None:
                value = float(value)
                category_values[category][name].append(value)
                values[name] = value
        latency = response.get("direct_latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        if values.get("lme_direct_recall_at_k", 0.0) < 1.0:
            failures.append(
                {
                    "question_id": request.get("question_id"),
                    "question_type": category,
                    "question": request.get("question"),
                    "relevant_ids": longmemeval_eval._relevant_ids(
                        {
                            "relevant_ids_json": row.get(
                                "relevant_ids_json/value", "[]"
                            )
                        }
                    ),
                    "ranked_ids": response.get("direct_ranked_ids", []),
                    **values,
                }
            )

    by_category = {
        category: {
            name: sum(values) / len(values)
            for name, values in metrics.items()
            if values
        }
        for category, metrics in sorted(category_values.items())
    }
    workspace_host = client.config.host.rstrip("/")
    run_url = (
        f"{workspace_host}/ml/experiments/{memory_eval.EXPERIMENT_ID}/evaluation-runs"
        f"?selectedRunUuid={result.run_id}"
    )
    print(
        json.dumps(
            {
                "phase": "complete",
                "run_id": result.run_id,
                "run_url": run_url,
                "question_count": len(result.result_df),
                "metrics": dict(getattr(result, "metrics", {}) or {}),
                "by_category": by_category,
                "failure_count": len(failures),
                "failures": failures,
                "latency_ms": {
                    "mean": sum(latencies) / len(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None,
                },
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

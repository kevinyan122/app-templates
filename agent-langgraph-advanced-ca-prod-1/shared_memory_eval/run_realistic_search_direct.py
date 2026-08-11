"""Run all realistic-pilot cases against direct managed-memory search."""

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
    from . import memory_eval, realistic_search_eval
    from .run_chbench_smoke import DEFAULT_APP_URL, DEFAULT_MEMORY_STORE
except ImportError:  # pragma: no cover
    import memory_eval  # type: ignore[no-redef]
    import realistic_search_eval  # type: ignore[no-redef]
    from run_chbench_smoke import DEFAULT_APP_URL, DEFAULT_MEMORY_STORE


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--memory-store", default=DEFAULT_MEMORY_STORE)
    parser.add_argument(
        "--dataset-name", default=realistic_search_eval.QUESTION_DATASET_NAME
    )
    parser.add_argument(
        "--corpus-dataset-name", default=realistic_search_eval.CORPUS_DATASET_NAME
    )
    parser.add_argument(
        "--eval-scope", default=realistic_search_eval.DEFAULT_PERSISTENT_SCOPE
    )
    parser.add_argument("--case-id", action="append", default=[])
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
    if args.top_k < 10 or args.workers <= 0:
        raise ValueError("--top-k must be at least 10 and --workers must be positive")

    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(args.workers)
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    client = WorkspaceClient(profile=args.profile)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    memory_eval.configure(
        app_url=DEFAULT_APP_URL,
        memory_store=args.memory_store,
        workspace_client=client,
    )

    records = realistic_search_eval.load_question_records(
        args.dataset_name,
        args.eval_scope,
        top_k=args.top_k,
    )
    selected_ids = set(args.case_id)
    selected = [
        row
        for row in records
        if not selected_ids or row["inputs"]["case_id"] in selected_ids
    ]
    if not selected:
        raise ValueError(f"No cases matched: {sorted(selected_ids)}")

    seeds = realistic_search_eval.load_seed_memories(args.corpus_dataset_name)
    realistic_search_eval.validate_question_corpus_records(selected, seeds)
    before = realistic_search_eval.validate_persistent_entries(args.eval_scope, seeds)
    before_snapshot = before["snapshot"]
    gold_by_case = {
        row["inputs"]["case_id"]: row["expectations"]["gold_memory_id"]
        for row in selected
    }

    print(
        json.dumps(
            {
                "phase": "validated",
                "cases": len(selected),
                "scope": before["full_scope"],
                "entries": before["actual_count"],
                "fingerprint": before["actual_fingerprint"],
                "top_k": args.top_k,
                "workers": args.workers,
            }
        ),
        flush=True,
    )

    result = None
    after = None
    corpus_diff = None
    with mlflow.start_run(
        run_name=f"memory-eval/realistic-search/{realistic_search_eval.PILOT_VERSION}/direct/{len(selected)}q",
        tags={
            "suite": realistic_search_eval.SUITE_NAME,
            "pilot_version": realistic_search_eval.PILOT_VERSION,
            "benchmark_as_of_date": "2026-06-05",
            "case_fingerprint": realistic_search_eval.EXPECTED_CASE_FINGERPRINT,
            "corpus_fingerprint": realistic_search_eval.EXPECTED_CORPUS_FINGERPRINT,
            "runner": "local-user-direct-memory-store",
            "search_mode": "direct-store-endpoint",
            "agent_invoked": "false",
            "judge_enabled": "false",
            "question_count": str(len(selected)),
            "top_k": str(args.top_k),
            "eval_scope": args.eval_scope,
            "question_dataset": args.dataset_name,
            "corpus_dataset": args.corpus_dataset_name,
        },
    ):
        try:
            result = mlflow.genai.evaluate(
                data=selected,
                predict_fn=realistic_search_eval.direct_predict_fn,
                scorers=realistic_search_eval.DIRECT_SEARCH_SCORERS,
            )
        finally:
            after = realistic_search_eval.inspect_persistent_entries(
                args.eval_scope, seeds
            )
            corpus_diff = memory_eval.diff_snapshots(
                before_snapshot, after["snapshot"]
            )
            corpus_changed = any(corpus_diff.values())
            mlflow.log_metric("realistic_search.corpus_changed", float(corpus_changed))
            mlflow.log_dict(
                {
                    "eval_scope": args.eval_scope,
                    "full_scope": before["full_scope"],
                    "expected_fingerprint": before["expected_fingerprint"],
                    "actual_fingerprint": after["actual_fingerprint"],
                    "before_count": before["actual_count"],
                    "after_count": after["actual_count"],
                    "after_exact": after["exact"],
                    "diff": corpus_diff,
                },
                "realistic_search/corpus_validation.json",
            )

    if result is None:
        raise RuntimeError("Direct realistic-search evaluation produced no result")
    if after is None or not after["exact"] or any(corpus_diff.values()):
        raise RuntimeError("Persistent realistic-search corpus changed during evaluation")

    score_names = [scorer.name for scorer in realistic_search_eval.DIRECT_SEARCH_SCORERS]
    by_category: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cases = []
    latencies = []
    for _, row in result.result_df.iterrows():
        request = _object(row.get("request"))
        response = _object(row.get("response"))
        case_id = str(request.get("case_id", ""))
        category = str(request.get("category", "unknown"))
        gold = gold_by_case[case_id]
        rank = realistic_search_eval.gold_rank(
            response, {"gold_memory_id": gold}
        )
        values = {}
        for name in score_names:
            value = row.get(f"{name}/value")
            if value is not None:
                numeric = float(value)
                by_category[category][name].append(numeric)
                values[name] = numeric
        latency = response.get("direct_latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "query": request.get("query"),
                "gold_memory_id": gold,
                "gold_rank": rank,
                "ranked_ids": response.get("direct_ranked_ids", []),
                **values,
            }
        )

    by_category_summary = {
        category: {
            name: sum(values) / len(values)
            for name, values in metrics.items()
            if values
        }
        for category, metrics in sorted(by_category.items())
    }
    failures = [case for case in cases if case["gold_rank"] is None]
    outside_top_1 = [case for case in cases if case["gold_rank"] != 1]
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
                "question_count": len(cases),
                "metrics": dict(getattr(result, "metrics", {}) or {}),
                "by_category": by_category_summary,
                "missing_gold_count": len(failures),
                "not_top_1_count": len(outside_top_1),
                "cases_not_top_1": outside_top_1,
                "latency_ms": {
                    "mean": sum(latencies) / len(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None,
                },
                "corpus_unchanged": True,
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

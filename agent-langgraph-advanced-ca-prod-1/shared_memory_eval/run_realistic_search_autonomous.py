"""Run the realistic-memory pilot through the deployed App autonomously.

Each natural question is sent unchanged in a fresh conversation thread. The
only scorer is an LLM judge over the final answer and that case's
``answer_should`` rubric. The persistent 60-memory fixture is validated before
and after the run and is never seeded or repaired here.
"""

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
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--memory-store", default=DEFAULT_MEMORY_STORE)
    parser.add_argument(
        "--dataset-name", default=realistic_search_eval.QUESTION_DATASET_NAME
    )
    parser.add_argument(
        "--corpus-dataset-name",
        default=realistic_search_eval.CORPUS_DATASET_NAME,
    )
    parser.add_argument(
        "--eval-scope", default=realistic_search_eval.DEFAULT_PERSISTENT_SCOPE
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--judge-model", default="databricks:/databricks-gpt-5-2"
    )
    return parser.parse_args()


def build_scorers(judge_model: str) -> list[Any]:
    """Return the intentionally one-item scorer suite for this track."""

    return [realistic_search_eval.load_autonomous_answer_judge(judge_model)]


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value or {}) if isinstance(value, dict) else {}


def _score(value: Any) -> float | None:
    if value is None or value != value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_label(value: float | None) -> str:
    if value is None:
        return "missing"
    if value >= 0.75:
        return "pass"
    if value >= 0.25:
        return "partial"
    return "fail"


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(args.workers)
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    client = WorkspaceClient(profile=args.profile)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    memory_eval.configure(
        app_url=args.app_url,
        memory_store=args.memory_store,
        workspace_client=client,
    )

    records = realistic_search_eval.load_autonomous_question_records(
        args.dataset_name, args.eval_scope
    )
    selected_ids = set(args.case_id)
    selected = [
        row
        for row in records
        if not selected_ids or row["inputs"]["case_id"] in selected_ids
    ]
    if not selected:
        raise ValueError(f"No cases matched: {sorted(selected_ids)}")
    unknown_ids = selected_ids - {
        str(row["inputs"]["case_id"]) for row in selected
    }
    if unknown_ids:
        raise ValueError(f"Unknown case ids: {sorted(unknown_ids)}")

    seeds = realistic_search_eval.load_seed_memories(args.corpus_dataset_name)
    realistic_search_eval.validate_question_corpus_records(selected, seeds)
    before = realistic_search_eval.validate_persistent_entries(args.eval_scope, seeds)
    before_snapshot = before["snapshot"]
    expectation_by_case = {
        str(row["inputs"]["case_id"]): str(row["expectations"]["answer_should"])
        for row in selected
    }

    scorers = build_scorers(args.judge_model)
    print(
        json.dumps(
            {
                "phase": "validated",
                "mode": "autonomous-final-answer",
                "cases": len(selected),
                "scope": before["full_scope"],
                "entries": before["actual_count"],
                "fingerprint": before["actual_fingerprint"],
                "app_url": args.app_url,
                "query_transform": "none",
                "memory_access": "read-only",
                "scorers": [scorer.name for scorer in scorers],
                "workers": args.workers,
            }
        ),
        flush=True,
    )

    result = None
    after = None
    corpus_diff = None
    with mlflow.start_run(
        run_name=(
            "memory-eval/realistic-search/"
            f"{realistic_search_eval.PILOT_VERSION}/autonomous-answer/{len(selected)}q"
        ),
        tags={
            "suite": realistic_search_eval.SUITE_NAME,
            "pilot_version": realistic_search_eval.PILOT_VERSION,
            "benchmark_as_of_date": "2026-06-05",
            "case_fingerprint": realistic_search_eval.EXPECTED_CASE_FINGERPRINT,
            "corpus_fingerprint": realistic_search_eval.EXPECTED_CORPUS_FINGERPRINT,
            "runner": "local-user-oauth-remote-app",
            "evaluation_track": "autonomous-final-answer",
            "agent_search_mode": "autonomous",
            "query_transform": "none",
            "direct_search_enabled": "false",
            "forced_search_enabled": "false",
            "memory_access": "read-only",
            "memory_scope_mode": "persistent-direct-eval-scope",
            "judge_enabled": "true",
            "judge_names": realistic_search_eval.AUTONOMOUS_ANSWER_JUDGE_NAME,
            "judge_model": args.judge_model,
            "judge_prompt_version": (
                realistic_search_eval.AUTONOMOUS_ANSWER_JUDGE_VERSION
            ),
            "question_count": str(len(selected)),
            "eval_scope": args.eval_scope,
            "app_url": args.app_url,
            "question_dataset": args.dataset_name,
            "corpus_dataset": args.corpus_dataset_name,
        },
    ):
        try:
            result = mlflow.genai.evaluate(
                data=selected,
                predict_fn=realistic_search_eval.autonomous_predict_fn,
                scorers=scorers,
            )
        finally:
            after = realistic_search_eval.inspect_persistent_entries(
                args.eval_scope, seeds
            )
            corpus_diff = memory_eval.diff_snapshots(
                before_snapshot, after["snapshot"]
            )
            corpus_changed = any(corpus_diff.values())
            mlflow.log_metric(
                "realistic_autonomous.corpus_changed", float(corpus_changed)
            )
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
                "realistic_autonomous/corpus_validation.json",
            )

    if result is None:
        raise RuntimeError("Autonomous realistic-memory evaluation produced no result")
    if after is None or not after["exact"] or any((corpus_diff or {}).values()):
        raise RuntimeError(
            "Persistent realistic-memory corpus changed during autonomous evaluation"
        )

    score_name = realistic_search_eval.AUTONOMOUS_ANSWER_JUDGE_NAME
    value_column = f"{score_name}/value"
    rationale_column = f"{score_name}/rationale"
    cases: list[dict[str, Any]] = []
    by_category: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []
    tokens: list[int] = []
    for _, row in result.result_df.iterrows():
        request = _object(row.get("request"))
        response = _object(row.get("response"))
        case_id = str(request.get("case_id", ""))
        category = str(request.get("category", "unknown"))
        value = _score(row.get(value_column))
        if value is not None:
            by_category[category].append(value)
        latency = response.get("app_latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        token_count = response.get("tokens")
        if isinstance(token_count, (int, float)):
            tokens.append(int(token_count))
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "query": request.get("query"),
                "answer_should": expectation_by_case.get(case_id, ""),
                "answer": response.get("answer", ""),
                "score": value,
                "label": _score_label(value),
                "rationale": row.get(rationale_column),
                "tool_calls": [
                    call.get("name")
                    for call in response.get("tool_calls", [])
                    if isinstance(call, dict)
                ],
                "app_latency_ms": latency,
            }
        )

    scored = [case["score"] for case in cases if case["score"] is not None]
    counts = {
        label: sum(case["label"] == label for case in cases)
        for label in ("pass", "partial", "fail", "missing")
    }
    category_summary = {
        category: {
            "mean": sum(values) / len(values),
            "cases": len(values),
        }
        for category, values in sorted(by_category.items())
        if values
    }
    mlflow_client = mlflow.MlflowClient()
    for category, summary in category_summary.items():
        mlflow_client.log_metric(
            result.run_id,
            f"category.{category}.{score_name}.mean",
            summary["mean"],
        )

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
                "judge": score_name,
                "mean_score": sum(scored) / len(scored) if scored else None,
                "counts": counts,
                "by_category": category_summary,
                "cases_below_full_credit": [
                    case for case in cases if case["label"] != "pass"
                ],
                "app_latency_ms": {
                    "mean": sum(latencies) / len(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None,
                },
                "tokens": {
                    "total": sum(tokens),
                    "mean": sum(tokens) / len(tokens) if tokens else None,
                },
                "corpus_unchanged": True,
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

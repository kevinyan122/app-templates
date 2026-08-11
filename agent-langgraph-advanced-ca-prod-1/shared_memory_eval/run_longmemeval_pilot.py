"""Run the LongMemEval-S raw-session pilot against the deployed testing App."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import mlflow  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402

try:
    from . import longmemeval_adapter, longmemeval_eval, memory_eval
except ImportError:  # pragma: no cover
    import longmemeval_adapter  # type: ignore[no-redef]
    import longmemeval_eval  # type: ignore[no-redef]
    import memory_eval  # type: ignore[no-redef]


DEFAULT_APP_URL = (
    "https://agent-langgraph-adv-kevin-1653573648247579.staging.aws.databricksapps.com"
)
DEFAULT_MEMORY_STORE = "kevinyan.default.kevin_test"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--memory-store", default=DEFAULT_MEMORY_STORE)
    parser.add_argument("--dataset-name", default=longmemeval_eval.QUESTION_DATASET_NAME)
    parser.add_argument("--question-id", default="")
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Evaluate all 30 cases instead of one --question-id.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index-timeout-seconds", type=int, default=180)
    parser.add_argument("--include-judge", action="store_true")
    parser.add_argument(
        "--judge-model", default="databricks:/databricks-gpt-5-2"
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    if args.all_cases == bool(args.question_id):
        raise ValueError("Choose exactly one of --all-cases or --question-id")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    print(
        json.dumps({"phase": "connecting", "profile": args.profile}),
        flush=True,
    )
    client = WorkspaceClient(profile=args.profile)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    memory_eval.configure(
        app_url=args.app_url,
        memory_store=args.memory_store,
        index_timeout_s=args.index_timeout_seconds,
        workspace_client=client,
    )
    print(
        json.dumps({"phase": "loading_dataset", "dataset": args.dataset_name}),
        flush=True,
    )
    questions = longmemeval_eval.load_question_records(
        args.dataset_name, top_k=args.top_k
    )
    selected = [
        row
        for row in questions
        if args.all_cases or row["inputs"]["question_id"] == args.question_id
    ]
    if not selected:
        raise ValueError(f"Unknown LongMemEval question id: {args.question_id}")

    print(
        json.dumps(
            {
                "phase": "validating_corpora",
                "cases": len(selected),
                "question_ids": [row["inputs"]["question_id"] for row in selected],
            }
        ),
        flush=True,
    )
    before = {}
    for row in selected:
        inputs = row["inputs"]
        status = longmemeval_eval.validate_persistent_scope(
            inputs["scope_selector"],
            inputs["expected_memory_count"],
            inputs["expected_corpus_fingerprint"],
        )
        before[inputs["case_id"]] = status
    print(
        json.dumps(
            {
                "phase": "validated",
                "cases": len(selected),
                "entries": sum(status["actual_count"] for status in before.values()),
                "question_ids": [row["inputs"]["question_id"] for row in selected],
            }
        ),
        flush=True,
    )

    scorers = list(longmemeval_eval.DETERMINISTIC_SCORERS)
    if args.include_judge:
        scorers.append(longmemeval_eval.load_answer_judge(args.judge_model))
    print(
        json.dumps(
            {
                "phase": "evaluating",
                "cases": len(selected),
                "deterministic_scorers": len(longmemeval_eval.DETERMINISTIC_SCORERS),
                "judge": args.judge_model if args.include_judge else None,
            }
        ),
        flush=True,
    )

    result = None
    after = {}
    validation_artifact = {}
    with mlflow.start_run(
        run_name=f"memory-eval/longmemeval-s/raw/forced-search/{len(selected)}q",
        tags={
            "suite": longmemeval_adapter.SUITE_NAME,
            "adapter_version": longmemeval_adapter.ADAPTER_VERSION,
            "source_commit": longmemeval_adapter.SOURCE_COMMIT,
            "bundle_fingerprint": longmemeval_adapter.EXPECTED_PILOT_BUNDLE_FINGERPRINT,
            "runner": "local-user-oauth-remote-app",
            "agent_search_mode": "explicit-search-memory-instruction",
            "memory_scope_mode": "persistent-per-case-obo-scope",
            "question_count": str(len(selected)),
            "top_k": str(args.top_k),
            "question_dataset": args.dataset_name,
            "judge_enabled": str(args.include_judge).lower(),
            "judge_model": args.judge_model if args.include_judge else "none",
        },
    ):
        try:
            result = mlflow.genai.evaluate(
                data=selected,
                predict_fn=longmemeval_eval.predict_fn,
                scorers=scorers,
            )
        finally:
            for row in selected:
                inputs = row["inputs"]
                case = inputs["case_id"]
                status = longmemeval_eval.inspect_persistent_scope(
                    inputs["scope_selector"],
                    inputs["expected_memory_count"],
                    inputs["expected_corpus_fingerprint"],
                )
                after[case] = status
                diff = memory_eval.diff_snapshots(
                    before[case]["snapshot"], status["snapshot"]
                )
                validation_artifact[case] = {
                    "question_id": inputs["question_id"],
                    "scope_selector": inputs["scope_selector"],
                    "full_scope": status["full_scope"],
                    "before_count": before[case]["actual_count"],
                    "after_count": status["actual_count"],
                    "expected_fingerprint": status["expected_fingerprint"],
                    "actual_fingerprint": status["actual_fingerprint"],
                    "exact": status["exact"],
                    "diff": diff,
                }
            corpus_changed = any(not status["exact"] for status in after.values()) or any(
                any(artifact["diff"][kind] for kind in ("created", "updated", "deleted"))
                for artifact in validation_artifact.values()
            )
            mlflow.log_metric("longmemeval.corpus_changed", float(corpus_changed))
            mlflow.log_dict(
                {
                    "bundle_fingerprint": longmemeval_adapter.EXPECTED_PILOT_BUNDLE_FINGERPRINT,
                    "cases": validation_artifact,
                },
                "longmemeval/corpus_validation.json",
            )

    if result is None:
        raise RuntimeError("LongMemEval evaluation did not produce a result")
    changed = [case for case, status in after.items() if not status["exact"]]
    if changed:
        raise RuntimeError(f"LongMemEval persistent corpora changed: {changed}")
    metric_values = dict(getattr(result, "metrics", {}) or {})
    case_result = None
    if len(result.result_df) == 1:
        row = result.result_df.iloc[0]
        scorer_names = [scorer.name for scorer in scorers]
        metric_values = {
            name: (
                row[f"{name}/value"].item()
                if hasattr(row[f"{name}/value"], "item")
                else row[f"{name}/value"]
            )
            for name in scorer_names
            if f"{name}/value" in result.result_df.columns
        }
        response = row.get("response")
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                pass
        if isinstance(response, dict):
            case_result = {
                "question_id": response.get("question_id"),
                "answer": response.get("answer"),
                "agent_search_queries": [
                    search.get("query")
                    for search in response.get("agent_searches", [])
                ],
                "direct_ranked_ids": response.get("direct_ranked_ids", []),
                "agent_ranked_ids": response.get("agent_ranked_ids", []),
                "mutation_tools": [
                    call.get("name") for call in response.get("mutation_calls", [])
                ],
                "direct_latency_ms": response.get("direct_latency_ms"),
                "agent_latency_ms": response.get("agent_latency_ms"),
            }
    print(
        json.dumps(
            {
                "phase": "evaluated",
                "run_id": result.run_id,
                "question_count": len(result.result_df),
                "metrics": metric_values,
                **({"case_result": case_result} if case_result is not None else {}),
                "corpus": {
                    "cases": len(after),
                    "entries": sum(status["actual_count"] for status in after.values()),
                    "unchanged": True,
                },
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

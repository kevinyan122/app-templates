"""Run one authenticated CH-Bench question against a persistent frozen corpus.

Seed the corpus separately with :mod:`seed_chbench_corpus`. This evaluator only
validates and reads the versioned scope; it never seeds, repairs, or deletes it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import mlflow  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402

try:  # Package import locally; top-level import from a Workspace folder.
    from . import chbench_eval, memory_eval
except ImportError:  # pragma: no cover
    import chbench_eval  # type: ignore[no-redef]
    import memory_eval  # type: ignore[no-redef]


DEFAULT_APP_URL = (
    "https://agent-langgraph-adv-kevin-1653573648247579.staging.aws.databricksapps.com"
)
DEFAULT_MEMORY_STORE = "kevinyan.default.kevin_test"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", default="dev-q2")
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="Evaluate all 35 questions instead of --question-id.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--memory-store", default=DEFAULT_MEMORY_STORE)
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--eval-scope", default=chbench_eval.DEFAULT_PERSISTENT_SCOPE)
    parser.add_argument("--dataset-name", default=chbench_eval.DATASET_NAME)
    parser.add_argument(
        "--corpus-dataset-name",
        default=chbench_eval.CORPUS_DATASET_NAME,
    )
    parser.add_argument("--index-timeout-seconds", type=int, default=180)
    parser.add_argument("--include-judges", action="store_true")
    parser.add_argument(
        "--judge-model",
        default="databricks:/databricks-gpt-5-2",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    client = WorkspaceClient(profile=args.profile)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    memory_eval.configure(
        app_url=args.app_url,
        memory_store=args.memory_store,
        index_timeout_s=args.index_timeout_seconds,
        workspace_client=client,
    )

    eval_scope = args.eval_scope
    full_scope = memory_eval._full_eval_scope(eval_scope)
    records = chbench_eval.load_records_from_dataset(
        args.dataset_name,
        eval_scope,
        top_k=args.top_k,
    )
    seed_memories = chbench_eval.load_seed_memories_from_dataset(
        args.corpus_dataset_name
    )
    chbench_eval.validate_question_corpus_records(records, seed_memories)
    selected = (
        records
        if args.all_questions
        else chbench_eval.filter_records(records, {args.question_id}, set())
    )
    question_label = "all" if args.all_questions else args.question_id
    before_status = chbench_eval.validate_persistent_entries(
        eval_scope, seed_memories
    )
    before_snapshot = before_status["snapshot"]
    print(
        json.dumps(
            {
                "phase": "configured",
                "scope": full_scope,
                "entries": before_status["actual_count"],
                "fingerprint": before_status["fingerprint"],
                "question_count": len(selected),
                "question_ids": question_label,
            }
        ),
        flush=True,
    )

    print(json.dumps({"phase": "validated", "scope_entries": len(before_snapshot)}), flush=True)

    scorers = list(chbench_eval.DETERMINISTIC_SCORERS)
    if args.include_judges:
        scorers.extend(chbench_eval.load_answer_judges(args.judge_model))

    result = None
    corpus_diff = None
    after_status = None
    with mlflow.start_run(
        run_name=f"memory-eval/chbench/forced-search/{len(selected)}q/{eval_scope}",
        tags={
            "suite": "chbench-contextheavy",
            "source_commit": chbench_eval.SOURCE_COMMIT,
            "corpus_fingerprint": before_status["fingerprint"],
            "runner": "local-user-oauth-remote-app",
            "agent_search_mode": "explicit-search-memory-instruction",
            "memory_scope_mode": "persistent-direct-eval-scope",
            "eval_scope": eval_scope,
            "question_count": str(len(selected)),
            "question_ids": question_label,
            "top_k": str(args.top_k),
            "question_dataset": args.dataset_name,
            "corpus_dataset": args.corpus_dataset_name,
            "judges_enabled": str(args.include_judges).lower(),
            "judge_model": args.judge_model if args.include_judges else "none",
        },
    ):
        try:
            result = mlflow.genai.evaluate(
                data=selected,
                predict_fn=chbench_eval.predict_fn,
                scorers=scorers,
            )
        finally:
            after_status = chbench_eval.inspect_persistent_entries(
                eval_scope, seed_memories
            )
            corpus_diff = memory_eval.diff_snapshots(
                before_snapshot, after_status["snapshot"]
            )
            corpus_changed = any(
                corpus_diff[kind] for kind in ("created", "updated", "deleted")
            )
            mlflow.log_metric("chbench.corpus_changed", float(corpus_changed))
            mlflow.log_dict(
                {
                    "eval_scope": eval_scope,
                    "full_scope": full_scope,
                    "fingerprint": before_status["fingerprint"],
                    "actual_fingerprint": after_status["actual_fingerprint"],
                    "before_count": before_status["actual_count"],
                    "after_count": after_status["actual_count"],
                    "after_exact": after_status["exact"],
                    "diff": corpus_diff,
                },
                "chbench/corpus_validation.json",
            )

    if result is None:
        raise RuntimeError("CH-Bench evaluation did not produce a result")
    if after_status is None or not after_status["exact"] or any(
        corpus_diff[kind] for kind in ("created", "updated", "deleted")
    ):
        raise RuntimeError(
            "Persistent CH-Bench corpus changed during evaluation. "
            "Inspect chbench/corpus_validation.json and repair it explicitly with "
            "seed_chbench_corpus.py --replace."
        )

    metric_values = dict(getattr(result, "metrics", {}) or {})
    response = None
    if len(result.result_df) == 1:
        row = result.result_df.iloc[0]
        metric_values = {
            column.removesuffix("/value"): row[column]
            for column in result.result_df.columns
            if column.endswith("/value")
        }
        response = row.get("response")
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                pass
    print(
        json.dumps(
            {
                "phase": "evaluated",
                "run_id": result.run_id,
                "question_count": len(result.result_df),
                "metrics": metric_values,
                **({"response": response} if response is not None else {}),
                "corpus": {
                    "entries": after_status["actual_count"],
                    "fingerprint": after_status["actual_fingerprint"],
                    "unchanged": True,
                },
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

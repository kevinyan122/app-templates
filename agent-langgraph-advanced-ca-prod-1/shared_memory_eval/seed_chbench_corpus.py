"""Seed or validate the frozen CH-Bench corpus in one persistent scope."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from databricks.sdk import WorkspaceClient  # noqa: E402

try:
    from . import chbench_eval, memory_eval
    from .run_chbench_smoke import DEFAULT_APP_URL, DEFAULT_MEMORY_STORE
except ImportError:  # pragma: no cover
    import chbench_eval  # type: ignore[no-redef]
    import memory_eval  # type: ignore[no-redef]
    from run_chbench_smoke import DEFAULT_APP_URL, DEFAULT_MEMORY_STORE


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--memory-store", default=DEFAULT_MEMORY_STORE)
    parser.add_argument("--eval-scope", default=chbench_eval.DEFAULT_PERSISTENT_SCOPE)
    parser.add_argument(
        "--corpus-dataset-name",
        default=chbench_eval.CORPUS_DATASET_NAME,
    )
    parser.add_argument("--index-timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace a non-empty mismatched dedicated scope. Exact corpora remain a no-op.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    client = WorkspaceClient(profile=args.profile)
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    memory_eval.configure(
        app_url=args.app_url,
        memory_store=args.memory_store,
        index_timeout_s=args.index_timeout_seconds,
        workspace_client=client,
    )
    seed_memories = chbench_eval.load_seed_memories_from_dataset(
        args.corpus_dataset_name
    )
    status = chbench_eval.seed_persistent_entries(
        args.eval_scope,
        seed_memories,
        replace=args.replace,
        index_timeout_s=args.index_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "action": status["action"],
                "eval_scope": status["eval_scope"],
                "full_scope": status["full_scope"],
                "entries": status["actual_count"],
                "fingerprint": status["fingerprint"],
                "corpus_dataset": args.corpus_dataset_name,
                "source_commit": chbench_eval.SOURCE_COMMIT,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

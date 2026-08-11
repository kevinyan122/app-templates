"""Seed or validate the frozen realistic-search pilot scope."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

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
        "--eval-scope", default=realistic_search_eval.DEFAULT_PERSISTENT_SCOPE
    )
    parser.add_argument(
        "--corpus-dataset-name", default=realistic_search_eval.CORPUS_DATASET_NAME
    )
    parser.add_argument("--index-timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace a non-empty drifted dedicated scope. Exact scopes remain a no-op.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    client = WorkspaceClient(profile=args.profile)
    memory_eval.configure(
        app_url=DEFAULT_APP_URL,
        memory_store=args.memory_store,
        index_timeout_s=args.index_timeout_seconds,
        workspace_client=client,
    )
    seeds = realistic_search_eval.load_seed_memories(args.corpus_dataset_name)
    status = realistic_search_eval.seed_persistent_entries(
        args.eval_scope,
        seeds,
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
                "fingerprint": status["actual_fingerprint"],
                "corpus_dataset": args.corpus_dataset_name,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

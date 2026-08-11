"""Explicitly delete the persistent CH-Bench scope.

Evaluation runners never call this command. The exact selector must be repeated
through ``--confirm-scope`` to avoid deleting the wrong persistent fixture.
"""

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
    parser.add_argument("--confirm-scope", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    if args.confirm_scope != args.eval_scope:
        raise ValueError("--confirm-scope must exactly match --eval-scope")
    client = WorkspaceClient(profile=args.profile)
    memory_eval.configure(
        app_url=args.app_url,
        memory_store=args.memory_store,
        workspace_client=client,
    )
    full_scope = memory_eval._full_eval_scope(args.eval_scope)
    before = memory_eval.snapshot_scope(full_scope)
    deleted = memory_eval.cleanup_scope(full_scope)
    remaining = memory_eval.snapshot_scope(full_scope)
    print(
        json.dumps(
            {
                "eval_scope": args.eval_scope,
                "full_scope": full_scope,
                "before": len(before),
                "deleted": deleted,
                "remaining": len(remaining),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

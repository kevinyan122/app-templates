"""Seed or validate the 30 persistent LongMemEval-S pilot scopes."""

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
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--memory-store", default=DEFAULT_MEMORY_STORE)
    parser.add_argument("--dataset-name", default=longmemeval_eval.QUESTION_DATASET_NAME)
    parser.add_argument(
        "--source",
        default=os.getenv("LONGMEMEVAL_SOURCE", ""),
        help="Path to the pinned longmemeval_s_cleaned.json source file.",
    )
    parser.add_argument("--question-id", default="")
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Seed all 30 case scopes instead of one --question-id.",
    )
    parser.add_argument("--index-timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--search-check",
        choices=("none", "gold", "all"),
        default="gold",
        help=(
            "Search-index validation after durable fingerprinting: none, only gold "
            "sessions (default), or every entry."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace non-empty mismatched dedicated scopes. Exact scopes remain no-ops.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    if args.all_cases == bool(args.question_id):
        raise ValueError("Choose exactly one of --all-cases or --question-id")
    if not args.source:
        raise ValueError(
            "Pass --source or set LONGMEMEVAL_SOURCE. "
            f"The pinned source is {longmemeval_adapter.SOURCE_URL}"
        )
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
    questions = longmemeval_eval.load_question_records(args.dataset_name)
    print(
        json.dumps({"phase": "loading_source", "source": args.source}),
        flush=True,
    )
    source = longmemeval_adapter.load_source(args.source)
    bundle = longmemeval_adapter.build_pilot_bundle(source)
    longmemeval_adapter.validate_official_pilot_bundle(bundle)
    corpus_by_case: dict[str, list[dict[str, str]]] = {}
    for row in bundle["corpus"]:
        inputs = row["inputs"]
        corpus_by_case.setdefault(inputs["case_id"], []).append(
            {
                "path": inputs["path"],
                "description": inputs["description"],
                "contents": inputs["contents"],
            }
        )
    selected = [
        row
        for row in questions
        if args.all_cases or row["inputs"]["question_id"] == args.question_id
    ]
    if not selected:
        raise ValueError(f"Unknown LongMemEval question id: {args.question_id}")

    statuses = []
    for index, row in enumerate(selected, start=1):
        inputs = row["inputs"]
        print(
            json.dumps(
                {
                    "phase": "processing",
                    "index": index,
                    "total": len(selected),
                    "question_id": inputs["question_id"],
                }
            ),
            flush=True,
        )
        case_id = inputs["case_id"]
        entries = corpus_by_case[case_id]
        actual_fingerprint = longmemeval_adapter.corpus_fingerprint(entries)
        if len(entries) != int(inputs["expected_memory_count"]) or (
            actual_fingerprint != inputs["expected_corpus_fingerprint"]
        ):
            raise RuntimeError(
                f"Local source corpus for {inputs['question_id']} does not match "
                "the published question dataset"
            )
        relevant_ids = set(
            json.loads(row.get("expectations", {}).get("relevant_ids_json", "[]"))
        )
        if args.search_check == "none":
            search_check_entries = []
        elif args.search_check == "all":
            search_check_entries = entries
        else:
            search_check_entries = [
                entry
                for entry in entries
                if longmemeval_adapter.memory_id_from_path(entry["path"])
                in relevant_ids
            ]
            if len(search_check_entries) != len(relevant_ids):
                raise RuntimeError(
                    f"Could not map every gold memory for {inputs['question_id']} "
                    "to its local source entry"
                )
        status = longmemeval_eval.seed_persistent_case(
            inputs["scope_selector"],
            entries,
            replace=args.replace,
            index_timeout_s=args.index_timeout_seconds,
            search_check_entries=search_check_entries,
        )
        summary = {
            "index": index,
            "total": len(selected),
            "question_id": inputs["question_id"],
            "case_id": case_id,
            "scope_selector": inputs["scope_selector"],
            "full_scope": status["full_scope"],
            "action": status["action"],
            "entries": status["actual_count"],
            "fingerprint": status["actual_fingerprint"],
            "search_check": args.search_check,
            "search_check_entries": status["search_check_count"],
        }
        statuses.append(summary)
        print(json.dumps(summary), flush=True)
    print(
        json.dumps(
            {
                "phase": "complete",
                "cases": len(statuses),
                "entries": sum(status["entries"] for status in statuses),
                "actions": {
                    action: sum(status["action"] == action for status in statuses)
                    for action in sorted({status["action"] for status in statuses})
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

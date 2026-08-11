"""Publish the pinned LongMemEval-S pilot question/gold dataset.

Raw transcripts are intentionally not published as one managed dataset: their
table synchronization exceeds the platform's 4 MB gRPC message limit. The
one-time seeder transforms the pinned source directly, while this compact
dataset carries each case's expected entry count and corpus fingerprint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import mlflow  # noqa: E402

try:  # Package import locally; top-level import from a Workspace folder.
    from . import longmemeval_adapter, memory_eval
    from .sync_chbench_dataset import _sync_records
except ImportError:  # pragma: no cover
    import longmemeval_adapter  # type: ignore[no-redef]
    import memory_eval  # type: ignore[no-redef]
    from sync_chbench_dataset import _sync_records  # type: ignore[no-redef]


QUESTION_DATASET_NAME = "kevinyan.default.longmemeval_s_raw_pilot_v1"
def sync_datasets(
    source_path: str | Path,
    *,
    question_dataset_name: str = QUESTION_DATASET_NAME,
    attempts: int = 4,
) -> dict[str, Any]:
    source = longmemeval_adapter.load_source(source_path)
    bundle = longmemeval_adapter.build_pilot_bundle(source)
    longmemeval_adapter.validate_official_pilot_bundle(bundle)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    question_result = _sync_records(
        question_dataset_name,
        bundle["questions"],
        "question_id",
        attempts=attempts,
    )
    return {
        "questions": question_result,
        "metadata": bundle["metadata"],
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument(
        "--source",
        default=os.getenv("LONGMEMEVAL_SOURCE", ""),
        help="Path to the pinned longmemeval_s_cleaned.json file.",
    )
    parser.add_argument("--dataset-name", default=QUESTION_DATASET_NAME)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    if not args.source:
        raise ValueError(
            "Pass --source or set LONGMEMEVAL_SOURCE. "
            f"The pinned source is {longmemeval_adapter.SOURCE_URL}"
        )
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    result = sync_datasets(
        args.source,
        question_dataset_name=args.dataset_name,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

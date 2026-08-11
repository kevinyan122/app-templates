"""Publish the frozen realistic-search pilot as two MLflow datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import mlflow  # noqa: E402

try:
    from . import memory_eval, realistic_search_eval
    from .sync_chbench_dataset import _sync_records
except ImportError:  # pragma: no cover
    import memory_eval  # type: ignore[no-redef]
    import realistic_search_eval  # type: ignore[no-redef]
    from sync_chbench_dataset import _sync_records  # type: ignore[no-redef]


def sync_datasets(
    question_dataset_name: str = realistic_search_eval.QUESTION_DATASET_NAME,
    corpus_dataset_name: str = realistic_search_eval.CORPUS_DATASET_NAME,
) -> dict:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    pilot = realistic_search_eval.load_pilot()
    questions = _sync_records(
        question_dataset_name,
        realistic_search_eval.build_question_dataset_records(pilot),
        "case_id",
    )
    corpus = _sync_records(
        corpus_dataset_name,
        realistic_search_eval.build_corpus_dataset_records(pilot),
        "memory_id",
    )
    loaded_questions = realistic_search_eval.load_question_records(
        question_dataset_name
    )
    loaded_memories = realistic_search_eval.load_seed_memories(corpus_dataset_name)
    realistic_search_eval.validate_question_corpus_records(
        loaded_questions, loaded_memories
    )
    return {
        "questions": questions,
        "corpus": corpus,
        "case_fingerprint": realistic_search_eval.EXPECTED_CASE_FINGERPRINT,
        "corpus_fingerprint": realistic_search_eval.EXPECTED_CORPUS_FINGERPRINT,
        "validated_question_rows": len(loaded_questions),
        "validated_memory_rows": len(loaded_memories),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument(
        "--dataset-name", default=realistic_search_eval.QUESTION_DATASET_NAME
    )
    parser.add_argument(
        "--corpus-dataset-name", default=realistic_search_eval.CORPUS_DATASET_NAME
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    result = sync_datasets(args.dataset_name, args.corpus_dataset_name)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Publish pinned CH-Bench questions and memories to versioned MLflow datasets.

The vendored JSON is used only by this publisher. Runtime notebooks load the two
managed datasets and never need a Workspace copy of the raw benchmark files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import mlflow  # noqa: E402
from databricks.sdk.errors import NotFound  # noqa: E402
from mlflow.genai import datasets  # noqa: E402

try:  # Package import locally; top-level import from a Workspace folder.
    from . import chbench_eval, memory_eval
except ImportError:  # pragma: no cover
    import chbench_eval  # type: ignore[no-redef]
    import memory_eval  # type: ignore[no-redef]


def _record_inputs(managed_record) -> dict[str, Any]:
    return {item.key: item.value for item in managed_record.inputs}


def _managed_records(dataset) -> list[Any]:
    from databricks.rag_eval.datasets.entities import _get_client

    return list(_get_client().list_dataset_records(dataset.dataset_id))


def _missing_records(
    dataset,
    records: list[dict[str, Any]],
    id_key: str,
) -> list[dict[str, Any]]:
    existing_ids = {
        _record_inputs(managed_record).get(id_key)
        for managed_record in _managed_records(dataset)
    }
    return [
        record
        for record in records
        if record["inputs"][id_key] not in existing_ids
    ]


def _replace_changed_records(
    dataset,
    records: list[dict[str, Any]],
    *,
    id_key: str,
    attempts: int,
) -> int:
    """Replace full rows so edits cannot leave stale expectations or tags."""

    from databricks.rag_eval.datasets import rest_entities
    from databricks.rag_eval.datasets.entities import _get_client

    desired_by_id = {record["inputs"][id_key]: record for record in records}
    client = _get_client()
    replaced = 0
    seen: set[str] = set()
    for managed_record in client.list_dataset_records(dataset.dataset_id):
        inputs = _record_inputs(managed_record)
        record_id = inputs.get(id_key)
        desired = desired_by_id.get(record_id)
        if desired is None:
            continue
        if record_id in seen:
            raise RuntimeError(f"Duplicate managed-dataset row for {record_id}")
        seen.add(record_id)
        current_expectations = {
            key: expectation.value
            for key, expectation in (managed_record.expectations or {}).items()
        }
        if (
            inputs == desired["inputs"]
            and current_expectations == desired.get("expectations", {})
            and managed_record.tags == desired["tags"]
        ):
            continue

        managed_record.inputs = [
            rest_entities.Input(key=key, value=value)
            for key, value in desired["inputs"].items()
        ]
        managed_record.expectations = {
            key: rest_entities.ExpectationValue(value=value)
            for key, value in desired.get("expectations", {}).items()
        }
        managed_record.tags = dict(desired["tags"])
        for attempt in range(1, attempts + 1):
            try:
                client.update_dataset_record(
                    dataset.dataset_id,
                    managed_record,
                    "inputs,expectations,tags",
                )
                break
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(5 * attempt)
        replaced += 1

    missing = set(desired_by_id) - seen
    if missing:
        raise RuntimeError(
            "Managed dataset is missing questions after merge: "
            + ", ".join(sorted(missing))
        )
    if replaced:
        client.sync_dataset_to_uc(dataset.dataset_id, dataset.name)
    return replaced


def _validate_exact_ids(
    dataset,
    records: list[dict[str, Any]],
    id_key: str,
) -> None:
    desired_ids = {record["inputs"][id_key] for record in records}
    actual_ids = [
        _record_inputs(managed_record).get(id_key)
        for managed_record in _managed_records(dataset)
    ]
    duplicates = sorted(
        question_id
        for question_id in set(actual_ids)
        if actual_ids.count(question_id) > 1
    )
    unexpected = sorted(set(actual_ids) - desired_ids, key=str)
    missing = sorted(desired_ids - set(actual_ids), key=str)
    if duplicates or unexpected or missing:
        raise RuntimeError(
            f"Managed dataset {id_key} values are not exact: "
            f"duplicates={duplicates}, unexpected={unexpected}, missing={missing}"
        )


def _sync_records(
    dataset_name: str,
    records: list[dict[str, Any]],
    id_key: str,
    *,
    attempts: int = 4,
) -> dict[str, Any]:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            try:
                dataset = datasets.get_dataset(name=dataset_name)
                action = "found"
            except NotFound:
                dataset = datasets.create_dataset(
                    name=dataset_name,
                    experiment_id=memory_eval.EXPERIMENT_ID,
                )
                action = "created"

            missing = _missing_records(dataset, records, id_key)
            if missing:
                dataset.merge_records(missing)
            replaced = _replace_changed_records(
                dataset,
                records,
                id_key=id_key,
                attempts=attempts,
            )
            _validate_exact_ids(dataset, records, id_key)
            return {
                "action": action,
                "dataset_name": dataset_name,
                "dataset_id": dataset.dataset_id,
                "rows": len(records),
                "inserted": len(missing),
                "replaced": replaced,
                "id_key": id_key,
            }
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(10)
    raise RuntimeError(
        f"Could not publish {dataset_name} after {attempts} attempts"
    ) from last_error


def sync_datasets(
    question_dataset_name: str = chbench_eval.DATASET_NAME,
    corpus_dataset_name: str = chbench_eval.CORPUS_DATASET_NAME,
    *,
    attempts: int = 4,
) -> dict[str, Any]:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
    suite = chbench_eval.load_suite()
    question_result = _sync_records(
        question_dataset_name,
        chbench_eval.build_dataset_records(suite),
        "question_id",
        attempts=attempts,
    )
    corpus_result = _sync_records(
        corpus_dataset_name,
        chbench_eval.build_corpus_dataset_records(suite),
        "memory_id",
        attempts=attempts,
    )
    loaded_questions = chbench_eval.load_records_from_dataset(
        question_dataset_name,
        chbench_eval.DEFAULT_PERSISTENT_SCOPE,
    )
    loaded_memories = chbench_eval.load_seed_memories_from_dataset(corpus_dataset_name)
    chbench_eval.validate_question_corpus_records(loaded_questions, loaded_memories)
    return {
        "questions": question_result,
        "corpus": corpus_result,
        "source_commit": chbench_eval.SOURCE_COMMIT,
        "corpus_fingerprint": chbench_eval.EXPECTED_CORPUS_FINGERPRINT,
        "validated_question_rows": len(loaded_questions),
        "validated_memory_rows": len(loaded_memories),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--dataset-name", default=chbench_eval.DATASET_NAME)
    parser.add_argument(
        "--corpus-dataset-name",
        default=chbench_eval.CORPUS_DATASET_NAME,
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    result = sync_datasets(args.dataset_name, args.corpus_dataset_name)
    print(result, flush=True)


if __name__ == "__main__":
    main()

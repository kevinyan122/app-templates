"""Publish the local managed-memory scenarios to an MLflow evaluation dataset.

The readable Python case file remains the authoring source of truth. Scenario turns
and seed memories are published as native structured inputs; per-turn requirements
are published as scalar expectation fields and each record gets a category tag.

Run: uv run python sync_eval_dataset.py
"""

from __future__ import annotations

import time
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

import mlflow  # noqa: E402
from databricks.sdk.errors import NotFound  # noqa: E402
from mlflow.genai import datasets  # noqa: E402

from eval_memory import DATASET_NAME, EXPERIMENT_ID, build_records  # noqa: E402


def _replace_managed_records(
    dataset, records: list[dict[str, Any]], attempts: int = 4
) -> int:
    """Replace, rather than merge, inputs, expectations, and tags on managed records.

    The public managed-dataset merge API matches records by their full inputs and
    preserves old fields. Direct record updates let the Python case file remain the
    source of truth when a scenario's turns, seed memories, expectations, or tags change.
    """

    from databricks.rag_eval.datasets import rest_entities
    from databricks.rag_eval.datasets.entities import _get_client

    desired_by_id = {record["inputs"]["scenario_id"]: record for record in records}
    client = _get_client()
    existing = client.list_dataset_records(dataset.dataset_id)
    replaced = 0
    seen: set[str] = set()
    for managed_record in existing:
        inputs = {item.key: item.value for item in managed_record.inputs}
        scenario_id = inputs.get("scenario_id")
        desired = desired_by_id.get(scenario_id)
        if desired is None:
            continue
        if scenario_id in seen:
            raise RuntimeError(f"Duplicate managed-dataset row for {scenario_id}")
        seen.add(scenario_id)
        desired_inputs = [
            rest_entities.Input(key=key, value=value)
            for key, value in desired["inputs"].items()
        ]
        desired_expectations = {
            key: rest_entities.ExpectationValue(value=value)
            for key, value in desired["expectations"].items()
        }
        current_expectations = {
            key: expectation.value
            for key, expectation in (managed_record.expectations or {}).items()
        }
        if (
            inputs == desired["inputs"]
            and current_expectations == desired["expectations"]
            and managed_record.tags == desired["tags"]
        ):
            continue

        managed_record.inputs = desired_inputs
        managed_record.expectations = desired_expectations
        managed_record.tags = dict(desired["tags"])
        for attempt in range(1, attempts + 1):
            try:
                client.update_dataset_record(
                    dataset.dataset_id,
                    managed_record,
                    "inputs,expectations,tags",
                )
                break
            except Exception as exc:
                if attempt == attempts:
                    raise
                print(
                    f"update {scenario_id} attempt {attempt}/{attempts} failed: "
                    f"{type(exc).__name__}: {exc}; retrying",
                    flush=True,
                )
                time.sleep(5 * attempt)
        replaced += 1
        print(f"replaced inputs, expectations, and tags: {scenario_id}", flush=True)

    missing = set(desired_by_id) - seen
    if missing:
        raise RuntimeError(
            "Managed dataset is missing scenarios after merge: " + ", ".join(sorted(missing))
        )
    if replaced:
        client.sync_dataset_to_uc(dataset.dataset_id, dataset.name)
    return replaced


def _missing_records(dataset, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only local scenarios that do not exist in the managed dataset."""

    from databricks.rag_eval.datasets.entities import _get_client

    existing_ids = {
        inputs.get("scenario_id")
        for managed_record in _get_client().list_dataset_records(dataset.dataset_id)
        for inputs in ({item.key: item.value for item in managed_record.inputs},)
    }
    return [record for record in records if record["inputs"]["scenario_id"] not in existing_ids]


def sync_dataset(dataset_name: str = DATASET_NAME, attempts: int = 4) -> None:
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
    records = build_records()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            try:
                dataset = datasets.get_dataset(name=dataset_name)
                print(f"found existing dataset {dataset_name}", flush=True)
            except NotFound:
                dataset = datasets.create_dataset(
                    name=dataset_name,
                    experiment_id=EXPERIMENT_ID,
                )
                print(f"created dataset {dataset_name}", flush=True)

            missing = _missing_records(dataset, records)
            if missing:
                dataset.merge_records(missing)
                print(f"inserted {len(missing)} new scenarios", flush=True)
            replaced = _replace_managed_records(dataset, records)
            print(
                f"published {len(records)} scenarios to {dataset_name}; "
                f"replaced {replaced} changed records",
                flush=True,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"dataset sync attempt {attempt}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}; retrying",
                flush=True,
            )
            # A timed-out create may still finish in the service. The next pass
            # deliberately tries get_dataset before attempting another create.
            time.sleep(10)
    raise RuntimeError(f"Could not publish {dataset_name} after {attempts} attempts") from last_error


if __name__ == "__main__":
    sync_dataset()

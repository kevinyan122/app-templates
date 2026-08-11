"""Run managed-memory evaluations against a deployed Databricks App.

The dataset contains one row per scenario. ``predict_fn`` executes that scenario's
turns sequentially in one isolated memory scope, rotating the conversation thread
when requested. Expectations are consumed only by scorers and judges.

This module is designed to live beside the Shared Databricks evaluation notebook.
It has no dependency on the agent source tree: predictions go through the deployed
App's ``/invocations`` endpoint and verification goes directly to the evaluation
memory store.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
import os
import re
import subprocess
import time
import uuid
from functools import lru_cache
from typing import Any

import mlflow  # noqa: E402
import requests  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.errors import DatabricksError  # noqa: E402
from mlflow.entities import Feedback  # noqa: E402
from mlflow.genai.scorers import get_scorer, scorer  # noqa: E402


logger = logging.getLogger(__name__)

EXPERIMENT_ID = "2879499460172053"
DATASET_NAME = "kevinyan.default.memory_eval_scenarios"
INDEX_POLL_INTERVAL_S = 5
INDEX_TIMEOUT_S = 180
JUDGE_RETRY_ATTEMPTS = 3
JUDGE_RETRY_INTERVAL_S = 2
HTTP_TIMEOUT_S = 300
STORE_IO_MAX_WORKERS = max(1, int(os.getenv("MEMORY_EVAL_STORE_WORKERS", "8")))
MEMORY_MUTATION_TOOLS = {"save_memory", "update_memory", "delete_memory"}
MUTATION_SUCCESS_PREFIXES = {
    "save_memory": "Saved memory at ",
    "update_memory": "Updated ",
    "delete_memory": "Deleted ",
}
RUN_TOKEN = uuid.uuid4().hex[:10]
EVAL_SCOPE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
APP_URL = ""
MEMORY_STORE = ""
_workspace_client: WorkspaceClient | None = None


def configure(
    *,
    app_url: str,
    memory_store: str,
    index_timeout_s: int = INDEX_TIMEOUT_S,
    workspace_client: WorkspaceClient | None = None,
) -> dict[str, str]:
    """Configure the remote runner and return its target summary."""

    global APP_URL, MEMORY_STORE, INDEX_TIMEOUT_S, _workspace_client
    APP_URL = app_url.rstrip("/")
    if APP_URL.endswith("/invocations"):
        APP_URL = APP_URL[: -len("/invocations")]
    MEMORY_STORE = memory_store.strip()
    INDEX_TIMEOUT_S = int(index_timeout_s)
    if not APP_URL:
        raise ValueError("app_url cannot be empty")
    if not MEMORY_STORE:
        raise ValueError("memory_store cannot be empty")
    _workspace_client = workspace_client or WorkspaceClient()
    return {
        "app_url": APP_URL,
        "memory_store": MEMORY_STORE,
    }


def _ws() -> WorkspaceClient:
    if _workspace_client is None:
        raise RuntimeError("Call memory_eval.configure(...) before running an evaluation.")
    return _workspace_client


def _full_eval_scope(scope: str) -> str:
    """Validate and return the exact scope sent to the evaluation App."""

    scope = str(scope)
    if not EVAL_SCOPE_PATTERN.fullmatch(scope):
        raise ValueError(
            "eval_scope must be 1-128 characters using only letters, numbers, "
            "'.', '_', ':', or '-'."
        )
    return scope


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------


def scenario_to_record(scenario: dict[str, Any]) -> dict[str, Any]:
    """Convert the readable local schema to one MLflow evaluation record.

    Runtime inputs stay as native lists and dictionaries so the scenario is
    readable in the MLflow dataset. Per-turn expectations remain separate from
    predictor inputs and are published as short, scalar fields that render cleanly
    in the MLflow UI.
    """

    turn_expectations = [turn["expect"] for turn in scenario["turns"]]
    return {
        "inputs": {
            "scenario_id": scenario["id"],
            "category": scenario["category"],
            "turns": [
                {
                    "message": turn["message"],
                    "new_session": bool(turn.get("new_session", False)),
                }
                for turn in scenario["turns"]
            ],
            "initial_memories": [
                {
                    "path": memory["path"],
                    "description": memory["description"],
                    "contents": memory.get("contents", ""),
                }
                for memory in scenario.get("initial_memories", [])
            ],
        },
        "expectations": expectation_fields(turn_expectations),
        # Category is also an input because the runner filters and reports by it.
        # The record tag makes the same category visible and filterable in MLflow.
        "tags": {"category": scenario["category"]},
    }


EXPECTATION_FIELD_NAMES = {
    "write",
    "search",
    "search_result_should_include",
    "memory_should",
    "answer_should",
}


def expectation_fields(turn_expectations: list[dict[str, Any]]) -> dict[str, str]:
    """Flatten per-turn requirements into scalar MLflow expectation fields.

    Managed datasets render ``expectations`` as an object. Keeping every value on
    one line avoids the escaped-newline blob produced by one multiline value.
    """

    fields: dict[str, str] = {}
    for index, expectation in enumerate(turn_expectations, start=1):
        prefix = f"turn_{index}_"
        fields[f"{prefix}write"] = str(expectation["write"])
        if search := expectation.get("search"):
            fields[f"{prefix}search"] = str(search)
        if facts := expectation.get("search_should_contain"):
            fields[f"{prefix}search_result_should_include"] = "; ".join(facts)
        if memory_should := expectation.get("memory_should"):
            fields[f"{prefix}memory_should"] = str(memory_should)
        if answer_should := expectation.get("answer_should"):
            fields[f"{prefix}answer_should"] = str(answer_should)
    return fields


def format_expected_behavior(turn_expectations: list[dict[str, Any]]) -> str:
    """Render expectations as prose for notebook display and terminal output."""

    sections: list[str] = []
    for index, expectation in enumerate(turn_expectations, start=1):
        lines = [f"Turn {index}", f"  Write: {expectation['write']}"]
        if search := expectation.get("search"):
            lines.append(f"  Search: {search}")
        if facts := expectation.get("search_should_contain"):
            lines.append(f"  Search result includes: {json.dumps(facts, ensure_ascii=False)}")
        if memory_should := expectation.get("memory_should"):
            lines.append(f"  Memory requirement: {memory_should}")
        if answer_should := expectation.get("answer_should"):
            lines.append(f"  Answer requirement: {answer_should}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def parse_expected_behavior(value: str) -> list[dict[str, Any]]:
    """Parse the legacy multiline expectation format."""

    turns: list[dict[str, Any]] = []
    for section in value.strip().split("\n\n"):
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines:
            continue
        if not re.fullmatch(r"Turn \d+", lines[0]):
            raise ValueError(f"Invalid expected behavior section: {lines[0]!r}")
        expectation: dict[str, Any] = {}
        for line in lines[1:]:
            label, separator, raw_value = line.partition(": ")
            if not separator:
                raise ValueError(f"Invalid expected behavior line: {line!r}")
            if label == "Write":
                expectation["write"] = raw_value
            elif label == "Search":
                expectation["search"] = raw_value
            elif label == "Search result includes":
                expectation["search_should_contain"] = json.loads(raw_value)
            elif label == "Memory requirement":
                expectation["memory_should"] = raw_value
            elif label == "Answer requirement":
                expectation["answer_should"] = raw_value
            else:
                raise ValueError(f"Unknown expected behavior label: {label!r}")
        turns.append(expectation)
    return turns


def parse_expectation_fields(values: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return per-turn requirements from current or legacy dataset rows."""

    expectations = values or {}
    turn_fields: dict[int, dict[str, Any]] = {}
    for key, value in expectations.items():
        match = re.fullmatch(r"turn_(\d+)_(.+)", str(key))
        if not match or match.group(2) not in EXPECTATION_FIELD_NAMES:
            continue
        turn_index = int(match.group(1))
        field = match.group(2)
        if field == "search_result_should_include":
            field = "search_should_contain"
            value = [fact.strip() for fact in str(value).split(";") if fact.strip()]
        turn_fields.setdefault(turn_index, {})[field] = value

    if turn_fields:
        expected_indexes = list(range(1, max(turn_fields) + 1))
        if sorted(turn_fields) != expected_indexes:
            raise ValueError("Expectation turn numbers must be consecutive and start at 1.")
        return [turn_fields[index] for index in expected_indexes]

    if readable := expectations.get("expected_behavior"):
        return parse_expected_behavior(str(readable))
    value = expectations.get("turn_expectations_json", "[]")
    return json.loads(value) if isinstance(value, str) else list(value)


def build_records(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert explicitly supplied cases; Shared runs normally use the managed dataset."""

    return [scenario_to_record(scenario) for scenario in scenarios]


def load_records_from_dataset(
    dataset_name: str = DATASET_NAME, attempts: int = 3
) -> list[dict[str, Any]]:
    from mlflow.genai import datasets

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            dataset = datasets.get_dataset(name=dataset_name)
            frame = dataset.to_df()
            return [
                {
                    "inputs": dict(row["inputs"]),
                    "expectations": dict(row["expectations"] or {}),
                    "tags": dict(row.get("tags") or {}),
                }
                for _, row in frame.iterrows()
            ]
        except Exception as exc:  # managed dataset endpoints can be transient
            last_error = exc
            logger.warning("Dataset read attempt %s/%s failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(5)
    raise RuntimeError(f"Could not read MLflow dataset {dataset_name}") from last_error


def _record_identity(record: dict[str, Any]) -> tuple[str, str]:
    inputs = record["inputs"]
    return str(inputs["category"]), str(inputs["scenario_id"])


def filter_records(
    records: list[dict[str, Any]], scenario_ids: set[str], categories: set[str]
) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        category, scenario_id = _record_identity(record)
        if scenario_ids and scenario_id not in scenario_ids:
            continue
        if categories and category not in categories:
            continue
        filtered.append(record)
    if not filtered:
        raise ValueError("No scenarios matched the requested filters.")
    return sorted(filtered, key=lambda record: _record_identity(record)[1])


# ---------------------------------------------------------------------------
# Agent response parsing
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def parse_agent_output(response: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    """Return final assistant text and tool calls with their correlated outputs."""

    items = (response or {}).get("output", [])
    call_outputs = {
        item.get("call_id"): _text(item.get("output", ""))
        for item in items
        if item.get("type") == "function_call_output"
    }
    calls: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    for item in items:
        item_type = item.get("type")
        if item_type == "function_call":
            try:
                arguments = json.loads(item.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            call_id = item.get("call_id") or item.get("id")
            calls.append(
                {
                    "name": item.get("name"),
                    "arguments": arguments,
                    "output": call_outputs.get(call_id, ""),
                }
            )
        elif item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    answer_parts.append(part.get("text", ""))
    return "\n".join(answer_parts), calls


# ---------------------------------------------------------------------------
# Managed-memory store operations used for independent verification
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _memory_module():
    class StandaloneMemoryAPI:
        @staticmethod
        def _ws():
            return _ws()

        @staticmethod
        def _entries(suffix: str = "") -> str:
            if not MEMORY_STORE:
                raise RuntimeError("Call memory_eval.configure(...) before using the memory store.")
            return (
                f"/api/2.1/unity-catalog/memory-stores/{MEMORY_STORE}/entries{suffix}"
            )

        @staticmethod
        def _headers() -> dict[str, str] | None:
            traffic_id = os.getenv("DATABRICKS_MEMORY_TRAFFIC_ID")
            return {"x-databricks-traffic-id": traffic_id} if traffic_id else None

        @classmethod
        def _save(cls, scope, path, description, contents=""):
            try:
                cls._ws().api_client.do(
                    "POST",
                    cls._entries(),
                    query={"scope": scope},
                    headers=cls._headers(),
                    body={
                        "path": path,
                        "contents": contents,
                        "description": description,
                        "creation_reason": "CREATION_REASON_AGENT_INFERRED",
                        "creation_source": "CREATION_SOURCE_ONLINE_AGENT",
                    },
                )
            except DatabricksError as exc:
                return f"Could not save {path}: {getattr(exc, 'message', str(exc))}"
            return f"Saved memory at {path}."

        @classmethod
        def _delete(cls, scope, path):
            try:
                cls._ws().api_client.do(
                    "DELETE",
                    cls._entries(),
                    query={"scope": scope, "path": path},
                    headers=cls._headers(),
                )
            except DatabricksError as exc:
                if exc.error_code == "NOT_FOUND":
                    return f"No memory at {path} (already gone)."
                return f"Could not delete {path}: {getattr(exc, 'message', str(exc))}"
            return f"Deleted {path}."

    return StandaloneMemoryAPI


def _clean_entry(entry: dict[str, Any]) -> dict[str, str]:
    return {
        "path": str(entry.get("path", "")),
        "description": str(entry.get("description", "") or ""),
        "contents": str(entry.get("contents", "") or ""),
    }


def _store_io_map(function, items):
    """Run independent memory-entry API calls with a small, bounded worker pool."""

    values = list(items)
    if len(values) <= 1 or STORE_IO_MAX_WORKERS == 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(
        max_workers=min(STORE_IO_MAX_WORKERS, len(values)),
        thread_name_prefix="memory-eval-store",
    ) as executor:
        return list(executor.map(function, values))


def snapshot_scope(scope: str) -> dict[str, dict[str, str]]:
    um = _memory_module()
    response = um._ws().api_client.do(
        "GET",
        um._entries(),
        query={"scope": scope, "page_size": 500},
        headers=um._headers(),
    )
    def fetch(listed):
        path = listed["path"]
        entry = um._ws().api_client.do(
            "GET",
            um._entries(":get"),
            query={"scope": scope, "path": path},
            headers=um._headers(),
        )
        return path, _clean_entry(entry)

    return dict(_store_io_map(fetch, response.get("entries", [])))


def diff_snapshots(
    before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]
) -> dict[str, list[Any]]:
    before_paths = set(before)
    after_paths = set(after)
    created = [after[path] for path in sorted(after_paths - before_paths)]
    deleted = [before[path] for path in sorted(before_paths - after_paths)]
    updated = [
        {"before": before[path], "after": after[path]}
        for path in sorted(before_paths & after_paths)
        if before[path] != after[path]
    ]
    return {"created": created, "updated": updated, "deleted": deleted}


def seed_scope(scope: str, memories: list[dict[str, Any]]) -> None:
    um = _memory_module()

    def save(memory):
        result = um._save(
            scope,
            memory["path"],
            memory["description"],
            memory.get("contents", ""),
        )
        if not result.startswith("Saved memory"):
            raise RuntimeError(f"Could not seed {memory['path']}: {result}")
        return result

    _store_io_map(save, memories)


def cleanup_scope(scope: str) -> int:
    um = _memory_module()
    deleted = 0
    for path in list(snapshot_scope(scope)):
        result = um._delete(scope, path)
        if result.startswith("Deleted") or "already gone" in result:
            deleted += 1
        else:
            logger.warning("Could not clean %s in %s: %s", path, scope, result)
    return deleted


def _search_entries(scope: str, query: str) -> list[dict[str, str]]:
    um = _memory_module()
    response = um._ws().api_client.do(
        "POST",
        um._entries(":search"),
        query={"scope": scope},
        headers=um._headers(),
        body={"query": query, "top_k": 50},
    )
    return [
        _clean_entry(result.get("memory_entry", {}))
        for result in response.get("results", [])
    ]


def _search_query(entry: dict[str, str]) -> str:
    text = f"{entry.get('description', '')} {entry.get('contents', '')}".strip()
    return text[:500] or entry["path"]


def _same_entry(actual: dict[str, str], expected: dict[str, str]) -> bool:
    return all(actual.get(field, "") == expected.get(field, "") for field in _clean_entry(expected))


def wait_until_searchable(
    scope: str,
    entries: list[dict[str, str]],
    timeout_s: int = INDEX_TIMEOUT_S,
    poll_interval_s: int = INDEX_POLL_INTERVAL_S,
) -> None:
    """Poll until search returns the current version of every target entry."""

    if not entries:
        return
    deadline = time.monotonic() + timeout_s
    pending = {entry["path"]: _clean_entry(entry) for entry in entries}
    while pending:
        for path, expected in list(pending.items()):
            results = _search_entries(scope, _search_query(expected))
            if any(_same_entry(result, expected) for result in results if result["path"] == path):
                pending.pop(path)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Search index did not expose current entries in {timeout_s}s: {sorted(pending)}"
            )
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------


def _invoke_turn_sync(
    message: str,
    eval_scope: str,
    thread_id: str,
    *,
    eval_read_only: bool = False,
) -> dict[str, Any]:
    custom_inputs: dict[str, Any] = {
        "eval_scope": eval_scope,
        "thread_id": thread_id,
    }
    if eval_read_only:
        custom_inputs["eval_read_only"] = True
    payload = {
        "input": [{"role": "user", "content": message}],
        "custom_inputs": custom_inputs,
    }
    headers = dict(_ws().config.authenticate())
    headers["Content-Type"] = "application/json"
    response = requests.post(
        f"{APP_URL}/invocations",
        headers=headers,
        json=payload,
        timeout=HTTP_TIMEOUT_S,
        allow_redirects=False,
    )
    if response.status_code == 302:
        raise RuntimeError(
            "The App redirected the request. Databricks Apps require an OAuth token; "
            "the current notebook authentication did not produce one accepted by the App."
        )
    if not response.ok:
        raise RuntimeError(
            f"App invocation failed with HTTP {response.status_code}: {response.text[:1000]}"
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"App invocation returned non-JSON content: {response.text[:1000]}"
        ) from exc
    if "output" not in body:
        raise RuntimeError(f"App response did not contain output: {body}")
    return body


async def _invoke_turn(message: str, eval_scope: str, thread_id: str) -> dict[str, Any]:
    return await asyncio.to_thread(_invoke_turn_sync, message, eval_scope, thread_id)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Memory eval predict_fn must run outside an active asyncio event loop.")


def _entries_changed_by(diff: dict[str, list[Any]]) -> list[dict[str, str]]:
    return [
        *diff["created"],
        *(change["after"] for change in diff["updated"]),
    ]


def _successful_mutations(tool_calls: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Return paths for memory mutations whose tool outputs report success."""

    successful = {"created": set(), "updated": set(), "deleted": set()}
    result_kind = {
        "save_memory": "created",
        "update_memory": "updated",
        "delete_memory": "deleted",
    }
    for call in tool_calls:
        name = str(call.get("name", ""))
        prefix = MUTATION_SUCCESS_PREFIXES.get(name)
        output = str(call.get("output", "")).strip()
        path = str((call.get("arguments") or {}).get("path", "")).strip()
        if prefix and path and output.startswith(prefix):
            successful[result_kind[name]].add(path)
    return successful


def _diff_paths(diff: dict[str, list[Any]]) -> dict[str, set[str]]:
    return {
        "created": {entry["path"] for entry in diff["created"]},
        "updated": {change["after"]["path"] for change in diff["updated"]},
        "deleted": {entry["path"] for entry in diff["deleted"]},
    }


def wait_for_turn_snapshot(
    scope: str,
    before: dict[str, dict[str, str]],
    tool_calls: list[dict[str, Any]],
    timeout_s: int = INDEX_TIMEOUT_S,
    poll_interval_s: int = INDEX_POLL_INTERVAL_S,
) -> dict[str, dict[str, str]]:
    """Wait until direct reads reflect the turn's confirmed mutation results.

    A fresh scope gives the runner a known baseline. Successful mutation tool outputs
    define which paths must change; if no mutation succeeded, the store must settle
    back to that baseline. This prevents an eventually consistent read from assigning
    a write to the following turn or inventing a write on a read-only turn.
    """

    expected_paths = _successful_mutations(tool_calls)
    deadline = time.monotonic() + timeout_s
    last_paths = {"created": set(), "updated": set(), "deleted": set()}
    while True:
        after = snapshot_scope(scope)
        last_paths = _diff_paths(diff_snapshots(before, after))
        if last_paths == expected_paths:
            return after
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Memory store did not expose the expected post-turn state within "
                f"{timeout_s}s; expected changes={expected_paths}, last observed={last_paths}"
            )
        time.sleep(poll_interval_s)


@contextmanager
def _turn_trace(turn_index: int, turn: dict[str, Any], session_number: int):
    """Add a readable child span when run inside the scenario-level eval trace."""

    if mlflow.get_current_active_span() is None:
        yield None
        return
    with mlflow.start_span(
        name=f"turn_{turn_index + 1}",
        span_type="CHAIN",
        attributes={
            "eval.turn_index": turn_index,
            "eval.new_session": bool(turn.get("new_session", False)),
            "eval.session_number": session_number,
        },
    ) as span:
        span.set_inputs(
            {
                "message": turn["message"],
                "new_session": bool(turn.get("new_session", False)),
                "session_number": session_number,
            }
        )
        yield span


def run_scenario(
    scenario_id: str,
    category: str,
    scenario: dict[str, Any],
    index_timeout_s: int = INDEX_TIMEOUT_S,
) -> dict[str, Any]:
    safe_id = re.sub(r"[^a-z0-9-]+", "-", scenario_id.lower()).strip("-")
    invocation_token = uuid.uuid4().hex[:8]
    eval_scope = f"{RUN_TOKEN}-{invocation_token}-{safe_id}"[:128]
    scope = _full_eval_scope(eval_scope)
    session_number = 1
    turn_results: list[dict[str, Any]] = []

    try:
        initial_memories = [_clean_entry(memory) for memory in scenario.get("initial_memories", [])]
        if initial_memories:
            seed_scope(scope, initial_memories)
            wait_until_searchable(scope, initial_memories, timeout_s=index_timeout_s)
        confirmed_snapshot = {entry["path"]: entry for entry in initial_memories}

        turns = scenario.get("turns", [])
        for turn_index, turn in enumerate(turns):
            if turn_index and turn.get("new_session", False):
                session_number += 1
            thread_id = f"memeval-{eval_scope}-session-{session_number}"
            with _turn_trace(turn_index, turn, session_number) as turn_span:
                before = confirmed_snapshot
                response = _run_async(_invoke_turn(turn["message"], eval_scope, thread_id))
                answer, tool_calls = parse_agent_output(response)
                after = wait_for_turn_snapshot(
                    scope,
                    before,
                    tool_calls,
                    timeout_s=index_timeout_s,
                )
                store_diff = diff_snapshots(before, after)
                confirmed_snapshot = after
                turn_result = {
                    "turn_index": turn_index,
                    "message": turn["message"],
                    "new_session": bool(turn.get("new_session", False)),
                    "session_number": session_number,
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "store_diff": store_diff,
                }
                turn_results.append(turn_result)
                if turn_span is not None:
                    turn_span.set_outputs(
                        {
                            "answer": answer,
                            "tool_calls": tool_calls,
                            "store_diff": store_diff,
                        }
                    )

            changed_entries = _entries_changed_by(store_diff)
            if changed_entries and turn_index < len(turns) - 1:
                wait_until_searchable(scope, changed_entries, timeout_s=index_timeout_s)

        return {
            "scenario_id": scenario_id,
            "category": category,
            "turns": turn_results,
        }
    finally:
        try:
            cleanup_scope(scope)
        except Exception:
            logger.warning("Failed to clean eval scope %s", scope, exc_info=True)


@mlflow.trace(name="memory_eval_scenario", span_type="CHAIN")
def predict_fn(
    scenario_id: str,
    category: str,
    turns: list[dict[str, Any]],
    initial_memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scenario = {
        "turns": list(turns),
        "initial_memories": list(initial_memories or []),
    }
    turn_count = len(turns)
    mlflow.update_current_trace(
        tags={
            "eval.scenario_id": scenario_id,
            "eval.category": category,
            "eval.suite": "managed-memory-v2",
            "eval.turn_count": str(turn_count),
        },
        request_preview=f"{scenario_id} [{category}] · {turn_count} turn{'s' if turn_count != 1 else ''}",
    )
    result = run_scenario(
        scenario_id,
        category,
        scenario,
        index_timeout_s=INDEX_TIMEOUT_S,
    )
    tool_names = [
        (
            f"T{turn['turn_index'] + 1}: "
            + ", ".join(call.get("name", "") for call in turn.get("tool_calls", []))
        )
        for turn in result["turns"]
        if turn.get("tool_calls")
    ]
    mlflow.update_current_trace(
        response_preview=(
            f"Completed {turn_count} turn{'s' if turn_count != 1 else ''}; "
            f"memory tools: {' · '.join(tool_names) if tool_names else 'none'}"
        )
    )
    return result


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------


def _load_turn_expectations(expectations: dict[str, Any] | None) -> list[dict[str, Any]]:
    return parse_expectation_fields(expectations)


def _output_turns(outputs: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((outputs or {}).get("turns", []))


def _tool_names(turn: dict[str, Any]) -> list[str]:
    return [str(call.get("name")) for call in turn.get("tool_calls", [])]


def _has_store_changes(turn: dict[str, Any]) -> bool:
    diff = turn.get("store_diff", {})
    return any(diff.get(kind) for kind in ("created", "updated", "deleted"))


def _rationale_line(turn_index: int, passed: bool, detail: str) -> str:
    return f"Turn {turn_index + 1} — {'PASS' if passed else 'FAIL'}: {detail}"


@scorer(name="write_behavior")
def write_behavior(outputs, expectations):
    expected_turns = _load_turn_expectations(expectations)
    actual_turns = _output_turns(outputs)
    checks: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_turns):
        if index >= len(actual_turns):
            checks.append(
                {
                    "turn": index,
                    "passed": False,
                    "rationale": _rationale_line(index, False, "the scenario produced no output for this turn."),
                }
            )
            continue
        turn = actual_turns[index]
        names = _tool_names(turn)
        diff = turn.get("store_diff", {})
        write = expected["write"]
        if write == "save":
            correct_tools = "save_memory" in names and not ({"update_memory", "delete_memory"} & set(names))
            correct_store = len(diff.get("created", [])) == 1 and not diff.get("updated") and not diff.get("deleted")
            write_index = names.index("save_memory") if "save_memory" in names else -1
            searched_first = "search_memory" in names[:write_index] if write_index >= 0 else False
            passed = correct_tools and correct_store and searched_first
        elif write == "update":
            correct_tools = "update_memory" in names and not ({"save_memory", "delete_memory"} & set(names))
            correct_store = len(diff.get("updated", [])) == 1 and not diff.get("created") and not diff.get("deleted")
            write_index = names.index("update_memory") if "update_memory" in names else -1
            searched_first = "search_memory" in names[:write_index] if write_index >= 0 else False
            passed = correct_tools and correct_store and searched_first
        else:
            passed = not (MEMORY_MUTATION_TOOLS & set(names)) and not _has_store_changes(turn)
            searched_first = None
        checks.append(
            {
                "turn": index,
                "expected": write,
                "tools": names,
                "store_counts": {kind: len(diff.get(kind, [])) for kind in ("created", "updated", "deleted")},
                "searched_first": searched_first,
                "passed": passed,
                "rationale": _rationale_line(
                    index,
                    passed,
                    (
                        f"expected write={write}; tools={', '.join(names) if names else 'none'}; "
                        f"store changes: {len(diff.get('created', []))} created, "
                        f"{len(diff.get('updated', []))} updated, "
                        f"{len(diff.get('deleted', []))} deleted"
                        + (
                            (
                                f"; search before write={'yes' if searched_first else 'no'}."
                                if f"{write}_memory" in names
                                else f"; required {write}_memory call did not occur."
                            )
                            if write in {"save", "update"}
                            else "; store remained unchanged."
                        )
                    ),
                ),
            }
        )
    return Feedback(
        value=all(check["passed"] for check in checks),
        rationale="\n".join(check["rationale"] for check in checks),
    )


@scorer(name="write_structure")
def write_structure(outputs, expectations):
    expected_turns = _load_turn_expectations(expectations)
    actual_turns = _output_turns(outputs)
    checks: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_turns):
        if expected["write"] not in {"save", "update"} or index >= len(actual_turns):
            continue
        diff = actual_turns[index].get("store_diff", {})
        entries = [
            *diff.get("created", []),
            *(change["after"] for change in diff.get("updated", [])),
        ]
        entry_checks = []
        for entry in entries:
            description = str(entry.get("description", ""))
            contents = str(entry.get("contents", ""))
            entry_checks.append(
                {
                    "path": entry.get("path"),
                    "description_short_one_line": bool(description.strip())
                    and len(description) <= 150
                    and "\n" not in description,
                    "contents_not_description_duplicate": not contents.strip()
                    or contents.strip() != description.strip(),
                }
            )
        passed = bool(entry_checks) and all(
            all(value for key, value in check.items() if key != "path")
            for check in entry_checks
        )
        if not entry_checks:
            detail = "no created or updated store entry was available to inspect."
        else:
            entry_details = []
            for check in entry_checks:
                entry_details.append(
                    f"{check['path']} has a short one-line description="
                    f"{'yes' if check['description_short_one_line'] else 'no'} and distinct contents="
                    f"{'yes' if check['contents_not_description_duplicate'] else 'no'}"
                )
            detail = "; ".join(entry_details) + "."
        checks.append(
            {
                "turn": index,
                "entries": entry_checks,
                "passed": passed,
                "rationale": _rationale_line(index, passed, detail),
            }
        )
    if not checks:
        return None
    return Feedback(
        value=all(check["passed"] for check in checks),
        rationale="\n".join(check["rationale"] for check in checks),
    )


@scorer(name="search_behavior")
def search_behavior(outputs, expectations):
    expected_turns = _load_turn_expectations(expectations)
    actual_turns = _output_turns(outputs)
    checks: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_turns):
        search = expected.get("search")
        if search is None:
            continue
        names = _tool_names(actual_turns[index]) if index < len(actual_turns) else []
        called = "search_memory" in names
        passed = called if search == "required" else not called
        checks.append(
            {
                "turn": index,
                "expected": search,
                "search_called": called,
                "passed": passed,
                "rationale": _rationale_line(
                    index,
                    passed,
                    f"search was {search} and was {'called' if called else 'not called'}.",
                ),
            }
        )
    if not checks:
        return None
    return Feedback(
        value=all(check["passed"] for check in checks),
        rationale="\n".join(check["rationale"] for check in checks),
    )


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


@scorer(name="search_result")
def search_result(outputs, expectations):
    expected_turns = _load_turn_expectations(expectations)
    actual_turns = _output_turns(outputs)
    checks: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_turns):
        facts = expected.get("search_should_contain")
        if not facts:
            continue
        search_outputs = "\n".join(
            call.get("output", "")
            for call in (actual_turns[index].get("tool_calls", []) if index < len(actual_turns) else [])
            if call.get("name") == "search_memory"
        )
        normalized_output = _normalized_text(search_outputs)
        hits = {
            fact: _normalized_text(fact) in normalized_output
            for fact in facts
        }
        passed = all(hits.values())
        found = [fact for fact, hit in hits.items() if hit]
        missing = [fact for fact, hit in hits.items() if not hit]
        detail = f"retrieved expected facts: {', '.join(found) if found else 'none'}"
        if missing:
            detail += f"; missing: {', '.join(missing)}"
        detail += "."
        checks.append(
            {
                "turn": index,
                "facts": hits,
                "passed": passed,
                "rationale": _rationale_line(index, passed, detail),
            }
        )
    if not checks:
        return None
    return Feedback(
        value=all(check["passed"] for check in checks),
        rationale="\n".join(check["rationale"] for check in checks),
    )


DETERMINISTIC_SCORERS = [write_behavior, write_structure, search_behavior, search_result]


# ---------------------------------------------------------------------------
# Registered MLflow judges and result reporting
# ---------------------------------------------------------------------------


def _has_requirement(expectations: dict[str, Any] | None, field: str) -> bool:
    return any(turn.get(field) for turn in parse_expectation_fields(expectations))


def _invoke_judge_with_retry(judge, *, inputs, outputs, expectations):
    """Retry transient judge endpoint failures without changing judge semantics."""

    for attempt in range(1, JUDGE_RETRY_ATTEMPTS + 1):
        try:
            return judge(inputs=inputs, outputs=outputs, expectations=expectations)
        except Exception as exc:
            if attempt == JUDGE_RETRY_ATTEMPTS:
                raise
            logger.warning(
                "Judge call attempt %s/%s failed; retrying: %s",
                attempt,
                JUDGE_RETRY_ATTEMPTS,
                exc,
            )
            time.sleep(JUDGE_RETRY_INTERVAL_S * attempt)


def load_registered_judges(
    experiment_id: str = EXPERIMENT_ID,
) -> list[Any]:
    """Load the existing MLflow-managed judges with case applicability guards."""

    write_judge = get_scorer(
        name="memory_write_quality",
        experiment_id=experiment_id,
    )
    answer_judge = get_scorer(
        name="memory_answer_quality",
        experiment_id=experiment_id,
    )

    @scorer(name="memory_write_quality")
    def conditional_write_judge(inputs, outputs, expectations):
        if not _has_requirement(expectations, "memory_should"):
            return None
        return _invoke_judge_with_retry(
            write_judge,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
        )

    @scorer(name="memory_answer_quality")
    def conditional_answer_judge(inputs, outputs, expectations):
        if not _has_requirement(expectations, "answer_should"):
            return None
        return _invoke_judge_with_retry(
            answer_judge,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
        )

    return [conditional_write_judge, conditional_answer_judge]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _run_label(records: list[dict[str, Any]]) -> str:
    scenario_ids = [record["inputs"]["scenario_id"] for record in records]
    categories = sorted({record["inputs"]["category"] for record in records})
    if len(scenario_ids) == 1:
        return str(scenario_ids[0])
    if len(categories) == 1:
        return f"{categories[0]}-{len(scenario_ids)}-cases"
    return f"{len(scenario_ids)}-scenario-suite"


def _result_identity(row: Any) -> tuple[str, str]:
    request = row.get("request", {})
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except json.JSONDecodeError:
            request = {}
    return str(request.get("scenario_id", "unknown-scenario")), str(
        request.get("category", "unknown-category")
    )


def log_result_breakdown(result, client: mlflow.MlflowClient) -> None:
    """Print per-scenario results and log category-level metrics to MLflow."""

    if result.result_df is None or result.result_df.empty:
        return
    scorer_names = [scorer.name for scorer in DETERMINISTIC_SCORERS] + [
        "memory_write_quality",
        "memory_answer_quality",
    ]
    category_values: dict[tuple[str, str], list[float]] = {}
    print("\nScenario results:")
    for _, row in result.result_df.iterrows():
        scenario_id, category = _result_identity(row)
        values = []
        for scorer_name in scorer_names:
            column = f"{scorer_name}/value"
            if column not in row or row[column] is None or row[column] != row[column]:
                continue
            value = bool(row[column])
            values.append(f"{scorer_name}={'PASS' if value else 'FAIL'}")
            category_values.setdefault((category, scorer_name), []).append(float(value))
        print(f"  {scenario_id} [{category}]")
        print(f"    {', '.join(values) if values else 'no applicable scores'}")

    for (category, scorer_name), values in category_values.items():
        metric_name = f"category.{category}.{scorer_name}.mean"
        client.log_metric(result.run_id, metric_name, sum(values) / len(values))

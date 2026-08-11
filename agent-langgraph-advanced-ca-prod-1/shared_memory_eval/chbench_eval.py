"""Run ContextHeavy-Bench against managed memory and the deployed App.

The benchmark corpus is seeded directly into one isolated evaluation scope. Each
question then gets two retrieval paths:

* a direct memory-store search using the question verbatim; and
* an end-to-end deployed-App invocation, including the agent's search decisions,
  generated queries, ranked results, and final answer.

This module deliberately reuses :mod:`memory_eval` for OAuth App invocation and
direct test-only evaluation scopes. It has no dependency on agent source.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable
import uuid

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

try:  # Package import locally; top-level import beside a Workspace notebook.
    from . import memory_eval
except ImportError:  # pragma: no cover - exercised in Databricks Workspace
    import memory_eval  # type: ignore[no-redef]


SOURCE_REPOSITORY = "https://github.com/shihabshahrier/CH-Bench"
SOURCE_COMMIT = "5a08786a9b74896a139183a7ee8b8a4e066c9428"
SUITE_NAME = "contextheavy"
DATASET_NAME = f"kevinyan.default.chbench_contextheavy_{SOURCE_COMMIT[:8]}"
CORPUS_DATASET_NAME = (
    f"kevinyan.default.chbench_contextheavy_memories_{SOURCE_COMMIT[:8]}"
)
DEFAULT_PERSISTENT_SCOPE = f"chbench-contextheavy-{SOURCE_COMMIT[:8]}"
DEFAULT_DATA_DIR = Path(__file__).with_name("chbench_data") / SUITE_NAME
MEMORY_PATH_PREFIX = "/memories/eval/chbench/"
EXPECTED_QUESTION_COUNT = 35
EXPECTED_MEMORY_COUNT = 44
EXPECTED_CORPUS_FINGERPRINT = (
    "b1fa74bbe7823d0383db72fc280fffd989be25b9475602e196a00d0d119ee2a8"
)
MEMORY_MUTATION_TOOLS = {"save_memory", "update_memory", "delete_memory"}
FORCED_MEMORY_INSTRUCTION = (
    "Use the search_memory tool before answering. It returns only path, description, and "
    "has_contents. For any relevant result with has_contents=true that is worth looking into, "
    "call get_memory on its path before answering. If the saved memories do not contain the "
    "answer, say you do not know based on memory."
)
LEGACY_SEARCH_RESULT_PATTERN = re.compile(
    r"^- (?P<path>\S+) \(score (?P<score>[-+0-9.eE]+)\):",
    re.MULTILINE,
)
INDEX_SEARCH_RESULT_PATTERN = re.compile(r"^- path: (?P<path>\S+)\s*$", re.MULTILINE)


def search_result_paths(output: str) -> list[str]:
    """Extract ranked paths from current multiline/JSON indexes or historical scored text."""

    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, list):
        return [
            str(entry["path"])
            for entry in parsed
            if isinstance(entry, dict) and entry.get("path")
        ]
    current_paths = [
        match.group("path") for match in INDEX_SEARCH_RESULT_PATTERN.finditer(output)
    ]
    if current_paths:
        return current_paths
    return [
        match.group("path")
        for match in LEGACY_SEARCH_RESULT_PATTERN.finditer(output)
    ]


def load_suite(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the vendored ContextHeavy-Bench tracks."""

    root = Path(data_dir or DEFAULT_DATA_DIR)
    files = sorted(root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No CH-Bench track files found in {root}")

    tracks = [json.loads(path.read_text()) for path in files]
    memories: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    memory_ids: set[str] = set()
    question_ids: set[str] = set()

    for track_data in tracks:
        track = str(track_data.get("track", "")).strip()
        if not track:
            raise ValueError("Every CH-Bench file must define a non-empty track")
        for memory in track_data.get("memories", []):
            memory_id = str(memory["id"])
            if memory_id in memory_ids:
                raise ValueError(f"Duplicate CH-Bench memory id: {memory_id}")
            memory_ids.add(memory_id)
            memories.append({**memory, "track": track})
        for question in track_data.get("questions", []):
            question_id = str(question["id"])
            if question_id in question_ids:
                raise ValueError(f"Duplicate CH-Bench question id: {question_id}")
            question_ids.add(question_id)
            questions.append({**question, "track": track})

    for question in questions:
        relevant = set(question.get("relevant_ids", []))
        missing = relevant - memory_ids
        if missing:
            raise ValueError(
                f"Question {question['id']} references missing memories: {sorted(missing)}"
            )
        expect_abstain = bool(question.get("expect_abstain", False))
        if expect_abstain and relevant:
            raise ValueError(f"Abstention question {question['id']} cannot have relevant_ids")
        if not expect_abstain and not relevant:
            raise ValueError(f"Answerable question {question['id']} needs relevant_ids")
        if not expect_abstain and question.get("answer") is None:
            raise ValueError(f"Answerable question {question['id']} needs an answer")

    return {
        "name": SUITE_NAME,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "tracks": tracks,
        "memories": memories,
        "questions": questions,
    }


def memory_path(memory_id: str) -> str:
    """Return the stable managed-memory path that preserves a CH-Bench id."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", memory_id):
        raise ValueError(f"Invalid CH-Bench memory id for a path: {memory_id!r}")
    return f"{MEMORY_PATH_PREFIX}{memory_id}.md"


def memory_id_from_path(path: str) -> str | None:
    """Resolve a benchmark-owned managed-memory path back to its suite id."""

    if not path.startswith(MEMORY_PATH_PREFIX) or not path.endswith(".md"):
        return None
    memory_id = path[len(MEMORY_PATH_PREFIX) : -len(".md")]
    return memory_id or None


def _metadata_contents(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    lines = ["Benchmark metadata:"]
    lines.extend(f"{key}: {metadata[key]}" for key in sorted(metadata))
    return "\n".join(lines)


def build_seed_memories(suite: dict[str, Any]) -> list[dict[str, str]]:
    """Map benchmark memories to managed-memory entries without changing their text."""

    return [
        {
            "path": memory_path(str(memory["id"])),
            "description": str(memory["text"]),
            "contents": _metadata_contents(dict(memory.get("metadata", {}))),
        }
        for memory in suite["memories"]
    ]


def corpus_fingerprint(seed_memories: list[dict[str, str]]) -> str:
    """Return a stable content hash for one frozen benchmark corpus."""

    canonical = json.dumps(
        sorted(seed_memories, key=lambda entry: entry["path"]),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_corpus_dataset_records(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen managed-memory corpus as versioned dataset rows."""

    seeds_by_id = {
        str(memory["id"]): seed
        for memory, seed in zip(suite["memories"], build_seed_memories(suite))
    }
    fingerprint = corpus_fingerprint(list(seeds_by_id.values()))
    return [
        {
            "inputs": {
                "memory_id": str(memory["id"]),
                **seeds_by_id[str(memory["id"])],
            },
            "tags": {
                "suite": f"chbench-{SUITE_NAME}",
                "source_commit": SOURCE_COMMIT,
                "corpus_fingerprint": fingerprint,
                "track": str(memory["track"]),
            },
        }
        for memory in suite["memories"]
    ]


def _dataset_rows(dataset_name: str, attempts: int = 3) -> list[dict[str, Any]]:
    """Load normalized MLflow managed-dataset rows by name."""

    from mlflow.genai import datasets

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            frame = datasets.get_dataset(name=dataset_name).to_df()
            break
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                raise RuntimeError(
                    f"Could not read MLflow dataset {dataset_name} after {attempts} attempts"
                ) from exc
            time.sleep(5 * attempt)
    else:  # pragma: no cover - the loop either breaks or raises
        raise RuntimeError(f"Could not read MLflow dataset {dataset_name}") from last_error
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        inputs = raw.get("inputs")
        expectations = raw.get("expectations")
        tags = raw.get("tags")
        if not isinstance(inputs, dict):
            raise ValueError(f"Dataset {dataset_name} contains a row without inputs")
        rows.append(
            {
                "inputs": dict(inputs),
                "expectations": dict(expectations) if isinstance(expectations, dict) else {},
                "tags": dict(tags) if isinstance(tags, dict) else {},
            }
        )
    return rows


def _validate_provenance(
    dataset_name: str,
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"Dataset {dataset_name} has {len(rows)} rows; expected {expected_count}"
        )
    bad_commits = {
        str(row["tags"].get("source_commit", ""))
        for row in rows
        if row["tags"].get("source_commit") != SOURCE_COMMIT
    }
    bad_fingerprints = {
        str(row["tags"].get("corpus_fingerprint", ""))
        for row in rows
        if row["tags"].get("corpus_fingerprint") != EXPECTED_CORPUS_FINGERPRINT
    }
    if bad_commits:
        raise ValueError(
            f"Dataset {dataset_name} is not pinned to source commit {SOURCE_COMMIT}: "
            f"{sorted(bad_commits)}"
        )
    if bad_fingerprints:
        raise ValueError(
            f"Dataset {dataset_name} is not pinned to corpus fingerprint "
            f"{EXPECTED_CORPUS_FINGERPRINT}: {sorted(bad_fingerprints)}"
        )


def load_records_from_dataset(
    dataset_name: str,
    eval_scope: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Load the 35 question/gold rows and attach this run's settings."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows = _dataset_rows(dataset_name)
    _validate_provenance(dataset_name, rows, expected_count=EXPECTED_QUESTION_COUNT)
    question_ids = [str(row["inputs"].get("question_id", "")) for row in rows]
    if "" in question_ids or len(set(question_ids)) != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"Dataset {dataset_name} must contain 35 unique question IDs")
    for row in rows:
        relevant = _relevant_ids(row["expectations"])
        expect_abstain = _expects_abstention(row["expectations"])
        if expect_abstain == bool(relevant):
            raise ValueError(
                f"Question {row['inputs']['question_id']} has inconsistent abstention gold"
            )
        row["inputs"] = {
            **row["inputs"],
            "eval_scope": eval_scope,
            "top_k": top_k,
        }
    return rows


def load_seed_memories_from_dataset(dataset_name: str) -> list[dict[str, str]]:
    """Load and validate the exact 44-entry frozen corpus dataset."""

    rows = _dataset_rows(dataset_name)
    _validate_provenance(dataset_name, rows, expected_count=EXPECTED_MEMORY_COUNT)
    memory_ids: set[str] = set()
    paths: set[str] = set()
    seeds: list[dict[str, str]] = []
    for row in rows:
        inputs = row["inputs"]
        memory_id = str(inputs.get("memory_id", ""))
        path = str(inputs.get("path", ""))
        if not memory_id or memory_id in memory_ids:
            raise ValueError(f"Dataset {dataset_name} has a missing or duplicate memory ID")
        if path != memory_path(memory_id) or path in paths:
            raise ValueError(
                f"Dataset {dataset_name} has an invalid or duplicate path for {memory_id}"
            )
        memory_ids.add(memory_id)
        paths.add(path)
        seeds.append(
            {
                "path": path,
                "description": str(inputs.get("description", "")),
                "contents": str(inputs.get("contents", "")),
            }
        )
    fingerprint = corpus_fingerprint(seeds)
    if fingerprint != EXPECTED_CORPUS_FINGERPRINT:
        raise ValueError(
            f"Dataset {dataset_name} content fingerprint is {fingerprint}; "
            f"expected {EXPECTED_CORPUS_FINGERPRINT}"
        )
    return seeds


def validate_question_corpus_records(
    records: list[dict[str, Any]],
    seed_memories: list[dict[str, str]],
) -> None:
    """Ensure every question's gold IDs resolve in the loaded corpus dataset."""

    memory_ids = {
        memory_id
        for entry in seed_memories
        if (memory_id := memory_id_from_path(entry["path"])) is not None
    }
    for record in records:
        missing = set(_relevant_ids(record.get("expectations"))) - memory_ids
        if missing:
            raise ValueError(
                f"Question {record['inputs']['question_id']} references missing corpus IDs: "
                f"{sorted(missing)}"
            )


def inspect_persistent_entries(
    eval_scope: str,
    seed_memories: list[dict[str, str]],
) -> dict[str, Any]:
    """Compare a persistent scope with exact dataset-loaded memory entries."""

    expected = {
        entry["path"]: memory_eval._clean_entry(entry) for entry in seed_memories
    }
    expected_fingerprint = corpus_fingerprint(seed_memories)
    if expected_fingerprint != EXPECTED_CORPUS_FINGERPRINT:
        raise ValueError(
            f"Refusing corpus with fingerprint {expected_fingerprint}; "
            f"expected {EXPECTED_CORPUS_FINGERPRINT}"
        )
    full_scope = memory_eval._full_eval_scope(eval_scope)
    actual = memory_eval.snapshot_scope(full_scope)
    actual_fingerprint = corpus_fingerprint(list(actual.values()))
    diff = memory_eval.diff_snapshots(expected, actual)
    return {
        "eval_scope": eval_scope,
        "full_scope": full_scope,
        "fingerprint": expected_fingerprint,
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual_fingerprint,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "exact": expected_fingerprint == actual_fingerprint
        and not any(diff[kind] for kind in ("created", "updated", "deleted")),
        "missing": diff["deleted"],
        "unexpected": diff["created"],
        "changed": diff["updated"],
        "snapshot": actual,
    }


def inspect_persistent_corpus(
    eval_scope: str,
    suite: dict[str, Any],
) -> dict[str, Any]:
    """Compare a persistent scope with the exact vendored benchmark corpus."""

    return inspect_persistent_entries(eval_scope, build_seed_memories(suite))


def persistent_corpus_error(status: dict[str, Any]) -> str:
    return (
        f"Persistent CH-Bench corpus {status['full_scope']} does not match "
        f"{status['fingerprint'][:12]}: {len(status['missing'])} missing, "
        f"{len(status['unexpected'])} unexpected, {len(status['changed'])} changed."
    )


def validate_persistent_corpus(
    eval_scope: str,
    suite: dict[str, Any],
) -> dict[str, Any]:
    """Return corpus status or fail before evaluation touches a bad fixture."""

    return validate_persistent_entries(eval_scope, build_seed_memories(suite))


def validate_persistent_entries(
    eval_scope: str,
    seed_memories: list[dict[str, str]],
) -> dict[str, Any]:
    """Return exact corpus status or fail before evaluation starts."""

    status = inspect_persistent_entries(eval_scope, seed_memories)
    if not status["exact"]:
        raise RuntimeError(
            persistent_corpus_error(status)
            + " Run seed_chbench_corpus.py; use --replace only to repair this dedicated scope."
        )
    return status


def seed_persistent_corpus(
    eval_scope: str,
    suite: dict[str, Any],
    *,
    replace: bool = False,
    index_timeout_s: int = 180,
) -> dict[str, Any]:
    """Idempotently create a persistent corpus, replacing drift only on request."""

    return seed_persistent_entries(
        eval_scope,
        build_seed_memories(suite),
        replace=replace,
        index_timeout_s=index_timeout_s,
    )


def seed_persistent_entries(
    eval_scope: str,
    seed_memories: list[dict[str, str]],
    *,
    replace: bool = False,
    index_timeout_s: int = 180,
) -> dict[str, Any]:
    """Idempotently seed exact entries, replacing drift only when requested."""

    status = inspect_persistent_entries(eval_scope, seed_memories)
    if status["exact"]:
        memory_eval.wait_until_searchable(
            status["full_scope"],
            seed_memories,
            timeout_s=index_timeout_s,
        )
        return {**status, "action": "already_exact"}
    if status["actual_count"] and not replace:
        raise RuntimeError(
            persistent_corpus_error(status)
            + " Refusing to overwrite it without --replace."
        )

    full_scope = status["full_scope"]
    if status["actual_count"]:
        memory_eval.cleanup_scope(full_scope)
    try:
        memory_eval.seed_scope(full_scope, seed_memories)
        stored_status = inspect_persistent_entries(eval_scope, seed_memories)
        if not stored_status["exact"]:
            raise RuntimeError(persistent_corpus_error(stored_status))
    except Exception:
        # A partial corpus is worse than an empty one because evaluation could
        # silently score against it. Restore write failures to an empty scope.
        memory_eval.cleanup_scope(full_scope)
        raise

    # At this point all entries are durably exact. Indexing is asynchronous, so
    # a visibility timeout should report failure but retain the valid corpus;
    # rerunning this idempotent command will only repeat the visibility check.
    memory_eval.wait_until_searchable(
        full_scope,
        seed_memories,
        timeout_s=index_timeout_s,
    )
    final_status = inspect_persistent_entries(eval_scope, seed_memories)
    if not final_status["exact"]:
        raise RuntimeError(persistent_corpus_error(final_status))
    return {**final_status, "action": "replaced" if status["actual_count"] else "seeded"}


def build_dataset_records(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable managed-dataset rows without evaluator runtime settings."""

    fingerprint = corpus_fingerprint(build_seed_memories(suite))
    records: list[dict[str, Any]] = []
    for question in suite["questions"]:
        metadata = dict(question.get("metadata", {}))
        question_type = str(metadata.get("type", "unspecified"))
        records.append(
            {
                "inputs": {
                    "question_id": str(question["id"]),
                    "question": str(question["question"]),
                    "track": str(question["track"]),
                    "question_type": question_type,
                },
                "expectations": {
                    "expected_answer": ""
                    if question.get("answer") is None
                    else str(question["answer"]),
                    "relevant_ids_json": json.dumps(question.get("relevant_ids", [])),
                    "expect_abstain": bool(question.get("expect_abstain", False)),
                },
                "tags": {
                    "suite": f"chbench-{SUITE_NAME}",
                    "source_commit": SOURCE_COMMIT,
                    "corpus_fingerprint": fingerprint,
                    "track": str(question["track"]),
                    "question_type": question_type,
                },
            }
        )
    return records


def build_records(
    suite: dict[str, Any], eval_scope: str, *, top_k: int = 10
) -> list[dict[str, Any]]:
    """Add one run's scope and retrieval depth to stable dataset records."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    records = build_dataset_records(suite)
    return [
        {
            **record,
            "inputs": {
                **record["inputs"],
                "eval_scope": eval_scope,
                "top_k": top_k,
            },
        }
        for record in records
    ]


def filter_records(
    records: list[dict[str, Any]], question_ids: set[str], tracks: set[str]
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        inputs = record["inputs"]
        if question_ids and inputs["question_id"] not in question_ids:
            continue
        if tracks and inputs["track"] not in tracks:
            continue
        selected.append(record)
    if not selected:
        raise ValueError("No CH-Bench questions matched the requested filters")
    return selected


def _source_from_result(result: dict[str, Any], rank: int) -> dict[str, Any]:
    entry = result.get("memory_entry", {})
    path = str(entry.get("path", ""))
    return {
        "id": memory_id_from_path(path),
        "path": path,
        "score": float(result.get("score", 0.0) or 0.0),
        "description": str(entry.get("description", "") or ""),
        "contents": str(entry.get("contents", "") or ""),
        "rank": rank,
    }


def direct_store_search(question: str, eval_scope: str, top_k: int) -> dict[str, Any]:
    """Search the same isolated corpus directly using the question verbatim."""

    scope = memory_eval._full_eval_scope(eval_scope)
    api = memory_eval._memory_module()
    started = time.perf_counter()
    response = api._ws().api_client.do(
        "POST",
        api._entries(":search"),
        query={"scope": scope},
        headers=api._headers(),
        body={"query": question, "top_k": top_k},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    sources = [
        _source_from_result(result, rank)
        for rank, result in enumerate(response.get("results", []), start=1)
    ]
    return {
        "query": question,
        "sources": sources,
        "ranked_ids": [source["id"] for source in sources],
        "latency_ms": latency_ms,
    }


def parse_agent_sources(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Create one first-seen ranking across all agent search calls.

    If the agent searches more than once, results are concatenated in tool-call
    order and duplicate paths retain their first position. This scores the order
    in which evidence became available to the agent without selecting its best
    search after the fact.
    """

    searches: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for search_index, call in enumerate(
        (call for call in tool_calls if call.get("name") == "search_memory"),
        start=1,
    ):
        output = str(call.get("output", "") or "")
        query = str(dict(call.get("arguments") or {}).get("query", ""))
        per_search: list[dict[str, Any]] = []
        for rank, path in enumerate(search_result_paths(output), start=1):
            source = {
                "id": memory_id_from_path(path),
                "path": path,
                "search_index": search_index,
                "search_rank": rank,
            }
            per_search.append(source)
            if path not in seen_paths:
                seen_paths.add(path)
                sources.append(source)
        searches.append({"query": query, "sources": per_search, "output": output})
    return {
        "searches": searches,
        "sources": sources,
        "ranked_ids": [source["id"] for source in sources],
    }


def _response_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    for key in ("total_tokens", "total_token_count"):
        value = usage.get(key) if isinstance(usage, dict) else None
        if isinstance(value, (int, float)):
            return int(value)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
            return int(input_tokens + output_tokens)
    return 0


def forced_memory_question(question: str) -> str:
    """Require managed-memory search while preserving the source question text."""

    return f"{FORCED_MEMORY_INSTRUCTION}\n\nQuestion: {question}"


@mlflow.trace(name="chbench_question", span_type="CHAIN")
def predict_fn(
    question_id: str,
    question: str,
    track: str,
    question_type: str,
    eval_scope: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run direct retrieval and a forced-memory deployed-App turn."""

    direct = direct_store_search(question, eval_scope, int(top_k))
    agent_question = forced_memory_question(question)
    safe_id = re.sub(r"[^a-z0-9-]+", "-", question_id.lower()).strip("-")
    # The corpus persists across runs, but short-term conversation state must not.
    # A random suffix prevents a repeated question from inheriting an older
    # checkpoint for the same corpus selector.
    thread_id = f"chbench-{eval_scope}-{safe_id}-{uuid.uuid4().hex[:10]}"
    started = time.perf_counter()
    response = memory_eval._invoke_turn_sync(agent_question, eval_scope, thread_id)
    app_latency_ms = (time.perf_counter() - started) * 1000
    answer, tool_calls = memory_eval.parse_agent_output(response)
    agent_retrieval = parse_agent_sources(tool_calls)
    mutation_calls = [
        call for call in tool_calls if call.get("name") in MEMORY_MUTATION_TOOLS
    ]

    mlflow.update_current_trace(
        tags={
            "eval.suite": "chbench-contextheavy",
            "eval.question_id": question_id,
            "eval.track": track,
            "eval.question_type": question_type,
            "eval.agent_search_mode": "explicit",
        },
        request_preview=f"{question_id} [{track}/{question_type}] {question}",
        response_preview=(
            f"searches={len(agent_retrieval['searches'])}; "
            f"ranked={agent_retrieval['ranked_ids'][:int(top_k)]}; "
            f"answer={answer[:240]}"
        ),
    )
    return {
        "question_id": question_id,
        "agent_question": agent_question,
        "answer": answer,
        "tool_calls": tool_calls,
        "agent_searches": agent_retrieval["searches"],
        "agent_sources": agent_retrieval["sources"],
        "agent_ranked_ids": agent_retrieval["ranked_ids"],
        "direct_sources": direct["sources"],
        "direct_ranked_ids": direct["ranked_ids"],
        "mutation_calls": mutation_calls,
        "agent_latency_ms": app_latency_ms,
        "direct_latency_ms": direct["latency_ms"],
        "tokens": _response_tokens(response),
    }


def recall_at_k(ranked: list[str | None], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def precision_at_k(ranked: list[str | None], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in rel) / len(top)


def hit_at_k(ranked: list[str | None], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(item in rel for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str | None], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for rank, item in enumerate(ranked, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str | None], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _relevant_ids(expectations: dict[str, Any] | None) -> list[str]:
    raw = (expectations or {}).get("relevant_ids_json", "[]")
    values = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    return [str(value) for value in values]


def _expects_abstention(expectations: dict[str, Any] | None) -> bool:
    value = (expectations or {}).get("expect_abstain", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _top_k(inputs: dict[str, Any] | None) -> int:
    return int((inputs or {}).get("top_k", 10))


def _ranked(outputs: dict[str, Any] | None, axis: str) -> list[str | None]:
    return list((outputs or {}).get(f"{axis}_ranked_ids", []))


def _metric_feedback(
    axis: str,
    metric_name: str,
    inputs: dict[str, Any] | None,
    outputs: dict[str, Any] | None,
    expectations: dict[str, Any] | None,
) -> Feedback | None:
    relevant = _relevant_ids(expectations)
    if not relevant:
        return None  # Abstention rows have no ranking gold and are excluded.
    ranked = _ranked(outputs, axis)
    k = _top_k(inputs)
    functions = {
        "recall_at_k": lambda: recall_at_k(ranked, relevant, k),
        "precision_at_k": lambda: precision_at_k(ranked, relevant, k),
        "hit_at_k": lambda: hit_at_k(ranked, relevant, k),
        "mrr": lambda: reciprocal_rank(ranked, relevant),
        "ndcg_at_k": lambda: ndcg_at_k(ranked, relevant, k),
    }
    value = functions[metric_name]()
    return Feedback(
        value=value,
        rationale=(
            f"{axis} {metric_name}: relevant={relevant}; "
            f"ranked={ranked[:k]}; k={k}."
        ),
    )


@scorer(name="agent_recall_at_k")
def agent_recall_at_k(inputs, outputs, expectations):
    return _metric_feedback("agent", "recall_at_k", inputs, outputs, expectations)


@scorer(name="agent_precision_at_k")
def agent_precision_at_k(inputs, outputs, expectations):
    return _metric_feedback("agent", "precision_at_k", inputs, outputs, expectations)


@scorer(name="agent_hit_at_k")
def agent_hit_at_k(inputs, outputs, expectations):
    return _metric_feedback("agent", "hit_at_k", inputs, outputs, expectations)


@scorer(name="agent_mrr")
def agent_mrr(inputs, outputs, expectations):
    return _metric_feedback("agent", "mrr", inputs, outputs, expectations)


@scorer(name="agent_ndcg_at_k")
def agent_ndcg_at_k(inputs, outputs, expectations):
    return _metric_feedback("agent", "ndcg_at_k", inputs, outputs, expectations)


@scorer(name="direct_recall_at_k")
def direct_recall_at_k(inputs, outputs, expectations):
    return _metric_feedback("direct", "recall_at_k", inputs, outputs, expectations)


@scorer(name="direct_precision_at_k")
def direct_precision_at_k(inputs, outputs, expectations):
    return _metric_feedback("direct", "precision_at_k", inputs, outputs, expectations)


@scorer(name="direct_hit_at_k")
def direct_hit_at_k(inputs, outputs, expectations):
    return _metric_feedback("direct", "hit_at_k", inputs, outputs, expectations)


@scorer(name="direct_mrr")
def direct_mrr(inputs, outputs, expectations):
    return _metric_feedback("direct", "mrr", inputs, outputs, expectations)


@scorer(name="direct_ndcg_at_k")
def direct_ndcg_at_k(inputs, outputs, expectations):
    return _metric_feedback("direct", "ndcg_at_k", inputs, outputs, expectations)


@scorer(name="agent_search_called")
def agent_search_called(outputs):
    searches = list((outputs or {}).get("agent_searches", []))
    return Feedback(
        value=bool(searches),
        rationale=f"The agent made {len(searches)} search_memory call(s).",
    )


@scorer(name="agent_no_memory_write")
def agent_no_memory_write(outputs):
    calls = list((outputs or {}).get("mutation_calls", []))
    names = [str(call.get("name")) for call in calls]
    return Feedback(
        value=not calls,
        rationale=(
            "No memory mutation tools were called."
            if not calls
            else f"Unexpected query-time memory mutations: {names}."
        ),
    )


@scorer(name="agent_latency_ms")
def agent_latency_ms(outputs):
    return float((outputs or {}).get("agent_latency_ms", 0.0))


@scorer(name="direct_latency_ms")
def direct_latency_ms(outputs):
    return float((outputs or {}).get("direct_latency_ms", 0.0))


@scorer(name="agent_tokens")
def agent_tokens(outputs):
    return int((outputs or {}).get("tokens", 0))


DETERMINISTIC_SCORERS = [
    agent_recall_at_k,
    agent_precision_at_k,
    agent_hit_at_k,
    agent_mrr,
    agent_ndcg_at_k,
    direct_recall_at_k,
    direct_precision_at_k,
    direct_hit_at_k,
    direct_mrr,
    direct_ndcg_at_k,
    agent_search_called,
    agent_no_memory_write,
    agent_latency_ms,
    direct_latency_ms,
    agent_tokens,
]


ANSWER_CORRECTNESS_INSTRUCTIONS = """You are grading one answer from a long-term-memory benchmark.

The question and its metadata are in {{ inputs }}. The deployed agent execution is in {{ outputs }}.
The gold answer and retrieval ids are in {{ expectations }}.

This scorer is invoked only for answerable questions. Compare outputs.answer with
expectations.expected_answer. Return 1.0 when the answer contains all material facts and relations,
0.5 when it is directionally correct but misses a material detail, and 0.0 when it is wrong,
contradictory, unsupported, empty, or refuses to answer. Ignore wording and style. Use the retrieved
search evidence to detect entity mixups or claims that do not follow from the benchmark memories.
Give a short, concrete rationale.
"""


ABSTENTION_INSTRUCTIONS = """You are grading an unanswerable long-term-memory benchmark question.

The question is in {{ inputs }}, the deployed agent execution is in {{ outputs }}, and
expectations.expect_abstain is true. Return true only if outputs.answer clearly says the requested
fact is not known or not present in memory and does not invent a specific answer. A helpful statement
of what is known is allowed, but any unsupported answer to the requested fact is a failure. Give a
short, concrete rationale.
"""


def load_answer_judges(model: str = "databricks:/databricks-gpt-5-2") -> list[Any]:
    """Build optional MLflow judges for correctness and abstention."""

    from mlflow.genai.judges import make_judge

    correctness_judge = make_judge(
        name="chbench_answer_correctness",
        model=model,
        instructions=ANSWER_CORRECTNESS_INSTRUCTIONS,
        feedback_value_type=float,
    )
    abstention_judge = make_judge(
        name="chbench_abstention_correct",
        model=model,
        instructions=ABSTENTION_INSTRUCTIONS,
        feedback_value_type=bool,
    )

    @scorer(name="chbench_answer_correctness")
    def conditional_correctness(inputs, outputs, expectations):
        if _expects_abstention(expectations):
            return None
        return correctness_judge(inputs=inputs, outputs=outputs, expectations=expectations)

    @scorer(name="chbench_abstention_correct")
    def conditional_abstention(inputs, outputs, expectations):
        if not _expects_abstention(expectations):
            return None
        return abstention_judge(inputs=inputs, outputs=outputs, expectations=expectations)

    return [conditional_correctness, conditional_abstention]

"""Frozen realistic-memory pilot adapter for direct managed-memory retrieval.

The pilot source remains a readable local artifact. MLflow receives separate
question and corpus datasets, while evaluation reads one immutable, versioned
managed-memory scope and never invokes the deployed agent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
import uuid

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

try:  # Package import locally; top-level import beside a Workspace notebook.
    from . import memory_eval
except ImportError:  # pragma: no cover
    import memory_eval  # type: ignore[no-redef]


PILOT_VERSION = "pilot-v2"
SUITE_NAME = "realistic-memory-search-pilot-v2"
SOURCE_ARTIFACT = (
    Path(__file__).with_name("realistic_search_benchmark")
    / "artifacts"
    / "pilot_enriched.json"
)
QUESTION_DATASET_NAME = "kevinyan.default.realistic_memory_search_pilot_v2_queries"
CORPUS_DATASET_NAME = "kevinyan.default.realistic_memory_search_pilot_v2_memories"
MEMORY_PATH_PREFIX = "/memories/eval/realistic-search-pilot-v2/"
EXPECTED_CASE_COUNT = 25
EXPECTED_MEMORY_COUNT = 60
EXPECTED_CASE_FINGERPRINT = (
    "2245c72c21e6a521b298a5cb14cb90505298c904edd9f91b161e37a3ce44cfb2"
)
EXPECTED_CORPUS_FINGERPRINT = (
    "6faac0b1801c1e83e8e283323c5fb9fb5a6e62ff04ebcb4b0eb5e943d7c5589e"
)
DEFAULT_PERSISTENT_SCOPE = (
    f"realistic-search-{PILOT_VERSION}-{EXPECTED_CORPUS_FINGERPRINT[:8]}"
)


def memory_path(memory_id: str) -> str:
    if not memory_id.startswith("mem-") or not memory_id[4:].isdigit():
        raise ValueError(f"Invalid realistic-pilot memory id: {memory_id!r}")
    return f"{MEMORY_PATH_PREFIX}{memory_id}.md"


def memory_id_from_path(path: str) -> str | None:
    if not path.startswith(MEMORY_PATH_PREFIX) or not path.endswith(".md"):
        return None
    memory_id = path[len(MEMORY_PATH_PREFIX) : -len(".md")]
    return memory_id or None


def corpus_fingerprint(seed_memories: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        sorted(seed_memories, key=lambda entry: entry["path"]),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def case_fingerprint(cases: list[dict[str, Any]]) -> str:
    fields = (
        "case_id",
        "category",
        "situation",
        "query",
        "gold_memory_id",
        "context_memory_ids",
        "answer_should",
        "secondary_tags",
    )
    canonical_cases = [
        {field: case.get(field) for field in fields} for case in cases
    ]
    canonical = json.dumps(
        sorted(canonical_cases, key=lambda case: case["case_id"]),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_seed_memories(pilot: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "path": memory_path(str(memory["memory_id"])),
            "description": str(memory["description"]),
            "contents": str(memory.get("contents", "")),
        }
        for memory in pilot["corpus"]
    ]


def load_pilot(path: str | Path = SOURCE_ARTIFACT) -> dict[str, Any]:
    pilot = json.loads(Path(path).read_text())
    if pilot.get("schema_version") != "realistic-search-pilot-enriched-v2":
        raise ValueError("Realistic search pilot must use the enriched-v2 schema")
    if pilot.get("benchmark_as_of_date") != "2026-06-05":
        raise ValueError("Realistic search pilot has the wrong benchmark as-of date")
    if len(pilot.get("corpus", [])) != EXPECTED_MEMORY_COUNT:
        raise ValueError(f"Realistic search pilot must contain {EXPECTED_MEMORY_COUNT} memories")
    if len(pilot.get("cases", [])) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Realistic search pilot must contain {EXPECTED_CASE_COUNT} cases")

    memory_ids = [str(memory["memory_id"]) for memory in pilot["corpus"]]
    case_ids = [str(case["case_id"]) for case in pilot["cases"]]
    if len(set(memory_ids)) != EXPECTED_MEMORY_COUNT:
        raise ValueError("Realistic search pilot memory ids must be unique")
    if len(set(case_ids)) != EXPECTED_CASE_COUNT:
        raise ValueError("Realistic search pilot case ids must be unique")
    for case in pilot["cases"]:
        if case["gold_memory_id"] not in memory_ids:
            raise ValueError(f"Case {case['case_id']} has a missing gold memory")
        missing = set(case["context_memory_ids"]) - set(memory_ids)
        if missing:
            raise ValueError(
                f"Case {case['case_id']} has missing context memories: {sorted(missing)}"
            )

    seeds = build_seed_memories(pilot)
    actual_corpus_fingerprint = corpus_fingerprint(seeds)
    actual_case_fingerprint = case_fingerprint(pilot["cases"])
    if actual_corpus_fingerprint != EXPECTED_CORPUS_FINGERPRINT:
        raise ValueError(
            "Realistic pilot corpus fingerprint changed: "
            f"{actual_corpus_fingerprint}; expected {EXPECTED_CORPUS_FINGERPRINT}"
        )
    if actual_case_fingerprint != EXPECTED_CASE_FINGERPRINT:
        raise ValueError(
            "Realistic pilot case fingerprint changed: "
            f"{actual_case_fingerprint}; expected {EXPECTED_CASE_FINGERPRINT}"
        )
    return pilot


def _base_tags(category: str) -> dict[str, str]:
    return {
        "suite": SUITE_NAME,
        "pilot_version": PILOT_VERSION,
        "benchmark_as_of_date": "2026-06-05",
        "corpus_fingerprint": EXPECTED_CORPUS_FINGERPRINT,
        "case_fingerprint": EXPECTED_CASE_FINGERPRINT,
        "category": category,
    }


def build_question_dataset_records(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "case_id": str(case["case_id"]),
                "query": str(case["query"]),
                "category": str(case["category"]),
                "situation": str(case["situation"]),
            },
            "expectations": {
                "gold_memory_id": str(case["gold_memory_id"]),
                "answer_should": str(case["answer_should"]),
            },
            "tags": {
                **_base_tags(str(case["category"])),
                "secondary_tags": ",".join(case.get("secondary_tags", [])),
            },
        }
        for case in pilot["cases"]
    ]


def build_corpus_dataset_records(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = pilot["enrichment"]["profile_by_memory_id"]
    seeds_by_id = {
        memory["memory_id"]: seed
        for memory, seed in zip(pilot["corpus"], build_seed_memories(pilot))
    }
    return [
        {
            "inputs": {
                "memory_id": str(memory["memory_id"]),
                "category": str(memory["category"]),
                "content_profile": str(profiles[memory["memory_id"]]),
                **seeds_by_id[memory["memory_id"]],
            },
            "tags": {
                **_base_tags(str(memory["category"])),
                "content_profile": str(profiles[memory["memory_id"]]),
            },
        }
        for memory in pilot["corpus"]
    ]


def _dataset_rows(dataset_name: str, attempts: int = 3) -> list[dict[str, Any]]:
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
    else:  # pragma: no cover
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


def _validate_dataset_tags(
    dataset_name: str, rows: list[dict[str, Any]], expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Dataset {dataset_name} has {len(rows)} rows; expected {expected_count}")
    expected = {
        "suite": SUITE_NAME,
        "pilot_version": PILOT_VERSION,
        "benchmark_as_of_date": "2026-06-05",
        "corpus_fingerprint": EXPECTED_CORPUS_FINGERPRINT,
        "case_fingerprint": EXPECTED_CASE_FINGERPRINT,
    }
    for row in rows:
        mismatched = {
            key: row["tags"].get(key)
            for key, value in expected.items()
            if row["tags"].get(key) != value
        }
        if mismatched:
            raise ValueError(f"Dataset {dataset_name} has mismatched tags: {mismatched}")


def _load_question_records(dataset_name: str) -> list[dict[str, Any]]:
    rows = _dataset_rows(dataset_name)
    _validate_dataset_tags(dataset_name, rows, EXPECTED_CASE_COUNT)
    case_ids = [str(row["inputs"].get("case_id", "")) for row in rows]
    if "" in case_ids or len(set(case_ids)) != EXPECTED_CASE_COUNT:
        raise ValueError("Question dataset must contain 25 unique case ids")
    for row in rows:
        if not row["expectations"].get("gold_memory_id"):
            raise ValueError(f"Case {row['inputs']['case_id']} is missing gold_memory_id")
        if not row["expectations"].get("answer_should"):
            raise ValueError(f"Case {row['inputs']['case_id']} is missing answer_should")
    return rows


def load_question_records(
    dataset_name: str = QUESTION_DATASET_NAME,
    eval_scope: str = DEFAULT_PERSISTENT_SCOPE,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if top_k < 10:
        raise ValueError("top_k must be at least 10 for Hit@10")
    rows = _load_question_records(dataset_name)
    for row in rows:
        row["inputs"] = {
            **row["inputs"],
            "eval_scope": eval_scope,
            "top_k": top_k,
        }
    return rows


def load_autonomous_question_records(
    dataset_name: str = QUESTION_DATASET_NAME,
    eval_scope: str = DEFAULT_PERSISTENT_SCOPE,
) -> list[dict[str, Any]]:
    """Load natural questions for the end-to-end App track.

    Unlike the direct-retrieval adapter, this adds no retrieval parameters. The
    deployed agent receives the dataset query exactly as written.
    """

    rows = _load_question_records(dataset_name)
    for row in rows:
        row["inputs"] = {**row["inputs"], "eval_scope": eval_scope}
    return rows


def load_seed_memories(
    dataset_name: str = CORPUS_DATASET_NAME,
) -> list[dict[str, str]]:
    rows = _dataset_rows(dataset_name)
    _validate_dataset_tags(dataset_name, rows, EXPECTED_MEMORY_COUNT)
    memory_ids: set[str] = set()
    paths: set[str] = set()
    seeds: list[dict[str, str]] = []
    for row in rows:
        inputs = row["inputs"]
        memory_id = str(inputs.get("memory_id", ""))
        path = str(inputs.get("path", ""))
        if not memory_id or memory_id in memory_ids:
            raise ValueError("Corpus dataset has a missing or duplicate memory id")
        if path != memory_path(memory_id) or path in paths:
            raise ValueError(f"Corpus dataset has an invalid path for {memory_id}")
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
            f"Corpus dataset fingerprint is {fingerprint}; expected {EXPECTED_CORPUS_FINGERPRINT}"
        )
    return seeds


def validate_question_corpus_records(
    records: list[dict[str, Any]], seed_memories: list[dict[str, str]]
) -> None:
    memory_ids = {
        memory_id
        for entry in seed_memories
        if (memory_id := memory_id_from_path(entry["path"])) is not None
    }
    for record in records:
        gold = str(record["expectations"].get("gold_memory_id", ""))
        if gold not in memory_ids:
            raise ValueError(f"Case {record['inputs']['case_id']} references missing gold {gold}")


def inspect_persistent_entries(
    eval_scope: str, seed_memories: list[dict[str, str]]
) -> dict[str, Any]:
    expected = {
        entry["path"]: memory_eval._clean_entry(entry) for entry in seed_memories
    }
    fingerprint = corpus_fingerprint(seed_memories)
    if fingerprint != EXPECTED_CORPUS_FINGERPRINT:
        raise ValueError(f"Refusing unpinned corpus fingerprint {fingerprint}")
    full_scope = memory_eval._full_eval_scope(eval_scope)
    actual = memory_eval.snapshot_scope(full_scope)
    actual_fingerprint = corpus_fingerprint(list(actual.values()))
    diff = memory_eval.diff_snapshots(expected, actual)
    return {
        "eval_scope": eval_scope,
        "full_scope": full_scope,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "expected_fingerprint": fingerprint,
        "actual_fingerprint": actual_fingerprint,
        "exact": fingerprint == actual_fingerprint
        and not any(diff[kind] for kind in ("created", "updated", "deleted")),
        "missing": diff["deleted"],
        "unexpected": diff["created"],
        "changed": diff["updated"],
        "snapshot": actual,
    }


def persistent_corpus_error(status: dict[str, Any]) -> str:
    return (
        f"Persistent realistic-search corpus {status['full_scope']} does not match "
        f"{status['expected_fingerprint'][:12]}: {len(status['missing'])} missing, "
        f"{len(status['unexpected'])} unexpected, {len(status['changed'])} changed."
    )


def validate_persistent_entries(
    eval_scope: str, seed_memories: list[dict[str, str]]
) -> dict[str, Any]:
    status = inspect_persistent_entries(eval_scope, seed_memories)
    if not status["exact"]:
        raise RuntimeError(persistent_corpus_error(status))
    return status


def seed_persistent_entries(
    eval_scope: str,
    seed_memories: list[dict[str, str]],
    *,
    replace: bool = False,
    index_timeout_s: int = 600,
) -> dict[str, Any]:
    status = inspect_persistent_entries(eval_scope, seed_memories)
    if status["exact"]:
        memory_eval.wait_until_searchable(
            status["full_scope"], seed_memories, timeout_s=index_timeout_s
        )
        return {**status, "action": "already_exact"}
    if status["actual_count"] and not replace:
        raise RuntimeError(
            persistent_corpus_error(status) + " Refusing to overwrite without --replace."
        )

    if status["actual_count"]:
        memory_eval.cleanup_scope(status["full_scope"])
    try:
        memory_eval.seed_scope(status["full_scope"], seed_memories)
        stored = inspect_persistent_entries(eval_scope, seed_memories)
        if not stored["exact"]:
            raise RuntimeError(persistent_corpus_error(stored))
    except Exception:
        memory_eval.cleanup_scope(status["full_scope"])
        raise

    memory_eval.wait_until_searchable(
        status["full_scope"], seed_memories, timeout_s=index_timeout_s
    )
    final = inspect_persistent_entries(eval_scope, seed_memories)
    if not final["exact"]:
        raise RuntimeError(persistent_corpus_error(final))
    return {
        **final,
        "action": "replaced" if status["actual_count"] else "seeded",
    }


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


def direct_store_search(query: str, eval_scope: str, top_k: int) -> dict[str, Any]:
    scope = memory_eval._full_eval_scope(eval_scope)
    api = memory_eval._memory_module()
    started = time.perf_counter()
    response = api._ws().api_client.do(
        "POST",
        api._entries(":search"),
        query={"scope": scope},
        headers=api._headers(),
        body={"query": query, "top_k": top_k},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    sources = [
        _source_from_result(result, rank)
        for rank, result in enumerate(response.get("results", []), start=1)
    ]
    return {
        "query": query,
        "direct_sources": sources,
        "direct_ranked_ids": [source["id"] for source in sources],
        "direct_latency_ms": latency_ms,
    }


@mlflow.trace(name="realistic_memory_search", span_type="RETRIEVER")
def direct_predict_fn(
    case_id: str,
    query: str,
    category: str,
    situation: str,
    eval_scope: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search the memory store with the natural question verbatim."""

    result = direct_store_search(query, eval_scope, int(top_k))
    mlflow.update_current_trace(
        tags={
            "eval.suite": SUITE_NAME,
            "eval.case_id": case_id,
            "eval.category": category,
            "eval.search_mode": "direct-store-endpoint",
        },
        request_preview=f"{case_id} [{category}] {query}",
        response_preview=f"ranked={result['direct_ranked_ids'][:int(top_k)]}",
    )
    return {
        "case_id": case_id,
        "category": category,
        "situation": situation,
        **result,
    }


def _response_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    for key in ("total_tokens", "total_token_count"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        return int(input_tokens + output_tokens)
    return 0


@mlflow.trace(name="realistic_memory_autonomous_answer", span_type="CHAIN")
def autonomous_predict_fn(
    case_id: str,
    query: str,
    category: str,
    situation: str,
    eval_scope: str,
) -> dict[str, Any]:
    """Ask the deployed agent the natural question without forcing memory use."""

    safe_case_id = re.sub(r"[^a-z0-9-]+", "-", case_id.lower()).strip("-")
    thread_id = f"realistic-auto-{safe_case_id}-{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()
    response = memory_eval._invoke_turn_sync(
        query,
        eval_scope,
        thread_id,
        eval_read_only=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    answer, tool_calls = memory_eval.parse_agent_output(response)

    mlflow.update_current_trace(
        tags={
            "eval.suite": SUITE_NAME,
            "eval.case_id": case_id,
            "eval.category": category,
            "eval.agent_search_mode": "autonomous",
        },
        request_preview=f"{case_id} [{category}] {query}",
        response_preview=f"answer={answer[:300]}",
    )
    return {
        "case_id": case_id,
        "category": category,
        "situation": situation,
        "answer": answer,
        # Retained only for trace debugging. The autonomous scorer below never
        # receives tool calls or retrieval results.
        "tool_calls": tool_calls,
        "thread_id": thread_id,
        "app_latency_ms": latency_ms,
        "tokens": _response_tokens(response),
    }


AUTONOMOUS_ANSWER_JUDGE_NAME = "realistic_memory_answer_correctness"
AUTONOMOUS_ANSWER_JUDGE_VERSION = "v1"
AUTONOMOUS_ANSWER_JUDGE_INSTRUCTIONS = """You are grading only the final answer from a
realistic long-term-memory question.

The natural user question is in {{ inputs }}. The answer to grade is in {{ outputs }}. The only
reference rubric is in {{ expectations }}. These objects contain only query, answer, and answer_should,
respectively. You receive no tool calls, retrieved memories, gold memory ids, or other execution
evidence; do not assume any.

Return exactly one numeric score:
- 1.0: the answer satisfies every material requirement in answer_should and does not contradict it;
- 0.5: the core answer is directionally correct but omits an important requested fact or relation;
- 0.0: the answer is wrong, contradictory, vague enough to avoid the requested facts, unsupported by
  the rubric, empty, or refuses despite the expected answer being available.

Accept concise semantic paraphrases. Ignore prose style and do not require facts that answer_should
does not require. Give a short, concrete rationale naming any missing or incorrect requirement.
"""


def load_autonomous_answer_judge(
    model: str = "databricks:/databricks-gpt-5-2",
):
    """Build the sole scorer for the autonomous final-answer track.

    The wrapper deliberately strips all retrieval and trace evidence before the
    LLM judge is called, so only the answer and per-case answer rubric can affect
    the score.
    """

    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name=AUTONOMOUS_ANSWER_JUDGE_NAME,
        model=model,
        instructions=AUTONOMOUS_ANSWER_JUDGE_INSTRUCTIONS,
        feedback_value_type=float,
    )

    @scorer(name=AUTONOMOUS_ANSWER_JUDGE_NAME)
    def final_answer_correctness(inputs, outputs, expectations):
        return memory_eval._invoke_judge_with_retry(
            judge,
            inputs={"query": str((inputs or {}).get("query", ""))},
            outputs={"answer": str((outputs or {}).get("answer", ""))},
            expectations={
                "answer_should": str((expectations or {}).get("answer_should", ""))
            },
        )

    return final_answer_correctness


def _gold_id(expectations: dict[str, Any] | None) -> str:
    return str((expectations or {}).get("gold_memory_id", ""))


def _ranked(outputs: dict[str, Any] | None) -> list[str | None]:
    return list((outputs or {}).get("direct_ranked_ids", []))


def gold_rank(outputs: dict[str, Any] | None, expectations: dict[str, Any] | None) -> int | None:
    gold = _gold_id(expectations)
    try:
        return _ranked(outputs).index(gold) + 1
    except ValueError:
        return None


def _hit_feedback(
    k: int, outputs: dict[str, Any] | None, expectations: dict[str, Any] | None
) -> Feedback:
    gold = _gold_id(expectations)
    rank = gold_rank(outputs, expectations)
    return Feedback(
        value=rank is not None and rank <= k,
        rationale=f"gold={gold}; rank={rank}; top_{k}={_ranked(outputs)[:k]}",
    )


@scorer(name="realistic_search_hit_at_1")
def hit_at_1(outputs, expectations):
    return _hit_feedback(1, outputs, expectations)


@scorer(name="realistic_search_hit_at_3")
def hit_at_3(outputs, expectations):
    return _hit_feedback(3, outputs, expectations)


@scorer(name="realistic_search_hit_at_5")
def hit_at_5(outputs, expectations):
    return _hit_feedback(5, outputs, expectations)


@scorer(name="realistic_search_hit_at_10")
def hit_at_10(outputs, expectations):
    return _hit_feedback(10, outputs, expectations)


@scorer(name="realistic_search_mrr")
def mrr(outputs, expectations):
    gold = _gold_id(expectations)
    rank = gold_rank(outputs, expectations)
    return Feedback(
        value=0.0 if rank is None else 1.0 / rank,
        rationale=f"gold={gold}; rank={rank}; ranked={_ranked(outputs)}",
    )


@scorer(name="realistic_search_latency_ms")
def latency_ms(outputs):
    return float((outputs or {}).get("direct_latency_ms", 0.0))


DIRECT_SEARCH_SCORERS = [hit_at_1, hit_at_3, hit_at_5, hit_at_10, mrr, latency_ms]

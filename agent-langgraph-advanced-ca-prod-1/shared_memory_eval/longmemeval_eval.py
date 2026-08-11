"""Persistent-corpus evaluation helpers for the LongMemEval-S raw pilot."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Iterable
import uuid

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

try:  # Package import locally; top-level import beside a Workspace notebook.
    from . import longmemeval_adapter, memory_eval
    from .sync_longmemeval_datasets import QUESTION_DATASET_NAME
except ImportError:  # pragma: no cover
    import longmemeval_adapter  # type: ignore[no-redef]
    import memory_eval  # type: ignore[no-redef]
    from sync_longmemeval_datasets import QUESTION_DATASET_NAME  # type: ignore[no-redef]


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


def full_scope(scope_selector: str) -> str:
    """Return the exact test scope selected in deployed direct-eval mode."""

    selector = str(scope_selector)
    if not memory_eval.EVAL_SCOPE_PATTERN.fullmatch(selector):
        raise ValueError(f"Invalid LongMemEval scope selector: {selector!r}")
    return selector


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
        if not isinstance(raw.get("inputs"), dict):
            raise ValueError(f"Dataset {dataset_name} contains a row without inputs")
        rows.append(
            {
                "inputs": dict(raw["inputs"]),
                "expectations": (
                    dict(raw["expectations"])
                    if isinstance(raw.get("expectations"), dict)
                    else {}
                ),
                "tags": dict(raw["tags"]) if isinstance(raw.get("tags"), dict) else {},
            }
        )
    return rows


def _validate_provenance(dataset_name: str, rows: list[dict[str, Any]]) -> None:
    bad = [
        row
        for row in rows
        if row["tags"].get("suite") != longmemeval_adapter.SUITE_NAME
        or row["tags"].get("adapter_version")
        != longmemeval_adapter.ADAPTER_VERSION
        or row["tags"].get("source_commit") != longmemeval_adapter.SOURCE_COMMIT
    ]
    if bad:
        raise ValueError(
            f"Dataset {dataset_name} has {len(bad)} rows with invalid provenance"
        )


def load_question_records(
    question_dataset_name: str = QUESTION_DATASET_NAME,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Load and validate the compact question/gold managed dataset."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    questions = _dataset_rows(question_dataset_name)
    if len(questions) != longmemeval_adapter.EXPECTED_PILOT_QUESTION_COUNT:
        raise ValueError(
            f"Question dataset has {len(questions)} rows; expected "
            f"{longmemeval_adapter.EXPECTED_PILOT_QUESTION_COUNT}"
        )
    _validate_provenance(question_dataset_name, questions)
    actual_fingerprint = longmemeval_adapter.question_dataset_fingerprint(questions)
    if actual_fingerprint != longmemeval_adapter.EXPECTED_PILOT_QUESTION_FINGERPRINT:
        raise ValueError(
            f"Managed LongMemEval question fingerprint is {actual_fingerprint}; expected "
            f"{longmemeval_adapter.EXPECTED_PILOT_QUESTION_FINGERPRINT}"
        )
    case_ids: set[str] = set()
    for row in questions:
        inputs = row["inputs"]
        case = str(inputs.get("case_id", ""))
        if not case or case in case_ids:
            raise ValueError(f"Question dataset contains an invalid case id: {case!r}")
        case_ids.add(case)
        relevant = _relevant_ids(row.get("expectations"))
        if not relevant:
            raise ValueError(f"Question case {case} has no gold memory ids")
        expected_count = int(inputs.get("expected_memory_count", 0))
        expected_fingerprint = str(inputs.get("expected_corpus_fingerprint", ""))
        if expected_count <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint):
            raise ValueError(f"Question case {case} has invalid corpus expectations")
        if row["tags"].get("case_corpus_fingerprint") != expected_fingerprint:
            raise ValueError(f"Question case {case} has inconsistent corpus fingerprints")
        row["inputs"] = {**inputs, "top_k": top_k}
    return questions


def inspect_persistent_scope(
    scope_selector: str,
    expected_count: int,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Validate a fixture from the compact expectations stored with its question."""

    scope = full_scope(scope_selector)
    actual = memory_eval.snapshot_scope(scope)
    actual_fingerprint = longmemeval_adapter.corpus_fingerprint(list(actual.values()))
    return {
        "scope_selector": scope_selector,
        "full_scope": scope,
        "expected_count": int(expected_count),
        "actual_count": len(actual),
        "expected_fingerprint": str(expected_fingerprint),
        "actual_fingerprint": actual_fingerprint,
        "exact": len(actual) == int(expected_count)
        and actual_fingerprint == str(expected_fingerprint),
        "snapshot": actual,
    }


def validate_persistent_scope(
    scope_selector: str,
    expected_count: int,
    expected_fingerprint: str,
) -> dict[str, Any]:
    status = inspect_persistent_scope(
        scope_selector, expected_count, expected_fingerprint
    )
    if not status["exact"]:
        raise RuntimeError(
            f"LongMemEval scope {status['full_scope']} has "
            f"{status['actual_count']} entries and fingerprint "
            f"{status['actual_fingerprint'][:12]}; expected "
            f"{status['expected_count']} and {status['expected_fingerprint'][:12]}."
        )
    return status


def inspect_persistent_case(
    scope_selector: str, entries: list[dict[str, str]]
) -> dict[str, Any]:
    expected = {entry["path"]: memory_eval._clean_entry(entry) for entry in entries}
    scope = full_scope(scope_selector)
    actual = memory_eval.snapshot_scope(scope)
    diff = memory_eval.diff_snapshots(expected, actual)
    expected_fingerprint = longmemeval_adapter.corpus_fingerprint(entries)
    actual_fingerprint = longmemeval_adapter.corpus_fingerprint(list(actual.values()))
    return {
        "scope_selector": scope_selector,
        "full_scope": scope,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual_fingerprint,
        "exact": expected_fingerprint == actual_fingerprint
        and not any(diff[kind] for kind in ("created", "updated", "deleted")),
        "diff": diff,
        "snapshot": actual,
    }


def _case_error(status: dict[str, Any]) -> str:
    diff = status["diff"]
    return (
        f"LongMemEval scope {status['full_scope']} does not match "
        f"{status['expected_fingerprint'][:12]}: {len(diff['deleted'])} missing, "
        f"{len(diff['created'])} unexpected, {len(diff['updated'])} changed."
    )


def validate_persistent_case(
    scope_selector: str, entries: list[dict[str, str]]
) -> dict[str, Any]:
    status = inspect_persistent_case(scope_selector, entries)
    if not status["exact"]:
        raise RuntimeError(_case_error(status))
    return status


def seed_persistent_case(
    scope_selector: str,
    entries: list[dict[str, str]],
    *,
    replace: bool = False,
    index_timeout_s: int = 600,
    search_check_entries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Idempotently seed one case and optionally verify selected search-index entries."""

    status = inspect_persistent_case(scope_selector, entries)
    if status["exact"]:
        if search_check_entries:
            memory_eval.wait_until_searchable(
                status["full_scope"],
                search_check_entries,
                timeout_s=index_timeout_s,
            )
        return {
            **status,
            "action": "already_exact",
            "search_check_count": len(search_check_entries or []),
        }
    if status["actual_count"] and not replace:
        raise RuntimeError(_case_error(status) + " Refusing to overwrite without --replace.")

    if status["actual_count"]:
        memory_eval.cleanup_scope(status["full_scope"])
    try:
        memory_eval.seed_scope(status["full_scope"], entries)
        stored = inspect_persistent_case(scope_selector, entries)
        if not stored["exact"]:
            raise RuntimeError(_case_error(stored))
    except Exception:
        memory_eval.cleanup_scope(status["full_scope"])
        raise
    if search_check_entries:
        memory_eval.wait_until_searchable(
            status["full_scope"],
            search_check_entries,
            timeout_s=index_timeout_s,
        )
    final = inspect_persistent_case(scope_selector, entries)
    if not final["exact"]:
        raise RuntimeError(_case_error(final))
    return {
        **final,
        "action": "replaced" if status["actual_count"] else "seeded",
        "search_check_count": len(search_check_entries or []),
    }


def _source_from_result(result: dict[str, Any], rank: int) -> dict[str, Any]:
    entry = result.get("memory_entry", {})
    path = str(entry.get("path", ""))
    return {
        "id": longmemeval_adapter.memory_id_from_path(path),
        "path": path,
        "score": float(result.get("score", 0.0) or 0.0),
        "description": str(entry.get("description", "") or ""),
        "rank": rank,
    }


def direct_store_search(question: str, scope_selector: str, top_k: int) -> dict[str, Any]:
    scope = full_scope(scope_selector)
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
    searches: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for search_index, call in enumerate(
        (call for call in tool_calls if call.get("name") == "search_memory"), start=1
    ):
        output = str(call.get("output", "") or "")
        query = str(dict(call.get("arguments") or {}).get("query", ""))
        per_search: list[dict[str, Any]] = []
        for rank, path in enumerate(search_result_paths(output), start=1):
            source = {
                "id": longmemeval_adapter.memory_id_from_path(path),
                "path": path,
                "search_index": search_index,
                "search_rank": rank,
            }
            per_search.append(source)
            if path not in seen_paths:
                seen_paths.add(path)
                sources.append(source)
        searches.append({"query": query, "sources": per_search})
    return {
        "searches": searches,
        "sources": sources,
        "ranked_ids": [source["id"] for source in sources],
    }


def forced_memory_question(question: str) -> str:
    return f"{FORCED_MEMORY_INSTRUCTION}\n\nQuestion: {question}"


@mlflow.trace(name="longmemeval_direct_search", span_type="RETRIEVER")
def direct_predict_fn(
    case_id: str,
    question_id: str,
    question: str,
    question_type: str,
    scope_selector: str,
    question_date: str,
    expected_memory_count: int,
    expected_corpus_fingerprint: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search the managed-memory endpoint directly without invoking the App."""

    direct = direct_store_search(question, scope_selector, int(top_k))
    mlflow.update_current_trace(
        tags={
            "eval.suite": longmemeval_adapter.SUITE_NAME,
            "eval.question_id": question_id,
            "eval.question_type": question_type,
            "eval.search_mode": "direct-store-endpoint",
        },
        request_preview=f"{question_id} [{question_type}] {question}",
        response_preview=(
            f"ranked={direct['ranked_ids'][:int(top_k)]}; "
            f"latency_ms={direct['latency_ms']:.1f}"
        ),
    )
    return {
        "case_id": case_id,
        "question_id": question_id,
        "question_type": question_type,
        "question_date": question_date,
        "direct_sources": direct["sources"],
        "direct_ranked_ids": direct["ranked_ids"],
        "direct_latency_ms": direct["latency_ms"],
    }


@mlflow.trace(name="longmemeval_question", span_type="CHAIN")
def predict_fn(
    case_id: str,
    question_id: str,
    question: str,
    question_type: str,
    scope_selector: str,
    question_date: str,
    expected_memory_count: int,
    expected_corpus_fingerprint: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run direct retrieval plus a forced-search deployed-App answer."""

    direct = direct_store_search(question, scope_selector, int(top_k))
    agent_question = forced_memory_question(question)
    thread_id = f"lme-{case_id}-{uuid.uuid4().hex[:10]}"
    started = time.perf_counter()
    response = memory_eval._invoke_turn_sync(agent_question, scope_selector, thread_id)
    app_latency_ms = (time.perf_counter() - started) * 1000
    answer, tool_calls = memory_eval.parse_agent_output(response)
    agent_retrieval = parse_agent_sources(tool_calls)
    mutation_calls = [
        call for call in tool_calls if call.get("name") in MEMORY_MUTATION_TOOLS
    ]
    mlflow.update_current_trace(
        tags={
            "eval.suite": longmemeval_adapter.SUITE_NAME,
            "eval.question_id": question_id,
            "eval.question_type": question_type,
            "eval.agent_search_mode": "explicit",
        },
        request_preview=f"{question_id} [{question_type}] {question}",
        response_preview=(
            f"searches={len(agent_retrieval['searches'])}; "
            f"ranked={agent_retrieval['ranked_ids'][:int(top_k)]}; "
            f"answer={answer[:240]}"
        ),
    )
    return {
        "case_id": case_id,
        "question_id": question_id,
        "question_date": question_date,
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
    }


def _relevant_ids(expectations: dict[str, Any] | None) -> list[str]:
    raw = (expectations or {}).get("relevant_ids_json", "[]")
    values = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    return [str(value) for value in values]


def _metric(
    axis: str,
    metric_name: str,
    inputs: dict[str, Any] | None,
    outputs: dict[str, Any] | None,
    expectations: dict[str, Any] | None,
) -> Feedback:
    relevant = set(_relevant_ids(expectations))
    ranked = list((outputs or {}).get(f"{axis}_ranked_ids", []))
    k = int((inputs or {}).get("top_k", 10))
    top = ranked[:k]
    if metric_name == "recall_at_k":
        value = len(set(top) & relevant) / len(relevant)
    elif metric_name == "hit_at_k":
        value = float(any(item in relevant for item in top))
    elif metric_name == "mrr":
        value = next(
            (1.0 / rank for rank, item in enumerate(ranked, start=1) if item in relevant),
            0.0,
        )
    elif metric_name == "ndcg_at_k":
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(top, start=1)
            if item in relevant
        )
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), k) + 1)
        )
        value = dcg / idcg if idcg else 0.0
    else:  # pragma: no cover
        raise ValueError(metric_name)
    return Feedback(
        value=value,
        rationale=f"{axis} {metric_name}: relevant={sorted(relevant)}; ranked={top}; k={k}.",
    )


@scorer(name="lme_direct_recall_at_k")
def direct_recall_at_k(inputs, outputs, expectations):
    return _metric("direct", "recall_at_k", inputs, outputs, expectations)


@scorer(name="lme_direct_hit_at_k")
def direct_hit_at_k(inputs, outputs, expectations):
    return _metric("direct", "hit_at_k", inputs, outputs, expectations)


@scorer(name="lme_direct_mrr")
def direct_mrr(inputs, outputs, expectations):
    return _metric("direct", "mrr", inputs, outputs, expectations)


@scorer(name="lme_direct_ndcg_at_k")
def direct_ndcg_at_k(inputs, outputs, expectations):
    return _metric("direct", "ndcg_at_k", inputs, outputs, expectations)


@scorer(name="lme_agent_recall_at_k")
def agent_recall_at_k(inputs, outputs, expectations):
    return _metric("agent", "recall_at_k", inputs, outputs, expectations)


@scorer(name="lme_agent_hit_at_k")
def agent_hit_at_k(inputs, outputs, expectations):
    return _metric("agent", "hit_at_k", inputs, outputs, expectations)


@scorer(name="lme_agent_mrr")
def agent_mrr(inputs, outputs, expectations):
    return _metric("agent", "mrr", inputs, outputs, expectations)


@scorer(name="lme_agent_ndcg_at_k")
def agent_ndcg_at_k(inputs, outputs, expectations):
    return _metric("agent", "ndcg_at_k", inputs, outputs, expectations)


@scorer(name="lme_agent_search_called")
def agent_search_called(outputs):
    searches = list((outputs or {}).get("agent_searches", []))
    return Feedback(
        value=bool(searches),
        rationale=f"The agent made {len(searches)} search_memory call(s).",
    )


@scorer(name="lme_no_memory_write")
def no_memory_write(outputs):
    calls = list((outputs or {}).get("mutation_calls", []))
    return Feedback(
        value=not calls,
        rationale=(
            "No memory mutation tools were called."
            if not calls
            else f"Unexpected query-time memory mutations: {[c.get('name') for c in calls]}."
        ),
    )


DETERMINISTIC_SCORERS = [
    direct_recall_at_k,
    direct_hit_at_k,
    direct_mrr,
    direct_ndcg_at_k,
    agent_recall_at_k,
    agent_hit_at_k,
    agent_mrr,
    agent_ndcg_at_k,
    agent_search_called,
    no_memory_write,
]

DIRECT_SEARCH_SCORERS = [
    direct_recall_at_k,
    direct_hit_at_k,
    direct_mrr,
    direct_ndcg_at_k,
]


ANSWER_CORRECTNESS_INSTRUCTIONS = """You are grading an answer from LongMemEval.

The question and question_type are in {{ inputs }}. The deployed agent result is in {{ outputs }}.
The reference data is in {{ expectations }}. Use expectations.expected_answer as the reference answer.
Do not treat fixture metadata in inputs, such as expected_memory_count, as answer evidence.

Score outputs.answer as 1.0 when it gives the materially correct answer, 0.5 when it is directionally
correct but misses an important requested detail, and 0.0 when it is wrong, contradictory, empty,
unsupported, or refuses despite sufficient information. Ignore wording and style.

Apply these LongMemEval category rules:
- temporal-reasoning: tolerate an off-by-one result when counting days, weeks, or months;
- knowledge-update: the updated value must be present; mentioning an older value as history is allowed;
- single-session-preference: the response need not repeat the rubric verbatim, but it must correctly
  use the remembered preference in its recommendation;
- all other categories: all material facts and relations in the reference answer must be present.

Give a short, concrete rationale.
"""


def load_answer_judge(model: str = "databricks:/databricks-gpt-5-2") -> Any:
    from mlflow.genai.judges import make_judge

    return make_judge(
        name="longmemeval_answer_correctness",
        model=model,
        instructions=ANSWER_CORRECTNESS_INSTRUCTIONS,
        feedback_value_type=float,
    )

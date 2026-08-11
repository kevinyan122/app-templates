"""Convert LongMemEval-S histories into leakage-safe managed-memory fixtures.

LongMemEval gives every question its own synthetic conversation history.  This
adapter keeps that boundary: each selected question gets a distinct evaluation
scope and every source session becomes one memory entry in that scope.

The raw-session adapter is deliberately lossless.  It strips benchmark labels
(``has_answer`` and the source session id) from the searchable entry but keeps
the complete user/assistant transcript.  A later distilled adapter can be
compared with this baseline without changing question selection or gold ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


SOURCE_REPOSITORY = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
SOURCE_COMMIT = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
SOURCE_FILENAME = "longmemeval_s_cleaned.json"
SOURCE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SOURCE_URL = f"{SOURCE_REPOSITORY}/resolve/{SOURCE_COMMIT}/{SOURCE_FILENAME}"

ADAPTER_VERSION = "lme-s-raw-v1"
SUITE_NAME = "longmemeval-s-raw-pilot"
DEFAULT_CASES_PER_CATEGORY = 5
EXPECTED_PILOT_QUESTION_COUNT = 30
EXPECTED_PILOT_MEMORY_COUNT = 1444
EXPECTED_PILOT_GOLD_MEMORY_COUNT = 48
EXPECTED_PILOT_BUNDLE_FINGERPRINT = (
    "25d4387180fdc1f2e7f5fcc57e6be9d0ddf744b0215ebf27bb1577b840a952c3"
)
EXPECTED_PILOT_QUESTION_FINGERPRINT = (
    "ca39789cc63dadb6f0468815ac7006e541e0782506da72ef877cee006d9d9c60"
)
QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)
MEMORY_PATH_PREFIX = "/memories/eval/longmemeval/raw-v1/"
EXPECTED_SOURCE_QUESTION_COUNT = 500
_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<date>\d{4}/\d{2}/\d{2})(?: \([^)]+\))? (?P<time>\d{2}:\d{2})$"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = _canonical_json(value)
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: str | Path, *, verify_hash: bool = True) -> list[dict[str, Any]]:
    """Load the pinned cleaned LongMemEval-S source file."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"LongMemEval source not found: {source_path}. Download {SOURCE_URL}"
        )
    if verify_hash:
        actual_hash = _file_sha256(source_path)
        if actual_hash != SOURCE_SHA256:
            raise ValueError(
                f"LongMemEval source hash is {actual_hash}; expected {SOURCE_SHA256}"
            )

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LongMemEval source must be a JSON array")
    if verify_hash and len(payload) != EXPECTED_SOURCE_QUESTION_COUNT:
        raise ValueError(
            f"LongMemEval source has {len(payload)} questions; "
            f"expected {EXPECTED_SOURCE_QUESTION_COUNT}"
        )
    return payload


def _selection_key(case: dict[str, Any]) -> str:
    return _sha256(f"{ADAPTER_VERSION}:{case['question_id']}")


def select_pilot_cases(
    cases: Iterable[dict[str, Any]],
    *,
    cases_per_category: int = DEFAULT_CASES_PER_CATEGORY,
) -> list[dict[str, Any]]:
    """Select a stable, non-abstention sample from every question type."""

    if cases_per_category <= 0:
        raise ValueError("cases_per_category must be positive")
    all_cases = list(cases)
    selected: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for case in all_cases:
        question_id = str(case.get("question_id", ""))
        if not question_id or question_id in seen_question_ids:
            raise ValueError(f"Missing or duplicate LongMemEval question id: {question_id!r}")
        seen_question_ids.add(question_id)

    for question_type in QUESTION_TYPES:
        candidates = [
            case
            for case in all_cases
            if case.get("question_type") == question_type
            and not str(case["question_id"]).endswith("_abs")
        ]
        if len(candidates) < cases_per_category:
            raise ValueError(
                f"Only {len(candidates)} non-abstention {question_type} cases; "
                f"need {cases_per_category}"
            )
        selected.extend(sorted(candidates, key=_selection_key)[:cases_per_category])
    return selected


def case_id(question_id: str) -> str:
    """Return an opaque stable id for dataset joins and scope selection."""

    return f"c-{_sha256(f'{ADAPTER_VERSION}:case:{question_id}')[:16]}"


def scope_selector(question_id: str) -> str:
    """Return the selector namespaced beneath verified OBO identity by the App."""

    return f"{ADAPTER_VERSION}-{case_id(question_id)}"


def memory_id(
    question_id: str,
    source_session_id: str,
    source_timestamp: str,
    source_position: int,
) -> str:
    """Return an opaque id that never exposes LongMemEval's ``answer_`` marker."""

    source_key = {
        "adapter_version": ADAPTER_VERSION,
        "question_id": question_id,
        "source_session_id": source_session_id,
        "source_timestamp": source_timestamp,
        "source_position": source_position,
    }
    return f"m-{_sha256(source_key)[:20]}"


def memory_path(value: str) -> str:
    if not re.fullmatch(r"m-[0-9a-f]{20}", value):
        raise ValueError(f"Invalid LongMemEval memory id: {value!r}")
    return f"{MEMORY_PATH_PREFIX}{value}.md"


def memory_id_from_path(path: str) -> str | None:
    if not path.startswith(MEMORY_PATH_PREFIX) or not path.endswith(".md"):
        return None
    value = path[len(MEMORY_PATH_PREFIX) : -len(".md")]
    return value if re.fullmatch(r"m-[0-9a-f]{20}", value) else None


def sanitize_session(session: Any) -> list[dict[str, str]]:
    """Keep only role/content, removing all benchmark-only turn labels."""

    if not isinstance(session, list) or not session:
        raise ValueError("Every LongMemEval session must be a non-empty list")
    clean: list[dict[str, str]] = []
    for turn in session:
        if not isinstance(turn, dict):
            raise ValueError("Every LongMemEval turn must be an object")
        role = str(turn.get("role", ""))
        content = turn.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unexpected LongMemEval role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Every LongMemEval turn must have non-empty string content")
        clean.append({"role": role, "content": content})
    return clean


def _description(timestamp: str) -> str:
    match = _TIMESTAMP_PATTERN.fullmatch(timestamp)
    if not match:
        raise ValueError(f"Unexpected LongMemEval timestamp: {timestamp!r}")
    date = match.group("date").replace("/", "-")
    return f"Conversation on {date} at {match.group('time')}"


def _contents(timestamp: str, session: list[dict[str, str]]) -> str:
    lines = [f"Conversation timestamp: {timestamp}"]
    for turn in session:
        lines.extend(("", f"{turn['role'].title()}: {turn['content']}"))
    return "\n".join(lines)


def _validated_case_sessions(
    case: dict[str, Any],
) -> list[tuple[int, str, str, list[dict[str, str]]]]:
    source_ids = case.get("haystack_session_ids")
    timestamps = case.get("haystack_dates")
    sessions = case.get("haystack_sessions")
    if not all(isinstance(value, list) for value in (source_ids, timestamps, sessions)):
        raise ValueError(f"Case {case.get('question_id')} has malformed session arrays")
    if not (len(source_ids) == len(timestamps) == len(sessions)):
        raise ValueError(f"Case {case.get('question_id')} has misaligned session arrays")
    return [
        (position, str(source_id), str(timestamp), sanitize_session(session))
        for position, (source_id, timestamp, session) in enumerate(
            zip(source_ids, timestamps, sessions)
        )
    ]


def build_case_memories(case: dict[str, Any]) -> list[dict[str, str]]:
    """Build exactly the entries persisted into this question's scope."""

    question_id = str(case["question_id"])
    entries: list[dict[str, str]] = []
    for source_position, source_id, timestamp, session in _validated_case_sessions(case):
        opaque_id = memory_id(
            question_id, source_id, timestamp, source_position
        )
        entry = {
            "path": memory_path(opaque_id),
            "description": _description(timestamp),
            "contents": _contents(timestamp, session),
        }
        if source_id in entry["path"] or "answer_" in entry["path"]:
            raise AssertionError("Source labels must never appear in managed-memory paths")
        entries.append(entry)
    return entries


def corpus_fingerprint(entries: list[dict[str, str]]) -> str:
    canonical = sorted(entries, key=lambda entry: entry["path"])
    return _sha256(canonical)


def build_corpus_dataset_records(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Return corpus rows, including provenance that is never seeded into memory."""

    question_id = str(case["question_id"])
    opaque_case_id = case_id(question_id)
    selector = scope_selector(question_id)
    sessions = _validated_case_sessions(case)
    entries = build_case_memories(case)
    fingerprint = corpus_fingerprint(entries)
    records: list[dict[str, Any]] = []
    for (source_position, source_id, timestamp, clean_session), entry in zip(
        sessions, entries
    ):
        opaque_memory_id = memory_id(
            question_id, source_id, timestamp, source_position
        )
        records.append(
            {
                "inputs": {
                    "case_id": opaque_case_id,
                    "scope_selector": selector,
                    "memory_id": opaque_memory_id,
                    **entry,
                    "source_session_id": source_id,
                    "source_timestamp": timestamp,
                    "source_position": source_position,
                    "source_content_sha256": _sha256(clean_session),
                },
                "tags": {
                    "suite": SUITE_NAME,
                    "adapter_version": ADAPTER_VERSION,
                    "source_commit": SOURCE_COMMIT,
                    "case_id": opaque_case_id,
                    "question_type": str(case["question_type"]),
                    "case_corpus_fingerprint": fingerprint,
                },
            }
        )
    return records


def build_question_dataset_record(case: dict[str, Any]) -> dict[str, Any]:
    question_id = str(case["question_id"])
    gold_source_ids = set(map(str, case["answer_session_ids"]))
    relevant_ids = [
        memory_id(question_id, source_id, timestamp, source_position)
        for source_position, source_id, timestamp, _session in _validated_case_sessions(case)
        if source_id in gold_source_ids
    ]
    if not relevant_ids:
        raise ValueError(f"Pilot case {question_id} must have at least one gold session")
    entries = build_case_memories(case)
    case_fingerprint = corpus_fingerprint(entries)
    return {
        "inputs": {
            "case_id": case_id(question_id),
            "question_id": question_id,
            "question": str(case["question"]),
            "question_type": str(case["question_type"]),
            "scope_selector": scope_selector(question_id),
            "question_date": str(case["question_date"]),
            "expected_memory_count": len(entries),
            "expected_corpus_fingerprint": case_fingerprint,
        },
        "expectations": {
            "expected_answer": str(case["answer"]),
            "relevant_ids_json": json.dumps(relevant_ids),
            "expect_abstain": False,
        },
        "tags": {
            "suite": SUITE_NAME,
            "adapter_version": ADAPTER_VERSION,
            "source_commit": SOURCE_COMMIT,
            "case_id": case_id(question_id),
            "question_type": str(case["question_type"]),
            "case_corpus_fingerprint": case_fingerprint,
        },
    }


def bundle_fingerprint(
    question_records: list[dict[str, Any]],
    corpus_records: list[dict[str, Any]],
) -> str:
    return _sha256(
        {
            "questions": sorted(
                question_records, key=lambda row: row["inputs"]["case_id"]
            ),
            "corpus": sorted(
                corpus_records,
                key=lambda row: (
                    row["inputs"]["case_id"],
                    row["inputs"]["memory_id"],
                ),
            ),
        }
    )


def question_dataset_fingerprint(question_records: list[dict[str, Any]]) -> str:
    normalized = []
    for record in question_records:
        row = {
            "inputs": dict(record["inputs"]),
            "expectations": dict(record.get("expectations") or {}),
            "tags": dict(record.get("tags") or {}),
        }
        # MLflow managed datasets round integral numeric inputs through a
        # floating Arrow column. Normalize that lossless representation before
        # hashing so local source rows and their remote round-trip stay equal.
        row["inputs"]["expected_memory_count"] = int(
            row["inputs"]["expected_memory_count"]
        )
        normalized.append(row)
    return _sha256(
        sorted(normalized, key=lambda row: row["inputs"]["case_id"])
    )


def build_pilot_bundle(
    cases: Iterable[dict[str, Any]],
    *,
    cases_per_category: int = DEFAULT_CASES_PER_CATEGORY,
) -> dict[str, Any]:
    selected = select_pilot_cases(cases, cases_per_category=cases_per_category)
    questions = [build_question_dataset_record(case) for case in selected]
    corpus = [
        record
        for case in selected
        for record in build_corpus_dataset_records(case)
    ]
    fingerprint = bundle_fingerprint(questions, corpus)
    questions_fingerprint = question_dataset_fingerprint(questions)
    category_counts = {
        question_type: sum(
            row["inputs"]["question_type"] == question_type for row in questions
        )
        for question_type in QUESTION_TYPES
    }
    return {
        "metadata": {
            "suite": SUITE_NAME,
            "adapter_version": ADAPTER_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "source_filename": SOURCE_FILENAME,
            "source_sha256": SOURCE_SHA256,
            "cases_per_category": cases_per_category,
            "question_count": len(questions),
            "memory_count": len(corpus),
            "gold_memory_count": sum(
                len(json.loads(row["expectations"]["relevant_ids_json"]))
                for row in questions
            ),
            "category_counts": category_counts,
            "selected_cases": [
                {
                    "question_id": row["inputs"]["question_id"],
                    "case_id": row["inputs"]["case_id"],
                    "question_type": row["inputs"]["question_type"],
                    "scope_selector": row["inputs"]["scope_selector"],
                    "memory_count": sum(
                        corpus_row["inputs"]["case_id"] == row["inputs"]["case_id"]
                        for corpus_row in corpus
                    ),
                    "gold_memory_count": len(
                        json.loads(row["expectations"]["relevant_ids_json"])
                    ),
                }
                for row in questions
            ],
            "bundle_fingerprint": fingerprint,
            "question_dataset_fingerprint": questions_fingerprint,
        },
        "questions": questions,
        "corpus": corpus,
    }


def validate_official_pilot_bundle(bundle: dict[str, Any]) -> None:
    """Fail if the pinned 5-per-category fixture drifts unexpectedly."""

    metadata = bundle.get("metadata", {})
    expected = {
        "cases_per_category": DEFAULT_CASES_PER_CATEGORY,
        "question_count": EXPECTED_PILOT_QUESTION_COUNT,
        "memory_count": EXPECTED_PILOT_MEMORY_COUNT,
        "gold_memory_count": EXPECTED_PILOT_GOLD_MEMORY_COUNT,
        "bundle_fingerprint": EXPECTED_PILOT_BUNDLE_FINGERPRINT,
        "question_dataset_fingerprint": EXPECTED_PILOT_QUESTION_FINGERPRINT,
    }
    changed = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if changed:
        raise ValueError(f"Pinned LongMemEval pilot bundle drifted: {changed}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_pilot_bundle(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"
    questions_path = output_root / "questions.jsonl"
    corpus_path = output_root / "corpus.jsonl"
    metadata_path.write_text(
        json.dumps(bundle["metadata"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(questions_path, bundle["questions"])
    _write_jsonl(corpus_path, bundle["corpus"])
    return {
        "metadata": str(metadata_path),
        "questions": str(questions_path),
        "corpus": str(corpus_path),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the leakage-safe LongMemEval-S raw-session pilot bundle."
    )
    parser.add_argument(
        "--source",
        default=os.getenv("LONGMEMEVAL_SOURCE", ""),
        help="Path to the pinned longmemeval_s_cleaned.json source file.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cases-per-category", type=int, default=DEFAULT_CASES_PER_CATEGORY
    )
    parser.add_argument(
        "--skip-source-hash-check",
        action="store_true",
        help="For local schema tests only; published fixtures must verify the pinned hash.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.source:
        raise ValueError(
            "Pass --source or set LONGMEMEVAL_SOURCE. "
            f"The pinned source is {SOURCE_URL}"
        )
    source = load_source(args.source, verify_hash=not args.skip_source_hash_check)
    bundle = build_pilot_bundle(
        source, cases_per_category=args.cases_per_category
    )
    if not args.skip_source_hash_check and (
        args.cases_per_category == DEFAULT_CASES_PER_CATEGORY
    ):
        validate_official_pilot_bundle(bundle)
    paths = write_pilot_bundle(bundle, args.output_dir)
    print(json.dumps({"metadata": bundle["metadata"], "files": paths}, indent=2))


if __name__ == "__main__":
    main()

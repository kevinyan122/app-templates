import json
from pathlib import Path

import pytest

from shared_memory_eval import longmemeval_adapter, longmemeval_eval


def _session(label: str, *, answer: bool = False):
    return [
        {
            "role": "user",
            "content": f"User fact {label}",
            "has_answer": answer,
        },
        {
            "role": "assistant",
            "content": f"Assistant detail {label}",
            "has_answer": False,
        },
    ]


def _case(question_type: str, index: int, *, abstention: bool = False):
    suffix = "_abs" if abstention else ""
    question_id = f"{question_type}-{index}{suffix}"
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": f"Question {index}?",
        "answer": f"Answer {index}",
        "question_date": "2023/06/01 (Thu) 12:00",
        "haystack_dates": [
            "2023/05/01 (Mon) 08:00",
            "2023/05/02 (Tue) 09:30",
        ],
        "haystack_session_ids": [
            f"distractor_{question_id}",
            f"answer_{question_id}",
        ],
        "haystack_sessions": [
            _session(f"noise-{index}"),
            _session(f"gold-{index}", answer=True),
        ],
        "answer_session_ids": [f"answer_{question_id}"],
    }


def _suite(per_type: int = 4):
    return [
        _case(question_type, index)
        for question_type in longmemeval_adapter.QUESTION_TYPES
        for index in range(per_type)
    ] + [
        _case(question_type, 99, abstention=True)
        for question_type in longmemeval_adapter.QUESTION_TYPES
    ]


def test_selection_is_balanced_stable_and_excludes_abstention():
    cases = _suite()
    selected = longmemeval_adapter.select_pilot_cases(cases, cases_per_category=2)
    reversed_selected = longmemeval_adapter.select_pilot_cases(
        reversed(cases), cases_per_category=2
    )

    assert len(selected) == 12
    assert [case["question_id"] for case in selected] == [
        case["question_id"] for case in reversed_selected
    ]
    assert not any(case["question_id"].endswith("_abs") for case in selected)
    assert {
        question_type: sum(case["question_type"] == question_type for case in selected)
        for question_type in longmemeval_adapter.QUESTION_TYPES
    } == {question_type: 2 for question_type in longmemeval_adapter.QUESTION_TYPES}


def test_raw_entries_remove_benchmark_labels_but_keep_the_full_transcript():
    case = _case("knowledge-update", 1)
    entries = longmemeval_adapter.build_case_memories(case)

    assert len(entries) == 2
    gold = entries[1]
    assert gold["description"] == "Conversation on 2023-05-02 at 09:30"
    assert "User fact gold-1" in gold["contents"]
    assert "Assistant detail gold-1" in gold["contents"]
    assert "has_answer" not in gold["contents"]
    assert "answer_knowledge-update-1" not in json.dumps(gold)
    assert longmemeval_adapter.memory_id_from_path(gold["path"]).startswith("m-")


def test_question_gold_ids_join_to_opaque_corpus_ids():
    case = _case("multi-session", 1)
    question = longmemeval_adapter.build_question_dataset_record(case)
    corpus = longmemeval_adapter.build_corpus_dataset_records(case)

    relevant = json.loads(question["expectations"]["relevant_ids_json"])
    corpus_ids = {row["inputs"]["memory_id"] for row in corpus}
    assert len(relevant) == 1
    assert set(relevant) <= corpus_ids
    assert all("answer" not in value for value in relevant)
    assert all("answer" not in row["inputs"]["path"] for row in corpus)
    assert any(
        row["inputs"]["source_session_id"].startswith("answer_") for row in corpus
    )


def test_each_case_gets_a_distinct_scope_and_preserves_conflicting_sessions():
    old = _case("knowledge-update", 1)
    new = _case("knowledge-update", 2)
    old["haystack_sessions"][1][0]["content"] = "Current personal best is 27:12"
    new["haystack_sessions"][1][0]["content"] = "Current personal best is 25:50"

    assert longmemeval_adapter.scope_selector(old["question_id"]) != (
        longmemeval_adapter.scope_selector(new["question_id"])
    )
    assert "27:12" in longmemeval_adapter.build_case_memories(old)[1]["contents"]
    assert "25:50" in longmemeval_adapter.build_case_memories(new)[1]["contents"]


def test_repeated_source_session_ids_remain_distinct_occurrences():
    case = _case("single-session-user", 1)
    case["haystack_session_ids"] = ["reused", "reused"]

    records = longmemeval_adapter.build_corpus_dataset_records(case)

    assert len({row["inputs"]["memory_id"] for row in records}) == 2
    assert [row["inputs"]["source_position"] for row in records] == [0, 1]


def test_bundle_and_written_files_are_reproducible(tmp_path: Path):
    cases = _suite(per_type=3)
    bundle = longmemeval_adapter.build_pilot_bundle(cases, cases_per_category=2)
    reversed_bundle = longmemeval_adapter.build_pilot_bundle(
        reversed(cases), cases_per_category=2
    )

    assert bundle["metadata"]["question_count"] == 12
    assert bundle["metadata"]["memory_count"] == 24
    assert bundle["metadata"]["gold_memory_count"] == 12
    assert bundle["metadata"]["bundle_fingerprint"] == (
        reversed_bundle["metadata"]["bundle_fingerprint"]
    )

    paths = longmemeval_adapter.write_pilot_bundle(bundle, tmp_path)
    metadata = json.loads(Path(paths["metadata"]).read_text())
    question_rows = Path(paths["questions"]).read_text().splitlines()
    corpus_rows = Path(paths["corpus"]).read_text().splitlines()
    assert metadata["bundle_fingerprint"] == bundle["metadata"]["bundle_fingerprint"]
    assert len(question_rows) == 12
    assert len(corpus_rows) == 24


def test_source_hash_mismatch_fails_closed(tmp_path: Path):
    path = tmp_path / "source.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="source hash"):
        longmemeval_adapter.load_source(path)


def test_eval_scope_uses_the_exact_selector_on_the_dedicated_test_app():
    selector = longmemeval_adapter.scope_selector("question-1")

    assert longmemeval_eval.full_scope(selector) == selector


def test_retrieval_metrics_keep_recall_all_separate_from_hit():
    inputs = {"top_k": 3}
    expectations = {"relevant_ids_json": '["gold-a", "gold-b"]'}
    outputs = {"direct_ranked_ids": ["noise", "gold-a", "noise-2"]}

    assert longmemeval_eval.direct_hit_at_k(
        inputs=inputs, outputs=outputs, expectations=expectations
    ).value == 1.0
    assert longmemeval_eval.direct_recall_at_k(
        inputs=inputs, outputs=outputs, expectations=expectations
    ).value == 0.5


def test_agent_search_parser_uses_opaque_longmemeval_paths():
    memory = "m-0123456789abcdefabcd"
    path = longmemeval_adapter.memory_path(memory)
    calls = [
        {
            "name": "search_memory",
            "arguments": {"query": "personal best"},
            "output": (
                "1 memory match (ranked index):\n"
                f"- path: {path}\n"
                "  description: Conversation on 2023-05-30 at 13:53\n"
                "  has_contents: true"
            ),
        }
    ]

    parsed = longmemeval_eval.parse_agent_sources(calls)

    assert parsed["ranked_ids"] == [memory]
    assert parsed["searches"][0]["query"] == "personal best"


def test_predictor_accepts_every_published_input_field(monkeypatch):
    row = longmemeval_adapter.build_question_dataset_record(
        _case("single-session-user", 1)
    )
    row["inputs"]["top_k"] = 10
    gold_id = json.loads(row["expectations"]["relevant_ids_json"])[0]
    gold_path = longmemeval_adapter.memory_path(gold_id)

    monkeypatch.setattr(
        longmemeval_eval,
        "direct_store_search",
        lambda question, scope_selector, top_k: {
            "query": question,
            "sources": [{"id": gold_id, "path": gold_path, "rank": 1}],
            "ranked_ids": [gold_id],
            "latency_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        longmemeval_eval.memory_eval,
        "_invoke_turn_sync",
        lambda question, scope_selector, thread_id: {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "search_memory",
                    "arguments": '{"query":"User fact gold-1"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": json.dumps(
                        [
                            {
                                "path": gold_path,
                                "description": "Conversation on 2023-05-02 at 09:30",
                                "has_contents": True,
                            }
                        ]
                    ),
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Answer 1"}],
                },
            ]
        },
    )
    monkeypatch.setattr(
        longmemeval_eval.mlflow, "update_current_trace", lambda **kwargs: None
    )

    result = longmemeval_eval.predict_fn.__wrapped__(**row["inputs"])

    assert result["question_id"] == row["inputs"]["question_id"]
    assert result["direct_ranked_ids"] == [gold_id]
    assert result["agent_ranked_ids"] == [gold_id]
    assert result["answer"] == "Answer 1"


def test_direct_predictor_never_invokes_the_deployed_app(monkeypatch):
    row = longmemeval_adapter.build_question_dataset_record(
        _case("single-session-user", 1)
    )
    row["inputs"]["top_k"] = 10
    gold_id = json.loads(row["expectations"]["relevant_ids_json"])[0]
    monkeypatch.setattr(
        longmemeval_eval,
        "direct_store_search",
        lambda question, scope_selector, top_k: {
            "query": question,
            "sources": [{"id": gold_id, "rank": 1}],
            "ranked_ids": [gold_id],
            "latency_ms": 2.0,
        },
    )
    monkeypatch.setattr(
        longmemeval_eval.memory_eval,
        "_invoke_turn_sync",
        lambda *args, **kwargs: pytest.fail("direct search must not invoke the App"),
    )
    monkeypatch.setattr(
        longmemeval_eval.mlflow, "update_current_trace", lambda **kwargs: None
    )

    result = longmemeval_eval.direct_predict_fn.__wrapped__(**row["inputs"])

    assert result["direct_ranked_ids"] == [gold_id]
    assert result["direct_latency_ms"] == 2.0
    assert "agent_ranked_ids" not in result


def test_answer_judge_receives_reference_expectations_explicitly():
    instructions = longmemeval_eval.ANSWER_CORRECTNESS_INSTRUCTIONS

    assert "{{ expectations }}" in instructions
    assert "expectations.expected_answer" in instructions
    assert "expected_memory_count" in instructions


def test_persistent_seeding_only_checks_requested_search_entries(monkeypatch):
    entries = longmemeval_adapter.build_case_memories(
        _case("single-session-user", 1)
    )
    exact = {
        "exact": True,
        "actual_count": len(entries),
        "full_scope": "scope",
    }
    monkeypatch.setattr(
        longmemeval_eval, "inspect_persistent_case", lambda *args: exact
    )
    observed = {}
    monkeypatch.setattr(
        longmemeval_eval.memory_eval,
        "wait_until_searchable",
        lambda scope, targets, timeout_s: observed.update(
            scope=scope,
            paths=[entry["path"] for entry in targets],
            timeout_s=timeout_s,
        ),
    )

    result = longmemeval_eval.seed_persistent_case(
        "selector",
        entries,
        search_check_entries=[entries[1]],
        index_timeout_s=45,
    )

    assert result["action"] == "already_exact"
    assert result["search_check_count"] == 1
    assert observed == {
        "scope": "scope",
        "paths": [entries[1]["path"]],
        "timeout_s": 45,
    }


def test_persistent_seeding_skips_search_checks_by_default(monkeypatch):
    entries = longmemeval_adapter.build_case_memories(
        _case("single-session-user", 1)
    )
    monkeypatch.setattr(
        longmemeval_eval,
        "inspect_persistent_case",
        lambda *args: {
            "exact": True,
            "actual_count": len(entries),
            "full_scope": "scope",
        },
    )
    monkeypatch.setattr(
        longmemeval_eval.memory_eval,
        "wait_until_searchable",
        lambda *args, **kwargs: pytest.fail("search check should be opt-in"),
    )

    result = longmemeval_eval.seed_persistent_case("selector", entries)

    assert result["search_check_count"] == 0


def test_longmemeval_evaluator_is_read_only_and_seeding_is_separate():
    root = Path(longmemeval_eval.__file__).parent
    runner = (root / "run_longmemeval_pilot.py").read_text()
    seeder = (root / "seed_longmemeval_corpora.py").read_text()

    assert "validate_persistent_scope" in runner
    assert "inspect_persistent_scope" in runner
    assert "seed_persistent_case" not in runner
    assert "cleanup_scope" not in runner
    assert "seed_persistent_case" in seeder
    assert '"--search-check"' in seeder
    direct_runner = (root / "run_longmemeval_direct.py").read_text()
    assert "direct_predict_fn" in direct_runner
    assert "DIRECT_SEARCH_SCORERS" in direct_runner
    assert "_invoke_turn_sync" not in direct_runner


def test_workspace_notebooks_keep_longmemeval_seeding_separate():
    root = Path(longmemeval_eval.__file__).parent
    seed_notebook = (root / "Seed_longmemeval_memories.py").read_text()
    run_notebook = (root / "Run_longmemeval_evaluation.py").read_text()

    assert "# Databricks notebook source" in seed_notebook
    assert "seed_persistent_case" in seed_notebook
    assert "mlflow.genai.evaluate" not in seed_notebook
    assert "# Databricks notebook source" in run_notebook
    assert "mlflow.genai.evaluate" in run_notebook
    assert "validate_persistent_scope" in run_notebook
    assert "seed_persistent_case" not in run_notebook
    assert "cleanup_scope" not in run_notebook

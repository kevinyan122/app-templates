import json

import pytest

from shared_memory_eval import realistic_search_eval
from shared_memory_eval import run_realistic_search_autonomous


def test_frozen_pilot_has_expected_shape_and_fingerprints():
    pilot = realistic_search_eval.load_pilot()
    seeds = realistic_search_eval.build_seed_memories(pilot)

    assert len(pilot["cases"]) == 25
    assert len(seeds) == 60
    assert realistic_search_eval.corpus_fingerprint(seeds) == (
        realistic_search_eval.EXPECTED_CORPUS_FINGERPRINT
    )
    assert realistic_search_eval.case_fingerprint(pilot["cases"]) == (
        realistic_search_eval.EXPECTED_CASE_FINGERPRINT
    )
    assert realistic_search_eval.DEFAULT_PERSISTENT_SCOPE.endswith("-6faac0b1")


def test_seed_entries_preserve_production_memory_shape_and_opaque_ids():
    pilot = realistic_search_eval.load_pilot()
    seeds = {
        realistic_search_eval.memory_id_from_path(seed["path"]): seed
        for seed in realistic_search_eval.build_seed_memories(pilot)
    }

    assert seeds["mem-0007"]["contents"] == ""
    assert "Aurora Calendar" in seeds["mem-0007"]["description"]
    assert seeds["mem-0042"]["contents"].startswith("Summer Reading Ops 2026")
    assert realistic_search_eval.memory_id_from_path(
        "/memories/preferences/calendar.md"
    ) is None


def test_question_rows_are_readable_and_keep_gold_out_of_inputs():
    pilot = realistic_search_eval.load_pilot()
    records = realistic_search_eval.build_question_dataset_records(pilot)
    record = next(
        row
        for row in records
        if row["inputs"]["case_id"] == "project-kb-refresh-it-approver"
    )

    assert record["inputs"]["query"].startswith("For KB Refresh changes")
    assert record["inputs"]["situation"]
    assert "gold_memory_id" not in record["inputs"]
    assert record["expectations"]["gold_memory_id"] == "mem-0038"
    assert record["expectations"]["answer_should"]
    assert record["tags"]["category"] == "project"


def test_corpus_dataset_rows_recreate_exact_frozen_fixture():
    pilot = realistic_search_eval.load_pilot()
    rows = realistic_search_eval.build_corpus_dataset_records(pilot)
    seeds = [
        {
            "path": row["inputs"]["path"],
            "description": row["inputs"]["description"],
            "contents": row["inputs"]["contents"],
        }
        for row in rows
    ]

    assert len(rows) == 60
    assert {row["inputs"]["content_profile"] for row in rows} == {
        "short",
        "medium",
        "episodic",
    }
    assert realistic_search_eval.corpus_fingerprint(seeds) == (
        realistic_search_eval.EXPECTED_CORPUS_FINGERPRINT
    )


def test_dataset_loaders_validate_and_join_questions_to_corpus(monkeypatch):
    pilot = realistic_search_eval.load_pilot()
    questions = realistic_search_eval.build_question_dataset_records(pilot)
    corpus = realistic_search_eval.build_corpus_dataset_records(pilot)
    monkeypatch.setattr(
        realistic_search_eval,
        "_dataset_rows",
        lambda name: questions if name == "questions" else corpus,
    )

    records = realistic_search_eval.load_question_records(
        "questions", "fixture", top_k=10
    )
    seeds = realistic_search_eval.load_seed_memories("corpus")
    realistic_search_eval.validate_question_corpus_records(records, seeds)

    assert len(records) == 25
    assert len(seeds) == 60
    assert all(record["inputs"]["eval_scope"] == "fixture" for record in records)
    assert all(record["inputs"]["top_k"] == 10 for record in records)


def test_loader_rejects_top_k_below_reported_hit_at_10(monkeypatch):
    pilot = realistic_search_eval.load_pilot()
    monkeypatch.setattr(
        realistic_search_eval,
        "_dataset_rows",
        lambda name: realistic_search_eval.build_question_dataset_records(pilot),
    )

    with pytest.raises(ValueError, match="at least 10"):
        realistic_search_eval.load_question_records("questions", top_k=5)


def test_persistent_scope_accepts_exact_snapshot(monkeypatch):
    pilot = realistic_search_eval.load_pilot()
    seeds = realistic_search_eval.build_seed_memories(pilot)
    exact = {
        entry["path"]: realistic_search_eval.memory_eval._clean_entry(entry)
        for entry in seeds
    }
    monkeypatch.setattr(
        realistic_search_eval.memory_eval, "snapshot_scope", lambda scope: exact
    )

    status = realistic_search_eval.inspect_persistent_entries("fixture", seeds)

    assert status["exact"] is True
    assert status["actual_count"] == 60
    assert status["actual_fingerprint"] == (
        realistic_search_eval.EXPECTED_CORPUS_FINGERPRINT
    )


def test_seed_is_noop_for_exact_scope(monkeypatch):
    pilot = realistic_search_eval.load_pilot()
    seeds = realistic_search_eval.build_seed_memories(pilot)
    status = {
        "exact": True,
        "actual_count": 60,
        "full_scope": "fixture",
    }
    monkeypatch.setattr(
        realistic_search_eval, "inspect_persistent_entries", lambda *args: status
    )
    monkeypatch.setattr(
        realistic_search_eval.memory_eval,
        "seed_scope",
        lambda *args: pytest.fail("exact scope must not be seeded"),
    )
    observed = {}
    monkeypatch.setattr(
        realistic_search_eval.memory_eval,
        "wait_until_searchable",
        lambda scope, entries, timeout_s: observed.update(
            scope=scope, entries=len(entries), timeout_s=timeout_s
        ),
    )

    result = realistic_search_eval.seed_persistent_entries(
        "fixture", seeds, index_timeout_s=30
    )

    assert result["action"] == "already_exact"
    assert observed == {"scope": "fixture", "entries": 60, "timeout_s": 30}


def test_gold_rank_and_fixed_hit_metrics_are_intuitive():
    outputs = {"direct_ranked_ids": ["mem-0002", "mem-0007", "mem-0001"]}
    expectations = {"gold_memory_id": "mem-0007"}

    assert realistic_search_eval.gold_rank(outputs, expectations) == 2
    assert realistic_search_eval.hit_at_1(
        outputs=outputs, expectations=expectations
    ).value is False
    assert realistic_search_eval.hit_at_3(
        outputs=outputs, expectations=expectations
    ).value is True
    assert realistic_search_eval.mrr(
        outputs=outputs, expectations=expectations
    ).value == 0.5


def test_direct_predictor_uses_natural_query_verbatim(monkeypatch):
    observed = {}

    def fake_search(query, eval_scope, top_k):
        observed.update(query=query, eval_scope=eval_scope, top_k=top_k)
        return {
            "query": query,
            "direct_sources": [{"id": "mem-0038", "rank": 1}],
            "direct_ranked_ids": ["mem-0038"],
            "direct_latency_ms": 4.0,
        }

    monkeypatch.setattr(realistic_search_eval, "direct_store_search", fake_search)
    monkeypatch.setattr(realistic_search_eval.mlflow, "update_current_trace", lambda **kwargs: None)

    result = realistic_search_eval.direct_predict_fn.__wrapped__(
        case_id="project-kb-refresh-it-approver",
        query="Who approves this?",
        category="project",
        situation="Mara needs the approver.",
        eval_scope="fixture",
        top_k=10,
    )

    assert observed == {
        "query": "Who approves this?",
        "eval_scope": "fixture",
        "top_k": 10,
    }
    assert result["direct_ranked_ids"] == ["mem-0038"]


def test_autonomous_loader_adds_scope_without_retrieval_parameters(monkeypatch):
    pilot = realistic_search_eval.load_pilot()
    monkeypatch.setattr(
        realistic_search_eval,
        "_dataset_rows",
        lambda name: realistic_search_eval.build_question_dataset_records(pilot),
    )

    records = realistic_search_eval.load_autonomous_question_records(
        "questions", "fixture"
    )

    assert len(records) == 25
    assert all(record["inputs"]["eval_scope"] == "fixture" for record in records)
    assert all("top_k" not in record["inputs"] for record in records)


def test_autonomous_predictor_sends_query_verbatim_in_fresh_threads(monkeypatch):
    calls = []

    def fake_invoke(message, eval_scope, thread_id, *, eval_read_only=False):
        calls.append((message, eval_scope, thread_id, eval_read_only))
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Ada approves it."}],
                }
            ]
        }

    monkeypatch.setattr(
        realistic_search_eval.memory_eval, "_invoke_turn_sync", fake_invoke
    )
    monkeypatch.setattr(
        realistic_search_eval.mlflow, "update_current_trace", lambda **kwargs: None
    )
    inputs = {
        "case_id": "project-kb-refresh-it-approver",
        "query": "Who approves this?",
        "category": "project",
        "situation": "Mara needs the approver.",
        "eval_scope": "fixture",
    }

    first = realistic_search_eval.autonomous_predict_fn.__wrapped__(**inputs)
    second = realistic_search_eval.autonomous_predict_fn.__wrapped__(**inputs)

    assert [(call[0], call[1], call[3]) for call in calls] == [
        ("Who approves this?", "fixture", True),
        ("Who approves this?", "fixture", True),
    ]
    assert calls[0][2] != calls[1][2]
    assert first["answer"] == "Ada approves it."
    assert first["thread_id"] != second["thread_id"]
    assert "agent_question" not in first
    assert "direct_ranked_ids" not in first


def test_autonomous_judge_receives_only_query_answer_and_answer_rubric(monkeypatch):
    observed = {}
    fake_judge = object()
    monkeypatch.setattr(
        "mlflow.genai.judges.make_judge", lambda **kwargs: fake_judge
    )

    def fake_invoke(judge, *, inputs, outputs, expectations):
        observed.update(
            judge=judge,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
        )
        return 1.0

    monkeypatch.setattr(
        realistic_search_eval.memory_eval,
        "_invoke_judge_with_retry",
        fake_invoke,
    )
    scorer = realistic_search_eval.load_autonomous_answer_judge("fake-model")

    result = scorer(
        inputs={"query": "Who approves?", "situation": "hidden context"},
        outputs={
            "answer": "Ada approves.",
            "tool_calls": [{"name": "search_memory", "output": "secret"}],
        },
        expectations={
            "answer_should": "Name Ada.",
            "gold_memory_id": "mem-0038",
        },
    )

    assert result == 1.0
    assert observed == {
        "judge": fake_judge,
        "inputs": {"query": "Who approves?"},
        "outputs": {"answer": "Ada approves."},
        "expectations": {"answer_should": "Name Ada."},
    }


def test_autonomous_runner_has_exactly_one_final_answer_scorer(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        realistic_search_eval, "load_autonomous_answer_judge", lambda model: expected
    )

    assert run_realistic_search_autonomous.build_scorers("fake-model") == [expected]


def test_case_fingerprint_changes_when_a_gold_mapping_changes():
    pilot = realistic_search_eval.load_pilot()
    cases = json.loads(json.dumps(pilot["cases"]))
    cases[0]["gold_memory_id"] = "mem-0001"

    assert realistic_search_eval.case_fingerprint(cases) != (
        realistic_search_eval.EXPECTED_CASE_FINGERPRINT
    )

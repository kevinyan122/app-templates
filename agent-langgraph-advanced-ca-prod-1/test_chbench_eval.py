import json
from pathlib import Path

import pytest

from shared_memory_eval import chbench_eval


def test_vendored_suite_has_expected_shape_and_valid_gold_ids():
    suite = chbench_eval.load_suite()

    assert len(suite["memories"]) == 44
    assert len(suite["questions"]) == 35
    assert {memory["track"] for memory in suite["memories"]} == {
        "developer",
        "founder",
        "pm",
        "researcher",
        "writer",
    }
    assert sum(bool(question.get("expect_abstain")) for question in suite["questions"]) == 5

    memory_ids = {memory["id"] for memory in suite["memories"]}
    assert all(set(question.get("relevant_ids", [])) <= memory_ids for question in suite["questions"])


def test_seed_entries_preserve_ids_text_and_temporal_metadata():
    suite = chbench_eval.load_suite()
    entries = {entry["path"]: entry for entry in chbench_eval.build_seed_memories(suite)}

    entry = entries["/memories/eval/chbench/dev-006.md"]
    assert entry["description"].startswith("Context-Heavy now embeds with BGE-M3")
    assert "timestamp: 2026-06-12" in entry["contents"]
    assert "supersedes: dev-007" in entry["contents"]
    assert chbench_eval.memory_id_from_path(entry["path"]) == "dev-006"
    assert chbench_eval.memory_id_from_path("/memories/preferences/editor.md") is None


def test_corpus_fingerprint_is_stable_order_independent_and_content_sensitive():
    entries = chbench_eval.build_seed_memories(chbench_eval.load_suite())

    fingerprint = chbench_eval.corpus_fingerprint(entries)
    assert fingerprint == chbench_eval.corpus_fingerprint(list(reversed(entries)))

    changed = [dict(entry) for entry in entries]
    changed[0]["description"] += " changed"
    assert chbench_eval.corpus_fingerprint(changed) != fingerprint


def test_persistent_corpus_reports_missing_unexpected_and_changed(monkeypatch):
    suite = chbench_eval.load_suite()
    expected = {
        entry["path"]: chbench_eval.memory_eval._clean_entry(entry)
        for entry in chbench_eval.build_seed_memories(suite)
    }
    actual = {path: dict(entry) for path, entry in expected.items()}
    missing_path = sorted(actual)[0]
    actual.pop(missing_path)
    changed_path = sorted(actual)[0]
    actual[changed_path]["description"] += " drift"
    unexpected_path = "/memories/eval/chbench/unexpected.md"
    actual[unexpected_path] = {
        "path": unexpected_path,
        "description": "unexpected",
        "contents": "",
    }
    monkeypatch.setattr(
        chbench_eval.memory_eval,
        "_full_eval_scope",
        lambda selector: f"user:eval:{selector}",
    )
    monkeypatch.setattr(chbench_eval.memory_eval, "snapshot_scope", lambda scope: actual)

    status = chbench_eval.inspect_persistent_corpus("fixture", suite)

    assert status["exact"] is False
    assert [entry["path"] for entry in status["missing"]] == [missing_path]
    assert [entry["path"] for entry in status["unexpected"]] == [unexpected_path]
    assert [entry["before"]["path"] for entry in status["changed"]] == [changed_path]
    assert status["actual_fingerprint"] != status["expected_fingerprint"]


def test_persistent_corpus_accepts_exact_snapshot(monkeypatch):
    suite = chbench_eval.load_suite()
    expected = {
        entry["path"]: chbench_eval.memory_eval._clean_entry(entry)
        for entry in chbench_eval.build_seed_memories(suite)
    }
    monkeypatch.setattr(
        chbench_eval.memory_eval,
        "_full_eval_scope",
        lambda selector: f"user:eval:{selector}",
    )
    monkeypatch.setattr(chbench_eval.memory_eval, "snapshot_scope", lambda scope: expected)

    status = chbench_eval.inspect_persistent_corpus("fixture", suite)

    assert status["exact"] is True
    assert status["actual_count"] == 44
    assert status["actual_fingerprint"] == status["expected_fingerprint"]
    assert status["missing"] == status["unexpected"] == status["changed"] == []


def test_seed_persistent_corpus_is_noop_when_fixture_is_already_exact(monkeypatch):
    suite = chbench_eval.load_suite()
    exact = {
        "exact": True,
        "actual_count": 44,
        "full_scope": "user:eval:fixture",
    }
    monkeypatch.setattr(chbench_eval, "inspect_persistent_entries", lambda *args: exact)
    monkeypatch.setattr(
        chbench_eval.memory_eval,
        "seed_scope",
        lambda *args: pytest.fail("exact fixture must not be seeded again"),
    )
    monkeypatch.setattr(
        chbench_eval.memory_eval,
        "cleanup_scope",
        lambda *args: pytest.fail("exact fixture must not be cleaned"),
    )
    observed = {}
    monkeypatch.setattr(
        chbench_eval.memory_eval,
        "wait_until_searchable",
        lambda scope, entries, timeout_s: observed.update(
            scope=scope,
            entry_count=len(entries),
            timeout_s=timeout_s,
        ),
    )

    result = chbench_eval.seed_persistent_corpus("fixture", suite)

    assert result["action"] == "already_exact"
    assert observed == {
        "scope": "user:eval:fixture",
        "entry_count": 44,
        "timeout_s": 180,
    }


def test_index_timeout_retains_fully_stored_corpus(monkeypatch):
    suite = chbench_eval.load_suite()
    empty = {
        "exact": False,
        "actual_count": 0,
        "full_scope": "user:eval:fixture",
    }
    exact = {
        "exact": True,
        "actual_count": 44,
        "full_scope": "user:eval:fixture",
    }
    statuses = iter([empty, exact])
    monkeypatch.setattr(
        chbench_eval,
        "inspect_persistent_entries",
        lambda *args: next(statuses),
    )
    monkeypatch.setattr(chbench_eval.memory_eval, "seed_scope", lambda *args: None)
    cleanups = []
    monkeypatch.setattr(
        chbench_eval.memory_eval,
        "cleanup_scope",
        lambda scope: cleanups.append(scope),
    )
    monkeypatch.setattr(
        chbench_eval.memory_eval,
        "wait_until_searchable",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("index lag")),
    )

    with pytest.raises(TimeoutError, match="index lag"):
        chbench_eval.seed_persistent_corpus("fixture", suite)

    assert cleanups == []


def test_smoke_runner_only_validates_persistent_fixture():
    source = Path(chbench_eval.__file__).with_name("run_chbench_smoke.py").read_text()

    assert "validate_persistent_entries" in source
    assert "inspect_persistent_entries" in source
    assert "load_records_from_dataset" in source
    assert "load_seed_memories_from_dataset" in source
    assert '"--all-questions"' in source
    assert "memory_eval.seed_scope" not in source
    assert "memory_eval.cleanup_scope" not in source


def test_records_keep_runtime_inputs_separate_from_gold_expectations():
    suite = chbench_eval.load_suite()
    records = chbench_eval.build_records(suite, "run-123", top_k=7)
    record = next(row for row in records if row["inputs"]["question_id"] == "dev-q2")

    assert record["inputs"] == {
        "question_id": "dev-q2",
        "question": "Why did Context-Heavy switch its embedding provider to NVIDIA NIM?",
        "track": "developer",
        "question_type": "causal",
        "eval_scope": "run-123",
        "top_k": 7,
    }
    assert json.loads(record["expectations"]["relevant_ids_json"]) == ["dev-005"]
    assert "expected_answer" in record["expectations"]
    assert "expected_answer" not in record["inputs"]


def test_managed_dataset_records_are_stable_and_provenance_tagged():
    suite = chbench_eval.load_suite()
    records = chbench_eval.build_dataset_records(suite)
    record = next(row for row in records if row["inputs"]["question_id"] == "dev-q2")

    assert len(records) == 35
    assert "eval_scope" not in record["inputs"]
    assert "top_k" not in record["inputs"]
    assert record["tags"] == {
        "suite": "chbench-contextheavy",
        "source_commit": chbench_eval.SOURCE_COMMIT,
        "corpus_fingerprint": chbench_eval.corpus_fingerprint(
            chbench_eval.build_seed_memories(suite)
        ),
        "track": "developer",
        "question_type": "causal",
    }


def test_corpus_dataset_records_recreate_the_exact_frozen_fixture():
    suite = chbench_eval.load_suite()
    records = chbench_eval.build_corpus_dataset_records(suite)

    assert len(records) == 44
    assert len({record["inputs"]["memory_id"] for record in records}) == 44
    seeds = [
        {
            "path": record["inputs"]["path"],
            "description": record["inputs"]["description"],
            "contents": record["inputs"]["contents"],
        }
        for record in records
    ]
    assert chbench_eval.corpus_fingerprint(seeds) == chbench_eval.EXPECTED_CORPUS_FINGERPRINT


def test_dataset_loaders_validate_and_join_questions_to_corpus(monkeypatch):
    suite = chbench_eval.load_suite()
    questions = chbench_eval.build_dataset_records(suite)
    corpus = chbench_eval.build_corpus_dataset_records(suite)

    monkeypatch.setattr(
        chbench_eval,
        "_dataset_rows",
        lambda name: questions if name == "questions" else corpus,
    )

    records = chbench_eval.load_records_from_dataset("questions", "fixture", top_k=7)
    seeds = chbench_eval.load_seed_memories_from_dataset("corpus")
    chbench_eval.validate_question_corpus_records(records, seeds)

    assert len(records) == 35
    assert len(seeds) == 44
    assert all(record["inputs"]["eval_scope"] == "fixture" for record in records)
    assert all(record["inputs"]["top_k"] == 7 for record in records)


def test_agent_ranking_concatenates_searches_and_deduplicates_first_seen():
    calls = [
        {
            "name": "search_memory",
            "arguments": {"query": "video api"},
            "output": (
                "2 memory matches (ranked index):\n"
                "- path: /memories/eval/chbench/dev-008.md\n"
                "  description: offSchool uses the video API\n"
                "  has_contents: true\n\n"
                "- path: /memories/eval/chbench/dev-001.md\n"
                "  description: letX uses the video API\n"
                "  has_contents: true"
            ),
        },
        {"name": "get_memory", "arguments": {"path": "ignored"}, "output": "ignored"},
        {
            "name": "search_memory",
            "arguments": {"query": "Quantum Sketch consumers"},
            "output": json.dumps(
                [
                    {
                        "path": "/memories/eval/chbench/dev-001.md",
                        "description": "duplicate",
                        "has_contents": True,
                    },
                    {
                        "path": "/memories/eval/chbench/dev-002.md",
                        "description": "new result",
                        "has_contents": True,
                    },
                ]
            ),
        },
    ]

    result = chbench_eval.parse_agent_sources(calls)

    assert [search["query"] for search in result["searches"]] == [
        "video api",
        "Quantum Sketch consumers",
    ]
    assert result["ranked_ids"] == ["dev-008", "dev-001", "dev-002"]
    assert result["sources"][1]["search_index"] == 1
    assert result["sources"][1]["search_rank"] == 2


def test_chbench_retrieval_metrics_match_ranked_list_definitions():
    ranked = ["noise", "a", "c", "b"]
    relevant = ["a", "b"]

    assert chbench_eval.recall_at_k(ranked, relevant, 3) == 0.5
    assert chbench_eval.precision_at_k(ranked, relevant, 3) == pytest.approx(1 / 3)
    assert chbench_eval.hit_at_k(ranked, relevant, 3) == 1.0
    assert chbench_eval.reciprocal_rank(ranked, relevant) == 0.5
    assert 0.0 < chbench_eval.ndcg_at_k(ranked, relevant, 3) < 1.0


def test_retrieval_scorers_compare_direct_and_agent_independently():
    inputs = {"top_k": 3}
    expectations = {"relevant_ids_json": '["dev-005"]', "expect_abstain": False}
    outputs = {
        "agent_ranked_ids": ["dev-001", "dev-005"],
        "direct_ranked_ids": ["dev-005", "dev-001"],
    }

    assert chbench_eval.agent_recall_at_k(
        inputs=inputs, outputs=outputs, expectations=expectations
    ).value == 1.0
    assert chbench_eval.agent_mrr(
        inputs=inputs, outputs=outputs, expectations=expectations
    ).value == 0.5
    assert chbench_eval.direct_mrr(
        inputs=inputs, outputs=outputs, expectations=expectations
    ).value == 1.0


def test_ranking_metrics_skip_abstention_rows_without_gold():
    feedback = chbench_eval.agent_recall_at_k(
        inputs={"top_k": 10},
        outputs={"agent_ranked_ids": []},
        expectations={"relevant_ids_json": "[]", "expect_abstain": True},
    )

    assert feedback is None


def test_query_time_mutation_scorer_uses_observed_tool_calls():
    clean = chbench_eval.agent_no_memory_write(outputs={"mutation_calls": []})
    dirty = chbench_eval.agent_no_memory_write(
        outputs={"mutation_calls": [{"name": "update_memory"}]}
    )

    assert clean.value is True
    assert dirty.value is False
    assert "update_memory" in dirty.rationale


def test_predictor_combines_direct_baseline_and_deployed_app_evidence(monkeypatch):
    direct_observed = {}

    def fake_direct_search(question, eval_scope, top_k):
        direct_observed.update(question=question, eval_scope=eval_scope, top_k=top_k)
        return {
            "sources": [{"id": "dev-005", "rank": 1}],
            "ranked_ids": ["dev-005"],
            "latency_ms": 3.5,
        }

    monkeypatch.setattr(
        chbench_eval,
        "direct_store_search",
        fake_direct_search,
    )
    observed = {}

    def fake_invoke(question, eval_scope, thread_id):
        observed.update(
            question=question,
            eval_scope=eval_scope,
            thread_id=thread_id,
        )
        return {
            "usage": {"input_tokens": 12, "output_tokens": 8},
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "search_memory",
                    "arguments": '{"query":"NVIDIA NIM switch"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": json.dumps(
                        [
                            {
                                "path": "/memories/eval/chbench/dev-005.md",
                                "description": "provider decision",
                                "has_contents": True,
                            }
                        ]
                    ),
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The OpenRouter key hit its spending limit and returned 403 errors.",
                        }
                    ],
                },
            ],
        }

    monkeypatch.setattr(chbench_eval.memory_eval, "_invoke_turn_sync", fake_invoke)
    monkeypatch.setattr(chbench_eval.mlflow, "update_current_trace", lambda **kwargs: None)

    result = chbench_eval.predict_fn.__wrapped__(
        question_id="dev-q2",
        question="Why did the provider change?",
        track="developer",
        question_type="causal",
        eval_scope="run-123",
        top_k=10,
    )

    assert direct_observed["question"] == "Why did the provider change?"
    assert observed["question"] == chbench_eval.forced_memory_question(
        "Why did the provider change?"
    )
    assert "Use the search_memory tool before answering" in observed["question"]
    assert observed["eval_scope"] == "run-123"
    assert observed["thread_id"].startswith("chbench-run-123-dev-q2-")
    assert result["direct_ranked_ids"] == ["dev-005"]
    assert result["agent_ranked_ids"] == ["dev-005"]
    assert result["agent_question"] == observed["question"]
    assert result["mutation_calls"] == []
    assert result["tokens"] == 20

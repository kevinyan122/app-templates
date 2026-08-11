from __future__ import annotations

import json

import eval_memory as em
import memory_eval_judges as judges
from memory_eval_cases import CATEGORIES, SCENARIOS, validate_scenarios
from memory_eval_judges import _has_requirement


def _expectations(*turns):
    return {"turn_expectations_json": json.dumps(list(turns))}


def test_case_inventory_covers_all_categories_and_multiturn_cases():
    validate_scenarios()
    assert len(SCENARIOS) == 35
    assert {scenario["category"] for scenario in SCENARIOS} == CATEGORIES
    assert sum(len(scenario["turns"]) > 1 for scenario in SCENARIOS) == 9
    assert len({scenario["id"] for scenario in SCENARIOS}) == len(SCENARIOS)


def test_dataset_adapter_keeps_expectations_out_of_predictor_inputs():
    record = em.scenario_to_record(SCENARIOS[0])
    inputs = record["inputs"]
    expectations = em.parse_expectation_fields(record["expectations"])

    assert inputs["category"] == "explicit-save"
    assert set(inputs) == {"scenario_id", "category", "turns", "initial_memories"}
    assert "scenario_json" not in inputs
    assert isinstance(inputs["turns"], list)
    assert isinstance(inputs["initial_memories"], list)
    assert record["expectations"]["turn_1_write"] == "save"
    assert "black" in record["expectations"]["turn_1_memory_should"]
    assert record["expectations"]["turn_2_search_result_should_include"] == "black; sweetener"
    assert record["tags"] == {"category": "explicit-save"}
    assert len(inputs["turns"]) == len(expectations) == 2
    assert all("expect" not in turn for turn in inputs["turns"])
    assert expectations[0]["memory_should"]


def test_expectation_fields_round_trip_multiturn_requirements():
    original = [
        {"write": "update", "search": "required", "memory_should": "Keep the current value."},
        {
            "write": "none",
            "search": "required",
            "search_should_contain": ["20,000", "January 15"],
            "answer_should": "Use the current value.",
        },
    ]

    fields = em.expectation_fields(original)

    assert fields == {
        "turn_1_write": "update",
        "turn_1_search": "required",
        "turn_1_memory_should": "Keep the current value.",
        "turn_2_write": "none",
        "turn_2_search": "required",
        "turn_2_search_result_should_include": "20,000; January 15",
        "turn_2_answer_should": "Use the current value.",
    }
    assert em.parse_expectation_fields(fields) == original


def test_parse_agent_output_correlates_tool_outputs_and_final_text():
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "search_memory",
                "arguments": '{"query":"editor"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "Neovim for Python",
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "You use Neovim."}],
            },
        ]
    }

    answer, calls = em.parse_agent_output(response)

    assert answer == "You use Neovim."
    assert calls == [
        {
            "name": "search_memory",
            "arguments": {"query": "editor"},
            "output": "Neovim for Python",
        }
    ]


def test_diff_snapshots_reports_only_relevant_changes():
    before = {
        "/memories/a.md": {"path": "/memories/a.md", "description": "old", "contents": "A"},
        "/memories/b.md": {"path": "/memories/b.md", "description": "gone", "contents": "B"},
    }
    after = {
        "/memories/a.md": {"path": "/memories/a.md", "description": "new", "contents": "A"},
        "/memories/c.md": {"path": "/memories/c.md", "description": "made", "contents": "C"},
    }

    diff = em.diff_snapshots(before, after)

    assert [entry["path"] for entry in diff["created"]] == ["/memories/c.md"]
    assert [entry["path"] for entry in diff["deleted"]] == ["/memories/b.md"]
    assert diff["updated"][0]["before"]["description"] == "old"
    assert diff["updated"][0]["after"]["description"] == "new"


def test_write_behavior_requires_search_before_save_and_real_store_change():
    passing = {
        "turns": [
            {
                "tool_calls": [
                    {"name": "search_memory"},
                    {"name": "save_memory"},
                ],
                "store_diff": {
                    "created": [
                        {"path": "/memories/x.md", "description": "X", "contents": ""}
                    ],
                    "updated": [],
                    "deleted": [],
                },
            }
        ]
    }
    failing = {
        "turns": [
            {
                "tool_calls": [{"name": "save_memory"}],
                "store_diff": passing["turns"][0]["store_diff"],
            }
        ]
    }
    expectations = _expectations({"write": "save"})

    assert em.write_behavior(outputs=passing, expectations=expectations).value is True
    failure = em.write_behavior(outputs=failing, expectations=expectations)
    assert failure.value is False
    assert failure.rationale.startswith("Turn 1 — FAIL:")
    assert "search before write=no" in failure.rationale

    no_save = {
        "turns": [
            {
                "tool_calls": [{"name": "search_memory"}],
                "store_diff": {"created": [], "updated": [], "deleted": []},
            }
        ]
    }
    missing_save = em.write_behavior(outputs=no_save, expectations=expectations)
    assert missing_save.value is False
    assert "required save_memory call did not occur" in missing_save.rationale


def test_no_write_checks_tool_calls_and_store_state():
    expectations = _expectations({"write": "none"})
    clean = {
        "turns": [
            {
                "tool_calls": [],
                "store_diff": {"created": [], "updated": [], "deleted": []},
            }
        ]
    }
    hidden_mutation = {
        "turns": [
            {
                "tool_calls": [],
                "store_diff": {
                    "created": [
                        {"path": "/memories/x.md", "description": "X", "contents": ""}
                    ],
                    "updated": [],
                    "deleted": [],
                },
            }
        ]
    }

    assert em.write_behavior(outputs=clean, expectations=expectations).value is True
    assert em.write_behavior(outputs=hidden_mutation, expectations=expectations).value is False


def test_search_result_is_case_and_punctuation_tolerant():
    outputs = {
        "turns": [
            {
                "tool_calls": [
                    {
                        "name": "search_memory",
                        "output": "Budget: $20,000 CAD; diet is strictly gluten-free.",
                    }
                ]
            }
        ]
    }
    expectations = _expectations(
        {
            "write": "none",
            "search": "required",
            "search_should_contain": ["20,000", "Gluten Free"],
        }
    )

    assert em.search_result(outputs=outputs, expectations=expectations).value is True


def test_run_scenario_rotates_only_on_new_session(monkeypatch):
    thread_ids = []

    async def fake_invoke(message, scope, thread_id):
        thread_ids.append(thread_id)
        return {"output": [{"type": "message", "content": []}]}

    monkeypatch.setattr(em, "_invoke_turn", fake_invoke)
    monkeypatch.setattr(em, "snapshot_scope", lambda scope: {})
    monkeypatch.setattr(em, "cleanup_scope", lambda scope: 0)

    result = em.run_scenario(
        "thread-test",
        "multi-turn-lifecycle",
        {
            "turns": [
                {"message": "one", "new_session": False},
                {"message": "two", "new_session": True},
                {"message": "three", "new_session": False},
            ]
        },
    )

    assert thread_ids[0].endswith("session-1")
    assert thread_ids[1].endswith("session-2")
    assert thread_ids[2] == thread_ids[1]
    assert [turn["session_number"] for turn in result["turns"]] == [1, 2, 2]


def test_each_scenario_prediction_uses_a_fresh_scope(monkeypatch):
    scopes = []

    async def fake_invoke(message, scope, thread_id):
        scopes.append(scope)
        return {"output": [{"type": "message", "content": []}]}

    monkeypatch.setattr(em, "_invoke_turn", fake_invoke)
    monkeypatch.setattr(em, "snapshot_scope", lambda scope: {})
    monkeypatch.setattr(em, "cleanup_scope", lambda scope: 0)
    scenario = {"turns": [{"message": "hello"}]}

    em.run_scenario("scope-test", "no-search", scenario)
    em.run_scenario("scope-test", "no-search", scenario)

    assert len(scopes) == 2
    assert scopes[0] != scopes[1]


def test_wait_until_searchable_waits_for_current_entry(monkeypatch):
    current = {
        "path": "/memories/project.md",
        "description": "Launch is October 24",
        "contents": "Blocked on SSO validation",
    }
    stale = {**current, "description": "Launch is October 10"}
    responses = [[stale], [current]]

    monkeypatch.setattr(em, "_search_entries", lambda scope, query: responses.pop(0))
    monkeypatch.setattr(em.time, "sleep", lambda seconds: None)

    em.wait_until_searchable("scope", [current], timeout_s=1, poll_interval_s=0)
    assert responses == []


def test_wait_for_turn_snapshot_attributes_a_delayed_update_to_its_turn(monkeypatch):
    path = "/memories/project.md"
    before = {
        path: {
            "path": path,
            "description": "Launch is October 10",
            "contents": "Owner: Priya",
        }
    }
    after = {
        path: {
            "path": path,
            "description": "Launch is October 24",
            "contents": "Owner: Priya",
        }
    }
    responses = [before, after]
    tool_calls = [
        {
            "name": "update_memory",
            "arguments": {"path": path},
            "output": f"Updated {path}.",
        }
    ]

    monkeypatch.setattr(em, "snapshot_scope", lambda scope: responses.pop(0))
    monkeypatch.setattr(em.time, "sleep", lambda seconds: None)

    observed = em.wait_for_turn_snapshot(
        "scope", before, tool_calls, timeout_s=1, poll_interval_s=0
    )

    assert observed == after
    assert responses == []


def test_wait_for_turn_snapshot_rejects_a_stale_reversal_on_read_only_turn(monkeypatch):
    path = "/memories/work/current-role.md"
    confirmed = {
        path: {
            "path": path,
            "description": "Works as shift supervisor",
            "contents": "Role: shift supervisor",
        }
    }
    stale = {
        path: {
            "path": path,
            "description": "Works as a barista",
            "contents": "Role: barista",
        }
    }
    responses = [stale, confirmed]

    monkeypatch.setattr(em, "snapshot_scope", lambda scope: responses.pop(0))
    monkeypatch.setattr(em.time, "sleep", lambda seconds: None)

    observed = em.wait_for_turn_snapshot(
        "scope",
        confirmed,
        [{"name": "search_memory", "arguments": {}, "output": "shift supervisor"}],
        timeout_s=1,
        poll_interval_s=0,
    )

    assert observed == confirmed
    assert responses == []


def test_judge_requirements_are_conditional():
    assert _has_requirement(
        em.expectation_fields(
            [{"write": "save", "memory_should": "Store the preference."}]
        ),
        "memory_should",
    )
    assert _has_requirement(
        _expectations({"write": "save", "memory_should": "Store the preference."}),
        "memory_should",
    )
    assert not _has_requirement(_expectations({"write": "none"}), "answer_should")


def test_local_judge_retries_transient_failures(monkeypatch):
    attempts = []

    def flaky_judge(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("temporary endpoint failure")
        return "passed"

    monkeypatch.setattr(judges.time, "sleep", lambda seconds: None)

    result = judges._invoke_judge_with_retry(
        flaky_judge,
        inputs={"case": "x"},
        outputs={"answer": "y"},
        expectations={"required": "z"},
    )

    assert result == "passed"
    assert len(attempts) == 2

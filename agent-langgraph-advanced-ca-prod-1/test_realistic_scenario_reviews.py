import json
from pathlib import Path

import pytest

from shared_memory_eval.realistic_search_benchmark.review_scenarios import (
    EVIDENCE_REVIEWER,
    _normalize_critic_output,
    critic_prompt,
    global_consistency_checks,
)
from shared_memory_eval.realistic_search_benchmark.schema import scenario_bundle_from_dict


_DRAFTS = (
    Path(__file__).parent
    / "shared_memory_eval"
    / "realistic_search_benchmark"
    / "artifacts"
    / "pilot_scenario_drafts.json"
)


def _bundles():
    artifact = json.loads(_DRAFTS.read_text())
    return tuple(scenario_bundle_from_dict(item) for item in artifact["bundles"][:2])


def _valid_output():
    return {
        "reviews": [
            {
                "case_id": bundle.case.case_id,
                "realistic": True,
                "answerable": True,
                "gold_correct": True,
                "unambiguous": True,
                "category_fit": True,
                "issue_codes": [],
                "alternative_memory_ids": [],
                "notes": "The gold alone supports the requested answer and contexts do not.",
                "suggested_repair": "",
            }
            for bundle in _bundles()
        ]
    }


def test_normalizes_one_review_per_case():
    normalized = _normalize_critic_output(
        _valid_output(), EVIDENCE_REVIEWER, _bundles()
    )

    assert set(normalized) == {bundle.case.case_id for bundle in _bundles()}
    assert all(decision.accepted for decision in normalized.values())


def test_rejects_missing_case_review():
    output = _valid_output()
    output["reviews"].pop()

    with pytest.raises(ValueError, match="case ids mismatch"):
        _normalize_critic_output(output, EVIDENCE_REVIEWER, _bundles())


def test_rejects_string_instead_of_json_boolean():
    output = _valid_output()
    output["reviews"][0]["gold_correct"] = "false"

    with pytest.raises(TypeError, match="JSON boolean"):
        _normalize_critic_output(output, EVIDENCE_REVIEWER, _bundles())


def test_critic_does_not_treat_partial_context_as_an_alternate_gold():
    prompt = critic_prompt(
        _bundles(), EVIDENCE_REVIEWER, benchmark_as_of_date="2026-06-05"
    )

    assert "supports one part of a multi-part answer" in prompt
    assert "does not invalidate the gold" in prompt
    assert "independently satisfies every material requirement" in prompt
    assert "complete answer_should rubric" in prompt
    assert "description and contents together" in prompt
    assert "Empty contents are" in prompt
    assert "project-specific escalation contact" in prompt
    assert "Never reject gold uniqueness" in prompt
    assert '"benchmark_as_of_date": "2026-06-05"' in prompt


def test_global_checks_surface_candidate_memory_duplicates():
    checks = global_consistency_checks(
        tuple(
            scenario_bundle_from_dict(item)
            for item in json.loads(_DRAFTS.read_text())["bundles"]
        )
    )

    assert checks["candidate_memory_occurrences"] == 93
    assert checks["unique_fact_sets"] < 93
    assert checks["duplicate_fact_set_groups"]

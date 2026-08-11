import json
from pathlib import Path

from shared_memory_eval.realistic_search_benchmark.curate_pilot import (
    curate_pilot,
    validate_curated_pilot,
)
from shared_memory_eval.realistic_search_benchmark.schema import (
    scenario_bundle_from_dict,
    world_bible_from_dict,
)


_ROOT = Path(__file__).parent / "shared_memory_eval" / "realistic_search_benchmark" / "artifacts"


def _inputs():
    world = world_bible_from_dict(json.loads((_ROOT / "pilot_world.json").read_text())["world"])
    drafts = json.loads((_ROOT / "pilot_scenario_drafts.json").read_text())
    bundles = tuple(scenario_bundle_from_dict(item) for item in drafts["bundles"])
    return world, bundles


def test_curates_exactly_sixty_unique_memories_and_twenty_five_cases():
    world, drafts = _inputs()
    corpus, bundles, metadata = curate_pilot(world, drafts)

    validate_curated_pilot(corpus, bundles, world)
    assert len(corpus) == 60
    assert len(bundles) == 25
    assert len({tuple(sorted(memory.fact_ids)) for memory in corpus}) == 60
    assert metadata["deduplicated_candidate_memories"] == 50
    assert metadata["added_unused_world_facts"] == 10
    assert len(metadata["repairs"]) == 7


def test_repairs_substantive_and_cross_corpus_critic_findings():
    world, drafts = _inputs()
    _, bundles, _ = curate_pilot(world, drafts)
    cases = {bundle.case.case_id: bundle.case for bundle in bundles}

    assert "exact job title" in cases["people-sasha-comms-training-partner"].query
    assert "what activity" in cases["people-jordan-package-backup"].query
    freeze = cases["decision-kb-freeze-window"]
    assert "Why did Nina approve" in freeze.query
    assert "clean review pass" in freeze.answer_should
    assert "tonight" not in cases["household-trash-night"].query.casefold()
    assert "general point of contact" in cases["people-owen-escalation-contact"].query
    expansion = cases["project-coverage-pilot-expansion"]
    assert "expand next" in expansion.query
    assert "HQ-based teams" in expansion.answer_should
    assert "cross-site escalation paths" in expansion.answer_should
    standing_role = cases["global-branch-leads-channel"]
    assert "Outside individual projects" in standing_role.query
    assert "before broad rollout" in standing_role.answer_should

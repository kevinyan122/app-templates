from dataclasses import asdict
import json

import pytest

from shared_memory_eval.realistic_search_benchmark.generate_scenarios import (
    _normalize_writer_output,
)
from shared_memory_eval.realistic_search_benchmark.schema import (
    Category,
    WorldBible,
    WorldFact,
    scenario_bundle_from_dict,
)


def _world() -> WorldBible:
    from shared_memory_eval.realistic_search_benchmark.schema import (
        CategoryBrief,
        EntityKind,
        WorldEntity,
    )

    return WorldBible(
        world_id="scenario-test-world",
        title="Scenario test world",
        premise="Kevin coordinates ordinary work and travel details with an assistant.",
        owner_entity_id="person-kevin",
        entities=(
            WorldEntity(
                entity_id="person-kevin",
                kind=EntityKind.PERSON,
                name="Kevin Park",
                aliases=("Kevin",),
                summary="The owner of the memory scope.",
            ),
            WorldEntity(
                entity_id="project-atlas",
                kind=EntityKind.PROJECT,
                name="Atlas Pilot",
                aliases=("Atlas",),
                summary="A fictional work pilot.",
            ),
        ),
        category_briefs=tuple(
            CategoryBrief(
                category=category,
                brief=f"Ordinary {category.value} details.",
                focus_entity_ids=("person-kevin",),
            )
            for category in Category
        ),
        facts=(
            WorldFact(
                fact_id="fact-preference-seat",
                category=Category.PREFERENCE,
                statement="Kevin prefers aisle seats on flights.",
                entity_ids=("person-kevin",),
            ),
            WorldFact(
                fact_id="fact-preference-briefing",
                category=Category.PREFERENCE,
                statement="Kevin prefers short morning briefings.",
                entity_ids=("person-kevin",),
            ),
            WorldFact(
                fact_id="fact-project-atlas-owner",
                category=Category.PROJECT,
                statement="Kevin owns the Atlas Pilot schedule.",
                entity_ids=("person-kevin", "project-atlas"),
            ),
        ),
    )


def _output():
    return {
        "bundles": [
            {
                "case": {
                    "case_id": "preference-flight-seat",
                    "category": "preference",
                    "situation": "Booking Kevin's flight for a work trip.",
                    "query": "Which kind of seat should I book for the flight?",
                    "gold_memory_key": "gold",
                    "context_memory_keys": ["context-1", "context-2"],
                    "answer_should": "Recommend an aisle seat.",
                    "why_gold": "Only the gold states Kevin's flight seating preference.",
                    "secondary_tags": ["paraphrase", "same-entity"],
                },
                "memories": [
                    {
                        "memory_key": "gold",
                        "category": "preference",
                        "description": "Kevin's preferred airplane seat",
                        "contents": "Book an aisle seat for Kevin when arranging flights.",
                        "fact_ids": ["fact-preference-seat"],
                        "entity_ids": ["person-kevin"],
                    },
                    {
                        "memory_key": "context-1",
                        "category": "preference",
                        "description": "Kevin likes concise morning briefings",
                        "contents": "Keep the morning briefing short and focused.",
                        "fact_ids": ["fact-preference-briefing"],
                        "entity_ids": ["person-kevin"],
                    },
                    {
                        "memory_key": "context-2",
                        "category": "project",
                        "description": "Kevin owns the Atlas Pilot schedule",
                        "contents": "Kevin coordinates the Atlas Pilot schedule and timing.",
                        "fact_ids": ["fact-project-atlas-owner"],
                        "entity_ids": ["person-kevin", "project-atlas"],
                    },
                ],
            }
        ]
    }


def test_normalizes_local_memory_keys_to_opaque_global_ids():
    bundles = _normalize_writer_output(
        _output(),
        _world(),
        (Category.PREFERENCE,),
        {Category.PREFERENCE: 1},
        memory_start=42,
    )

    assert len(bundles) == 1
    assert bundles[0].case.gold_memory_id == "mem-0042"
    assert bundles[0].case.context_memory_ids == ("mem-0043", "mem-0044")
    assert {memory.path for memory in bundles[0].memories} == {
        "/memories/benchmark/mem-0042.md",
        "/memories/benchmark/mem-0043.md",
        "/memories/benchmark/mem-0044.md",
    }

    persisted = json.loads(json.dumps(asdict(bundles[0])))
    reparsed = scenario_bundle_from_dict(persisted)
    assert reparsed == bundles[0]


def test_rejects_wrong_case_quota():
    with pytest.raises(ValueError, match="case count mismatches"):
        _normalize_writer_output(
            _output(),
            _world(),
            (Category.PREFERENCE,),
            {Category.PREFERENCE: 2},
            memory_start=1,
        )


def test_rejects_superseded_fact_as_gold():
    world = _world()
    old = WorldFact(
        fact_id="fact-preference-seat-old",
        category=Category.PREFERENCE,
        statement="Kevin previously preferred window seats.",
        entity_ids=("person-kevin",),
        valid_from="2026-01-01",
        valid_to="2026-03-31",
    )
    current = WorldFact(
        fact_id="fact-preference-seat-current",
        category=Category.PREFERENCE,
        statement="Kevin now prefers aisle seats.",
        entity_ids=("person-kevin",),
        valid_from="2026-04-01",
        supersedes_fact_id=old.fact_id,
    )
    temporal_world = WorldBible(
        world_id=world.world_id,
        title=world.title,
        premise=world.premise,
        owner_entity_id=world.owner_entity_id,
        entities=world.entities,
        category_briefs=world.category_briefs,
        facts=(old, current, *world.facts[1:]),
    )
    output = _output()
    output["bundles"][0]["memories"][0]["fact_ids"] = [old.fact_id]
    output["bundles"][0]["memories"][0]["contents"] = "Kevin used to choose window seats."

    with pytest.raises(ValueError, match="superseded fact"):
        _normalize_writer_output(
            output,
            temporal_world,
            (Category.PREFERENCE,),
            {Category.PREFERENCE: 1},
            memory_start=1,
        )

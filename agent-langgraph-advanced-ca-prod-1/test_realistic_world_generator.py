import pytest

from shared_memory_eval.realistic_search_benchmark.generate_world import (
    CATEGORY_GROUPS,
    _parse_json_object,
    generate_world_bible,
)
from shared_memory_eval.realistic_search_benchmark.schema import (
    Category,
    WorldBible,
    validate_world_bible,
)


def _architecture():
    return {
        "world_id": "realistic-search-pilot",
        "title": "A coherent pilot world",
        "premise": "Kevin balances product work, recurring collaborators, travel, and a household.",
        "owner_entity_id": "person-kevin",
        "entities": [
            {
                "entity_id": "person-kevin",
                "kind": "person",
                "name": "Kevin Park",
                "aliases": ["Kevin"],
                "summary": "The owner of this assistant memory scope.",
            },
            {
                "entity_id": "org-lantern",
                "kind": "organization",
                "name": "Lantern Works",
                "aliases": ["Lantern"],
                "summary": "Kevin's fictional employer.",
            },
        ],
        "category_briefs": [
            {
                "category": category.value,
                "brief": f"Realistic {category.value} facts in Kevin's life and work.",
                "focus_entity_ids": ["person-kevin", "org-lantern"],
            }
            for category in Category
        ],
        "facts": [],
    }


def test_world_generator_merges_parallel_role_contracts_and_meets_quotas():
    role_categories = {
        f"fact-builder-{index + 1}": group
        for index, group in enumerate(CATEGORY_GROUPS)
    }
    calls = []

    def fake_agent(role, prompt):
        calls.append(role)
        if role == "world-architect":
            return _architecture()
        return {
            "facts": [
                {
                    "fact_id": f"fact-{category.value.replace('_', '-')}-one",
                    "category": category.value,
                    "statement": f"One durable {category.value} fact about Kevin.",
                    "entity_ids": ["person-kevin"],
                    "aliases": [],
                    "valid_from": None,
                    "valid_to": None,
                    "supersedes_fact_id": None,
                }
                for category in role_categories[role]
            ]
        }

    quotas = {category: 1 for category in Category}
    world = generate_world_bible(fake_agent, fact_quotas=quotas, parallel=False)

    assert calls == ["world-architect", *role_categories]
    assert len(world.entities) == 2
    assert len(world.facts) == len(Category)
    assert {fact.category for fact in world.facts} == set(Category)
    validate_world_bible(world, fact_quotas=quotas)


def test_world_validation_rejects_fact_with_missing_entity():
    world = generate_world_bible(
        _minimal_fake_agent,
        fact_quotas={category: 1 for category in Category},
        parallel=False,
    )
    first = world.facts[0]
    broken_fact = first.__class__(
        fact_id=first.fact_id,
        category=first.category,
        statement=first.statement,
        entity_ids=("person-missing",),
    )
    broken = WorldBible(
        world_id=world.world_id,
        title=world.title,
        premise=world.premise,
        owner_entity_id=world.owner_entity_id,
        entities=world.entities,
        category_briefs=world.category_briefs,
        facts=(broken_fact, *world.facts[1:]),
    )

    with pytest.raises(ValueError, match="missing entities"):
        validate_world_bible(broken)


def test_world_validation_rejects_fact_count_above_exact_quota():
    quotas = {category: 1 for category in Category}
    world = generate_world_bible(
        _minimal_fake_agent,
        fact_quotas=quotas,
        parallel=False,
    )
    extra = world.facts[0].__class__(
        fact_id="fact-preference-extra",
        category=Category.PREFERENCE,
        statement="Another durable preference fact about Kevin.",
        entity_ids=("person-kevin",),
    )
    over_quota = WorldBible(
        world_id=world.world_id,
        title=world.title,
        premise=world.premise,
        owner_entity_id=world.owner_entity_id,
        entities=world.entities,
        category_briefs=world.category_briefs,
        facts=(*world.facts, extra),
    )

    with pytest.raises(ValueError, match="quota mismatches"):
        validate_world_bible(over_quota, fact_quotas=quotas)


def test_world_validation_rejects_supersedes_without_prior_end_date():
    world = generate_world_bible(
        _minimal_fake_agent,
        fact_quotas={category: 1 for category in Category},
        parallel=False,
    )
    prior = world.facts[0].__class__(
        fact_id="fact-preference-prior",
        category=Category.PREFERENCE,
        statement="Kevin previously preferred morning meetings.",
        entity_ids=("person-kevin",),
        valid_from="2026-01-01",
    )
    current = world.facts[0].__class__(
        fact_id="fact-preference-current",
        category=Category.PREFERENCE,
        statement="Kevin now prefers afternoon meetings.",
        entity_ids=("person-kevin",),
        valid_from="2026-04-01",
        supersedes_fact_id=prior.fact_id,
    )
    temporal_world = WorldBible(
        world_id=world.world_id,
        title=world.title,
        premise=world.premise,
        owner_entity_id=world.owner_entity_id,
        entities=world.entities,
        category_briefs=world.category_briefs,
        facts=(prior, current, *world.facts[1:]),
    )

    with pytest.raises(ValueError, match="must have valid_to"):
        validate_world_bible(temporal_world)


def test_world_generator_retries_only_builder_with_wrong_fact_count():
    role_calls = {f"fact-builder-{index + 1}": 0 for index in range(len(CATEGORY_GROUPS))}

    def flaky_agent(role, prompt):
        if role == "world-architect":
            return _architecture()
        role_calls[role] += 1
        index = int(role.rsplit("-", 1)[1]) - 1
        facts = [
            {
                "fact_id": f"fact-{category.value.replace('_', '-')}-one",
                "category": category.value,
                "statement": f"One durable {category.value} fact about Kevin.",
                "entity_ids": ["person-kevin"],
            }
            for category in CATEGORY_GROUPS[index]
        ]
        if role == "fact-builder-3" and role_calls[role] == 1:
            facts.append(
                {
                    "fact_id": "fact-workflow-extra",
                    "category": Category.WORKFLOW.value,
                    "statement": "An extra workflow fact that violates the exact quota.",
                    "entity_ids": ["person-kevin"],
                }
            )
        return {"facts": facts}

    world = generate_world_bible(
        flaky_agent,
        fact_quotas={category: 1 for category in Category},
        parallel=False,
    )

    assert len(world.facts) == len(Category)
    assert role_calls == {
        "fact-builder-1": 1,
        "fact-builder-2": 1,
        "fact-builder-3": 2,
        "fact-builder-4": 1,
    }


def test_json_parser_accepts_plain_and_fenced_objects():
    assert _parse_json_object('{"facts": []}') == {"facts": []}
    assert _parse_json_object('```json\n{"facts": []}\n```') == {"facts": []}


def _minimal_fake_agent(role, prompt):
    if role == "world-architect":
        return _architecture()
    index = int(role.rsplit("-", 1)[1]) - 1
    return {
        "facts": [
            {
                "fact_id": f"fact-{category.value.replace('_', '-')}-one",
                "category": category.value,
                "statement": f"One durable {category.value} fact about Kevin.",
                "entity_ids": ["person-kevin"],
            }
            for category in CATEGORY_GROUPS[index]
        ]
    }

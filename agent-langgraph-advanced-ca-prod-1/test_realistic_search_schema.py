import pytest

from shared_memory_eval.realistic_search_benchmark.schema import (
    BenchmarkMemory,
    Category,
    EntityKind,
    ReviewDecision,
    ScenarioBundle,
    ScenarioCase,
    WorldFact,
    validate_bundle,
)


def test_pet_is_a_first_class_world_entity_kind():
    assert EntityKind("pet") is EntityKind.PET


def _valid_bundle() -> ScenarioBundle:
    facts = (
        WorldFact(
            fact_id="fact-flight-seat",
            category=Category.PREFERENCE,
            statement="Kevin prefers aisle seats because his knee gets stiff.",
            entity_ids=("person-kevin",),
        ),
        WorldFact(
            fact_id="fact-train-seat",
            category=Category.PREFERENCE,
            statement="Kevin prefers the quiet car on trains.",
            entity_ids=("person-kevin",),
        ),
        WorldFact(
            fact_id="fact-maya-seat",
            category=Category.PREFERENCE,
            statement="Maya prefers window seats on flights.",
            entity_ids=("person-maya",),
        ),
    )
    memories = (
        BenchmarkMemory(
            memory_id="mem-0001",
            category=Category.PREFERENCE,
            description="Kevin prefers aisle seats on flights",
            contents="His knee gets stiff, so he likes being able to stand without disturbing anyone.",
            fact_ids=("fact-flight-seat",),
            entity_ids=("person-kevin",),
        ),
        BenchmarkMemory(
            memory_id="mem-0002",
            category=Category.PREFERENCE,
            description="Kevin prefers the quiet car on trains",
            contents="",
            fact_ids=("fact-train-seat",),
            entity_ids=("person-kevin",),
        ),
        BenchmarkMemory(
            memory_id="mem-0003",
            category=Category.PREFERENCE,
            description="Maya prefers window seats on flights",
            contents="She enjoys taking photographs during takeoff.",
            fact_ids=("fact-maya-seat",),
            entity_ids=("person-maya",),
        ),
    )
    case = ScenarioCase(
        case_id="preference-flight-seat",
        category=Category.PREFERENCE,
        situation="Booking Kevin's seat on an upcoming Toronto work trip.",
        query="Which seat should I choose for Kevin on the Toronto flight?",
        gold_memory_id="mem-0001",
        context_memory_ids=("mem-0002", "mem-0003"),
        answer_should="Recommend an aisle seat and mention that Kevin's knee gets stiff.",
        why_gold="Only mem-0001 describes Kevin's preference for seating on flights.",
        secondary_tags=("similar-entity",),
    )
    reviews = (
        ReviewDecision("critic-a", True, True, True, True, True),
        ReviewDecision("critic-b", True, True, True, True, True),
    )
    return ScenarioBundle(case=case, facts=facts, memories=memories, reviews=reviews)


def test_valid_bundle_passes_hard_gates_and_builds_opaque_seed_path():
    bundle = _valid_bundle()

    validate_bundle(bundle, require_reviews=True)

    gold = bundle.memories[0]
    assert gold.seed_entry() == {
        "path": "/memories/benchmark/mem-0001.md",
        "description": "Kevin prefers aisle seats on flights",
        "contents": "His knee gets stiff, so he likes being able to stand without disturbing anyone.",
    }


def test_bundle_rejects_missing_context_memory():
    bundle = _valid_bundle()
    incomplete = ScenarioBundle(
        case=bundle.case,
        facts=bundle.facts,
        memories=bundle.memories[:-1],
        reviews=bundle.reviews,
    )

    with pytest.raises(ValueError, match="missing memories"):
        validate_bundle(incomplete)


def test_bundle_rejects_gold_category_mismatch():
    bundle = _valid_bundle()
    wrong_gold = BenchmarkMemory(
        memory_id="mem-0001",
        category=Category.PROJECT,
        description=bundle.memories[0].description,
        contents=bundle.memories[0].contents,
        fact_ids=bundle.memories[0].fact_ids,
        entity_ids=bundle.memories[0].entity_ids,
    )
    mismatched = ScenarioBundle(
        case=bundle.case,
        facts=bundle.facts,
        memories=(wrong_gold, *bundle.memories[1:]),
        reviews=bundle.reviews,
    )

    with pytest.raises(ValueError, match="category"):
        validate_bundle(mismatched)


def test_bundle_requires_two_accepting_reviews_for_release():
    bundle = _valid_bundle()
    rejected = ScenarioBundle(
        case=bundle.case,
        facts=bundle.facts,
        memories=bundle.memories,
        reviews=(ReviewDecision("critic-a", True, True, False, True, True),),
    )

    with pytest.raises(ValueError, match="two independent reviews"):
        validate_bundle(rejected, require_reviews=True)


def test_memory_rejects_description_repeated_as_contents():
    with pytest.raises(ValueError, match="must not repeat"):
        BenchmarkMemory(
            memory_id="mem-0009",
            category=Category.PREFERENCE,
            description="Kevin prefers aisle seats",
            contents="Kevin prefers aisle seats",
            fact_ids=("fact-flight-seat",),
            entity_ids=("person-kevin",),
        )

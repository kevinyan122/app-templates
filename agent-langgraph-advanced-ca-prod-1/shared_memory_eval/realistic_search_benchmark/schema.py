"""Stable generation contracts for the realistic search benchmark.

Generation agents will eventually emit dictionaries matching these dataclasses. Keeping the
contract dependency-free makes it usable from local scripts and Databricks notebooks alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import re
from typing import Iterable


class Category(str, Enum):
    PREFERENCE = "preference"
    PROJECT = "project"
    WORKFLOW = "workflow"
    DECISION = "decision"
    PEOPLE = "people"
    PLAN = "plan"
    HOUSEHOLD_LOGISTICS = "household_logistics"
    GLOBAL_FACT = "global_fact"


class EntityKind(str, Enum):
    PERSON = "person"
    PET = "pet"
    ORGANIZATION = "organization"
    TEAM = "team"
    PROJECT = "project"
    PRODUCT = "product"
    PLACE = "place"
    HOUSEHOLD = "household"
    TOOL = "tool"


PILOT_CASE_QUOTAS: dict[Category, int] = {
    Category.PREFERENCE: 4,
    Category.PROJECT: 5,
    Category.WORKFLOW: 4,
    Category.DECISION: 3,
    Category.PEOPLE: 3,
    Category.PLAN: 2,
    Category.HOUSEHOLD_LOGISTICS: 2,
    Category.GLOBAL_FACT: 2,
}

V1_CASE_QUOTAS: dict[Category, int] = {
    Category.PREFERENCE: 15,
    Category.PROJECT: 20,
    Category.WORKFLOW: 15,
    Category.DECISION: 15,
    Category.PEOPLE: 15,
    Category.PLAN: 15,
    Category.HOUSEHOLD_LOGISTICS: 10,
    Category.GLOBAL_FACT: 15,
}

PILOT_MEMORY_TARGET = 60
V1_MEMORY_TARGET = 300

PILOT_FACT_QUOTAS: dict[Category, int] = {
    Category.PREFERENCE: 8,
    Category.PROJECT: 10,
    Category.WORKFLOW: 8,
    Category.DECISION: 8,
    Category.PEOPLE: 8,
    Category.PLAN: 8,
    Category.HOUSEHOLD_LOGISTICS: 6,
    Category.GLOBAL_FACT: 8,
}

_STABLE_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_MEMORY_ID = re.compile(r"^mem-[0-9]{4}$")
_MEMORY_PATH = re.compile(r"^/memories/benchmark/mem-[0-9]{4}\.md$")


def _require_text(value: str, field_name: str, *, max_length: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    if len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")


def _require_stable_id(value: str, field_name: str) -> None:
    if not _STABLE_ID.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase hyphenated stable id: {value!r}")


def _validate_iso_date(value: str | None, field_name: str) -> None:
    if value is None:
        return
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date: {value!r}") from exc


@dataclass(frozen=True)
class WorldEntity:
    entity_id: str
    kind: EntityKind
    name: str
    aliases: tuple[str, ...]
    summary: str

    def __post_init__(self) -> None:
        _require_stable_id(self.entity_id, "entity_id")
        _require_text(self.name, "name", max_length=120)
        _require_text(self.summary, "summary", max_length=600)
        if len(set(alias.casefold() for alias in self.aliases)) != len(self.aliases):
            raise ValueError(f"Entity {self.entity_id} contains duplicate aliases")


@dataclass(frozen=True)
class CategoryBrief:
    category: Category
    brief: str
    focus_entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.brief, "brief", max_length=800)
        if not self.focus_entity_ids:
            raise ValueError("CategoryBrief.focus_entity_ids must not be empty")
        for entity_id in self.focus_entity_ids:
            _require_stable_id(entity_id, "focus_entity_id")


@dataclass(frozen=True)
class WorldFact:
    fact_id: str
    category: Category
    statement: str
    entity_ids: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_to: str | None = None
    supersedes_fact_id: str | None = None

    def __post_init__(self) -> None:
        _require_stable_id(self.fact_id, "fact_id")
        _require_text(self.statement, "statement", max_length=800)
        if not self.entity_ids:
            raise ValueError("WorldFact.entity_ids must contain at least one entity")
        for entity_id in self.entity_ids:
            _require_stable_id(entity_id, "entity_id")
        _validate_iso_date(self.valid_from, "valid_from")
        _validate_iso_date(self.valid_to, "valid_to")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from cannot be after valid_to")
        if self.supersedes_fact_id:
            _require_stable_id(self.supersedes_fact_id, "supersedes_fact_id")


@dataclass(frozen=True)
class WorldBible:
    world_id: str
    title: str
    premise: str
    owner_entity_id: str
    entities: tuple[WorldEntity, ...]
    category_briefs: tuple[CategoryBrief, ...]
    facts: tuple[WorldFact, ...]

    def __post_init__(self) -> None:
        _require_stable_id(self.world_id, "world_id")
        _require_text(self.title, "title", max_length=160)
        _require_text(self.premise, "premise", max_length=1600)
        _require_stable_id(self.owner_entity_id, "owner_entity_id")


@dataclass(frozen=True)
class BenchmarkMemory:
    memory_id: str
    category: Category
    description: str
    contents: str
    fact_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _MEMORY_ID.fullmatch(self.memory_id):
            raise ValueError(f"memory_id must look like mem-0001: {self.memory_id!r}")
        _require_text(self.description, "description", max_length=240)
        if "\n" in self.description:
            raise ValueError("description must be one line")
        if len(self.contents) > 4000:
            raise ValueError("contents must be at most 4000 characters")
        if self.contents.strip() == self.description.strip():
            raise ValueError("contents must not repeat the description")
        if not self.fact_ids:
            raise ValueError("BenchmarkMemory.fact_ids must not be empty")
        if not self.entity_ids:
            raise ValueError("BenchmarkMemory.entity_ids must not be empty")
        for fact_id in self.fact_ids:
            _require_stable_id(fact_id, "fact_id")
        for entity_id in self.entity_ids:
            _require_stable_id(entity_id, "entity_id")

    @property
    def path(self) -> str:
        return f"/memories/benchmark/{self.memory_id}.md"

    def seed_entry(self) -> dict[str, str]:
        if not _MEMORY_PATH.fullmatch(self.path):  # Defensive assertion around path construction.
            raise ValueError(f"Invalid benchmark memory path: {self.path}")
        return {
            "path": self.path,
            "description": self.description,
            "contents": self.contents,
        }


@dataclass(frozen=True)
class ScenarioCase:
    case_id: str
    category: Category
    situation: str
    query: str
    gold_memory_id: str
    context_memory_ids: tuple[str, ...]
    answer_should: str
    why_gold: str
    secondary_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_stable_id(self.case_id, "case_id")
        _require_text(self.situation, "situation", max_length=600)
        _require_text(self.query, "query", max_length=500)
        _require_text(self.answer_should, "answer_should", max_length=500)
        _require_text(self.why_gold, "why_gold", max_length=500)
        if not _MEMORY_ID.fullmatch(self.gold_memory_id):
            raise ValueError("gold_memory_id must reference an opaque benchmark memory id")
        if not 2 <= len(self.context_memory_ids) <= 6:
            raise ValueError("Every case needs 2-6 realistic context memories")
        if len(set(self.context_memory_ids)) != len(self.context_memory_ids):
            raise ValueError("context_memory_ids must be unique")
        if self.gold_memory_id in self.context_memory_ids:
            raise ValueError("The gold memory cannot also be a context memory")
        if any(not _MEMORY_ID.fullmatch(memory_id) for memory_id in self.context_memory_ids):
            raise ValueError("Every context_memory_id must be an opaque benchmark memory id")
        for tag in self.secondary_tags:
            _require_stable_id(tag, "secondary_tag")


@dataclass(frozen=True)
class ReviewDecision:
    reviewer: str
    realistic: bool
    answerable: bool
    gold_correct: bool
    unambiguous: bool
    category_fit: bool
    notes: str = ""
    issue_codes: tuple[str, ...] = ()
    alternative_memory_ids: tuple[str, ...] = ()
    suggested_repair: str = ""

    def __post_init__(self) -> None:
        _require_text(self.reviewer, "reviewer", max_length=120)
        if len(self.notes) > 1200:
            raise ValueError("review notes must be at most 1200 characters")
        if len(self.suggested_repair) > 1200:
            raise ValueError("suggested_repair must be at most 1200 characters")
        for issue_code in self.issue_codes:
            _require_stable_id(issue_code, "issue_code")
        for memory_id in self.alternative_memory_ids:
            if not _MEMORY_ID.fullmatch(memory_id):
                raise ValueError("alternative_memory_ids must contain benchmark memory ids")

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.realistic,
                self.answerable,
                self.gold_correct,
                self.unambiguous,
                self.category_fit,
            )
        )


@dataclass(frozen=True)
class ScenarioBundle:
    case: ScenarioCase
    facts: tuple[WorldFact, ...]
    memories: tuple[BenchmarkMemory, ...]
    reviews: tuple[ReviewDecision, ...] = field(default_factory=tuple)


def validate_world_bible(
    world: WorldBible,
    *,
    fact_quotas: dict[Category, int] | None = None,
) -> None:
    """Validate entity references, category coverage, and temporal fact links."""

    entities_by_id = _unique_by_id(world.entities, "entity_id")
    facts_by_id = _unique_by_id(world.facts, "fact_id")
    if world.owner_entity_id not in entities_by_id:
        raise ValueError("owner_entity_id must reference an entity")
    if entities_by_id[world.owner_entity_id].kind != EntityKind.PERSON:
        raise ValueError("owner_entity_id must reference a person")

    briefs_by_category = {brief.category: brief for brief in world.category_briefs}
    if len(briefs_by_category) != len(world.category_briefs):
        raise ValueError("World bible contains duplicate category briefs")
    missing_briefs = set(Category) - set(briefs_by_category)
    if missing_briefs:
        raise ValueError(
            f"World bible is missing category briefs: {sorted(item.value for item in missing_briefs)}"
        )

    missing_entity_refs = {
        entity_id
        for brief in world.category_briefs
        for entity_id in brief.focus_entity_ids
        if entity_id not in entities_by_id
    }
    missing_entity_refs.update(
        entity_id
        for fact in world.facts
        for entity_id in fact.entity_ids
        if entity_id not in entities_by_id
    )
    if missing_entity_refs:
        raise ValueError(f"World bible references missing entities: {sorted(missing_entity_refs)}")

    missing_superseded = {
        fact.supersedes_fact_id
        for fact in world.facts
        if fact.supersedes_fact_id and fact.supersedes_fact_id not in facts_by_id
    }
    if missing_superseded:
        raise ValueError(
            f"World bible references missing superseded facts: {sorted(missing_superseded)}"
        )

    for fact in world.facts:
        if fact.supersedes_fact_id:
            prior = facts_by_id[fact.supersedes_fact_id]
            if prior.category != fact.category:
                raise ValueError("A fact can supersede only another fact in the same category")
            if not prior.valid_to:
                raise ValueError(
                    f"Superseded fact {prior.fact_id} must have valid_to"
                )
            if not fact.valid_from:
                raise ValueError(
                    f"Superseding fact {fact.fact_id} must have valid_from"
                )
            if prior.valid_to > fact.valid_from:
                raise ValueError(
                    f"Superseded fact {prior.fact_id} ends after {fact.fact_id} begins"
                )

    quotas = fact_quotas or {}
    counts = {category: 0 for category in Category}
    for fact in world.facts:
        counts[fact.category] += 1
    mismatches = {
        category.value: {"expected": required, "actual": counts[category]}
        for category, required in quotas.items()
        if counts[category] != required
    }
    if mismatches:
        raise ValueError(f"World fact quota mismatches: {mismatches}")


def validate_bundle(bundle: ScenarioBundle, *, require_reviews: bool = False) -> None:
    """Validate references and hard acceptance gates for one generated scenario bundle."""

    memories_by_id = _unique_by_id(bundle.memories, "memory_id")
    facts_by_id = _unique_by_id(bundle.facts, "fact_id")

    referenced_memory_ids = {
        bundle.case.gold_memory_id,
        *bundle.case.context_memory_ids,
    }
    missing_memories = referenced_memory_ids - set(memories_by_id)
    if missing_memories:
        raise ValueError(f"Case references missing memories: {sorted(missing_memories)}")

    gold = memories_by_id[bundle.case.gold_memory_id]
    if gold.category != bundle.case.category:
        raise ValueError("The gold memory category must match the case category")

    missing_facts = {
        fact_id
        for memory in bundle.memories
        for fact_id in memory.fact_ids
        if fact_id not in facts_by_id
    }
    if missing_facts:
        raise ValueError(f"Memories reference missing facts: {sorted(missing_facts)}")

    if require_reviews:
        if len(bundle.reviews) < 2:
            raise ValueError("Accepted bundles require at least two independent reviews")
        if not all(review.accepted for review in bundle.reviews):
            raise ValueError("Accepted bundles cannot contain a rejecting review")


def _unique_by_id(items: Iterable[object], field_name: str) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for item in items:
        item_id = str(getattr(item, field_name))
        if item_id in indexed:
            raise ValueError(f"Duplicate {field_name}: {item_id}")
        indexed[item_id] = item
    return indexed


def world_bible_from_dict(data: dict) -> WorldBible:
    """Parse one generated JSON object into the validated dataclass contract."""

    world = WorldBible(
        world_id=str(data["world_id"]),
        title=str(data["title"]),
        premise=str(data["premise"]),
        owner_entity_id=str(data["owner_entity_id"]),
        entities=tuple(
            WorldEntity(
                entity_id=str(item["entity_id"]),
                kind=EntityKind(str(item["kind"])),
                name=str(item["name"]),
                aliases=tuple(str(value) for value in item.get("aliases", [])),
                summary=str(item["summary"]),
            )
            for item in data["entities"]
        ),
        category_briefs=tuple(
            CategoryBrief(
                category=Category(str(item["category"])),
                brief=str(item["brief"]),
                focus_entity_ids=tuple(
                    str(value) for value in item.get("focus_entity_ids", [])
                ),
            )
            for item in data["category_briefs"]
        ),
        facts=tuple(
            WorldFact(
                fact_id=str(item["fact_id"]),
                category=Category(str(item["category"])),
                statement=str(item["statement"]),
                entity_ids=tuple(str(value) for value in item.get("entity_ids", [])),
                aliases=tuple(str(value) for value in item.get("aliases", [])),
                valid_from=(str(item["valid_from"]) if item.get("valid_from") else None),
                valid_to=(str(item["valid_to"]) if item.get("valid_to") else None),
                supersedes_fact_id=(
                    str(item["supersedes_fact_id"])
                    if item.get("supersedes_fact_id")
                    else None
                ),
            )
            for item in data.get("facts", [])
        ),
    )
    return world


def scenario_bundle_from_dict(data: dict) -> ScenarioBundle:
    """Parse a persisted scenario bundle back into the stable dataclass contract."""

    case_data = data["case"]
    bundle = ScenarioBundle(
        case=scenario_case_from_dict(case_data),
        facts=tuple(world_fact_from_dict(item) for item in data.get("facts", [])),
        memories=tuple(
            benchmark_memory_from_dict(item) for item in data.get("memories", [])
        ),
        reviews=tuple(
            ReviewDecision(
                reviewer=str(item["reviewer"]),
                realistic=bool(item["realistic"]),
                answerable=bool(item["answerable"]),
                gold_correct=bool(item["gold_correct"]),
                unambiguous=bool(item["unambiguous"]),
                category_fit=bool(item["category_fit"]),
                notes=str(item.get("notes", "")),
                issue_codes=tuple(str(value) for value in item.get("issue_codes", [])),
                alternative_memory_ids=tuple(
                    str(value) for value in item.get("alternative_memory_ids", [])
                ),
                suggested_repair=str(item.get("suggested_repair", "")),
            )
            for item in data.get("reviews", [])
        ),
    )
    validate_bundle(bundle)
    return bundle


def world_fact_from_dict(data: dict) -> WorldFact:
    return WorldFact(
        fact_id=str(data["fact_id"]),
        category=Category(str(data["category"])),
        statement=str(data["statement"]),
        entity_ids=tuple(str(value) for value in data.get("entity_ids", [])),
        aliases=tuple(str(value) for value in data.get("aliases", [])),
        valid_from=(str(data["valid_from"]) if data.get("valid_from") else None),
        valid_to=(str(data["valid_to"]) if data.get("valid_to") else None),
        supersedes_fact_id=(
            str(data["supersedes_fact_id"]) if data.get("supersedes_fact_id") else None
        ),
    )


def benchmark_memory_from_dict(data: dict) -> BenchmarkMemory:
    return BenchmarkMemory(
        memory_id=str(data["memory_id"]),
        category=Category(str(data["category"])),
        description=str(data["description"]),
        contents=str(data.get("contents", "")),
        fact_ids=tuple(str(value) for value in data.get("fact_ids", [])),
        entity_ids=tuple(str(value) for value in data.get("entity_ids", [])),
    )


def scenario_case_from_dict(data: dict) -> ScenarioCase:
    return ScenarioCase(
        case_id=str(data["case_id"]),
        category=Category(str(data["category"])),
        situation=str(data["situation"]),
        query=str(data["query"]),
        gold_memory_id=str(data["gold_memory_id"]),
        context_memory_ids=tuple(
            str(value) for value in data.get("context_memory_ids", [])
        ),
        answer_should=str(data["answer_should"]),
        why_gold=str(data["why_gold"]),
        secondary_tags=tuple(str(value) for value in data.get("secondary_tags", [])),
    )

"""Curate the draft bundles into one deterministic 60-memory pilot corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .schema import (
    BenchmarkMemory,
    Category,
    PILOT_CASE_QUOTAS,
    PILOT_MEMORY_TARGET,
    ScenarioBundle,
    ScenarioCase,
    WorldBible,
    scenario_bundle_from_dict,
    validate_bundle,
    validate_world_bible,
    world_bible_from_dict,
)


DEFAULT_WORLD = Path(__file__).with_name("artifacts") / "pilot_world.json"
DEFAULT_DRAFTS = Path(__file__).with_name("artifacts") / "pilot_scenario_drafts.json"
DEFAULT_REVIEWS = Path(__file__).with_name("artifacts") / "pilot_scenario_reviews.json"
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts") / "pilot_curated.json"

_RAW_ID = re.compile(
    r"\b(?:person|pet|team|tool|place|project|product|organization|household)-[a-z0-9-]+"
)

# Active, useful facts that were absent from the 93 generated candidates. The four omitted unused
# facts are three superseded historical records and a redundant owner-role fact.
PILOT_ADDED_FACT_DESCRIPTIONS: dict[str, str] = {
    "fact-people-eli-based-at-riverbend": "Eli Mercado works at Riverbend Branch Library",
    "fact-preference-mara-hawthorn-work-blocks": (
        "Mara prefers Hawthorn Coffee for change-of-scene work blocks"
    ),
    "fact-decision-coverage-pilot-site-selection-riverbend": (
        "Riverbend chosen as the first Shift Coverage Pilot site"
    ),
    "fact-decision-kb-refresh-review-location-hawthorn": (
        "KB Refresh final review session planned at Hawthorn Coffee"
    ),
    "fact-decision-coverage-pilot-template-owner-ops-programs": (
        "Ops Programs maintains the official Shift Coverage Pilot templates"
    ),
    "fact-global-fact-ops-programs-cross-branch-process": (
        "Ops Programs owns cross-branch process consistency"
    ),
    "fact-global-fact-riverbend-frequent-pilot-site": (
        "Riverbend is a frequent Brightgrove pilot site"
    ),
    "fact-plan-coverage-pilot-kickoff-at-hq": (
        "Shift Coverage Pilot kickoff scheduled for March 19 at HQ"
    ),
    "fact-plan-riverbend-pilot-site-visit": (
        "Mara's Riverbend pilot visit scheduled for April 23"
    ),
    "fact-plan-hawthorn-weekend-planning-block-new": (
        "Weekend planning moved to Sunday afternoons at home"
    ),
}


def curate_pilot(
    world: WorldBible,
    draft_bundles: tuple[ScenarioBundle, ...],
) -> tuple[tuple[BenchmarkMemory, ...], tuple[ScenarioBundle, ...], dict[str, Any]]:
    """Deduplicate candidates, add ten memories, repair four cases, and remap references."""

    validate_world_bible(world)
    facts_by_id = {fact.fact_id: fact for fact in world.facts}
    groups: dict[tuple[str, ...], list[BenchmarkMemory]] = defaultdict(list)
    gold_candidate_ids = {bundle.case.gold_memory_id for bundle in draft_bundles}
    for bundle in draft_bundles:
        for memory in bundle.memories:
            groups[tuple(sorted(memory.fact_ids))].append(memory)

    representatives: list[tuple[tuple[str, ...], BenchmarkMemory, list[str]]] = []
    for fact_set, occurrences in groups.items():
        by_id = {memory.memory_id: memory for memory in occurrences}
        gold_occurrences = sorted(set(by_id) & gold_candidate_ids)
        representative = (
            by_id[gold_occurrences[0]]
            if gold_occurrences
            else min(by_id.values(), key=lambda memory: (len(memory.description), memory.memory_id))
        )
        representatives.append((fact_set, representative, sorted(by_id)))
    representatives.sort(key=lambda item: item[0])

    corpus: list[BenchmarkMemory] = []
    old_to_final: dict[str, str] = {}
    provenance: dict[str, Any] = {}
    for number, (fact_set, representative, source_ids) in enumerate(representatives, start=1):
        memory_id = f"mem-{number:04d}"
        canonical = replace(representative, memory_id=memory_id)
        corpus.append(canonical)
        for source_id in source_ids:
            old_to_final[source_id] = memory_id
        provenance[memory_id] = {
            "fact_ids": list(fact_set),
            "source_candidate_memory_ids": source_ids,
            "source": "deduplicated-candidate",
        }

    used_fact_ids = {fact_id for memory in corpus for fact_id in memory.fact_ids}
    for fact_id, description in PILOT_ADDED_FACT_DESCRIPTIONS.items():
        if fact_id in used_fact_ids:
            raise ValueError(f"Added pilot fact is already represented: {fact_id}")
        fact = facts_by_id[fact_id]
        memory_id = f"mem-{len(corpus) + 1:04d}"
        memory = BenchmarkMemory(
            memory_id=memory_id,
            category=fact.category,
            description=description,
            contents=fact.statement,
            fact_ids=(fact.fact_id,),
            entity_ids=fact.entity_ids,
        )
        corpus.append(memory)
        provenance[memory_id] = {
            "fact_ids": [fact.fact_id],
            "source_candidate_memory_ids": [],
            "source": "unused-world-fact",
        }

    corpus_by_id = {memory.memory_id: memory for memory in corpus}
    memory_id_by_fact_id = {
        fact_id: memory.memory_id for memory in corpus for fact_id in memory.fact_ids
    }
    repaired_cases: list[ScenarioCase] = []
    repairs: list[dict[str, str]] = []
    for bundle in draft_bundles:
        contexts = _unique(
            old_to_final[memory_id] for memory_id in bundle.case.context_memory_ids
        )
        gold_id = old_to_final[bundle.case.gold_memory_id]
        contexts = tuple(memory_id for memory_id in contexts if memory_id != gold_id)
        case = replace(
            bundle.case,
            gold_memory_id=gold_id,
            context_memory_ids=contexts,
        )
        case, repair = _repair_case(case, memory_id_by_fact_id)
        repaired_cases.append(case)
        if repair:
            repairs.append(repair)

    curated_bundles = tuple(
        _bundle_for_case(case, corpus_by_id, world)
        for case in repaired_cases
    )
    validate_curated_pilot(tuple(corpus), curated_bundles, world)
    metadata = {
        "source_candidate_memories": sum(len(bundle.memories) for bundle in draft_bundles),
        "deduplicated_candidate_memories": len(representatives),
        "added_unused_world_facts": len(PILOT_ADDED_FACT_DESCRIPTIONS),
        "repairs": repairs,
        "provenance": provenance,
    }
    return tuple(corpus), curated_bundles, metadata


def _repair_case(
    case: ScenarioCase,
    memory_id_by_fact_id: dict[str, str],
) -> tuple[ScenarioCase, dict[str, str] | None]:
    original_query = case.query
    if case.case_id == "people-owen-escalation-contact":
        case = replace(
            case,
            situation=(
                "Mara is updating an internal contact guide and needs the general IT owner for systems "
                "rollouts and support workflows, not a project-specific escalation path."
            ),
            query=(
                "Who leads IT Services and is our general point of contact for systems rollouts and "
                "support workflows?"
            ),
            answer_should=(
                "Identify Owen Reed as the IT Services lead and Brightgrove's general point of contact "
                "for systems rollouts and support workflows."
            ),
            why_gold=(
                "The gold states Owen's leadership role and general contact responsibility; the pilot "
                "decision only names him for a project-specific escalation path."
            ),
        )
    elif case.case_id == "people-sasha-comms-training-partner":
        case = replace(
            case,
            situation=(
                "Mara is writing an orientation note and wants to describe Sasha's role and how they "
                "work together without reopening old notes."
            ),
            query=(
                "What's Sasha's exact job title at Brightgrove, and which working group are we in "
                "together?"
            ),
            answer_should=(
                "Identify Sasha Kim as a communications specialist and say she works closely with "
                "Mara in the Comms & Training Working Group."
            ),
            why_gold=(
                "The gold uniquely contains both Sasha's job role and the working group she shares "
                "with Mara; project memories contain only narrower collaboration details."
            ),
        )
    elif case.case_id == "people-jordan-package-backup":
        case = replace(
            case,
            situation=(
                "Mara sees Jordan's name in a neighborhood thread and wants to place how they know "
                "each other outside occasional household favors."
            ),
            query=(
                "Besides package pickup, how do I know Jordan socially, and what activity do we "
                "sometimes do together?"
            ),
            answer_should=(
                "Say Jordan Wu is Mara's neighbor and occasional running buddy."
            ),
            why_gold=(
                "Only the gold states both the neighbor relationship and their occasional runs; "
                "household memories mention Jordan only as a package fallback."
            ),
        )
    elif case.case_id == "decision-kb-freeze-window":
        case = replace(
            case,
            situation=(
                "Mara already has the approved late-April freeze dates in front of her, but needs to "
                "explain Nina's reason for the decision to contributors."
            ),
            query=(
                "Why did Nina approve the Knowledge Base Refresh documentation freeze from April 20 "
                "through April 30?"
            ),
            answer_should=(
                "Explain that Nina approved the freeze to allow a clean review pass and avoid "
                "midstream edits."
            ),
            why_gold=(
                "Only the gold records Nina's approved decision and its rationale; Mara's earlier "
                "personal calendar block was for finishing the outline and page inventory."
            ),
        )
    elif case.case_id == "household-trash-night":
        case = replace(
            case,
            situation=(
                "Mara and Devon are coordinating household responsibilities and want to confirm the "
                "standing trash and recycling routine."
            ),
            query="What night do trash and recycling go out, and who handles the bins by default?",
        )
    elif case.case_id == "project-coverage-pilot-primary-site":
        case = replace(
            case,
            case_id="project-coverage-pilot-expansion",
            situation=(
                "After the Riverbend pilot period, Mara is briefing IT and the HQ team leads on the "
                "next phase and wants to confirm both its audience and purpose."
            ),
            query=(
                "After the Riverbend pilot period, where is the Shift Coverage Pilot supposed to "
                "expand next, and what will that phase test?"
            ),
            answer_should=(
                "Say the pilot is supposed to expand to HQ-based teams and that the next phase will "
                "test cross-site escalation paths."
            ),
            why_gold=(
                "Only the gold describes the post-Riverbend expansion to HQ-based teams and its "
                "cross-site escalation purpose; nearby memories cover the current site, the kickoff "
                "location, or the existing escalation owner."
            ),
            gold_memory_id=memory_id_by_fact_id[
                "fact-project-coverage-pilot-expansion-plan-to-hq"
            ],
            context_memory_ids=tuple(
                memory_id_by_fact_id[fact_id]
                for fact_id in (
                    "fact-project-coverage-pilot-primary-site-riverbend",
                    "fact-decision-coverage-pilot-site-selection-riverbend",
                    "fact-plan-coverage-pilot-kickoff-at-hq",
                    "fact-decision-coverage-pilot-escalation-owner-it-services",
                )
            ),
            secondary_tags=("same-project", "temporal", "cross-category", "two-part-question"),
        )
    elif case.case_id == "global-branch-leads-channel":
        case = replace(
            case,
            situation=(
                "A new project lead knows the Branch Leads Council handles some pilot feedback but "
                "does not understand the council's permanent network-wide role."
            ),
            query=(
                "Outside individual projects, what is the Branch Leads Council's standing role at "
                "Brightgrove?"
            ),
            answer_should=(
                "Explain that the Branch Leads Council is Brightgrove's standing channel for frontline "
                "feedback on network-wide operational changes before broad rollout."
            ),
            why_gold=(
                "The gold defines the council's permanent network-wide role and when that feedback is "
                "used; the pilot decision only assigns it as the channel for one project."
            ),
        )
    else:
        return case, None
    return case, {"case_id": case.case_id, "original_query": original_query, "new_query": case.query}


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _bundle_for_case(
    case: ScenarioCase,
    corpus_by_id: dict[str, BenchmarkMemory],
    world: WorldBible,
) -> ScenarioBundle:
    referenced_ids = (case.gold_memory_id, *case.context_memory_ids)
    memories = tuple(corpus_by_id[memory_id] for memory_id in referenced_ids)
    fact_ids = {fact_id for memory in memories for fact_id in memory.fact_ids}
    facts = tuple(fact for fact in world.facts if fact.fact_id in fact_ids)
    bundle = ScenarioBundle(case=case, facts=facts, memories=memories)
    validate_bundle(bundle)
    return bundle


def validate_curated_pilot(
    corpus: tuple[BenchmarkMemory, ...],
    bundles: tuple[ScenarioBundle, ...],
    world: WorldBible,
) -> None:
    if len(corpus) != PILOT_MEMORY_TARGET:
        raise ValueError(
            f"Curated pilot must contain {PILOT_MEMORY_TARGET} memories, got {len(corpus)}"
        )
    memory_ids = [memory.memory_id for memory in corpus]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("Curated corpus memory ids must be unique")
    expected_ids = {f"mem-{number:04d}" for number in range(1, PILOT_MEMORY_TARGET + 1)}
    if set(memory_ids) != expected_ids:
        raise ValueError("Curated corpus memory ids must be contiguous")
    fact_sets = [tuple(sorted(memory.fact_ids)) for memory in corpus]
    if len(fact_sets) != len(set(fact_sets)):
        raise ValueError("Curated corpus cannot contain duplicate canonical fact sets")
    descriptions = [memory.description.casefold() for memory in corpus]
    if len(descriptions) != len(set(descriptions)):
        raise ValueError("Curated corpus cannot contain duplicate descriptions")

    facts_by_id = {fact.fact_id: fact for fact in world.facts}
    for memory in corpus:
        if any(fact_id not in facts_by_id for fact_id in memory.fact_ids):
            raise ValueError("Curated memory references a missing world fact")
        if any(facts_by_id[fact_id].category != memory.category for fact_id in memory.fact_ids):
            raise ValueError("Curated memory category must match all canonical facts")
        if not memory.contents.strip():
            raise ValueError("Curated memory contents must not be empty")
        if _RAW_ID.search(memory.description) or _RAW_ID.search(memory.contents):
            raise ValueError("Curated memory exposes a raw world id")

    counts = Counter(bundle.case.category for bundle in bundles)
    mismatches = {
        category.value: {"expected": required, "actual": counts[category]}
        for category, required in PILOT_CASE_QUOTAS.items()
        if counts[category] != required
    }
    if mismatches:
        raise ValueError(f"Curated case quota mismatches: {mismatches}")
    case_ids = [bundle.case.case_id for bundle in bundles]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Curated case ids must be unique")
    queries = [bundle.case.query.casefold() for bundle in bundles]
    if len(queries) != len(set(queries)):
        raise ValueError("Curated queries must be unique")
    corpus_by_id = {memory.memory_id: memory for memory in corpus}
    gold_ids: set[str] = set()
    for bundle in bundles:
        validate_bundle(bundle)
        case = bundle.case
        if case.gold_memory_id not in corpus_by_id:
            raise ValueError("Curated gold memory is absent from corpus")
        if any(memory_id not in corpus_by_id for memory_id in case.context_memory_ids):
            raise ValueError("Curated context memory is absent from corpus")
        if corpus_by_id[case.gold_memory_id].category != case.category:
            raise ValueError("Curated gold category must match case category")
        if case.gold_memory_id in gold_ids:
            raise ValueError("Curated cases must use unique gold memories")
        gold_ids.add(case.gold_memory_id)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _args()
    world_artifact = json.loads(args.world.read_text())
    draft_artifact = json.loads(args.drafts.read_text())
    review_artifact = json.loads(args.reviews.read_text())
    world = world_bible_from_dict(world_artifact["world"])
    draft_bundles = tuple(
        scenario_bundle_from_dict(item) for item in draft_artifact["bundles"]
    )
    corpus, bundles, metadata = curate_pilot(world, draft_bundles)
    artifact = {
        "schema_version": "realistic-search-pilot-curated-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "world": str(args.world),
            "drafts": str(args.drafts),
            "reviews": str(args.reviews),
            "review_schema_version": review_artifact.get("schema_version"),
        },
        "summary": {
            "memories": len(corpus),
            "cases": len(bundles),
            "case_counts": {
                category.value: sum(bundle.case.category == category for bundle in bundles)
                for category in Category
            },
            "repairs": len(metadata["repairs"]),
        },
        "corpus": [asdict(memory) for memory in corpus],
        "facts": [asdict(fact) for fact in world.facts],
        "cases": [asdict(bundle.case) for bundle in bundles],
        "bundles": [asdict(bundle) for bundle in bundles],
        "curation": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **artifact["summary"]}))


if __name__ == "__main__":
    main()

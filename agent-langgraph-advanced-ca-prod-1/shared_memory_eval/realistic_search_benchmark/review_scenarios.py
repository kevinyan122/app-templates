"""Run two independent critics over the pilot scenario drafts.

This stage produces local review and consistency artifacts only. It does not repair cases, publish an
MLflow dataset, or call the managed-memory APIs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from .generate_world import DEFAULT_MODEL_ENDPOINT, DatabricksJsonAgent
from .schema import (
    BenchmarkMemory,
    ReviewDecision,
    ScenarioBundle,
    WorldFact,
    benchmark_memory_from_dict,
    scenario_bundle_from_dict,
    world_fact_from_dict,
)


DEFAULT_DRAFTS = Path(__file__).with_name("artifacts") / "pilot_scenario_drafts.json"
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts") / "pilot_scenario_reviews.json"

EVIDENCE_REVIEWER = "critic-evidence"
REALISM_REVIEWER = "critic-realism"
REVIEWERS = (EVIDENCE_REVIEWER, REALISM_REVIEWER)

JsonAgent = Callable[[str, str], dict[str, Any]]

EVIDENCE_SYSTEM_PROMPT = """You are the evidence critic for a realistic memory-search benchmark.
Audit every case strictly against the supplied gold memory, local context ids, full candidate-memory
pool, and canonical facts. Compare answer_should requirement by requirement. Reject a case when the gold
does not fully support the rubric or another candidate memory independently satisfies every material
requirement.

Be precise, not adversarial. Related memories are desirable; only equally sufficient alternatives are a
problem. Return exactly one JSON object with no markdown or commentary."""

REALISM_SYSTEM_PROMPT = """You are the realism and ambiguity critic for a realistic memory-search
benchmark. Independently review every case as a future interaction with a personal/work assistant.
Focus on whether the situation and query sound natural, have one intended interpretation, use the right
category, handle dates and superseded facts correctly, and identify a uniquely sufficient gold memory.

Do not reward artificial search difficulty or reject ordinary paraphrases. Return exactly one JSON
object with no markdown or commentary."""


def critic_prompt(
    bundles: tuple[ScenarioBundle, ...],
    reviewer: str,
    candidate_memories: tuple[BenchmarkMemory, ...] | None = None,
    canonical_facts: tuple[WorldFact, ...] | None = None,
    benchmark_as_of_date: str | None = None,
) -> str:
    memories_by_id = {
        memory.memory_id: memory
        for memory in (
            candidate_memories
            or tuple(memory for bundle in bundles for memory in bundle.memories)
        )
    }
    facts_by_id = {
        fact.fact_id: fact
        for fact in (
            canonical_facts or tuple(fact for bundle in bundles for fact in bundle.facts)
        )
    }
    payload = {
        "cases": [asdict(bundle.case) for bundle in bundles],
        "candidate_memory_pool": [asdict(memory) for memory in memories_by_id.values()],
        "canonical_facts": [asdict(fact) for fact in facts_by_id.values()],
    }
    if benchmark_as_of_date is not None:
        payload["benchmark_as_of_date"] = benchmark_as_of_date
    perspective = (
        "evidence support, gold uniqueness, and alternative answer-bearing memories"
        if reviewer == EVIDENCE_REVIEWER
        else "realism, naturalness, temporal clarity, ambiguity, and category fit"
    )
    return f"""Review all {len(bundles)} scenario cases independently as `{reviewer}`.

Your primary perspective is: {perspective}. You must still score every field for every case.

Scoring fields:
- realistic: the situation, query, memories, and expected answer resemble an ordinary future use.
- answerable: the gold memory alone contains everything required by answer_should.
- gold_correct: the selected gold is the uniquely best answer source. Set false only if a local context
  or memory elsewhere in candidate_memory_pool independently satisfies every material requirement.
- unambiguous: the query and time frame have one clear intended answer and do not depend on unstated facts.
- category_fit: the case's category matches the kind of information being recalled.

Rules:
- Inspect the entire candidate pool, not only context_memory_ids.
- Treat each memory's description and contents together as its complete evidence. Empty contents are
  valid for an atomic memory when the description itself fully states the fact.
- Set gold_correct false only when a non-gold memory independently supports every material requirement
  in answer_should, or reaches the same complete answer without a meaningful inference. A memory that
  merely names the same entity, supports one part of a multi-part answer, or supplies related project
  context is a useful distractor and does not invalidate the gold.
- When answer_should asks for both an exact title or role and a relationship, group, or project, a
  memory that supplies only the relationship, group, or project is partial evidence, not an alternative.
- Match scope exactly: a project-specific escalation contact or one-off collaborator does not satisfy a
  requirement for a person's general organization-wide role or ownership, and vice versa.
- Treat a context that directly names the requested person, date, place, process, or decision as an
  alternative only when it is independently sufficient for the complete requested answer.
- For temporal cases, the gold must be current and superseded memories must be clearly historical.
- Do not judge expected retrieval rank, keyword overlap, or whether the query is difficult enough.
- Do not reject merely because context is closely related; realistic nearby context is intentional.
- issue_codes are short lowercase hyphenated labels such as context-answers-query,
  global-alternative-answer, gold-missing-requirement, temporal-ambiguity, query-ambiguous,
  category-mismatch, unsupported-memory, answer-rubric-overreach, or unrealistic-scenario.
- alternative_memory_ids must list every non-gold candidate memory that independently satisfies the
  complete answer_should rubric; do not list partial supporting memories.
- If all five booleans are true, suggested_repair may be empty. Otherwise give one concrete minimal fix.
- notes must concisely justify the booleans with specific evidence.
- Before returning, ensure gold_correct, issue_codes, alternative_memory_ids, notes, and suggested_repair
  agree with one another. Never reject gold uniqueness when the notes conclude that only the gold meets
  every requirement.

Return exactly:
{{
  "reviews": [
    {{
      "case_id": "...",
      "realistic": true,
      "answerable": true,
      "gold_correct": true,
      "unambiguous": true,
      "category_fit": true,
      "issue_codes": [],
      "alternative_memory_ids": [],
      "notes": "...",
      "suggested_repair": ""
    }}
  ]
}}

There must be exactly one review for every supplied case_id and no extra case ids.

Benchmark payload:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"""


def review_scenario_drafts(
    agents: dict[str, JsonAgent],
    bundles: tuple[ScenarioBundle, ...],
    *,
    attempts: int = 3,
    candidate_memories: tuple[BenchmarkMemory, ...] | None = None,
    canonical_facts: tuple[WorldFact, ...] | None = None,
    benchmark_as_of_date: str | None = None,
) -> dict[str, tuple[ReviewDecision, ...]]:
    """Run both critics concurrently and return reviews keyed by case id."""

    if set(agents) != set(REVIEWERS):
        raise ValueError(f"agents must contain exactly these reviewers: {REVIEWERS}")

    outputs: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(REVIEWERS)) as executor:
        futures = {
            executor.submit(
                _generate_critic_output,
                agents[reviewer],
                reviewer,
                bundles,
                attempts=attempts,
                candidate_memories=candidate_memories,
                canonical_facts=canonical_facts,
                benchmark_as_of_date=benchmark_as_of_date,
            ): reviewer
            for reviewer in REVIEWERS
        }
        for future in as_completed(futures):
            reviewer = futures[future]
            outputs[reviewer] = future.result()

    reviews_by_case: dict[str, list[ReviewDecision]] = defaultdict(list)
    for reviewer in REVIEWERS:
        normalized = _normalize_critic_output(
            outputs[reviewer], reviewer, bundles, candidate_memories=candidate_memories
        )
        for case_id, decision in normalized.items():
            reviews_by_case[case_id].append(decision)
    return {case_id: tuple(decisions) for case_id, decisions in reviews_by_case.items()}


def _generate_critic_output(
    agent: JsonAgent,
    reviewer: str,
    bundles: tuple[ScenarioBundle, ...],
    *,
    attempts: int,
    candidate_memories: tuple[BenchmarkMemory, ...] | None,
    canonical_facts: tuple[WorldFact, ...] | None,
    benchmark_as_of_date: str | None,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    prompt = critic_prompt(
        bundles,
        reviewer,
        candidate_memories,
        canonical_facts,
        benchmark_as_of_date,
    )
    current_prompt = prompt
    last_error: Exception | None = None
    for _ in range(attempts):
        output: dict[str, Any] | None = None
        try:
            output = agent(reviewer, current_prompt)
            _normalize_critic_output(
                output, reviewer, bundles, candidate_memories=candidate_memories
            )
            return output
        except (KeyError, TypeError, ValueError) as exc:
            last_error = exc
            current_prompt = (
                prompt
                + "\n\nCORRECTION: The prior review JSON failed validation: "
                + str(exc)
                + " Repair it minimally and return one complete review for every supplied case id."
                + "\n\nPrevious JSON:\n"
                + json.dumps(output or {}, ensure_ascii=False, sort_keys=True)
            )
    raise RuntimeError(f"{reviewer} failed review-output validation") from last_error


def _normalize_critic_output(
    output: dict[str, Any],
    reviewer: str,
    bundles: tuple[ScenarioBundle, ...],
    *,
    candidate_memories: tuple[BenchmarkMemory, ...] | None = None,
) -> dict[str, ReviewDecision]:
    raw_reviews = output["reviews"]
    if not isinstance(raw_reviews, list):
        raise TypeError("reviews must be a list")
    expected_case_ids = {bundle.case.case_id for bundle in bundles}
    all_memory_ids = {
        memory.memory_id
        for memory in (
            candidate_memories
            or tuple(memory for bundle in bundles for memory in bundle.memories)
        )
    }
    gold_by_case = {bundle.case.case_id: bundle.case.gold_memory_id for bundle in bundles}
    indexed: dict[str, ReviewDecision] = {}

    for item in raw_reviews:
        case_id = str(item["case_id"])
        if case_id in indexed:
            raise ValueError(f"duplicate critic review for {case_id}")
        booleans = {
            name: _require_bool(item[name], name)
            for name in (
                "realistic",
                "answerable",
                "gold_correct",
                "unambiguous",
                "category_fit",
            )
        }
        alternative_ids = tuple(
            str(value) for value in item.get("alternative_memory_ids", [])
        )
        missing_alternatives = set(alternative_ids) - all_memory_ids
        if missing_alternatives:
            raise ValueError(
                f"review references missing alternative memories: {sorted(missing_alternatives)}"
            )
        if case_id in gold_by_case and gold_by_case[case_id] in alternative_ids:
            raise ValueError("alternative_memory_ids cannot contain the case's gold memory")

        decision = ReviewDecision(
            reviewer=reviewer,
            **booleans,
            notes=str(item.get("notes", "")),
            issue_codes=tuple(str(value) for value in item.get("issue_codes", [])),
            alternative_memory_ids=alternative_ids,
            suggested_repair=str(item.get("suggested_repair", "")),
        )
        if not decision.notes.strip():
            raise ValueError("every critic review must include notes")
        if not decision.accepted and not decision.suggested_repair.strip():
            raise ValueError("a rejecting review must include suggested_repair")
        indexed[case_id] = decision

    actual_case_ids = set(indexed)
    if actual_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - actual_case_ids)
        extra = sorted(actual_case_ids - expected_case_ids)
        raise ValueError(f"critic case ids mismatch; missing={missing}, extra={extra}")
    return indexed


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a JSON boolean")
    return value


def global_consistency_checks(
    bundles: tuple[ScenarioBundle, ...],
    candidate_memories: tuple[BenchmarkMemory, ...] | None = None,
) -> dict[str, Any]:
    """Report cross-bundle duplication that the final curator must resolve."""

    fact_set_occurrences: dict[tuple[str, ...], list[str]] = defaultdict(list)
    description_occurrences: dict[str, list[str]] = defaultdict(list)
    memories_by_id = {
        memory.memory_id: memory
        for memory in (
            candidate_memories
            or tuple(memory for bundle in bundles for memory in bundle.memories)
        )
    }
    for memory in memories_by_id.values():
        fact_set_occurrences[tuple(sorted(memory.fact_ids))].append(memory.memory_id)
        description_occurrences[memory.description.casefold()].append(memory.memory_id)

    duplicate_fact_sets = [
        {"fact_ids": list(fact_ids), "memory_ids": memory_ids}
        for fact_ids, memory_ids in fact_set_occurrences.items()
        if len(memory_ids) > 1
    ]
    duplicate_descriptions = [
        {"description": description, "memory_ids": memory_ids}
        for description, memory_ids in description_occurrences.items()
        if len(memory_ids) > 1
    ]
    return {
        "candidate_memory_occurrences": len(memories_by_id),
        "unique_fact_sets": len(fact_set_occurrences),
        "duplicate_fact_set_groups": duplicate_fact_sets,
        "duplicate_description_groups": duplicate_descriptions,
    }


def review_summary(
    bundles: tuple[ScenarioBundle, ...],
    reviews_by_case: dict[str, tuple[ReviewDecision, ...]],
) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    for bundle in bundles:
        decisions = reviews_by_case[bundle.case.case_id]
        accepted = sum(decision.accepted for decision in decisions)
        statuses[bundle.case.case_id] = (
            "accepted" if accepted == len(REVIEWERS) else "disagreement" if accepted else "rejected"
        )
    return {
        "cases": len(bundles),
        "accepted_by_both": sum(status == "accepted" for status in statuses.values()),
        "disagreements": sum(status == "disagreement" for status in statuses.values()),
        "rejected_by_both": sum(status == "rejected" for status in statuses.values()),
        "review_queue_case_ids": [
            case_id for case_id, status in statuses.items() if status != "accepted"
        ],
        "critic_acceptance": {
            reviewer: sum(
                decision.accepted
                for decisions in reviews_by_case.values()
                for decision in decisions
                if decision.reviewer == reviewer
            )
            for reviewer in REVIEWERS
        },
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--model-endpoint", default=DEFAULT_MODEL_ENDPOINT)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    args = _args()
    profile = args.profile or os.getenv("DATABRICKS_CONFIG_PROFILE", "")
    if not profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")

    draft_artifact = json.loads(args.drafts.read_text())
    bundles = tuple(scenario_bundle_from_dict(item) for item in draft_artifact["bundles"])
    candidate_memories = tuple(
        benchmark_memory_from_dict(item) for item in draft_artifact.get("corpus", [])
    ) or None
    canonical_facts = tuple(
        world_fact_from_dict(item) for item in draft_artifact.get("facts", [])
    ) or None
    benchmark_as_of_date = draft_artifact.get("benchmark_as_of_date")

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=profile)
    agents = {
        EVIDENCE_REVIEWER: DatabricksJsonAgent(
            workspace_client=client,
            endpoint=args.model_endpoint,
            system_prompt=EVIDENCE_SYSTEM_PROMPT,
        ),
        REALISM_REVIEWER: DatabricksJsonAgent(
            workspace_client=client,
            endpoint=args.model_endpoint,
            system_prompt=REALISM_SYSTEM_PROMPT,
        ),
    }
    reviews_by_case = review_scenario_drafts(
        agents,
        bundles,
        candidate_memories=candidate_memories,
        canonical_facts=canonical_facts,
        benchmark_as_of_date=benchmark_as_of_date,
    )
    reviewed_bundles = tuple(
        replace(bundle, reviews=reviews_by_case[bundle.case.case_id])
        for bundle in bundles
    )
    summary = review_summary(reviewed_bundles, reviews_by_case)
    artifact = {
        "schema_version": "realistic-search-scenario-reviews-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_endpoint": args.model_endpoint,
        "source_drafts": str(args.drafts),
        "benchmark_as_of_date": benchmark_as_of_date,
        "summary": summary,
        "global_checks": global_consistency_checks(reviewed_bundles, candidate_memories),
        "cases": [
            {
                "case_id": bundle.case.case_id,
                "status": (
                    "accepted"
                    if all(review.accepted for review in bundle.reviews)
                    else "disagreement"
                    if any(review.accepted for review in bundle.reviews)
                    else "rejected"
                ),
                "reviews": [asdict(review) for review in bundle.reviews],
            }
            for bundle in reviewed_bundles
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **summary}))


if __name__ == "__main__":
    main()

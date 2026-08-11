"""Generate draft scenario bundles from the validated pilot world.

This stage writes a local JSON artifact only. It creates candidate memories for review; it does not
publish an MLflow dataset or call the managed-memory APIs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from dotenv import load_dotenv

from .generate_world import DEFAULT_MODEL_ENDPOINT, DatabricksJsonAgent
from .schema import (
    BenchmarkMemory,
    Category,
    PILOT_CASE_QUOTAS,
    ScenarioBundle,
    ScenarioCase,
    WorldBible,
    validate_bundle,
    validate_world_bible,
    world_bible_from_dict,
)


DEFAULT_WORLD = Path(__file__).with_name("artifacts") / "pilot_world.json"
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts") / "pilot_scenario_drafts.json"

SCENARIO_GROUPS: tuple[tuple[Category, ...], ...] = (
    (Category.PREFERENCE, Category.PEOPLE),
    (Category.PROJECT, Category.DECISION),
    (Category.WORKFLOW, Category.GLOBAL_FACT),
    (Category.PLAN, Category.HOUSEHOLD_LOGISTICS),
)

JsonAgent = Callable[[str, str], dict[str, Any]]

SCENARIO_SYSTEM_PROMPT = """You are a scenario writer for a realistic long-term-memory search
benchmark. Work only from the supplied fictional world and canonical facts. Create ordinary questions
that a person might naturally ask their assistant weeks or months later. Do not attack a retrieval
algorithm, insert false distractors, or invent facts.

Each case has one clearly best gold memory and two to six true contextual memories. Context should be
nearby in real life—about the same person, project, tool, place, or routine—but must not independently
answer the question. Return exactly one JSON object and no markdown or commentary."""

_RAW_ID = re.compile(
    r"\b(?:person|pet|team|tool|place|project|product|organization|household)-[a-z0-9-]+"
)


def scenario_prompt(
    world: WorldBible,
    categories: tuple[Category, ...],
    case_quotas: dict[Category, int],
) -> str:
    requested = {category.value: case_quotas[category] for category in categories}
    category_text = ", ".join(category.value for category in categories)
    return f"""Write draft scenario bundles for these categories only: {category_text}.

Exact case counts: {json.dumps(requested, sort_keys=True)}

Principles:
- Begin with a plausible future situation in this world, then derive the question and memories.
- The query must sound like a normal user request. Never mention tests, retrieval, search, memory ids,
  canonical facts, or instruct the assistant to use memory.
- Use a different gold fact or fact set for every case. Never make a superseded fact the gold.
- The gold memory alone must support answer_should. Context memories must be true and relevant but must
  not provide an equally complete answer.
- Give every case two to four context memories unless a fifth or sixth is genuinely useful.
- Prefer normal contextual overlap: the same project with a different detail, the same person on a
  different responsibility, a nearby workflow, a prior dated decision, or a similar household routine.
- Do not create keyword-stuffed or adversarial distractors. Do not copy the gold wording into the query.
- A memory may use one to three closely related canonical facts, all from its declared category.
- Every memory must stand alone. Its description is a short one-line label of at most 120 characters.
  Put the useful details, dates, numbers, reasons, and structure in non-empty contents; do not repeat the
  description verbatim in contents.
- fact_ids and entity_ids are provenance only. Never expose raw ids in descriptions, contents, queries,
  situations, answer_should, or why_gold.
- answer_should is a concise grading rubric stating only what a correct answer must include.
- why_gold briefly explains why the gold is uniquely sufficient. It is internal review metadata.
- secondary_tags may use natural analytical labels such as paraphrase, same-entity, same-topic,
  cross-category, alias, temporal, or implicit-context. They must not drive unnatural wording.

Use local memory keys within each bundle: `gold`, `context-1`, `context-2`, and so on. Keys are local to
the bundle and will be replaced with opaque global ids by the orchestrator.

World:
{json.dumps(asdict(world), ensure_ascii=False, sort_keys=True)}

Return this shape:
{{
  "bundles": [
    {{
      "case": {{
        "case_id": "preference-natural-short-id",
        "category": "preference",
        "situation": "Why this question naturally comes up later.",
        "query": "The user's natural question.",
        "gold_memory_key": "gold",
        "context_memory_keys": ["context-1", "context-2"],
        "answer_should": "A correct answer should ...",
        "why_gold": "The gold memory uniquely ...",
        "secondary_tags": ["paraphrase", "same-topic"]
      }},
      "memories": [
        {{
          "memory_key": "gold",
          "category": "preference",
          "description": "Short specific one-line label",
          "contents": "Standalone details supported only by the listed canonical facts.",
          "fact_ids": ["fact-preference-example"],
          "entity_ids": ["person-example"]
        }}
      ]
    }}
  ]
}}"""


def generate_scenario_drafts(
    agent: JsonAgent,
    world: WorldBible,
    *,
    case_quotas: dict[Category, int] | None = None,
    parallel: bool = True,
    writer_attempts: int = 3,
) -> tuple[ScenarioBundle, ...]:
    """Run four category writers and normalize their local keys to global memory ids."""

    quotas = dict(case_quotas or PILOT_CASE_QUOTAS)
    validate_world_bible(world)
    jobs = [
        (
            f"scenario-writer-{index + 1}",
            group,
            scenario_prompt(world, group, quotas),
        )
        for index, group in enumerate(SCENARIO_GROUPS)
    ]

    if parallel:
        outputs = _run_parallel(agent, world, jobs, quotas, writer_attempts)
    else:
        outputs = [
            _generate_writer_output(
                agent,
                world,
                role,
                prompt,
                categories,
                quotas,
                attempts=writer_attempts,
            )
            for role, categories, prompt in jobs
        ]

    bundles: list[ScenarioBundle] = []
    memory_number = 1
    for (_, categories, _), output in zip(jobs, outputs):
        normalized = _normalize_writer_output(
            output,
            world,
            categories,
            quotas,
            memory_start=memory_number,
        )
        bundles.extend(normalized)
        memory_number += sum(len(bundle.memories) for bundle in normalized)

    _validate_scenario_collection(bundles, quotas)
    return tuple(bundles)


def _run_parallel(
    agent: JsonAgent,
    world: WorldBible,
    jobs: list[tuple[str, tuple[Category, ...], str]],
    quotas: dict[Category, int],
    attempts: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                _generate_writer_output,
                agent,
                world,
                role,
                prompt,
                categories,
                quotas,
                attempts=attempts,
            ): index
            for index, (role, categories, prompt) in enumerate(jobs)
        }
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    return [output for output in outputs if output is not None]


def _generate_writer_output(
    agent: JsonAgent,
    world: WorldBible,
    role: str,
    prompt: str,
    categories: tuple[Category, ...],
    quotas: dict[Category, int],
    *,
    attempts: int,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("writer_attempts must be at least 1")

    current_prompt = prompt
    last_error: Exception | None = None
    for _ in range(attempts):
        output: dict[str, Any] | None = None
        try:
            output = agent(role, current_prompt)
            _normalize_writer_output(output, world, categories, quotas, memory_start=1)
            return output
        except (KeyError, TypeError, ValueError) as exc:
            last_error = exc
            expected = {category.value: quotas[category] for category in categories}
            current_prompt = (
                prompt
                + "\n\nCORRECTION: The previous JSON failed validation: "
                + str(exc)
                + " Repair it minimally while preserving valid bundles. Return the complete JSON with "
                + f"these exact case counts: {json.dumps(expected, sort_keys=True)}.\n\nPrevious JSON:\n"
                + json.dumps(output or {}, ensure_ascii=False, sort_keys=True)
            )
    raise RuntimeError(f"{role} failed scenario-output validation") from last_error


def _normalize_writer_output(
    output: dict[str, Any],
    world: WorldBible,
    categories: tuple[Category, ...],
    quotas: dict[Category, int],
    *,
    memory_start: int,
) -> tuple[ScenarioBundle, ...]:
    raw_bundles = output["bundles"]
    if not isinstance(raw_bundles, list):
        raise TypeError("bundles must be a list")

    allowed = set(categories)
    raw_counts = Counter(Category(str(item["case"]["category"])) for item in raw_bundles)
    if set(raw_counts) - allowed:
        raise ValueError("writer returned a category outside its assignment")
    mismatches = {
        category.value: {"expected": quotas[category], "actual": raw_counts[category]}
        for category in categories
        if raw_counts[category] != quotas[category]
    }
    if mismatches:
        raise ValueError(f"case count mismatches: {mismatches}")

    facts_by_id = {fact.fact_id: fact for fact in world.facts}
    entity_ids = {entity.entity_id for entity in world.entities}
    superseded_ids = {fact.supersedes_fact_id for fact in world.facts if fact.supersedes_fact_id}
    bundles: list[ScenarioBundle] = []
    next_memory = memory_start

    for raw_bundle in raw_bundles:
        raw_case = raw_bundle["case"]
        raw_memories = raw_bundle["memories"]
        if not isinstance(raw_memories, list):
            raise TypeError("memories must be a list")

        keys = [str(item["memory_key"]) for item in raw_memories]
        if len(keys) != len(set(keys)):
            raise ValueError("memory_key values must be unique within a bundle")
        gold_key = str(raw_case["gold_memory_key"])
        context_keys = [str(value) for value in raw_case["context_memory_keys"]]
        referenced_keys = {gold_key, *context_keys}
        if referenced_keys != set(keys):
            raise ValueError("bundle memories must exactly match the gold and context keys")

        key_to_id = {
            key: f"mem-{next_memory + index:04d}"
            for index, key in enumerate(keys)
        }
        next_memory += len(keys)
        memories: list[BenchmarkMemory] = []
        for raw_memory in raw_memories:
            fact_ids = tuple(str(value) for value in raw_memory["fact_ids"])
            missing_facts = set(fact_ids) - set(facts_by_id)
            if missing_facts:
                raise ValueError(f"memory references missing facts: {sorted(missing_facts)}")
            memory_entity_ids = tuple(str(value) for value in raw_memory["entity_ids"])
            missing_entities = set(memory_entity_ids) - entity_ids
            if missing_entities:
                raise ValueError(
                    f"memory references missing entities: {sorted(missing_entities)}"
                )
            category = Category(str(raw_memory["category"]))
            if any(facts_by_id[fact_id].category != category for fact_id in fact_ids):
                raise ValueError("all facts in a memory must match its category")

            memory = BenchmarkMemory(
                memory_id=key_to_id[str(raw_memory["memory_key"])],
                category=category,
                description=str(raw_memory["description"]),
                contents=str(raw_memory["contents"]),
                fact_ids=fact_ids,
                entity_ids=memory_entity_ids,
            )
            if len(memory.description) > 120:
                raise ValueError("candidate memory descriptions must be at most 120 characters")
            if not memory.contents.strip():
                raise ValueError("candidate memories must put supporting detail in contents")
            _reject_raw_ids(memory.description, "memory description")
            _reject_raw_ids(memory.contents, "memory contents")
            memories.append(memory)

        case = ScenarioCase(
            case_id=str(raw_case["case_id"]),
            category=Category(str(raw_case["category"])),
            situation=str(raw_case["situation"]),
            query=str(raw_case["query"]),
            gold_memory_id=key_to_id[gold_key],
            context_memory_ids=tuple(key_to_id[key] for key in context_keys),
            answer_should=str(raw_case["answer_should"]),
            why_gold=str(raw_case["why_gold"]),
            secondary_tags=tuple(str(value) for value in raw_case.get("secondary_tags", [])),
        )
        if case.category not in allowed:
            raise ValueError("case category is outside the writer assignment")
        for field_name, value in (
            ("situation", case.situation),
            ("query", case.query),
            ("answer_should", case.answer_should),
            ("why_gold", case.why_gold),
        ):
            _reject_raw_ids(value, field_name)

        memories_by_id = {memory.memory_id: memory for memory in memories}
        gold = memories_by_id[case.gold_memory_id]
        if any(fact_id in superseded_ids for fact_id in gold.fact_ids):
            raise ValueError("a superseded fact cannot be used as gold")
        gold_fact_ids = set(gold.fact_ids)
        context_fact_ids = {
            fact_id
            for memory_id in case.context_memory_ids
            for fact_id in memories_by_id[memory_id].fact_ids
        }
        if gold_fact_ids & context_fact_ids:
            raise ValueError("gold and context memories cannot repeat a canonical fact")

        used_fact_ids = {fact_id for memory in memories for fact_id in memory.fact_ids}
        bundle = ScenarioBundle(
            case=case,
            facts=tuple(fact for fact in world.facts if fact.fact_id in used_fact_ids),
            memories=tuple(memories),
        )
        validate_bundle(bundle)
        bundles.append(bundle)

    return tuple(bundles)


def _reject_raw_ids(value: str, field_name: str) -> None:
    if _RAW_ID.search(value):
        raise ValueError(f"{field_name} exposes a raw world id")


def _validate_scenario_collection(
    bundles: list[ScenarioBundle],
    quotas: dict[Category, int],
) -> None:
    case_ids = [bundle.case.case_id for bundle in bundles]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("scenario case ids must be globally unique")
    memory_ids = [memory.memory_id for bundle in bundles for memory in bundle.memories]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("candidate memory ids must be globally unique")

    counts = Counter(bundle.case.category for bundle in bundles)
    mismatches = {
        category.value: {"expected": required, "actual": counts[category]}
        for category, required in quotas.items()
        if counts[category] != required
    }
    if mismatches:
        raise ValueError(f"global case count mismatches: {mismatches}")

    seen_gold_facts: set[str] = set()
    for bundle in bundles:
        memories_by_id = {memory.memory_id: memory for memory in bundle.memories}
        gold_facts = set(memories_by_id[bundle.case.gold_memory_id].fact_ids)
        overlap = seen_gold_facts & gold_facts
        if overlap:
            raise ValueError(f"gold facts are reused across cases: {sorted(overlap)}")
        seen_gold_facts.update(gold_facts)


def _load_world(path: Path) -> tuple[dict[str, Any], WorldBible]:
    artifact = json.loads(path.read_text())
    world = world_bible_from_dict(artifact["world"])
    validate_world_bible(world)
    return artifact, world


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--model-endpoint", default=DEFAULT_MODEL_ENDPOINT)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--serial", action="store_true", help="Disable parallel scenario writers")
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    args = _args()
    profile = args.profile or os.getenv("DATABRICKS_CONFIG_PROFILE", "")
    if not profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")

    world_artifact, world = _load_world(args.world)
    from databricks.sdk import WorkspaceClient

    agent = DatabricksJsonAgent(
        workspace_client=WorkspaceClient(profile=profile),
        endpoint=args.model_endpoint,
        system_prompt=SCENARIO_SYSTEM_PROMPT,
    )
    bundles = generate_scenario_drafts(agent, world, parallel=not args.serial)
    memories = [memory for bundle in bundles for memory in bundle.memories]
    artifact = {
        "schema_version": "realistic-search-scenario-drafts-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_endpoint": args.model_endpoint,
        "source_world": {
            "path": str(args.world),
            "schema_version": world_artifact.get("schema_version"),
            "world_id": world.world_id,
        },
        "summary": {
            "cases": len(bundles),
            "candidate_memories": len(memories),
            "case_counts": {
                category.value: sum(bundle.case.category == category for bundle in bundles)
                for category in Category
            },
        },
        "bundles": [asdict(bundle) for bundle in bundles],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **artifact["summary"]}))


if __name__ == "__main__":
    main()

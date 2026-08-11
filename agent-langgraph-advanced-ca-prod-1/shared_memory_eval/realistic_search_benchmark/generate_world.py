"""Generate the coherent pilot world and canonical fact ledger.

This stage writes a local JSON artifact only. It never calls the managed-memory APIs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from .schema import (
    Category,
    EntityKind,
    PILOT_FACT_QUOTAS,
    WorldBible,
    validate_world_bible,
    world_bible_from_dict,
)


DEFAULT_MODEL_ENDPOINT = "databricks-gpt-5-2"
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts") / "pilot_world.json"

CATEGORY_GROUPS: tuple[tuple[Category, ...], ...] = (
    (Category.PREFERENCE, Category.PEOPLE),
    (Category.PROJECT, Category.DECISION),
    (Category.WORKFLOW, Category.GLOBAL_FACT),
    (Category.PLAN, Category.HOUSEHOLD_LOGISTICS),
)

JsonAgent = Callable[[str, str], dict[str, Any]]


ARCHITECT_SYSTEM_PROMPT = """You are the world architect for a realistic long-term-memory benchmark.
Create one coherent synthetic person and the organization and household around them. The world must
feel ordinary and internally consistent, with recurring people, overlapping projects, stable habits,
and a 2026 timeline. Do not write search tests, questions, memories, or adversarial distractors.

Use only fictional people and organizations. Avoid secrets, credentials, medical diagnoses, financial
account data, and sensational events. Mild accessibility needs and dietary constraints are allowed when
they are relevant to normal assistant personalization.

Return exactly one JSON object and no markdown or commentary."""


FACT_BUILDER_SYSTEM_PROMPT = """You are a canonical-fact writer for a realistic long-term-memory
benchmark. Work only inside the supplied synthetic world. Produce durable facts that a personal or work
assistant could plausibly remember and use weeks or months later. Facts must be true in the world, not
search traps. Reuse entities across categories so normal contextual overlap emerges naturally.

Do not write questions, memory descriptions, memory contents, evaluations, or explanations outside the
requested JSON. Return exactly one JSON object and no markdown."""


def architect_prompt() -> str:
    categories = ", ".join(category.value for category in Category)
    kinds = ", ".join(kind.value for kind in EntityKind)
    return f"""Design the pilot benchmark world.

Requirements:
- world_id must be `realistic-search-pilot`.
- The owner is one fictional working adult who uses an assistant for both personal and work context.
- Include 18-24 entities across these allowed kinds: {kinds}.
- Include the owner, 5-7 recurring people, one fictional organization, 2-4 teams, 3-5 projects or
  products, 2-4 places, and a small number of household or tool entities.
- If the household includes an animal, use kind `pet`; never classify an animal as a person.
- Names should be memorable but natural. Avoid famous names and real company names.
- The premise should explain the owner's role, household, work, and why these facts coexist.
- Create exactly one category brief for every category: {categories}.
- Each brief should describe realistic areas of life from which facts can be generated and list the
  relevant entity ids. Category briefs may overlap in their entity ids.
- Do not generate facts yet.

JSON shape:
{{
  "world_id": "realistic-search-pilot",
  "title": "...",
  "premise": "...",
  "owner_entity_id": "person-...",
  "entities": [
    {{
      "entity_id": "person-...",
      "kind": "person|organization|team|project|product|place|household|tool",
      "name": "...",
      "aliases": ["..."],
      "summary": "..."
    }}
  ],
  "category_briefs": [
    {{
      "category": "preference|project|workflow|decision|people|plan|household_logistics|global_fact",
      "brief": "...",
      "focus_entity_ids": ["..."]
    }}
  ],
  "facts": []
}}"""


def fact_builder_prompt(
    architecture: dict[str, Any],
    categories: tuple[Category, ...],
    fact_quotas: dict[Category, int],
) -> str:
    requested = {category.value: fact_quotas[category] for category in categories}
    category_text = ", ".join(category.value for category in categories)
    return f"""Generate canonical facts for these categories only: {category_text}.

Exact fact counts: {json.dumps(requested, sort_keys=True)}

Rules:
- Every fact must reference existing entity ids from the architecture.
- fact_id must be a unique lowercase hyphenated semantic id beginning with `fact-<category>-`.
  Replace underscores in category names with hyphens.
- Statements must be specific, standalone truths, normally one or two sentences.
- Statements must use natural entity names, never raw entity ids; ids belong only in entity_ids.
- Facts should be varied: stable preferences, concrete dates, roles, recurring procedures, current
  project state, constraints, ownership, and organization conventions as appropriate to the category.
- Keep category boundaries meaningful. For example, put a person's role or relationship in people,
  an individual preference in preference, and only organization-wide conventions in global_fact.
- Do not restate the same underlying truth in both assigned categories.
- Include a few intentional historical changes where a newer fact supersedes an older fact. Both facts
  must appear in this response, share the same category, and use valid_from/valid_to dates.
- Use aliases only when a natural nickname, abbreviation, or alternate phrasing exists.
- At least one third of the facts should relate to entities used by another assigned category.
- Do not make facts unnaturally similar merely to confuse retrieval.
- Dates, when present, must be ISO `YYYY-MM-DD` strings in 2026.

Architecture:
{json.dumps(architecture, ensure_ascii=False, sort_keys=True)}

Return:
{{
  "facts": [
    {{
      "fact_id": "fact-project-example",
      "category": "{categories[0].value}",
      "statement": "...",
      "entity_ids": ["..."],
      "aliases": ["..."],
      "valid_from": null,
      "valid_to": null,
      "supersedes_fact_id": null
    }}
  ]
}}"""


def generate_world_bible(
    agent: JsonAgent,
    *,
    fact_quotas: dict[Category, int] | None = None,
    parallel: bool = True,
    builder_attempts: int = 3,
) -> WorldBible:
    """Run one architect and four fact-builder roles, then validate the merged world."""

    quotas = dict(fact_quotas or PILOT_FACT_QUOTAS)
    architecture = agent("world-architect", architect_prompt())
    architecture["facts"] = []
    architecture_world = world_bible_from_dict(architecture)
    validate_world_bible(architecture_world)

    jobs = [
        (
            f"fact-builder-{index + 1}",
            group,
            fact_builder_prompt(architecture, group, quotas),
        )
        for index, group in enumerate(CATEGORY_GROUPS)
    ]

    if parallel:
        fact_outputs = _run_parallel(
            agent,
            jobs,
            fact_quotas=quotas,
            attempts=builder_attempts,
        )
    else:
        fact_outputs = [
            _generate_fact_output(
                agent,
                role,
                prompt,
                categories,
                quotas,
                attempts=builder_attempts,
            )
            for role, categories, prompt in jobs
        ]

    facts: list[dict[str, Any]] = []
    for (_, categories, _), output in zip(jobs, fact_outputs):
        generated = list(output.get("facts", []))
        allowed = set(categories)
        unexpected = {
            str(item.get("category"))
            for item in generated
            if Category(str(item.get("category"))) not in allowed
        }
        if unexpected:
            raise ValueError(f"Fact builder returned unexpected categories: {sorted(unexpected)}")
        facts.extend(generated)

    merged = {**architecture, "facts": facts}
    world = world_bible_from_dict(merged)
    validate_world_bible(world, fact_quotas=quotas)
    return world


def _run_parallel(
    agent: JsonAgent,
    jobs: list[tuple[str, tuple[Category, ...], str]],
    *,
    fact_quotas: dict[Category, int],
    attempts: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                _generate_fact_output,
                agent,
                role,
                prompt,
                categories,
                fact_quotas,
                attempts=attempts,
            ): index
            for index, (role, categories, prompt) in enumerate(jobs)
        }
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    return [output for output in outputs if output is not None]


def _generate_fact_output(
    agent: JsonAgent,
    role: str,
    prompt: str,
    categories: tuple[Category, ...],
    fact_quotas: dict[Category, int],
    *,
    attempts: int,
) -> dict[str, Any]:
    """Retry one builder without discarding valid outputs from the other builders."""

    if attempts < 1:
        raise ValueError("builder_attempts must be at least 1")

    current_prompt = prompt
    last_error: Exception | None = None
    for _ in range(attempts):
        output: dict[str, Any] | None = None
        try:
            output = agent(role, current_prompt)
            _validate_fact_output(output, categories, fact_quotas)
            return output
        except (KeyError, TypeError, ValueError) as exc:
            last_error = exc
            expected = {
                category.value: fact_quotas[category]
                for category in categories
            }
            current_prompt = (
                prompt
                + "\n\nCORRECTION: The previous output violated this contract: "
                + str(exc)
                + " Repair the previous JSON minimally: preserve its valid facts, add or remove only "
                + "what is necessary, and keep every supersedes_fact_id reference valid. Return the "
                + "complete corrected JSON object with exactly these fact counts and no other "
                + f"categories: {json.dumps(expected, sort_keys=True)}.\n\nPrevious JSON:\n"
                + json.dumps(output or {}, ensure_ascii=False, sort_keys=True)
            )
    raise RuntimeError(f"{role} failed exact fact-output validation") from last_error


def _validate_fact_output(
    output: dict[str, Any],
    categories: tuple[Category, ...],
    fact_quotas: dict[Category, int],
) -> None:
    facts = output["facts"]
    if not isinstance(facts, list):
        raise TypeError("facts must be a list")

    allowed = set(categories)
    counts = {category: 0 for category in categories}
    facts_by_id: dict[str, dict[str, Any]] = {}
    for fact in facts:
        category = Category(str(fact["category"]))
        if category not in allowed:
            raise ValueError(f"unexpected fact category: {category.value}")
        counts[category] += 1
        fact_id = str(fact["fact_id"])
        if fact_id in facts_by_id:
            raise ValueError(f"duplicate fact_id: {fact_id}")
        facts_by_id[fact_id] = fact

    mismatches = {
        category.value: {
            "expected": fact_quotas[category],
            "actual": counts[category],
        }
        for category in categories
        if counts[category] != fact_quotas[category]
    }
    if mismatches:
        raise ValueError(f"fact count mismatches: {mismatches}")

    for fact in facts:
        prior_id = fact.get("supersedes_fact_id")
        if not prior_id:
            continue
        if prior_id not in facts_by_id:
            raise ValueError(f"missing superseded fact: {prior_id}")
        prior = facts_by_id[prior_id]
        if prior.get("category") != fact.get("category"):
            raise ValueError("superseded fact must be in the same category")
        if not prior.get("valid_to"):
            raise ValueError(f"superseded fact {prior_id} must have valid_to")
        if not fact.get("valid_from"):
            raise ValueError(f"superseding fact {fact['fact_id']} must have valid_from")
        if str(prior["valid_to"]) > str(fact["valid_from"]):
            raise ValueError(
                f"superseded fact {prior_id} ends after {fact['fact_id']} begins"
            )


class DatabricksJsonAgent:
    """Thin JSON-only model caller; one fresh client per call is safe for parallel workers."""

    def __init__(
        self,
        *,
        workspace_client,
        endpoint: str,
        attempts: int = 3,
        system_prompt: str | None = None,
    ) -> None:
        self.workspace_client = workspace_client
        self.endpoint = endpoint
        self.attempts = attempts
        self.system_prompt = system_prompt

    def __call__(self, role: str, prompt: str) -> dict[str, Any]:
        from databricks_langchain import ChatDatabricks
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = self.system_prompt or (
            ARCHITECT_SYSTEM_PROMPT
            if role == "world-architect"
            else FACT_BUILDER_SYSTEM_PROMPT
        )
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            model = ChatDatabricks(
                endpoint=self.endpoint,
                workspace_client=self.workspace_client,
                temperature=0.7,
                max_tokens=12000,
            )
            repair = ""
            if attempt > 1:
                repair = (
                    "\n\nThe previous response was invalid. Return one complete JSON object only, "
                    "with every required field and no markdown."
                )
            try:
                response = model.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt + repair),
                    ]
                )
                return _parse_json_object(_message_text(response.content))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"{role} failed after {self.attempts} attempts") from last_error


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    raise ValueError(f"Model returned unsupported content: {type(content).__name__}")


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be one JSON object")
    return parsed


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--model-endpoint", default=DEFAULT_MODEL_ENDPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--serial", action="store_true", help="Disable parallel fact builders")
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    args = _args()
    profile = args.profile or os.getenv("DATABRICKS_CONFIG_PROFILE", "")
    if not profile:
        raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=profile)
    agent = DatabricksJsonAgent(
        workspace_client=client,
        endpoint=args.model_endpoint,
    )
    world = generate_world_bible(agent, parallel=not args.serial)
    artifact = {
        "schema_version": "realistic-search-world-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_endpoint": args.model_endpoint,
        "world": asdict(world),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "entities": len(world.entities),
                "facts": len(world.facts),
                "fact_counts": {
                    category.value: sum(fact.category == category for fact in world.facts)
                    for category in Category
                },
            }
        )
    )


if __name__ == "__main__":
    main()

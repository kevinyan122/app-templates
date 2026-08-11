"""Create and apply realistic content-length profiles to the curated pilot corpus.

Generation writes a reusable rewrite artifact. Applying that artifact is deterministic and never calls
the memory store or MLflow.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from dotenv import load_dotenv

from .generate_world import DEFAULT_MODEL_ENDPOINT, DatabricksJsonAgent


DEFAULT_CURATED = Path(__file__).with_name("artifacts") / "pilot_curated.json"
DEFAULT_WORLD = Path(__file__).with_name("artifacts") / "pilot_world.json"
DEFAULT_REWRITES = Path(__file__).with_name("artifacts") / "pilot_memory_rewrites.json"
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts") / "pilot_enriched.json"

PROFILE_SHORT = "short"
PROFILE_MEDIUM = "medium"
PROFILE_EPISODIC = "episodic"
PROFILE_QUOTAS = {PROFILE_SHORT: 30, PROFILE_MEDIUM: 20, PROFILE_EPISODIC: 10}
WORD_RANGES = {
    PROFILE_SHORT: (0, 0),
    PROFILE_MEDIUM: (35, 100),
    PROFILE_EPISODIC: (70, 180),
}
PILOT_AS_OF_DATE = "2026-06-05"

EPISODIC_MEMORY_IDS = {
    "mem-0001",
    "mem-0002",
    "mem-0003",
    "mem-0004",
    "mem-0015",
    "mem-0027",
    "mem-0042",
    "mem-0048",
    "mem-0053",
    "mem-0060",
}
SHORT_MEMORY_IDS = {
    "mem-0007",
    "mem-0008",
    "mem-0009",
    "mem-0010",
    "mem-0011",
    "mem-0012",
    "mem-0014",
    "mem-0017",
    "mem-0018",
    "mem-0020",
    "mem-0021",
    "mem-0023",
    "mem-0025",
    "mem-0026",
    "mem-0028",
    "mem-0030",
    "mem-0031",
    "mem-0032",
    "mem-0051",
    "mem-0052",
    "mem-0056",
    "mem-0057",
    "mem-0058",
    "mem-0059",
    "mem-0019",
    "mem-0033",
    "mem-0038",
    "mem-0039",
    "mem-0044",
    "mem-0050",
}
ALL_MEMORY_IDS = {f"mem-{number:04d}" for number in range(1, 61)}
MEDIUM_MEMORY_IDS = ALL_MEMORY_IDS - SHORT_MEMORY_IDS - EPISODIC_MEMORY_IDS

SHORT_DESCRIPTION_OVERRIDES = {
    "mem-0007": (
        "Brightgrove uses Aurora Calendar for scheduling and LanternDesk for operational and IT requests"
    ),
    "mem-0012": "The shared grocery list is pinned in Pine Notes for Mara and Devon",
    "mem-0014": "Poppy’s kibble is in the pantry bin; the reorder note is in Pine Notes",
    "mem-0018": "Eli is a branch supervisor and regular Branch Leads Council representative",
    "mem-0019": "Jordan is Mara’s neighbor, occasional running buddy, and package-pickup fallback",
    "mem-0023": "Poppy’s vet-planning reminder is in Aurora Calendar for September 14, 2026",
    "mem-0025": "Ops Programs team day is June 11 at HQ; Mara keeps it meeting-light",
    "mem-0026": (
        "Summer Reading Ops staffing-guidance draft is due May 8 for Comms & Training review"
    ),
    "mem-0031": "Mara wants summary and next steps first; details only when requested",
    "mem-0032": "Mara prefers async coordination with IT; call Owen only when blocked",
    "mem-0033": "Coverage Pilot expands to HQ teams after Riverbend to test cross-site escalation",
    "mem-0038": "Owen approves IT review for KB Refresh changes touching LanternDesk",
    "mem-0039": "Operations Programs owns KB Refresh; Mara coordinates scope and milestones",
    "mem-0044": (
        "Nina approves network-wide messages before Sasha sends them through Comms & Training"
    ),
    "mem-0050": "Mara’s Pine Notes meeting notes end with an action list labeled ‘Next steps’",
    "mem-0052": "Mara prefers Hawthorn Coffee over HQ for change-of-scene work blocks",
    "mem-0056": "Ops Programs owns cross-branch consistency and coordinates rollout support",
    "mem-0057": "Riverbend is a frequent Brightgrove pilot site for frontline testing",
    "mem-0058": "Coverage Pilot kickoff is March 19 at HQ; Mara leads and Nina handles approvals",
    "mem-0059": (
        "Mara’s April 23 Riverbend visit with Eli is for Coverage Pilot frontline feedback"
    ),
}

REWRITE_GROUPS = (
    (
        "mem-0001",
        "mem-0002",
        "mem-0003",
        "mem-0004",
        "mem-0005",
        "mem-0006",
        "mem-0053",
        "mem-0054",
        "mem-0055",
    ),
    (
        "mem-0013",
        "mem-0015",
        "mem-0016",
        "mem-0022",
        "mem-0024",
        "mem-0027",
        "mem-0029",
        "mem-0060",
    ),
    (
        "mem-0034",
        "mem-0035",
        "mem-0036",
        "mem-0037",
        "mem-0040",
        "mem-0041",
        "mem-0042",
    ),
    (
        "mem-0043",
        "mem-0045",
        "mem-0046",
        "mem-0047",
        "mem-0048",
        "mem-0049",
    ),
)

EPISODE_SUPPORT_FACT_IDS = {
    "mem-0001": (
        "fact-decision-coverage-pilot-escalation-owner-it-services",
        "fact-workflow-coverage-pilot-weekly-checkin-with-owen-eli",
        "fact-global-fact-lanterndesk-ownership-it-services",
    ),
    "mem-0002": (
        "fact-decision-coverage-pilot-feedback-channel-branch-leads-council",
        "fact-project-coverage-pilot-weekly-check-in-branch-leads-council",
        "fact-people-eli-branch-leads-rep",
    ),
    "mem-0003": (
        "fact-decision-kb-refresh-freeze-window-april",
        "fact-plan-kb-refresh-freeze-window",
        "fact-project-kb-refresh-draft-structure-in-pine-notes",
    ),
    "mem-0004": (
        "fact-decision-summer-reading-ops-training-format-hybrid",
        "fact-decision-summer-reading-ops-training-format-in-person",
        "fact-global-fact-comms-training-rollout-partner",
        "fact-global-fact-hq-primary-in-person-site",
    ),
    "mem-0015": (
        "fact-household-logistics-poppy-midday-walk-new",
        "fact-household-logistics-poppy-midday-walk-old",
        "fact-workflow-household-weekly-planning-calendar-sync",
    ),
    "mem-0027": (
        "fact-preference-household-poppy-walks-morning-new",
        "fact-preference-household-poppy-walks-evening-old",
        "fact-workflow-household-weekly-planning-calendar-sync",
    ),
    "mem-0042": (
        "fact-project-summer-reading-ops-kickoff-at-hq",
        "fact-project-summer-reading-ops-comms-training-partnership",
        "fact-global-fact-comms-training-rollout-partner",
    ),
    "mem-0048": (
        "fact-workflow-kb-refresh-intake-via-pine-notes-new",
        "fact-workflow-kb-refresh-intake-via-lanterndesk-old",
        "fact-project-kb-refresh-draft-structure-in-pine-notes",
        "fact-global-fact-pine-notes-internal-drafts",
    ),
    "mem-0053": (
        "fact-decision-coverage-pilot-site-selection-riverbend",
        "fact-project-coverage-pilot-primary-site-riverbend",
        "fact-project-coverage-pilot-weekly-check-in-branch-leads-council",
        "fact-plan-riverbend-pilot-site-visit",
    ),
    "mem-0060": (
        "fact-plan-hawthorn-weekend-planning-block-new",
        "fact-plan-hawthorn-weekend-planning-block-old",
        "fact-workflow-household-weekly-planning-calendar-sync",
        "fact-preference-mara-hawthorn-work-blocks",
    ),
}

REWRITE_SYSTEM_PROMPT = """You write realistic entries for a managed long-term-memory system.
Every entry has a one-line description and fuller contents. Use only the supplied canonical facts:
never invent a date, person, place, quote, cause, outcome, or commitment. Natural connective prose is
allowed, but every factual claim must be traceable to the supplied facts.

Medium entries are compact contextual notes, not padded restatements. Episodic entries are readable
records of one decision, change, kickoff, or handoff, combining the supplied facts into a chronological
account. They must remain useful durable memories rather than fictional scenes. Return exactly one JSON
object and no markdown or commentary."""

_RAW_ID = re.compile(
    r"\b(?:person|pet|team|tool|place|project|product|organization|household)-[a-z0-9-]+"
)

JsonAgent = Callable[[str, str], dict[str, Any]]


def profile_for(memory_id: str) -> str:
    if memory_id in SHORT_MEMORY_IDS:
        return PROFILE_SHORT
    if memory_id in EPISODIC_MEMORY_IDS:
        return PROFILE_EPISODIC
    if memory_id in MEDIUM_MEMORY_IDS:
        return PROFILE_MEDIUM
    raise ValueError(f"Unknown pilot memory id: {memory_id}")


def support_fact_ids(memory: dict[str, Any]) -> tuple[str, ...]:
    if memory["memory_id"] in EPISODE_SUPPORT_FACT_IDS:
        return EPISODE_SUPPORT_FACT_IDS[memory["memory_id"]]
    return tuple(memory["fact_ids"])


def rewrite_prompt(
    memory_ids: tuple[str, ...],
    corpus_by_id: dict[str, dict[str, Any]],
    facts_by_id: dict[str, dict[str, Any]],
    gold_cases_by_memory_id: dict[str, dict[str, Any]],
) -> str:
    targets = []
    for memory_id in memory_ids:
        memory = corpus_by_id[memory_id]
        fact_ids = support_fact_ids(memory)
        gold_case = gold_cases_by_memory_id.get(memory_id)
        targets.append(
            {
                "memory_id": memory_id,
                "profile": profile_for(memory_id),
                "current_memory": {
                    "category": memory["category"],
                    "description": memory["description"],
                    "contents": memory["contents"],
                },
                "supported_facts": [
                    {
                        key: facts_by_id[fact_id].get(key)
                        for key in (
                            "statement",
                            "aliases",
                            "valid_from",
                            "valid_to",
                        )
                    }
                    for fact_id in fact_ids
                ],
                "gold_case": (
                    {
                        "query": gold_case["query"],
                        "answer_should": gold_case["answer_should"],
                    }
                    if gold_case
                    else None
                ),
            }
        )
    return f"""Rewrite these {len(targets)} memory entries.

Requirements:
- Preserve memory_id and the assigned profile exactly.
- description: one specific summary line, 4-18 words, no raw ids, no newline.
- contents for medium: 35-100 words in 2-4 useful sentences.
- contents for episodic: 70-180 words in 4-8 sentences, written as a durable account of the supplied
  one-time decision, change, kickoff, or handoff. Do not add scene-setting details that are not supplied.
- The contents must add useful detail beyond the description and must not repeat the description verbatim.
- For entries with gold_case, the contents must fully support answer_should. For episodic gold entries,
  the description should identify the topic but should not by itself reveal the complete expected answer.
- Do not mention benchmarks, tests, gold answers, profiles, canonical facts, ids, or retrieval.
- Do not add markdown headings or bullet syntax inside contents.

Targets:
{json.dumps(targets, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "rewrites": [
    {{
      "memory_id": "mem-0001",
      "profile": "episodic",
      "description": "One-line summary",
      "contents": "The complete memory contents."
    }}
  ]
}}

Return exactly one rewrite for every target memory_id and no others."""


def generate_rewrites(
    agent: JsonAgent,
    curated: dict[str, Any],
    world: dict[str, Any],
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    corpus_by_id = {memory["memory_id"]: memory for memory in curated["corpus"]}
    facts_by_id = {fact["fact_id"]: fact for fact in world["world"]["facts"]}
    gold_cases = {case["gold_memory_id"]: case for case in curated["cases"]}
    outputs: dict[int, list[dict[str, Any]]] = {}

    def run_group(index: int, memory_ids: tuple[str, ...]) -> tuple[int, list[dict[str, Any]]]:
        base_prompt = rewrite_prompt(memory_ids, corpus_by_id, facts_by_id, gold_cases)
        prompt = base_prompt
        last_error: Exception | None = None
        for _ in range(attempts):
            output: dict[str, Any] | None = None
            try:
                output = agent(f"memory-rewriter-{index + 1}", prompt)
                rewrites = normalize_rewrites(output, set(memory_ids))
                return index, rewrites
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                prompt = (
                    base_prompt
                    + "\n\nCORRECTION: The prior JSON failed validation: "
                    + str(exc)
                    + " Repair every invalid entry and return the complete group. Previous JSON:\n"
                    + json.dumps(output or {}, ensure_ascii=False, sort_keys=True)
                )
        raise RuntimeError(f"memory rewrite group {index + 1} failed validation") from last_error

    with ThreadPoolExecutor(max_workers=len(REWRITE_GROUPS)) as executor:
        futures = {
            executor.submit(run_group, index, memory_ids): index
            for index, memory_ids in enumerate(REWRITE_GROUPS)
        }
        for future in as_completed(futures):
            index, rewrites = future.result()
            outputs[index] = rewrites

    rewrites = [rewrite for index in range(len(REWRITE_GROUPS)) for rewrite in outputs[index]]
    validate_rewrite_collection(rewrites, curated)
    return {
        "schema_version": "realistic-search-memory-rewrites-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_endpoint": DEFAULT_MODEL_ENDPOINT,
        "source_curated": str(DEFAULT_CURATED),
        "profile_quotas": PROFILE_QUOTAS,
        "rewrites": rewrites,
    }


def normalize_rewrites(
    output: dict[str, Any], expected_ids: set[str]
) -> list[dict[str, Any]]:
    raw_rewrites = output["rewrites"]
    if not isinstance(raw_rewrites, list):
        raise TypeError("rewrites must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in raw_rewrites:
        memory_id = str(raw["memory_id"])
        if memory_id in indexed:
            raise ValueError(f"duplicate rewrite for {memory_id}")
        profile = str(raw["profile"])
        if profile != profile_for(memory_id):
            raise ValueError(f"wrong profile for {memory_id}: {profile}")
        description = str(raw["description"]).strip()
        contents = str(raw["contents"]).strip()
        _validate_rewrite_text(memory_id, profile, description, contents)
        indexed[memory_id] = {
            "memory_id": memory_id,
            "profile": profile,
            "description": description,
            "contents": contents,
        }
    actual_ids = set(indexed)
    if actual_ids != expected_ids:
        raise ValueError(
            f"rewrite ids mismatch; missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    return [indexed[memory_id] for memory_id in sorted(indexed)]


def _validate_rewrite_text(
    memory_id: str, profile: str, description: str, contents: str
) -> None:
    description_words = _word_count(description)
    contents_words = _word_count(contents)
    minimum, maximum = WORD_RANGES[profile]
    if not 4 <= description_words <= 18:
        raise ValueError(f"{memory_id} description must be 4-18 words, got {description_words}")
    if "\n" in description:
        raise ValueError(f"{memory_id} description must be one line")
    if not minimum <= contents_words <= maximum:
        raise ValueError(
            f"{memory_id} {profile} contents must be {minimum}-{maximum} words, got {contents_words}"
        )
    minimum_sentences = 2 if profile == PROFILE_MEDIUM else 4
    if _sentence_count(contents) < minimum_sentences:
        raise ValueError(
            f"{memory_id} {profile} contents need at least {minimum_sentences} sentences"
        )
    if description.casefold() in contents.casefold():
        raise ValueError(f"{memory_id} contents repeat the description verbatim")
    if _RAW_ID.search(description) or _RAW_ID.search(contents):
        raise ValueError(f"{memory_id} exposes a raw id")


def validate_rewrite_collection(
    rewrites: list[dict[str, Any]], curated: dict[str, Any]
) -> None:
    expected_ids = MEDIUM_MEMORY_IDS | EPISODIC_MEMORY_IDS
    actual_ids = {rewrite["memory_id"] for rewrite in rewrites}
    if actual_ids != expected_ids:
        raise ValueError("rewrite collection must cover every medium and episodic memory")
    if len(actual_ids) != len(rewrites):
        raise ValueError("rewrite memory ids must be unique")
    descriptions = [rewrite["description"].casefold() for rewrite in rewrites]
    short_descriptions = [
        memory["description"].casefold()
        for memory in curated["corpus"]
        if memory["memory_id"] in SHORT_MEMORY_IDS
    ]
    if len(set(descriptions + short_descriptions)) != len(descriptions + short_descriptions):
        raise ValueError("enriched corpus descriptions must be unique")
    counts = Counter(rewrite["profile"] for rewrite in rewrites)
    expected_rewrites = Counter(
        {
            PROFILE_MEDIUM: PROFILE_QUOTAS[PROFILE_MEDIUM],
            PROFILE_EPISODIC: PROFILE_QUOTAS[PROFILE_EPISODIC],
        }
    )
    if counts != expected_rewrites:
        raise ValueError(f"rewrite profile counts are wrong: {counts}")


def apply_rewrites(
    curated: dict[str, Any],
    world: dict[str, Any],
    rewrite_artifact: dict[str, Any],
) -> dict[str, Any]:
    validate_rewrite_collection(rewrite_artifact["rewrites"], curated)
    rewrites_by_id = {
        rewrite["memory_id"]: rewrite for rewrite in rewrite_artifact["rewrites"]
    }
    facts_by_id = {fact["fact_id"]: fact for fact in world["world"]["facts"]}
    original_by_id = {memory["memory_id"]: memory for memory in curated["corpus"]}
    enriched_by_id: dict[str, dict[str, Any]] = {}
    profiles: dict[str, str] = {}

    for memory_id in sorted(original_by_id):
        original = original_by_id[memory_id]
        profile = profile_for(memory_id)
        profiles[memory_id] = profile
        if profile == PROFILE_SHORT:
            enriched = {
                **original,
                "description": SHORT_DESCRIPTION_OVERRIDES.get(
                    memory_id, original["description"]
                ),
                "contents": "",
            }
        else:
            rewrite = rewrites_by_id[memory_id]
            fact_ids = support_fact_ids(original)
            entity_ids = tuple(
                dict.fromkeys(
                    entity_id
                    for fact_id in fact_ids
                    for entity_id in facts_by_id[fact_id]["entity_ids"]
                )
            )
            enriched = {
                **original,
                "description": rewrite["description"],
                "contents": rewrite["contents"],
                "fact_ids": list(fact_ids),
                "entity_ids": list(entity_ids),
            }
        enriched_by_id[memory_id] = enriched

    bundles = []
    all_facts = curated["facts"]
    for bundle in curated["bundles"]:
        memory_ids = [memory["memory_id"] for memory in bundle["memories"]]
        memories = [enriched_by_id[memory_id] for memory_id in memory_ids]
        fact_ids = {fact_id for memory in memories for fact_id in memory["fact_ids"]}
        bundles.append(
            {
                **bundle,
                "facts": [fact for fact in all_facts if fact["fact_id"] in fact_ids],
                "memories": memories,
            }
        )

    provenance = json.loads(json.dumps(curated["curation"]["provenance"]))
    for memory_id, memory in enriched_by_id.items():
        provenance[memory_id]["fact_ids"] = memory["fact_ids"]
        provenance[memory_id]["content_profile"] = profiles[memory_id]

    artifact = {
        **curated,
        "schema_version": "realistic-search-pilot-enriched-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_as_of_date": PILOT_AS_OF_DATE,
        "sources": {
            **curated["sources"],
            "atomic_curated": str(DEFAULT_CURATED),
            "memory_rewrites": str(DEFAULT_REWRITES),
        },
        "summary": {
            **curated["summary"],
            "content_profiles": PROFILE_QUOTAS,
            "episodic_gold_cases": sum(
                profiles[case["gold_memory_id"]] == PROFILE_EPISODIC
                for case in curated["cases"]
            ),
        },
        "corpus": [enriched_by_id[memory_id] for memory_id in sorted(enriched_by_id)],
        "bundles": bundles,
        "curation": {**curated["curation"], "provenance": provenance},
        "enrichment": {
            "profile_by_memory_id": profiles,
            "word_ranges": {
                profile: {"minimum": bounds[0], "maximum": bounds[1]}
                for profile, bounds in WORD_RANGES.items()
            },
            "rewritten_memory_ids": sorted(rewrites_by_id),
        },
    }
    validate_enriched_artifact(artifact)
    return artifact


def validate_enriched_artifact(artifact: dict[str, Any]) -> None:
    corpus = artifact["corpus"]
    if len(corpus) != 60:
        raise ValueError("enriched corpus must contain 60 memories")
    if artifact.get("benchmark_as_of_date") != PILOT_AS_OF_DATE:
        raise ValueError("enriched corpus must declare the fixed pilot as-of date")
    profiles = artifact["enrichment"]["profile_by_memory_id"]
    counts = Counter(profiles.values())
    if counts != Counter(PROFILE_QUOTAS):
        raise ValueError(f"content profile quotas are wrong: {counts}")
    facts_by_id = {fact["fact_id"]: fact for fact in artifact["facts"]}
    descriptions: set[str] = set()
    corpus_by_id = {memory["memory_id"]: memory for memory in corpus}
    for memory in corpus:
        memory_id = memory["memory_id"]
        profile = profiles[memory_id]
        minimum, maximum = WORD_RANGES[profile]
        words = _word_count(memory["contents"])
        if not minimum <= words <= maximum:
            raise ValueError(
                f"{memory_id} {profile} contents must be {minimum}-{maximum} words, got {words}"
            )
        if any(fact_id not in facts_by_id for fact_id in memory["fact_ids"]):
            raise ValueError(f"{memory_id} references a missing fact")
        description = memory["description"].casefold()
        if description in descriptions:
            raise ValueError("enriched descriptions must be unique")
        descriptions.add(description)
    for bundle in artifact["bundles"]:
        case = bundle["case"]
        expected_ids = [case["gold_memory_id"], *case["context_memory_ids"]]
        actual_ids = [memory["memory_id"] for memory in bundle["memories"]]
        if actual_ids != expected_ids:
            raise ValueError(f"bundle memory ordering mismatch for {case['case_id']}")
        if any(memory_id not in corpus_by_id for memory_id in actual_ids):
            raise ValueError(f"bundle references missing memory for {case['case_id']}")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE", ""))
    parser.add_argument("--model-endpoint", default=DEFAULT_MODEL_ENDPOINT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--rewrites", type=Path, default=DEFAULT_REWRITES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    args = _args()
    curated = json.loads(args.curated.read_text())
    world = json.loads(args.world.read_text())

    if args.generate:
        profile = args.profile or os.getenv("DATABRICKS_CONFIG_PROFILE", "")
        if not profile:
            raise ValueError("Set DATABRICKS_CONFIG_PROFILE or pass --profile")
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient(profile=profile)
        agent = DatabricksJsonAgent(
            workspace_client=client,
            endpoint=args.model_endpoint,
            system_prompt=REWRITE_SYSTEM_PROMPT,
        )
        rewrite_artifact = generate_rewrites(agent, curated, world)
        rewrite_artifact["model_endpoint"] = args.model_endpoint
        rewrite_artifact["source_curated"] = str(args.curated)
        args.rewrites.parent.mkdir(parents=True, exist_ok=True)
        args.rewrites.write_text(
            json.dumps(rewrite_artifact, ensure_ascii=False, indent=2) + "\n"
        )
    else:
        rewrite_artifact = json.loads(args.rewrites.read_text())

    enriched = apply_rewrites(curated, world, rewrite_artifact)
    enriched["sources"]["atomic_curated"] = str(args.curated)
    enriched["sources"]["memory_rewrites"] = str(args.rewrites)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rewrites": str(args.rewrites),
                **enriched["summary"],
            }
        )
    )


if __name__ == "__main__":
    main()

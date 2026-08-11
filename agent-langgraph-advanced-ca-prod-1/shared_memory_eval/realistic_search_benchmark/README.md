# Realistic shared-corpus memory-search benchmark

## Objective

Build a frozen, believable long-term-memory corpus and natural user questions that measure whether
search retrieves the right memory. Categories represent real kinds of memory, not artificial search
attacks. Retrieval difficulty should emerge from normal overlap among projects, people, preferences,
plans, decisions, and workflows.

This benchmark complements the existing behavior suite. Saving, updating, deletion, and autonomous
search routing remain separate tests.

## Shape

| Release | Memories | Positive cases | Purpose |
|---|---:|---:|---|
| Pilot | 60 | 25 | Calibrate generation and review before spending on scale |
| V1 | 300 | 120 | Stable benchmark for comparing search techniques |

One corpus represents one coherent synthetic person and the organization around them. It includes
personal preferences, active work, recurring collaborators, household logistics, future plans, and
shared organizational facts. Memories use opaque paths such as `/memories/benchmark/mem-0042.md` so
the path cannot leak the answer.

The released corpus is seeded once into a versioned direct eval scope. Evaluation runs validate its
content fingerprint before and after and never mutate or reseed it.

### Content profiles

Every stored memory uses the production `description` and `contents` shape. The description is a
specific one-line summary. Atomic memories need no duplicate contents; medium and episodic contents
carry the detailed record without repeating the summary. Content profiles are generation and analysis
metadata only and are not added to searchable memory text.
The pilot is interpreted as of `2026-06-05`, so relative and superseded facts have one fixed reference
date during generation, critic review, and human review.

| Profile | Pilot | Contents length | Purpose |
|---|---:|---:|---|
| Short | 30 | Empty | Atomic facts fully captured by the description |
| Medium | 20 | 35–100 words | Compact contextual records with useful supporting detail |
| Episodic | 10 | 70–180 words | One-time decisions, changes, kickoffs, and handoffs |

Seven pilot questions use an episodic gold memory. For those cases, the description identifies the
topic without supplying the complete expected answer; the decisive details must be read from contents.

## Category quotas

| Category | Pilot | V1 | Examples |
|---|---:|---:|---|
| Preference | 4 | 15 | Food, travel, communication, tools, scheduling |
| Project | 5 | 20 | Goals, milestones, status, constraints, ownership |
| Workflow | 4 | 15 | How recurring work should be performed |
| Decision | 3 | 15 | What was chosen and why |
| People | 3 | 15 | Roles, relationships, responsibilities, aliases |
| Plan | 2 | 15 | Appointments, deadlines, future commitments |
| Household/logistics | 2 | 10 | Repairs, deliveries, purchases, travel details |
| Global fact | 2 | 15 | Team conventions, product facts, company policies |

Search characteristics such as paraphrase, similar entity, crowded topic, or temporal conflict are
secondary tags. They are useful for analysis but must not drive agents to invent benchmark-shaped
language.

## Generation principle

Create a believable situation first. Derive the facts, memories, future question, contextual
memories, and answer requirement from that situation. Do not begin by trying to make search fail.

Each scenario bundle contains:

- one natural future query;
- one gold memory;
- two to six nearby memories that are true and plausible in the same world;
- a brief `answer_should` rubric stating the facts a correct answer must contain;
- a short internal `why_gold` explanation;
- the canonical facts from which those memories were written.

Context memories should be related facts about the same project, another similar project, another
person, a prior plan, or a nearby workflow. They must not contain false statements just to trick the
retriever. A gold memory from one case may be context for another case.

The pilot uses exactly one gold memory per question. Multi-memory and no-answer cases can be added
after the single-gold contract is calibrated.

## Generation roles

The orchestration code is deterministic; logical LLM roles are reused in batches:

1. **World architect** — creates the person, organization, entity registry, and timeline.
2. **Three category generators** — generate situations and canonical facts across assigned categories.
3. **Context-memory writer** — adds two to six realistic neighboring memories per situation.
4. **Query writer** — writes the query, `answer_should`, and `why_gold` without copying final memory text.
5. **Content-profile writers** — turn selected atomic drafts into medium and episodic records using only
   approved canonical facts.
6. **Two independent critics** — review realism, answerability, gold correctness, ambiguity, and category fit.

A final curator is the orchestrator/main agent rather than another free-running worker. It merges
accepted bundles, resolves duplicate facts and memories, enforces quotas, and prepares human review.

Workers never write directly to the memory store. They emit structured draft artifacts. Only the
approved, frozen corpus seeder can write the benchmark scope.

## Pipeline

1. Human approves this specification and category quotas.
2. World architect produces a world bible, entity registry, and canonical fact ledger.
3. Category generators propose situations and gold facts in parallel.
4. Context and query writers turn each situation into a complete scenario bundle.
5. Deterministic validation rejects malformed IDs, missing references, bad paths, duplicate IDs, and
   descriptions that violate the memory contract.
6. Two critics independently review every draft bundle. Any rejection or disagreement enters the queue.
7. Curator merges accepted bundles and runs global contradiction, duplication, and quota checks.
8. Content-profile writers produce the 30 short, 20 medium, and 10 episodic memory mix; deterministic
   validation enforces lengths, references, and production field structure.
9. Two critics review every final case against the complete enriched corpus and canonical facts.
10. A human verifies every final query/gold mapping and double-labels at least 20% for calibration.
11. The pilot corpus and query datasets are published to MLflow and seeded into one immutable scope.
12. Direct retrieval is run first; forced-agent and autonomous-agent tracks remain separate views.

## Hard acceptance gates

A scenario cannot enter the benchmark unless:

- the query sounds like something a person could naturally ask an assistant;
- the gold memory alone supports the required answer;
- no context memory answers the question equally well;
- every memory is a plausible durable memory in this world;
- contextual memories are true, not fabricated negatives;
- the answer does not require facts outside the corpus;
- the gold memory uses a one-line description and puts detailed material in contents;
- IDs and paths are opaque and stable;
- two independent critics accept it;
- a human confirms the final gold mapping.

## Artifacts

`schema.py` defines the first stable contracts:

- `WorldFact`: canonical truth and temporal metadata;
- `BenchmarkMemory`: the exact seedable description and contents;
- `ScenarioCase`: query, gold mapping, context mapping, and answer rubric;
- `ReviewDecision`: one independent structured critique;
- `ScenarioBundle`: the unit passed through generation and review.

The MLflow publication keeps corpus and query rows separate:

- `kevinyan.default.realistic_memory_search_pilot_v2_memories` — 60 frozen memories;
- `kevinyan.default.realistic_memory_search_pilot_v2_queries` — 25 readable questions, with the gold
  memory ID and `answer_should` kept in expectations rather than predictor inputs.

`shared_memory_eval/realistic_search_eval.py` freezes both datasets by fingerprint and defines the
direct-search metrics. The publisher, idempotent seeder, and direct runner are separate commands:

```bash
uv run python -m shared_memory_eval.sync_realistic_search_datasets \
  --profile eng-ml-inference
uv run python -m shared_memory_eval.seed_realistic_search_corpus \
  --profile eng-ml-inference
uv run python -m shared_memory_eval.run_realistic_search_direct \
  --profile eng-ml-inference
```

The dedicated scope is `realistic-search-pilot-v2-6faac0b1`. Seeding is a no-op when all 60 entries
match. A non-empty drifted scope is refused unless the explicit `--replace` repair option is supplied.

Local draft generation uses the configured Databricks profile and never writes to the memory store:

```bash
uv run python -m shared_memory_eval.realistic_search_benchmark.generate_world \
  --profile eng-ml-inference
uv run python -m shared_memory_eval.realistic_search_benchmark.generate_scenarios \
  --profile eng-ml-inference
uv run python -m shared_memory_eval.realistic_search_benchmark.enrich_memories \
  --generate --profile eng-ml-inference
uv run python -m shared_memory_eval.realistic_search_benchmark.review_scenarios \
  --profile eng-ml-inference \
  --drafts shared_memory_eval/realistic_search_benchmark/artifacts/pilot_enriched.json \
  --output shared_memory_eval/realistic_search_benchmark/artifacts/pilot_enriched_reviews.json
```

## Primary evaluation

Direct retrieval uses the natural user query verbatim. Primary deterministic metrics are Hit@1/3/5/10,
MRR, per-case gold rank, and category breakdowns. A secondary forced-agent track can later measure query
formulation plus retrieval.
Answer quality is scored separately using `answer_should`; it does not change the retrieval score.

Development and held-out cases share the same corpus but are split by fact/intent family so paraphrases
of one fact cannot cross the split. Draft generation provenance and critic decisions are retained, but
are never seeded into searchable memory text.

## Current status

Pilot generation and machine curation are complete. `generate_world.py` produced the validated
24-entity, 64-fact world in `artifacts/pilot_world.json`. `generate_scenarios.py` used four parallel
category writers to create 25 cases at the exact category quotas and 93 candidate-memory occurrences.
The deterministic curator merged those candidates into 50 canonical memories, added ten useful unused
world facts, and repaired seven cases in `artifacts/pilot_curated.json`.

`enrich_memories.py` applied the retained `artifacts/pilot_memory_rewrites.json` artifact to create the
final candidate in `artifacts/pilot_enriched.json`. It contains exactly 30 description-only short, 20
medium, and 10 episodic memories. Medium contents average about 43.5 words and episodic contents average
87.5 words. Seven cases have episodic gold memories, whose descriptions identify the topic without
exposing the complete answer. The artifact fixes the benchmark reference date at `2026-06-05`.

The enriched corpus has unique contiguous memory ids, no duplicate descriptions, complete world-ledger
provenance, valid gold and context references, production-shaped descriptions and contents, and no raw
ids in user-facing text. One project case was replaced with a natural post-pilot expansion question
because another realistic memory could also answer the original single-gold site/contact question.
Related memories remain in the corpus; they were not deleted merely to make the critics pass.

`review_scenarios.py` then reran the evidence and realism critics against every final case, the full
60-memory corpus, all canonical facts, and the fixed reference date. The current
`artifacts/pilot_enriched_reviews.json` pass accepted 23/25 cases by both critics and sent two people
cases to human review. One critic disputed whether a pilot-only Owen escalation memory is too close to
his general IT-lead memory. For Sasha, the evidence critic treated communications work as equivalent to
her requested exact job title, while the realism critic required the explicit title. These are useful
calibration disagreements and should be adjudicated rather than repaired automatically. The rubric now
explicitly treats description and contents together, requires complete multi-part coverage, and
distinguishes project-specific contacts from general roles.

The approved pilot was published to the two versioned MLflow datasets and seeded once into
`realistic-search-pilot-v2-6faac0b1`. The stored corpus contains exactly 60 entries and matches fingerprint
`6faac0b1801c1e83e8e283323c5fb9fb5a6e62ff04ebcb4b0eb5e943d7c5589e` before and after evaluation.

The first direct-search run is
[`a8b7dd3058814d738e13de36c42aa589`](https://eng-ml-inference.staging.cloud.databricks.com/ml/experiments/2879499460172053/evaluation-runs?selectedRunUuid=a8b7dd3058814d738e13de36c42aa589).
It invoked only `entries:search`: no deployed agent, query rewriting, or LLM judge.

| Metric | Result |
|---|---:|
| Hit@1 | 84% |
| Hit@3 | 96% |
| Hit@5 | 96% |
| Hit@10 | 100% |
| MRR | 0.892 |
| Mean search latency | 289 ms |

Four cases were not Top 1: KB Refresh ownership ranked 2, the meeting-note label ranked 3, concise
project-update formatting ranked 3, and Coverage Pilot expansion ranked 8. The first three sit behind
plausible same-topic memories. The expansion case is the strongest diagnostic: seven realistic
Riverbend and Coverage Pilot memories outrank the exact future-expansion memory. All four gold mappings
remain correct, so these results are search-quality findings rather than dataset repairs.

### Autonomous final-answer track

The deployed-App track sends each natural query unchanged in a fresh thread. It does not force search,
call the direct search endpoint, or compute retrieval metrics. Its only scorer is the
`realistic_memory_answer_correctness` MLflow LLM judge, which receives only the query, final answer, and
the case's `answer_should` rubric. Tool calls remain visible in traces for debugging but are not judge
evidence.

The frozen fixture is read-only on this track. `custom_inputs.eval_read_only=true` is accepted only
alongside a valid `eval_scope` when `DATABRICKS_MEMORY_EVAL_MODE` is enabled; the deployed agent then has
search/get/list but no save/update/delete tools. Normal App requests retain all memory tools.

Run all 25 cases with:

```bash
uv run python -m shared_memory_eval.run_realistic_search_autonomous \
  --profile eng-ml-inference
```

The current clean run is
[`0f386ccd51554001a65e9414059297ed`](https://eng-ml-inference.staging.cloud.databricks.com/ml/experiments/2879499460172053/evaluation-runs?selectedRunUuid=0f386ccd51554001a65e9414059297ed).
It completed 25/25 assessments in about 45 seconds, left the 60-memory fingerprint unchanged, and the
agent autonomously chose `search_memory` for all 25 questions. The judge assigned 22 full-credit, three
partial-credit, and zero failing answers, for a mean of 0.94.

All three partials are currently expectation-calibration findings, not retrieval failures: the answers
directly answered the user's question but omitted extra context required by `answer_should` (Mara/working
group context for sign-off, Pine Notes for a question asking only the action-list label, and the May 1
effective date for a question asking the current intake route). These rubrics should be human-adjudicated
before treating 0.94 as the product's true answer-accuracy rate.

The development pilot can now be reused for search comparisons. Formal release still requires filling
the independent human-label sheet for all 25 cases and double-labeling the five calibration cases.

Generate the readable review packet and blank label sheet with:

```bash
uv run python -m shared_memory_eval.realistic_search_benchmark.prepare_human_review
```

Review `artifacts/pilot_human_review.md` and record decisions in
`artifacts/pilot_human_labels.csv`. Complete every `primary` row, then have a second person independently
complete the five `calibration` rows before comparing labels or opening the model-critic results.

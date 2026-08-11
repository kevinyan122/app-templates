# Managed-memory evaluations

Two complementary suites evaluate the deployed `agent-langgraph-advanced` testing App.

## Scenario suite: agent memory behavior

This suite tests whether the agent:

- saves explicit and implicit durable information;
- avoids unnecessary saves and searches;
- updates existing memories without losing required details;
- searches when prior user context is needed;
- retrieves the required facts and uses them in its answer; and
- keeps memory descriptions short while placing detail in contents.

The current dataset, `kevinyan.default.memory_eval_scenarios`, contains **35 scenarios** across
eight categories; nine scenarios are multi-turn. Each scenario receives a temporary direct
evaluation scope. The runner seeds its initial memories, waits for search indexing,
uses a fresh thread when requested, invokes the deployed App, waits until direct store reads reflect
the successful mutation tool outputs, scores the trace and store changes, and cleans the scope.
Registered judge calls retry transient endpoint failures up to three times.

Deterministic scorers cover write behavior, write structure, search behavior, and retrieved facts.
The registered `memory_write_quality` and `memory_answer_quality` LLM judges evaluate the
case-specific `memory_should` and `answer_should` requirements.

Sources:

- [Scenario definitions](./memory_eval_cases.py)
- [Shared notebook source](./shared_memory_eval/Run_managed_memory_evaluation.py)
- [Remote runner and deterministic scorers](./shared_memory_eval/memory_eval.py)
- [Registered judge definitions](./memory_eval_judges.py)
- [Dataset synchronization](./sync_eval_dataset.py)
- [Shared Databricks notebook](https://eng-ml-inference.staging.cloud.databricks.com/editor/notebooks/2266022481806828?o=1653573648247579)

Recent runs:

- [Latest full judged run: 35 scenarios](https://eng-ml-inference.staging.cloud.databricks.com/ml/experiments/2879499460172053/evaluation-runs?selectedRunUuid=203d5fc80d3d44afb8228d26a3168f90) — all 35 traces completed successfully; answer quality 75.9%, write quality 55.6%, search behavior 90.6%, search result 78.6%, write behavior 88.6%, and write structure 66.7%.
- [Update-consistency verification: 2 scenarios](https://eng-ml-inference.staging.cloud.databricks.com/ml/experiments/2879499460172053/evaluation-runs?selectedRunUuid=6e63e247e87a4cd29967abc4fc55bdbe) — both formerly misattributed update cases pass all deterministic scorers and both LLM judges with the fixed runner.

## CH-Bench: persistent-corpus retrieval

CH-Bench isolates retrieval quality from overall agent behavior. Its frozen ContextHeavy corpus
contains **44 memories and 35 questions** across five profiles, including five abstention questions.
The corpus is seeded once into a versioned direct evaluation scope and reused by every question; each
question still receives a fresh thread.

Every question evaluates two paths:

1. direct managed-memory search using the question verbatim;
2. end-to-end App retrieval, including search selection, generated queries, ranking, and the answer.

The runner reports recall@K, precision@K, hit@K, MRR, nDCG@K, answer correctness, abstention, and
latency. It verifies the corpus fingerprint before and after evaluation and fails if the agent
creates, updates, or deletes any benchmark memory.

Sources:

- [CH-Bench notebook source](./shared_memory_eval/Run_chbench_evaluation.py)
- [Runner, retrieval adapters, and scorers](./shared_memory_eval/chbench_eval.py)
- [One-time corpus seeder](./shared_memory_eval/seed_chbench_corpus.py)
- [Local authenticated runner](./shared_memory_eval/run_chbench_smoke.py)
- [Vendored benchmark data and provenance](./shared_memory_eval/chbench_data/README.md)
- [Explicit cleanup command](./shared_memory_eval/cleanup_chbench_corpus.py)

Most recent run:

- [Full 35-question judged run](https://eng-ml-inference.staging.cloud.databricks.com/ml/experiments/2879499460172053/evaluation-runs?selectedRunUuid=d7fbcfbe893e467fb1e63cd98ec641fc) — direct recall@10 86.7%, agent recall@10 60%, answer correctness 68.3%, abstention correctness 100%, and no corpus mutation.

The CH-Bench implementation and successful run currently exist locally. Its notebook, helper, and
vendored data have not yet been copied to the Shared Databricks folder.

## Realistic pilot: autonomous final answers

This track uses one shared, pre-seeded 60-memory world and 25 natural questions. Each question is sent
unchanged to the deployed App in a fresh thread. Search is autonomous: there is no forced-search text and
no direct-search scorer. The sole scorer is an MLflow LLM judge over the final answer and that case's
`answer_should` rubric. The evaluation request makes the frozen fixture read-only, without changing the
tool set for normal App traffic.

Sources:

- [Pilot data, design, and results](./shared_memory_eval/realistic_search_benchmark/README.md)
- [App predictor and final-answer judge](./shared_memory_eval/realistic_search_eval.py)
- [Autonomous runner](./shared_memory_eval/run_realistic_search_autonomous.py)

Most recent run:

- [25-case autonomous final-answer run](https://eng-ml-inference.staging.cloud.databricks.com/ml/experiments/2879499460172053/evaluation-runs?selectedRunUuid=0f386ccd51554001a65e9414059297ed) — 22 full-credit, three partial-credit, zero failing answers; mean 0.94; all 25 cases autonomously searched; corpus unchanged. The three partials need expectation calibration because each answer directly addressed the user's narrower question while omitting extra rubric context.

# Shared managed-memory evaluation

This directory is the source for the copy-first Databricks Workspace evaluation at
`/Workspace/Shared/managed-memory-eval-remote`.

- `Run_managed_memory_evaluation.py` is a Databricks source notebook.
- `memory_eval.py` loads the existing MLflow dataset and registered judges, invokes the deployed
  testing App, and independently seeds and verifies the evaluation memory store. It reconciles each
  turn against successful mutation tool outputs before scoring and retries transient judge failures.
- `Seed_chbench_memories.py` is the optional, guarded Databricks notebook that writes the frozen
  corpus into a named evaluation scope.
- `Run_chbench_evaluation.py` is the read-only Databricks evaluation notebook.
- `chbench_eval.py` converts CH-Bench questions into MLflow records, invokes direct store search and
  the deployed App, resolves ranked memory paths back to benchmark ids, and defines the scorers.
- `seed_chbench_corpus.py` idempotently creates or validates the frozen corpus in a versioned scope.
- `run_chbench_smoke.py` runs one local, user-OAuth question against that pre-seeded corpus;
  `dev-q2` is the default.
- `cleanup_chbench_corpus.py` is a guarded, explicit maintenance command that evaluators never call.
- `sync_chbench_dataset.py` idempotently publishes separate 35-row question/gold and 44-row corpus
  MLflow datasets. Runtime notebooks read those datasets; raw JSON is not copied to Workspace.
- `Seed_longmemeval_memories.py` is the guarded, one-time LongMemEval fixture notebook.
- `Run_longmemeval_evaluation.py` is the read-only, step-by-step LongMemEval evaluation notebook.
- `chbench_data/contextheavy/` pins the upstream 44-memory / 35-question suite and records its source
  commit in `chbench_data/README.md`.

The runner has no `agent_server` dependency. The deployed App must set
`DATABRICKS_MEMORY_EVAL_MODE=true`; each request supplies the complete
`custom_inputs.eval_scope`, which the App uses directly. This deliberately bypasses normal OBO
per-user scope resolution and is suitable only for the dedicated testing App and test memory store.

The existing local evaluator, notebook, dataset, and managed judges remain unchanged.

## CH-Bench evaluation

Publish only the two notebooks plus `chbench_eval.py` and `memory_eval.py` to a `ch-bench/`
Workspace subfolder. Seed the corpus once under the same interactive user who will run the
evaluation. The evaluation notebook defaults to all 35 questions with judges enabled; set
`question_id=dev-q2` when you only want a smoke case.

The CH-Bench run deliberately uses one corpus for all rows:

1. pin the direct scope to `chbench-contextheavy-5a08786a`;
2. load the questions and exact 44-entry corpus from separate versioned MLflow datasets;
3. validate all persistent entries and their content against the corpus fingerprint;
4. prefix every deployed-agent question with an explicit instruction to call `search_memory`, while
   direct retrieval continues to use the original question verbatim;
5. run each question with a fresh `thread_id` and `MLFLOW_GENAI_EVAL_MAX_WORKERS=1`;
6. score direct memory-store retrieval and forced-search deployed-agent retrieval separately;
7. validate the count, content, and fingerprint again without deleting the fixture.

Direct retrieval uses the exact original question as the search query. Every agent prompt explicitly
requires `search_memory`; agent retrieval concatenates results
from every `search_memory` call in tool-call order and deduplicates paths at their first appearance.
That gives a stable ranked list for recall@K, precision@K, hit@K, MRR, and nDCG@K while preserving
the order in which evidence reached the agent. Abstention rows have no gold memory ids and are
excluded from ranking aggregates.

The deterministic run does not require the upstream CH-Bench package. MLflow judges score answer
correctness and abstention; omit `--include-judges` only when debugging retrieval in isolation.

Seed or validate the fixture once (rerunning this exact command is a no-op):

```bash
uv run python shared_memory_eval/seed_chbench_corpus.py \
  --profile eng-ml-inference
```

Publish or validate the versioned MLflow dataset:

```bash
uv run python shared_memory_eval/sync_chbench_dataset.py \
  --profile eng-ml-inference
```

The datasets are `kevinyan.default.chbench_contextheavy_5a08786a` for questions/gold and
`kevinyan.default.chbench_contextheavy_memories_5a08786a` for the exact corpus. Evaluation scope and
`top_k` remain run settings; the persistent scope remains the source searched by both retrieval paths.

For a user-authenticated smoke run from this template directory:

```bash
uv run python shared_memory_eval/run_chbench_smoke.py \
  --profile eng-ml-inference \
  --question-id dev-q2 \
  --include-judges
```

Run the complete 35-question suite with judges enabled:

```bash
uv run python shared_memory_eval/run_chbench_smoke.py \
  --profile eng-ml-inference \
  --all-questions \
  --include-judges
```

The scope is versioned from the pinned CH-Bench source commit. If the vendored corpus ever
changes, use a new scope. If this dedicated scope drifts, the seeder refuses to overwrite it
unless you deliberately pass `--replace`.

Cleanup is never part of evaluation. To remove this fixture intentionally, repeat the exact selector:

```bash
uv run python shared_memory_eval/cleanup_chbench_corpus.py \
  --profile eng-ml-inference \
  --confirm-scope chbench-contextheavy-5a08786a
```

All evaluators using `chbench-contextheavy-5a08786a` share that exact read-only test scope. Seed it
once, validate its fingerprint before and after every run, and use a different versioned scope for
a different corpus. Anyone authorized to invoke the testing App can name a scope in its configured
test memory store, which is why direct scope mode must remain disabled on customer-facing Apps.

## LongMemEval-S raw-session pilot

The first LongMemEval pilot uses 30 answerable questions: five deterministic cases from each of its
six question types. Every question has a distinct persistent scope containing its complete history,
so the pinned pilot has 30 scopes, 1,444 raw session memories, and 48 gold session memories. Source
session ids and `has_answer` labels are excluded from searchable entries; opaque ids preserve the
gold-session mapping in the MLflow datasets.

This raw corpus is the lossless baseline. A later question-blind session-to-memory extraction job
will produce a distilled corpus for the same cases and ids. Comparing the two separates extraction
loss from search, routing, and answer failures.

Download the pinned 277 MB source file outside the repository and build the local bundle:

```bash
curl -L --fail -o /tmp/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json

uv run python -m shared_memory_eval.longmemeval_adapter \
  --source /tmp/longmemeval_s_cleaned.json \
  --output-dir /tmp/longmemeval-s-raw-pilot
```

The adapter verifies the source SHA-256 and the exact pilot fingerprint. Publish the 30 compact
question/gold rows to MLflow:

```bash
uv run python shared_memory_eval/sync_longmemeval_datasets.py \
  --profile eng-ml-inference \
  --source /tmp/longmemeval_s_cleaned.json
```

The dataset is `kevinyan.default.longmemeval_s_raw_pilot_v1`. Each row includes its persistent
scope's expected entry count and content fingerprint. The 15 MB raw corpus is not duplicated into a
managed dataset because its table synchronization exceeds the platform's 4 MB gRPC message limit;
the one-time seeder transforms the pinned source directly.

Seed and validate one case before creating all 30 persistent fixtures:

```bash
uv run python shared_memory_eval/seed_longmemeval_corpora.py \
  --profile eng-ml-inference \
  --source /tmp/longmemeval_s_cleaned.json \
  --question-id 8fb83627

uv run python shared_memory_eval/seed_longmemeval_corpora.py \
  --profile eng-ml-inference \
  --source /tmp/longmemeval_s_cleaned.json \
  --all-cases
```

Seeding is idempotent. It never overwrites a non-empty mismatched scope without `--replace`, and an
index-visibility timeout retains an already exact durable corpus for a later retry. By default the
seeder fingerprints every durable entry but checks search visibility only for the 48 gold sessions,
not all 1,444 entries. Use `--search-check none` for the fastest durable-only seed or
`--search-check all` for an intentionally exhaustive index check. Independent entry writes and exact
fingerprint reads use eight bounded workers by default; set `MEMORY_EVAL_STORE_WORKERS=1` to debug
them serially.

Run a single judged smoke case, then the complete pilot:

```bash
uv run python shared_memory_eval/run_longmemeval_pilot.py \
  --profile eng-ml-inference \
  --question-id 8fb83627 \
  --include-judge

uv run python shared_memory_eval/run_longmemeval_pilot.py \
  --profile eng-ml-inference \
  --all-cases \
  --include-judge
```

Run only the direct managed-memory search endpoint over all 30 questions—no deployed App and no LLM
judge—with:

```bash
uv run python shared_memory_eval/run_longmemeval_direct.py \
  --profile eng-ml-inference
```

That run reports only direct recall@K, hit@K, MRR, nDCG@K, endpoint latency, category aggregates,
and the exact failed question ids.

Each question records direct managed-memory retrieval and forced-search agent retrieval separately,
including recall@K, hit@K, MRR, nDCG@K, search invocation, unexpected writes, and category-aware
answer correctness. Evaluation validates every selected scope before and after and never seeds,
repairs, or deletes a fixture.

For a shared Workspace workflow, copy `Seed_longmemeval_memories.py`,
`Run_longmemeval_evaluation.py`, `memory_eval.py`, `longmemeval_adapter.py`,
`longmemeval_eval.py`, and `sync_longmemeval_datasets.py` into the same folder. The evaluation
notebook needs no laptop files; the seed notebook downloads and verifies the pinned source when its
configured driver-local source path is absent. Its `evaluation_mode` widget supports both the
retrieval-only `direct_search` run and the deployed-App `forced_agent` run.

## Required permissions

Each notebook runner needs:

- access to this Shared workspace folder;
- permission to use the deployed testing App;
- `READ_MEMORY_STORE` on the evaluation memory store for evaluation;
- `WRITE_MEMORY_STORE` only for the separate seed, repair, and cleanup commands;
- access to the MLflow experiment and managed dataset.

The notebook validates the exact frozen corpus before running and validates it again afterward. It
never changes the persistent scope itself.

# Databricks notebook source
# MAGIC %md
# MAGIC # CH-Bench ContextHeavy evaluation — deployed App
# MAGIC
# MAGIC This notebook loads the versioned 44-memory / 35-question ContextHeavy-Bench datasets and runs two
# MAGIC paths in the same MLflow evaluation:
# MAGIC
# MAGIC 1. direct managed-memory search with the question as the query;
# MAGIC 2. the deployed App, explicitly instructed to search memory, including its generated query,
# MAGIC    ranked sources, and answer.
# MAGIC
# MAGIC The frozen corpus lives in a versioned direct evaluation scope and is seeded separately. Every
# MAGIC evaluation validates that exact fixture before and after the run, while every question uses a
# MAGIC fresh conversation thread. This notebook never seeds, repairs, or deletes the corpus.

# COMMAND ----------

# MAGIC %pip install \
# MAGIC   "databricks-sdk==0.106.0" \
# MAGIC   "databricks-agents==1.10.0" \
# MAGIC   "mlflow==3.11.1" \
# MAGIC   "requests==2.32.5"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure the run

# COMMAND ----------

import os
import sys
from pathlib import Path


def _widget_text(name: str, default: str, label: str) -> None:
    try:
        dbutils.widgets.text(name, default, label)
    except Exception:
        pass


def _widget_dropdown(name: str, default: str, choices: list[str], label: str) -> None:
    try:
        dbutils.widgets.dropdown(name, default, choices, label)
    except Exception:
        pass


_widget_text(
    "app_url",
    "https://agent-langgraph-adv-kevin-1653573648247579.staging.aws.databricksapps.com",
    "Testing App URL",
)
_widget_text("memory_store", "kevinyan.default.kevin_test", "UC memory store")
_widget_text(
    "question_dataset",
    "kevinyan.default.chbench_contextheavy_5a08786a",
    "Question/gold MLflow dataset",
)
_widget_text(
    "corpus_dataset",
    "kevinyan.default.chbench_contextheavy_memories_5a08786a",
    "Memory corpus MLflow dataset",
)
_widget_text(
    "eval_scope",
    "chbench-contextheavy-5a08786a",
    "Pre-seeded CH-Bench scope selector",
)
_widget_text("question_id", "", "Question ids (comma-separated; blank for track/all)")
_widget_text("track", "", "Tracks (comma-separated; blank for all)")
_widget_text("top_k", "10", "Retrieval top K")
_widget_dropdown("include_judges", "true", ["false", "true"], "Run answer judges")
_widget_text("judge_model", "databricks:/databricks-gpt-5-2", "Judge model")
_widget_text("index_timeout_seconds", "180", "Search-index timeout")

app_url = dbutils.widgets.get("app_url").strip()
memory_store = dbutils.widgets.get("memory_store").strip()
question_dataset = dbutils.widgets.get("question_dataset").strip()
corpus_dataset = dbutils.widgets.get("corpus_dataset").strip()
eval_scope = dbutils.widgets.get("eval_scope").strip()
question_ids = {
    value.strip() for value in dbutils.widgets.get("question_id").split(",") if value.strip()
}
tracks = {value.strip() for value in dbutils.widgets.get("track").split(",") if value.strip()}
top_k = int(dbutils.widgets.get("top_k"))
include_judges = dbutils.widgets.get("include_judges") == "true"
judge_model = dbutils.widgets.get("judge_model").strip()
index_timeout_seconds = int(dbutils.widgets.get("index_timeout_seconds"))

if not app_url or not memory_store or not question_dataset or not corpus_dataset or not eval_scope:
    raise ValueError(
        "app_url, memory_store, question_dataset, corpus_dataset, and eval_scope are required"
    )
if top_k <= 0:
    raise ValueError("top_k must be positive")

notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
workspace_notebook_path = (
    notebook_path if notebook_path.startswith("/Workspace/") else "/Workspace" + notebook_path
)
shared_eval_root = str(Path(workspace_notebook_path).parent)
required_paths = [
    Path(shared_eval_root) / "memory_eval.py",
    Path(shared_eval_root) / "chbench_eval.py",
]
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Copy chbench_eval.py and memory_eval.py beside this notebook. Missing: "
        + ", ".join(missing_paths)
    )
if shared_eval_root not in sys.path:
    sys.path.insert(0, shared_eval_root)

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the managed datasets and verify access

# COMMAND ----------

import json

import mlflow
import pandas as pd

import chbench_eval
import memory_eval


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
memory_eval.configure(
    app_url=app_url,
    memory_store=memory_store,
    index_timeout_s=index_timeout_seconds,
)

full_scope = memory_eval._full_eval_scope(eval_scope)
records = chbench_eval.load_records_from_dataset(
    question_dataset,
    eval_scope,
    top_k=top_k,
)
seed_memories = chbench_eval.load_seed_memories_from_dataset(corpus_dataset)
chbench_eval.validate_question_corpus_records(records, seed_memories)
selected_records = chbench_eval.filter_records(records, question_ids, tracks)

try:
    before_status = chbench_eval.validate_persistent_entries(eval_scope, seed_memories)
except Exception as exc:
    raise RuntimeError(
        "The persistent corpus is unavailable or does not match the corpus dataset. Run "
        "the sibling 'Seed CH-Bench memories' notebook with this same exact scope."
    ) from exc
before_snapshot = before_status["snapshot"]

abstention_count = sum(
    chbench_eval._expects_abstention(record["expectations"]) for record in records
)
print(f"Source: {chbench_eval.SOURCE_REPOSITORY}@{chbench_eval.SOURCE_COMMIT}")
print(f"Question dataset: {question_dataset}")
print(f"Corpus dataset: {corpus_dataset}")
print(f"Suite: {len(seed_memories)} memories, {len(records)} questions, {abstention_count} abstentions")
print(f"Selected: {len(selected_records)} questions")
print(f"Testing App: {app_url}")
print(f"Memory scope: {full_scope}")
print(f"Corpus fingerprint: {before_status['fingerprint']}")
print(f"Corpus validation: exact ({before_status['actual_count']} entries)")
print(f"Direct and forced-search agent retrieval top_k: {top_k}")
print(f"Answer judges: {'on' if include_judges else 'off'}")

catalog_rows = [
    {
        "question_id": record["inputs"]["question_id"],
        "track": record["inputs"]["track"],
        "type": record["inputs"]["question_type"],
        "question": record["inputs"]["question"],
        "expected_answer": record["expectations"]["expected_answer"],
        "relevant_ids": record["expectations"]["relevant_ids_json"],
        "expect_abstain": record["expectations"]["expect_abstain"],
    }
    for record in selected_records
]
display(pd.DataFrame(catalog_rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluate the pre-seeded read-only fixture
# MAGIC
# MAGIC Direct-search metrics isolate the managed-memory index. Agent metrics include search policy,
# MAGIC query formulation, ranking, and answer synthesis. Retrieval metrics exclude the five abstention
# MAGIC rows because they intentionally have no relevant-memory ids. The post-run check fails if any
# MAGIC entry was created, changed, or deleted; repair is always a separate explicit operation.

# COMMAND ----------

scorers = list(chbench_eval.DETERMINISTIC_SCORERS)
if include_judges:
    scorers.extend(chbench_eval.load_answer_judges(judge_model))

run_name = f"memory-eval/chbench/forced-search/{len(selected_records)}q/{eval_scope}"
run_tags = {
    "suite": "chbench-contextheavy",
    "source_repository": chbench_eval.SOURCE_REPOSITORY,
    "source_commit": chbench_eval.SOURCE_COMMIT,
    "runner": "databricks-shared-remote-app",
    "agent_search_mode": "explicit-search-memory-instruction",
    "app_url": app_url,
    "memory_scope_mode": "persistent-direct-eval-scope",
    "eval_scope": eval_scope,
    "corpus_fingerprint": before_status["fingerprint"],
    "question_dataset": question_dataset,
    "corpus_dataset": corpus_dataset,
    "question_count": str(len(selected_records)),
    "corpus_memory_count": str(len(seed_memories)),
    "top_k": str(top_k),
    "judges_enabled": str(include_judges).lower(),
}

result = None
corpus_diff = None
after_status = None
print(f"Running {len(selected_records)} question(s) with {len(scorers)} scorer(s)...")
with mlflow.start_run(run_name=run_name, tags=run_tags):
    try:
        result = mlflow.genai.evaluate(
            data=selected_records,
            predict_fn=chbench_eval.predict_fn,
            scorers=scorers,
        )
    finally:
        after_status = chbench_eval.inspect_persistent_entries(eval_scope, seed_memories)
        corpus_diff = memory_eval.diff_snapshots(before_snapshot, after_status["snapshot"])
        corpus_changed = any(corpus_diff[kind] for kind in ("created", "updated", "deleted"))
        mlflow.log_metric("chbench.corpus_changed", float(corpus_changed))
        mlflow.log_dict(
            {
                "eval_scope": eval_scope,
                "full_scope": full_scope,
                "fingerprint": before_status["fingerprint"],
                "actual_fingerprint": after_status["actual_fingerprint"],
                "before_count": before_status["actual_count"],
                "after_count": after_status["actual_count"],
                "after_exact": after_status["exact"],
                "diff": corpus_diff,
            },
            "chbench/corpus_validation.json",
        )

if result is None:
    raise RuntimeError("CH-Bench evaluation did not produce a result")
if after_status is None or not after_status["exact"] or any(
    corpus_diff[kind] for kind in ("created", "updated", "deleted")
):
    raise RuntimeError(
        "Persistent CH-Bench corpus changed during evaluation. Inspect "
        "chbench/corpus_validation.json and repair it explicitly with "
        "the 'Seed CH-Bench memories' notebook with action=replace."
    )
print(
    f"Corpus unchanged: {after_status['actual_count']} entries, "
    f"fingerprint {after_status['actual_fingerprint']}"
)

workspace_host = memory_eval._ws().config.host.rstrip("/")
run_url = (
    f"{workspace_host}/ml/experiments/{memory_eval.EXPERIMENT_ID}/evaluation-runs"
    f"?selectedRunUuid={result.run_id}"
)
print(f"MLflow run: {result.run_id}")
print(run_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Compare direct retrieval with forced-search agent retrieval

# COMMAND ----------

result_frame = result.result_df.copy()
metric_names = [scorer.name for scorer in scorers]
score_columns = [
    f"{name}/value" for name in metric_names if f"{name}/value" in result_frame.columns
]
rationale_columns = [
    f"{name}/rationale" for name in metric_names if f"{name}/rationale" in result_frame.columns
]

display(result_frame[["request", "response", *score_columns]])
display(result_frame[["request", *rationale_columns]])

print("Interpretation:")
print("- direct succeeds, agent fails: tool choice or query formulation problem")
print("- both retrieval paths fail: managed-memory retrieval/index problem")
print("- retrieval succeeds, judge fails: answer synthesis or abstention problem")
print("- agent_no_memory_write fails: the recall-only benchmark changed its corpus")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rerun checklist
# MAGIC
# MAGIC - Seed once with the sibling `Seed CH-Bench memories` notebook under the same interactive user identity.
# MAGIC - Start with one question and judges on; clear `question_id` for the full 35-question run.
# MAGIC - The deployed App must have `DATABRICKS_MEMORY_EVAL_MODE=true`.
# MAGIC - Use interactive user authentication; a serverless Job credential is not an App OAuth token.
# MAGIC - Keep `MLFLOW_GENAI_EVAL_MAX_WORKERS=1`: all rows intentionally share one read-only corpus.
# MAGIC - Evaluation never repairs or deletes the fixture. Use the guarded maintenance scripts explicitly.

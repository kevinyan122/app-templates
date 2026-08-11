# Databricks notebook source
# MAGIC %md
# MAGIC # LongMemEval-S raw-session evaluation — deployed App
# MAGIC
# MAGIC This notebook evaluates the persistent raw-session baseline without changing it. Each selected
# MAGIC question runs two retrieval paths:
# MAGIC
# MAGIC 1. direct managed-memory search with the original question;
# MAGIC 2. the deployed App explicitly instructed to call `search_memory` before answering.
# MAGIC
# MAGIC Retrieval, agent tool use, answer correctness, and corpus integrity remain separate results.
# MAGIC Start with the default single case. Clear `question_id` to run all 30 after every scope is seeded.
# MAGIC Choose `direct_search` to benchmark only the managed-memory search endpoint, or `forced_agent`
# MAGIC to include the deployed App and optional answer judge.

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
    "kevinyan.default.longmemeval_s_raw_pilot_v1",
    "Question/gold MLflow dataset",
)
_widget_text("question_id", "8fb83627", "Question ids (comma-separated; blank for all)")
_widget_text("top_k", "10", "Retrieval top K")
_widget_dropdown(
    "evaluation_mode",
    "direct_search",
    ["direct_search", "forced_agent"],
    "Evaluation mode",
)
_widget_dropdown("include_judge", "true", ["false", "true"], "Run answer judge")
_widget_text("judge_model", "databricks:/databricks-gpt-5-2", "Judge model")
_widget_text("index_timeout_seconds", "180", "Memory visibility timeout")

app_url = dbutils.widgets.get("app_url").strip()
memory_store = dbutils.widgets.get("memory_store").strip()
question_dataset = dbutils.widgets.get("question_dataset").strip()
question_ids = {
    value.strip()
    for value in dbutils.widgets.get("question_id").split(",")
    if value.strip()
}
top_k = int(dbutils.widgets.get("top_k"))
evaluation_mode = dbutils.widgets.get("evaluation_mode").strip()
include_judge = dbutils.widgets.get("include_judge") == "true"
judge_model = dbutils.widgets.get("judge_model").strip()
index_timeout_seconds = int(dbutils.widgets.get("index_timeout_seconds"))

if not app_url or not memory_store or not question_dataset:
    raise ValueError("app_url, memory_store, and question_dataset are required")
if top_k <= 0:
    raise ValueError("top_k must be positive")

notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
workspace_notebook_path = (
    notebook_path if notebook_path.startswith("/Workspace/") else "/Workspace" + notebook_path
)
notebook_root = str(Path(workspace_notebook_path).parent)
required_paths = [
    Path(notebook_root) / "memory_eval.py",
    Path(notebook_root) / "longmemeval_adapter.py",
    Path(notebook_root) / "longmemeval_eval.py",
    Path(notebook_root) / "sync_longmemeval_datasets.py",
]
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Copy the LongMemEval Python helpers beside this notebook. Missing: "
        + ", ".join(missing_paths)
    )
if notebook_root not in sys.path:
    sys.path.insert(0, notebook_root)

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load cases and verify every selected persistent scope

# COMMAND ----------

import json

import mlflow
import pandas as pd

import longmemeval_adapter
import longmemeval_eval
import memory_eval


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
memory_eval.configure(
    app_url=app_url,
    memory_store=memory_store,
    index_timeout_s=index_timeout_seconds,
)
records = longmemeval_eval.load_question_records(question_dataset, top_k=top_k)
selected = [
    row
    for row in records
    if not question_ids or row["inputs"]["question_id"] in question_ids
]
if not selected:
    raise ValueError(f"No cases matched question ids: {sorted(question_ids)}")

catalog = pd.DataFrame(
    [
        {
            "question_id": row["inputs"]["question_id"],
            "category": row["inputs"]["question_type"],
            "memories": int(row["inputs"]["expected_memory_count"]),
            "gold_sessions": len(longmemeval_eval._relevant_ids(row["expectations"])),
            "question": row["inputs"]["question"],
            "expected_answer": row["expectations"]["expected_answer"],
        }
        for row in selected
    ]
)
display(catalog)

before = {}
for index, row in enumerate(selected, start=1):
    inputs = row["inputs"]
    print(f"[{index}/{len(selected)}] Validating {inputs['question_id']} ...")
    before[inputs["case_id"]] = longmemeval_eval.validate_persistent_scope(
        inputs["scope_selector"],
        inputs["expected_memory_count"],
        inputs["expected_corpus_fingerprint"],
    )

print(
    f"Ready: {len(selected)} exact scopes, "
    f"{sum(status['actual_count'] for status in before.values())} memories."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run direct retrieval, forced-search agent retrieval, and scoring

# COMMAND ----------

if evaluation_mode == "direct_search":
    predict_fn = longmemeval_eval.direct_predict_fn
    scorers = list(longmemeval_eval.DIRECT_SEARCH_SCORERS)
else:
    predict_fn = longmemeval_eval.predict_fn
    scorers = list(longmemeval_eval.DETERMINISTIC_SCORERS)
if evaluation_mode == "forced_agent" and include_judge:
    scorers.append(longmemeval_eval.load_answer_judge(judge_model))

result = None
after = {}
validation = {}
with mlflow.start_run(
    run_name=f"memory-eval/longmemeval-s/raw/{evaluation_mode}/{len(selected)}q",
    tags={
        "suite": longmemeval_adapter.SUITE_NAME,
        "adapter_version": longmemeval_adapter.ADAPTER_VERSION,
        "source_commit": longmemeval_adapter.SOURCE_COMMIT,
        "bundle_fingerprint": longmemeval_adapter.EXPECTED_PILOT_BUNDLE_FINGERPRINT,
        "runner": "shared-workspace-notebook",
        "evaluation_mode": evaluation_mode,
        "agent_invoked": str(evaluation_mode == "forced_agent").lower(),
        "memory_scope_mode": "persistent-per-case-direct-eval-scope",
        "question_count": str(len(selected)),
        "top_k": str(top_k),
        "question_dataset": question_dataset,
        "judge_enabled": str(
            evaluation_mode == "forced_agent" and include_judge
        ).lower(),
        "judge_model": (
            judge_model
            if evaluation_mode == "forced_agent" and include_judge
            else "none"
        ),
    },
):
    try:
        result = mlflow.genai.evaluate(
            data=selected,
            predict_fn=predict_fn,
            scorers=scorers,
        )
    finally:
        for row in selected:
            inputs = row["inputs"]
            case_id = inputs["case_id"]
            status = longmemeval_eval.inspect_persistent_scope(
                inputs["scope_selector"],
                inputs["expected_memory_count"],
                inputs["expected_corpus_fingerprint"],
            )
            after[case_id] = status
            diff = memory_eval.diff_snapshots(
                before[case_id]["snapshot"], status["snapshot"]
            )
            validation[case_id] = {
                "question_id": inputs["question_id"],
                "scope": status["full_scope"],
                "before_count": before[case_id]["actual_count"],
                "after_count": status["actual_count"],
                "expected_fingerprint": status["expected_fingerprint"],
                "actual_fingerprint": status["actual_fingerprint"],
                "exact": status["exact"],
                "diff": diff,
            }

        corpus_changed = any(not status["exact"] for status in after.values()) or any(
            any(case["diff"][kind] for kind in ("created", "updated", "deleted"))
            for case in validation.values()
        )
        mlflow.log_metric("longmemeval.corpus_changed", float(corpus_changed))
        mlflow.log_dict(
            {
                "bundle_fingerprint": longmemeval_adapter.EXPECTED_PILOT_BUNDLE_FINGERPRINT,
                "cases": validation,
            },
            "longmemeval/corpus_validation.json",
        )

if result is None:
    raise RuntimeError("LongMemEval evaluation did not produce a result")
changed = [case_id for case_id, status in after.items() if not status["exact"]]
if changed:
    raise RuntimeError(f"Persistent LongMemEval corpora changed: {changed}")

workspace_host = memory_eval._ws().config.host.rstrip("/")
run_url = (
    f"{workspace_host}/ml/experiments/{memory_eval.EXPERIMENT_ID}/evaluation-runs"
    f"?selectedRunUuid={result.run_id}"
)
print(f"MLflow run: {result.run_id}")
print(run_url)
print(
    f"Corpus unchanged: {len(after)} scopes, "
    f"{sum(status['actual_count'] for status in after.values())} memories."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Read the results
# MAGIC
# MAGIC - Direct succeeds, agent retrieval fails: agent query formulation or tool-path problem.
# MAGIC - Both retrieval paths fail: managed-memory retrieval or indexing problem.
# MAGIC - Retrieval succeeds, answer judge fails: synthesis, conflict resolution, or temporal reasoning problem.
# MAGIC - Search-called fails: deployed App did not follow the forced-search instruction.
# MAGIC - No-memory-write fails: the recall-only evaluation changed its fixture.
# MAGIC
# MAGIC In `direct_search` mode, only the first retrieval path and its four metrics are present.

# COMMAND ----------

frame = result.result_df.copy()
scorer_names = [scorer.name for scorer in scorers]
score_columns = [
    f"{name}/value" for name in scorer_names if f"{name}/value" in frame.columns
]
rationale_columns = [
    f"{name}/rationale" for name in scorer_names if f"{name}/rationale" in frame.columns
]

summary_rows = []
for _, row in frame.iterrows():
    request = row.get("request")
    response = row.get("response")
    if isinstance(request, str):
        request = json.loads(request)
    if isinstance(response, str):
        response = json.loads(response)
    summary = {
        "question_id": (request or {}).get("question_id"),
        "category": (request or {}).get("question_type"),
        "answer": (response or {}).get("answer"),
        "direct_top_ids": (response or {}).get("direct_ranked_ids", [])[:top_k],
        "agent_search_queries": [
            search.get("query")
            for search in (response or {}).get("agent_searches", [])
        ],
        "agent_top_ids": (response or {}).get("agent_ranked_ids", [])[:top_k],
    }
    for column in score_columns:
        summary[column.removesuffix("/value")] = row.get(column)
    summary_rows.append(summary)

display(pd.DataFrame(summary_rows))
display(frame[["request", *rationale_columns]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect MLflow trace assessments

# COMMAND ----------

traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
trace_rows = []
assessment_rows = []
for trace in traces:
    trace_rows.append(
        {
            "trace_id": trace.info.trace_id,
            "state": str(trace.info.state),
            "request_preview": trace.info.request_preview,
            "response_preview": trace.info.response_preview,
            "span_count": len(trace.data.spans),
        }
    )
    for assessment in trace.info.assessments:
        value = assessment.to_dictionary()
        assessment_value = (
            value.get("feedback") or value.get("expectation") or {}
        ).get("value")
        assessment_rows.append(
            {
                "trace_id": trace.info.trace_id,
                "assessment": value.get("assessment_name"),
                "value": "" if assessment_value is None else str(assessment_value),
                "source": (value.get("source") or {}).get("source_type"),
                "rationale": value.get("rationale", ""),
            }
        )

display(pd.DataFrame(trace_rows))
display(pd.DataFrame(assessment_rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rerun checklist
# MAGIC
# MAGIC - Use the sibling seed notebook once; repeated evaluation runs reuse persistent scopes.
# MAGIC - Keep one question selected until its direct retrieval, agent retrieval, and judge all look sane.
# MAGIC - Clear `question_id` only after all 30 scopes validate.
# MAGIC - Keep `MLFLOW_GENAI_EVAL_MAX_WORKERS=1`; this makes App load and trace review predictable.
# MAGIC - The deployed testing App must have `DATABRICKS_MEMORY_EVAL_MODE=true`.
# MAGIC - Run interactively as a user authorized for the App, memory store, dataset, and experiment.

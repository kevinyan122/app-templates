# Databricks notebook source
# MAGIC %md
# MAGIC # Managed-memory evaluation — deployed App
# MAGIC
# MAGIC Run the categorized, multi-turn suite against the deployed testing App. The notebook
# MAGIC loads the managed MLflow dataset and registered judges; it does not import agent code.
# MAGIC
# MAGIC **Suggested flow**
# MAGIC
# MAGIC 1. Run the setup and configuration cells.
# MAGIC 2. Browse the case catalog and inspect the selected cases.
# MAGIC 3. Run the evaluation.
# MAGIC 4. Review the compact score table, rationales, and traces.
# MAGIC
# MAGIC Cases always load from the managed MLflow dataset named below. The default is one
# MAGIC deterministic smoke case. Clear `scenario_id` to select by category or run the whole suite.
# MAGIC Enable the LLM judges only when you want semantic write/answer grading.
# MAGIC Notebook outputs can be cleared from **Edit → Clear outputs** without changing the dataset or
# MAGIC any MLflow runs already logged.

# COMMAND ----------

# MAGIC %pip install \
# MAGIC   "databricks-sdk==0.106.0" \
# MAGIC   "databricks-agents==1.10.0" \
# MAGIC   "mlflow==3.11.1" \
# MAGIC   "requests==2.32.5"

# COMMAND ----------

# Load the notebook-scoped library versions instead of modules preloaded by the runtime.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure the run
# MAGIC
# MAGIC Widget changes take effect when this cell and the cells below are rerun.

# COMMAND ----------

import os
import sys
from pathlib import Path


def _widget_text(name: str, default: str, label: str) -> None:
    try:
        dbutils.widgets.text(name, default, label)
    except Exception:
        # The widget already exists after a partial rerun.
        pass


def _widget_dropdown(name: str, default: str, choices: list[str], label: str) -> None:
    try:
        dbutils.widgets.dropdown(name, default, choices, label)
    except Exception:
        pass


_widget_text(
    "dataset_name",
    "kevinyan.default.memory_eval_scenarios",
    "MLflow dataset",
)
_widget_text(
    "app_url",
    "https://agent-langgraph-adv-kevin-1653573648247579.staging.aws.databricksapps.com",
    "Testing App URL",
)
# Remove the retired source selector after upgrading an existing notebook session.
try:
    dbutils.widgets.remove("case_source")
except Exception:
    pass
_widget_text(
    "memory_store",
    "kevinyan.default.kevin_test",
    "UC memory store",
)
_widget_text(
    "scenario_id",
    "explicit-save-coffee-preference",
    "Scenario ids (comma-separated; blank for category/all)",
)
_widget_text("category", "", "Category (blank for all)")
_widget_dropdown("include_judges", "false", ["false", "true"], "Run LLM judges")
_widget_text("index_timeout_seconds", "180", "Search-index timeout")

dataset_name = dbutils.widgets.get("dataset_name").strip()
app_url = dbutils.widgets.get("app_url").strip()
memory_store = dbutils.widgets.get("memory_store").strip()
selected_scenario_ids = [
    value.strip()
    for value in dbutils.widgets.get("scenario_id").split(",")
    if value.strip()
]
selected_category = dbutils.widgets.get("category").strip()
include_judges = dbutils.widgets.get("include_judges") == "true"
index_timeout_seconds = int(dbutils.widgets.get("index_timeout_seconds"))

if not dataset_name:
    raise ValueError("dataset_name cannot be empty")
if not app_url:
    raise ValueError("app_url cannot be empty")
if not memory_store:
    raise ValueError("memory_store cannot be empty")

notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
workspace_notebook_path = (
    notebook_path if notebook_path.startswith("/Workspace/") else "/Workspace" + notebook_path
)
shared_eval_root = str(Path(workspace_notebook_path).parent)
if not (Path(shared_eval_root) / "memory_eval.py").exists():
    raise FileNotFoundError(
        f"Expected memory_eval.py beside this notebook: {shared_eval_root}"
    )
if shared_eval_root not in sys.path:
    sys.path.insert(0, shared_eval_root)

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"

print(f"Dataset: {dataset_name}")
print(f"Testing App: {app_url}")
print(f"Memory store: {memory_store}")
print(f"Scenarios: {', '.join(selected_scenario_ids) or '(not filtered)'}")
print(f"Category: {selected_category or '(not filtered)'}")
print(f"LLM judges: {'on' if include_judges else 'off'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and browse the case catalog
# MAGIC
# MAGIC The managed dataset is the single runtime source. Each row must use scalar `turn_N_*`
# MAGIC expectation fields and carry a category tag. The table below turns those fields into readable
# MAGIC prose before anything is executed.

# COMMAND ----------

import importlib

import pandas as pd
import mlflow
import uuid

import memory_eval


memory_eval = importlib.reload(memory_eval)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
memory_eval.configure(
    app_url=app_url,
    memory_store=memory_store,
    index_timeout_s=index_timeout_seconds,
)
print("Remote runner ready: each scenario sends its complete temporary eval scope")
permission_probe_scope = memory_eval._full_eval_scope(
    f"notebook-permission-check-{uuid.uuid4().hex[:8]}"
)
try:
    memory_eval.snapshot_scope(permission_probe_scope)
except Exception as exc:
    raise PermissionError(
        "The notebook identity cannot read the evaluation memory store. It needs "
        "READ_MEMORY_STORE and WRITE_MEMORY_STORE before scenarios can seed and clean up."
    ) from exc
print("Memory-store preflight: read access confirmed")

records = memory_eval.load_records_from_dataset(dataset_name)
source_label = dataset_name

for record in records:
    inputs = record["inputs"]
    expectations = record.get("expectations") or {}
    tags = record.get("tags") or {}
    scenario_id = inputs["scenario_id"]
    if "expected_behavior" in expectations:
        raise ValueError(
            f"{scenario_id} still uses legacy expected_behavior. Sync the updated dataset first."
        )
    expected_turns = memory_eval.parse_expectation_fields(expectations)
    if len(expected_turns) != len(inputs.get("turns", [])):
        raise ValueError(
            f"{scenario_id} has {len(inputs.get('turns', []))} turns but "
            f"{len(expected_turns)} expectation groups."
        )
    if tags.get("category") != inputs.get("category"):
        raise ValueError(
            f"{scenario_id} category tag {tags.get('category')!r} does not match "
            f"inputs.category {inputs.get('category')!r}."
        )

catalog_rows = []
for record in records:
    inputs = record["inputs"]
    turns = inputs.get("turns", [])
    seeds = inputs.get("initial_memories", [])
    catalog_rows.append(
        {
            "scenario_id": inputs["scenario_id"],
            "category": inputs["category"],
            "turns": len(turns),
            "new_sessions": sum(bool(turn.get("new_session")) for turn in turns),
            "seed_memories": len(seeds),
            "conversation": "\n".join(
                f"Turn {index}: {turn['message']}"
                + ("  [new session]" if turn.get("new_session") else "")
                for index, turn in enumerate(turns, start=1)
            ),
            "expected_behavior": memory_eval.format_expected_behavior(
                memory_eval.parse_expectation_fields(record.get("expectations"))
            ),
        }
    )

case_catalog = pd.DataFrame(catalog_rows).sort_values(["category", "scenario_id"])
print(
    f"Loaded {len(case_catalog)} scenarios across "
    f"{case_catalog['category'].nunique()} categories from {source_label}"
)
print("Validated flattened expectations and category tags")
display(case_catalog)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inspect the selected cases
# MAGIC
# MAGIC `scenario_id` accepts one id or a comma-separated list. It is combined with `category` when
# MAGIC both are set. Leave both blank to select every case. The default selects only
# MAGIC `explicit-save-coffee-preference`.

# COMMAND ----------

scenario_ids = set(selected_scenario_ids)
categories = {selected_category} if selected_category else set()
selected_records = memory_eval.filter_records(records, scenario_ids, categories)

selection_rows = []
for record in selected_records:
    inputs = record["inputs"]
    expected_turns = memory_eval.parse_expectation_fields(record["expectations"])
    for index, turn in enumerate(inputs["turns"], start=1):
        expectation = expected_turns[index - 1]
        selection_rows.append(
            {
                "scenario_id": inputs["scenario_id"],
                "category": inputs["category"],
                "turn": index,
                "new_session": bool(turn.get("new_session")),
                "message": turn["message"],
                "write": expectation.get("write", ""),
                "search": expectation.get("search", ""),
                "search_result_should_include": "; ".join(
                    expectation.get("search_should_contain", [])
                ),
                "memory_should": expectation.get("memory_should", ""),
                "answer_should": expectation.get("answer_should", ""),
            }
        )

print(f"Selected {len(selected_records)} scenario(s)")
display(pd.DataFrame(selection_rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run the evaluation
# MAGIC
# MAGIC Every prediction uses a fresh direct long-term-memory eval scope. Turns run sequentially, `new_session`
# MAGIC rotates the thread id, search-index readiness is polled between dependent turns, and the scope
# MAGIC is deleted in a `finally` block. MLflow logs one parent scenario trace with child turn spans.

# COMMAND ----------

scorers = list(memory_eval.DETERMINISTIC_SCORERS)
judge_names = "none"
if include_judges:
    scorers.extend(memory_eval.load_registered_judges())
    judge_names = "memory_write_quality,memory_answer_quality"

scenario_ids_for_tags = [
    str(record["inputs"]["scenario_id"]) for record in selected_records
]
categories_for_tags = sorted(
    {str(record["inputs"]["category"]) for record in selected_records}
)
run_name = f"memory-eval/notebook/{memory_eval._run_label(selected_records)}"
run_tags = {
    "suite": "managed-memory-v2",
    "runner": "databricks-shared-remote-app",
    "case_source": source_label,
    "app_url": app_url,
    "memory_scope_mode": "direct-eval-scope",
    "judges_enabled": str(include_judges).lower(),
    "judge_names": judge_names,
    "scenario_count": str(len(selected_records)),
    "scenario_ids": ",".join(scenario_ids_for_tags),
    "categories": ",".join(categories_for_tags),
    "index_timeout_seconds": str(index_timeout_seconds),
}

print(f"Running {len(selected_records)} scenario(s)")
print(f"Scorers: {', '.join(scorer.name for scorer in scorers)}")

with mlflow.start_run(run_name=run_name, tags=run_tags):
    result = mlflow.genai.evaluate(
        data=selected_records,
        predict_fn=memory_eval.predict_fn,
        scorers=scorers,
    )

client = mlflow.MlflowClient()
memory_eval.log_result_breakdown(result, client)

workspace_host = memory_eval._ws().config.host.rstrip("/")
run_url = (
    f"{workspace_host}/ml/experiments/{memory_eval.EXPERIMENT_ID}/evaluation-runs"
    f"?selectedRunUuid={result.run_id}"
)
print(f"\nMLflow run: {result.run_id}")
print(run_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Review scores and rationales
# MAGIC
# MAGIC The first table is intentionally compact. The second table contains the detailed scorer
# MAGIC rationales; expand it only when investigating a failure.

# COMMAND ----------

result_frame = result.result_df.copy()
active_scorer_names = [scorer.name for scorer in scorers]
score_columns = [
    f"{name}/value"
    for name in active_scorer_names
    if f"{name}/value" in result_frame.columns
]
rationale_columns = [
    f"{name}/rationale"
    for name in active_scorer_names
    if f"{name}/rationale" in result_frame.columns
]

summary_rows = []
for _, row in result_frame.iterrows():
    scenario_id, category = memory_eval._result_identity(row)
    summary = {"scenario_id": scenario_id, "category": category}
    for column in score_columns:
        value = row.get(column)
        summary[column.removesuffix("/value")] = (
            None if pd.isna(value) else ("PASS" if bool(value) else "FAIL")
        )
    summary_rows.append(summary)

score_summary = pd.DataFrame(summary_rows)
display(score_summary)

rationale_view = result_frame[["request", *rationale_columns]]
display(rationale_view)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Inspect the logged traces
# MAGIC
# MAGIC One trace corresponds to one scenario. Assessments include the deterministic scores, optional
# MAGIC LLM-judge scores, and the case's scalar per-turn expectation fields.

# COMMAND ----------

traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
trace_rows = []
assessment_rows = []

for trace in traces:
    root = trace.data.spans[0]
    root_error = next(
        (
            event.attributes.get("exception.message", "")
            for event in root.events
            if event.name == "exception"
        ),
        "",
    )
    trace_rows.append(
        {
            "trace_id": trace.info.trace_id,
            "state": str(trace.info.state),
            "request_preview": trace.info.request_preview,
            "response_preview": trace.info.response_preview,
            "input_fields": ", ".join(sorted(root.inputs or {})),
            "span_count": len(trace.data.spans),
            "error": root_error,
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
                # Keep one display column type across scorer and expectation values.
                "value": "" if assessment_value is None else str(assessment_value),
                "source": (value.get("source") or {}).get("source_type"),
                "rationale": value.get("rationale", ""),
            }
        )

trace_summary = pd.DataFrame(trace_rows)
display(trace_summary)
display(pd.DataFrame(assessment_rows))

if (trace_summary["state"] == "ERROR").any():
    print(
        "WARNING: One or more predictions ended in an execution error. "
        "Treat their zero scores as infrastructure/runtime failures and inspect the error column."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rerun checklist
# MAGIC
# MAGIC - Change widgets at the top, then rerun from **Configure the run** downward.
# MAGIC - Keep judges off while debugging tool behavior; enable them for final semantic grading.
# MAGIC - Leave both case filters blank only when you intend to run the complete suite.
# MAGIC - Use the printed MLflow link for durable results. Clearing notebook outputs does not delete runs.
# MAGIC - Scenario memory scopes are temporary and are cleaned automatically, including after failures.

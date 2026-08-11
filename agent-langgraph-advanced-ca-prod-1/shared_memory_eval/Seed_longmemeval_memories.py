# Databricks notebook source
# MAGIC %md
# MAGIC # Seed LongMemEval-S raw-session memories
# MAGIC
# MAGIC This notebook creates or validates the 30 persistent LongMemEval pilot scopes. It is the only
# MAGIC LongMemEval notebook that writes memory. The sibling evaluation notebook is read-only.
# MAGIC
# MAGIC Start with the default single case. Clear `question_id` only when you are ready to seed all 30.
# MAGIC Durable count and fingerprint checks always cover every entry. The default `gold` search check
# MAGIC probes only the gold sessions instead of serially probing all 1,444 memories.

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
# MAGIC ## 1. Configure the fixture operation

# COMMAND ----------

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve


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
_widget_text(
    "source_path",
    "/local_disk0/tmp/longmemeval_s_cleaned.json",
    "Pinned source path",
)
_widget_text("question_id", "8fb83627", "Question id (blank for all 30)")
_widget_dropdown("action", "seed", ["seed", "validate", "replace"], "Action")
_widget_dropdown("search_check", "gold", ["none", "gold", "all"], "Search check")
_widget_text("index_timeout_seconds", "600", "Search-index timeout")
_widget_text(
    "replace_confirmation",
    "",
    "For replace only: type the selected question id, or ALL",
)

app_url = dbutils.widgets.get("app_url").strip()
memory_store = dbutils.widgets.get("memory_store").strip()
question_dataset = dbutils.widgets.get("question_dataset").strip()
source_path_text = dbutils.widgets.get("source_path").strip()
source_path = Path(source_path_text)
question_id = dbutils.widgets.get("question_id").strip()
action = dbutils.widgets.get("action").strip()
search_check = dbutils.widgets.get("search_check").strip()
index_timeout_seconds = int(dbutils.widgets.get("index_timeout_seconds"))
replace_confirmation = dbutils.widgets.get("replace_confirmation").strip()

if not app_url or not memory_store or not question_dataset or not source_path_text:
    raise ValueError(
        "app_url, memory_store, question_dataset, and source_path are required"
    )
expected_confirmation = question_id or "ALL"
if action == "replace" and replace_confirmation != expected_confirmation:
    raise ValueError(
        f"replace_confirmation must exactly equal {expected_confirmation!r}"
    )

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

print(f"Action: {action}")
print(f"Question selection: {question_id or 'all 30'}")
print(f"Search-index check: {search_check}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the pinned source and compact MLflow dataset
# MAGIC
# MAGIC If the pinned source is not already at `source_path`, this cell downloads the exact committed
# MAGIC file. Its SHA-256 is verified before any memory write.

# COMMAND ----------

import mlflow
import pandas as pd

import longmemeval_adapter
import longmemeval_eval
import memory_eval


if not source_path.exists():
    source_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pinned source to {source_path} ...")
    download_path = source_path.with_name(source_path.name + ".download")
    urlretrieve(longmemeval_adapter.SOURCE_URL, download_path)
    longmemeval_adapter.load_source(download_path)
    download_path.replace(source_path)

source = longmemeval_adapter.load_source(source_path)
bundle = longmemeval_adapter.build_pilot_bundle(source)
longmemeval_adapter.validate_official_pilot_bundle(bundle)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
memory_eval.configure(
    app_url=app_url,
    memory_store=memory_store,
    index_timeout_s=index_timeout_seconds,
)
questions = longmemeval_eval.load_question_records(question_dataset)
selected = [
    row
    for row in questions
    if not question_id or row["inputs"]["question_id"] == question_id
]
if not selected:
    raise ValueError(f"Unknown LongMemEval question id: {question_id}")

corpus_by_case = {}
for row in bundle["corpus"]:
    inputs = row["inputs"]
    corpus_by_case.setdefault(inputs["case_id"], []).append(
        {
            "path": inputs["path"],
            "description": inputs["description"],
            "contents": inputs["contents"],
        }
    )

catalog = pd.DataFrame(
    [
        {
            "question_id": row["inputs"]["question_id"],
            "category": row["inputs"]["question_type"],
            "memories": int(row["inputs"]["expected_memory_count"]),
            "gold_sessions": len(longmemeval_eval._relevant_ids(row["expectations"])),
            "question": row["inputs"]["question"],
        }
        for row in selected
    ]
)
display(catalog)
print(
    f"Selected {len(selected)} cases with "
    f"{sum(int(row['inputs']['expected_memory_count']) for row in selected)} memories."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validate, seed, or deliberately replace the selected scopes
# MAGIC
# MAGIC Rerunning `seed` is a no-op for an exact scope and refuses a non-empty drifted scope. `replace`
# MAGIC is destructive and requires the confirmation widget above.

# COMMAND ----------

summaries = []
for index, row in enumerate(selected, start=1):
    inputs = row["inputs"]
    case_id = inputs["case_id"]
    entries = corpus_by_case[case_id]
    expected_count = int(inputs["expected_memory_count"])
    expected_fingerprint = inputs["expected_corpus_fingerprint"]
    actual_fingerprint = longmemeval_adapter.corpus_fingerprint(entries)
    if len(entries) != expected_count or actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"Pinned source for {inputs['question_id']} does not match its MLflow row"
        )

    relevant_ids = set(longmemeval_eval._relevant_ids(row["expectations"]))
    if search_check == "none":
        search_check_entries = []
    elif search_check == "all":
        search_check_entries = entries
    else:
        search_check_entries = [
            entry
            for entry in entries
            if longmemeval_adapter.memory_id_from_path(entry["path"]) in relevant_ids
        ]
        if len(search_check_entries) != len(relevant_ids):
            raise RuntimeError(f"Could not resolve every gold entry for {inputs['question_id']}")

    print(f"[{index}/{len(selected)}] {inputs['question_id']}: {len(entries)} entries ...")
    if action == "validate":
        status = longmemeval_eval.validate_persistent_case(
            inputs["scope_selector"], entries
        )
        result_action = "validated"
    else:
        status = longmemeval_eval.seed_persistent_case(
            inputs["scope_selector"],
            entries,
            replace=action == "replace",
            index_timeout_s=index_timeout_seconds,
            search_check_entries=search_check_entries,
        )
        result_action = status["action"]

    summary = {
        "question_id": inputs["question_id"],
        "category": inputs["question_type"],
        "action": result_action,
        "entries": status["actual_count"],
        "search_checks": (
            0 if action == "validate" else status["search_check_count"]
        ),
        "scope": status["full_scope"],
        "fingerprint": status["actual_fingerprint"],
    }
    summaries.append(summary)
    print(json.dumps(summary))

display(pd.DataFrame(summaries))
print(
    f"Complete: {len(summaries)} cases, "
    f"{sum(summary['entries'] for summary in summaries)} exact durable entries."
)

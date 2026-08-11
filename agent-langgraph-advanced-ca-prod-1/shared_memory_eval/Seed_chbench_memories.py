# Databricks notebook source
# MAGIC %md
# MAGIC # Seed CH-Bench memories
# MAGIC
# MAGIC Run this once per evaluator identity to create the frozen 44-entry corpus. Rerunning `seed`
# MAGIC is safe when the corpus is already exact. `replace` is the only action that can overwrite a
# MAGIC drifted scope and requires typing the exact scope selector as confirmation.

# COMMAND ----------

# MAGIC %pip install \
# MAGIC   "databricks-sdk==0.106.0" \
# MAGIC   "databricks-agents==1.10.0" \
# MAGIC   "mlflow==3.11.1" \
# MAGIC   "requests==2.32.5"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

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
    "corpus_dataset",
    "kevinyan.default.chbench_contextheavy_memories_5a08786a",
    "Memory corpus MLflow dataset",
)
_widget_text(
    "eval_scope",
    "chbench-contextheavy-5a08786a",
    "CH-Bench scope selector",
)
_widget_text("index_timeout_seconds", "600", "Search-index timeout")
_widget_dropdown("action", "seed", ["seed", "validate", "replace"], "Action")
_widget_text(
    "replace_confirmation",
    "",
    "For replace only: type the exact scope selector",
)

app_url = dbutils.widgets.get("app_url").strip()
memory_store = dbutils.widgets.get("memory_store").strip()
corpus_dataset = dbutils.widgets.get("corpus_dataset").strip()
eval_scope = dbutils.widgets.get("eval_scope").strip()
index_timeout_seconds = int(dbutils.widgets.get("index_timeout_seconds"))
action = dbutils.widgets.get("action").strip()
replace_confirmation = dbutils.widgets.get("replace_confirmation").strip()

if not app_url or not memory_store or not corpus_dataset or not eval_scope:
    raise ValueError("app_url, memory_store, corpus_dataset, and eval_scope are required")
if action == "replace" and replace_confirmation != eval_scope:
    raise ValueError("replace_confirmation must exactly match eval_scope")

notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
workspace_notebook_path = (
    notebook_path if notebook_path.startswith("/Workspace/") else "/Workspace" + notebook_path
)
notebook_root = str(Path(workspace_notebook_path).parent)
required_paths = [Path(notebook_root) / "memory_eval.py", Path(notebook_root) / "chbench_eval.py"]
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Copy chbench_eval.py and memory_eval.py beside this notebook. Missing: "
        + ", ".join(missing_paths)
    )
if notebook_root not in sys.path:
    sys.path.insert(0, notebook_root)

# COMMAND ----------

import mlflow

import chbench_eval
import memory_eval


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=memory_eval.EXPERIMENT_ID)
memory_eval.configure(
    app_url=app_url,
    memory_store=memory_store,
    index_timeout_s=index_timeout_seconds,
)
seed_memories = chbench_eval.load_seed_memories_from_dataset(corpus_dataset)

if action == "validate":
    status = chbench_eval.inspect_persistent_entries(eval_scope, seed_memories)
    if not status["exact"]:
        raise RuntimeError(
            chbench_eval.persistent_corpus_error(status)
            + " Choose action=seed if empty, or action=replace with confirmation to repair drift."
        )
    result_action = "validated"
else:
    status = chbench_eval.seed_persistent_entries(
        eval_scope,
        seed_memories,
        replace=action == "replace",
        index_timeout_s=index_timeout_seconds,
    )
    result_action = status["action"]

print(f"Action: {result_action}")
print(f"Corpus dataset: {corpus_dataset}")
print(f"Memory scope: {status['full_scope']}")
print(f"Entries: {status['actual_count']}")
print(f"Fingerprint: {status['actual_fingerprint']}")

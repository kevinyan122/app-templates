"""Two focused MLflow judges for the managed-memory evaluation suite.

The conditional scorer wrappers skip scenarios that do not contain the relevant
case-specific expectation. Run this file to register stable judge definitions on
the memory experiment after human calibration:

    uv run python memory_eval_judges.py
"""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

import mlflow  # noqa: E402
from mlflow.genai.judges import make_judge  # noqa: E402
from mlflow.genai.scorers import (  # noqa: E402
    ScorerSamplingConfig,
    get_scorer,
    scorer,
)


EXPERIMENT_ID = "2879499460172053"
JUDGE_MODEL = os.getenv("MEMORY_EVAL_JUDGE_MODEL", "databricks:/databricks-gpt-5-2")
JUDGE_RETRY_ATTEMPTS = 3
JUDGE_RETRY_INTERVAL_S = 2


WRITE_JUDGE_INSTRUCTIONS = """You are evaluating only the semantic quality of durable-memory
writes in a multi-turn scenario.

The scenario input is {{ inputs }}. The observed execution evidence is {{ outputs }}. The per-turn
requirements are scalar fields in {{ expectations }}, such as turn_1_memory_should.

Evaluate only turns whose expectation contains memory_should. For each such turn, compare the user
message and memory_should requirement with the actual created or updated entry in that turn's
store_diff.

Pass only when all applicable writes:
- faithfully store the durable facts the user actually stated;
- preserve relevant existing details during an update unless the user changed them;
- remove or supersede information that is no longer current;
- do not invent facts, certainty, preferences, or decisions;
- keep description as a compact label and put the bulk of multi-fact detail in contents;
- produce one coherent memory rather than an overgeneralized or fragmented representation.

Do not grade whether a tool was called, whether search happened first, path wording, or general
assistant-answer quality; deterministic scorers handle those. Semantic paraphrases are acceptable.
Return true only if every applicable memory_should requirement is satisfied. In the rationale,
identify the failing turn and the concrete missing, invented, stale, or misplaced information.
"""


ANSWER_JUDGE_INSTRUCTIONS = """You are evaluating only whether durable memory meaningfully improved
the assistant's final answer in a multi-turn scenario.

The scenario input is {{ inputs }}. The observed execution evidence is {{ outputs }}. The per-turn
requirements are scalar fields in {{ expectations }}, such as turn_2_answer_should.

Evaluate only turns whose expectation contains answer_should. For each such turn, inspect the actual
search_memory tool output and the final answer, then apply that turn's answer_should requirement.

Pass only when all applicable answers:
- use the relevant retrieved facts accurately and in the correct relationship;
- let those facts materially shape the answer rather than merely mentioning them;
- do not mix up nearby people, preferences, projects, old values, or distractor memories;
- do not claim a deleted, superseded, temporary, or absent fact is current;
- satisfy the case-specific answer_should requirement without inventing unsupported personal facts.

Do not grade search-query wording, tool-call policy, prose style, or unrelated general answer quality;
deterministic scorers handle tool behavior and retrieval. Semantic paraphrases are acceptable.
Return true only if every applicable answer_should requirement is satisfied. In the rationale,
identify the failing turn and how the retrieved memory was ignored, distorted, or misapplied.
"""


def _write_judge():
    return make_judge(
        name="memory_write_quality",
        model=JUDGE_MODEL,
        instructions=WRITE_JUDGE_INSTRUCTIONS,
        feedback_value_type=bool,
    )


def _answer_judge():
    return make_judge(
        name="memory_answer_quality",
        model=JUDGE_MODEL,
        instructions=ANSWER_JUDGE_INSTRUCTIONS,
        feedback_value_type=bool,
    )


def _expectation_turns(expectations: dict[str, Any] | None) -> list[dict[str, Any]]:
    from eval_memory import parse_expectation_fields

    return parse_expectation_fields(expectations)


def _has_requirement(expectations: dict[str, Any] | None, field: str) -> bool:
    return any(turn.get(field) for turn in _expectation_turns(expectations))


def _invoke_judge_with_retry(judge, *, inputs, outputs, expectations):
    """Retry transient judge endpoint failures without changing judge semantics."""

    for attempt in range(1, JUDGE_RETRY_ATTEMPTS + 1):
        try:
            return judge(inputs=inputs, outputs=outputs, expectations=expectations)
        except Exception:
            if attempt == JUDGE_RETRY_ATTEMPTS:
                raise
            time.sleep(JUDGE_RETRY_INTERVAL_S * attempt)


def memory_write_quality():
    """Return a scorer that invokes the write judge only for relevant scenarios."""

    judge = _write_judge()

    @scorer(name="memory_write_quality")
    def conditional_write_judge(inputs, outputs, expectations):
        if not _has_requirement(expectations, "memory_should"):
            return None
        return _invoke_judge_with_retry(
            judge,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
        )

    return conditional_write_judge


def memory_answer_quality():
    """Return a scorer that invokes the answer judge only for relevant scenarios."""

    judge = _answer_judge()

    @scorer(name="memory_answer_quality")
    def conditional_answer_judge(inputs, outputs, expectations):
        if not _has_requirement(expectations, "answer_should"):
            return None
        return _invoke_judge_with_retry(
            judge,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
        )

    return conditional_answer_judge


def register_judges() -> None:
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
    for judge in (_write_judge(), _answer_judge()):
        try:
            managed = judge.register(experiment_id=EXPERIMENT_ID)
            action = "registered"
        except ValueError as exc:
            if "already been registered" not in str(exc):
                raise
            existing = get_scorer(name=judge.name, experiment_id=EXPERIMENT_ID)
            managed = judge.update(
                name=judge.name,
                experiment_id=EXPERIMENT_ID,
                sampling_config=ScorerSamplingConfig(
                    sample_rate=existing.sample_rate,
                    filter_string=existing.filter_string,
                ),
            )
            action = "updated"
        print(f"{action} {judge.name}: {managed}")


if __name__ == "__main__":
    register_judges()

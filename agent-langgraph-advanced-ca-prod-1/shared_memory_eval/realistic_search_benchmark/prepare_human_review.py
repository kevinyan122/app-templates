"""Create a readable, independent human-review packet for the curated pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CURATED = Path(__file__).with_name("artifacts") / "pilot_enriched.json"
DEFAULT_PACKET = Path(__file__).with_name("artifacts") / "pilot_human_review.md"
DEFAULT_LABELS = Path(__file__).with_name("artifacts") / "pilot_human_labels.csv"

# Five high-value cases spanning different categories and ambiguity patterns. These receive a second,
# independent label after the first reviewer has completed all 25 cases.
DOUBLE_LABEL_CASE_IDS = (
    "people-sasha-comms-training-partner",
    "preference-poppy-long-walk-morning",
    "project-coverage-pilot-expansion",
    "decision-kb-freeze-window",
    "global-branch-leads-channel",
)

LABEL_FIELDS = (
    "case_id",
    "category",
    "review_pass",
    "reviewer",
    "query_natural",
    "gold_supported",
    "no_equally_sufficient_alternative",
    "answer_should_correct",
    "overall_decision",
    "notes",
)


def prepare_human_review(
    curated_path: Path,
    packet_path: Path,
    labels_path: Path,
) -> dict[str, Any]:
    """Write the Markdown packet and blank CSV label sheet."""

    artifact = json.loads(curated_path.read_text())
    cases = artifact["cases"]
    corpus = artifact["corpus"]
    corpus_by_id = {memory["memory_id"]: memory for memory in corpus}
    case_ids = {case["case_id"] for case in cases}

    if len(cases) != 25 or len(case_ids) != 25:
        raise ValueError("Human review packet requires exactly 25 unique cases")
    missing_double_labels = set(DOUBLE_LABEL_CASE_IDS) - case_ids
    if missing_double_labels:
        raise ValueError(f"Missing double-label cases: {sorted(missing_double_labels)}")
    for case in cases:
        referenced_ids = [case["gold_memory_id"], *case["context_memory_ids"]]
        missing_ids = set(referenced_ids) - set(corpus_by_id)
        if missing_ids:
            raise ValueError(
                f"Case {case['case_id']} references missing memories: {sorted(missing_ids)}"
            )

    repairs_by_case = {
        repair["case_id"]: repair for repair in artifact.get("curation", {}).get("repairs", [])
    }
    packet = _render_packet(
        artifact,
        cases,
        corpus_by_id,
        repairs_by_case,
        curated_path,
        labels_path,
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(packet)

    label_rows = _label_rows(cases)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(label_rows)

    return {
        "packet": str(packet_path),
        "labels": str(labels_path),
        "cases": len(cases),
        "primary_label_rows": len(cases),
        "double_label_rows": len(DOUBLE_LABEL_CASE_IDS),
    }


def _render_packet(
    artifact: dict[str, Any],
    cases: list[dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
    repairs_by_case: dict[str, dict[str, Any]],
    curated_path: Path,
    labels_path: Path,
) -> str:
    lines = [
        "# Realistic memory-search pilot: human review",
        "",
        f"Source: `{curated_path.name}`  ",
        f"Label sheet: `{labels_path.name}`  ",
        f"Cases: {len(cases)}  ",
        f"Corpus memories: {len(corpus_by_id)}  ",
        f"Benchmark as of: `{artifact.get('benchmark_as_of_date', 'unspecified')}`  ",
        f"Source generated at: `{artifact.get('generated_at', 'unknown')}`",
        "",
        "## How to review",
        "",
        "Review all 25 cases without opening `pilot_enriched_reviews.json`. For each case, decide:",
        "",
        "1. Does the query sound natural and have one clear interpretation?",
        "2. Does the gold memory alone support every material part of `answer_should`?",
        "3. Are all nearby memories insufficient to produce the same complete answer?",
        "4. Is `answer_should` correct, concise, and neither missing facts nor demanding extra facts?",
        "",
        "Record `yes` or `no` for those four fields in the CSV, then set `overall_decision` to",
        "`approve` or `needs_changes`. Add a concrete note whenever a field is `no`. Partial overlap is",
        "allowed: a distractor invalidates the gold only if it independently supports the complete",
        "expected answer.",
        "",
        "The five cases marked **Second label required** must be reviewed independently by another",
        "person using the `calibration` rows in the CSV. Compare labels only after both reviewers finish.",
        "",
        "## Calibration subset",
        "",
        *[f"- `{case_id}`" for case_id in DOUBLE_LABEL_CASE_IDS],
        "",
        "---",
        "",
    ]
    profiles = artifact.get("enrichment", {}).get("profile_by_memory_id", {})

    for number, case in enumerate(cases, start=1):
        gold = corpus_by_id[case["gold_memory_id"]]
        contexts = [corpus_by_id[memory_id] for memory_id in case["context_memory_ids"]]
        repair = repairs_by_case.get(case["case_id"])
        second_label = "**Yes**" if case["case_id"] in DOUBLE_LABEL_CASE_IDS else "No"

        lines.extend(
            [
                f"## Case {number}: `{case['case_id']}`",
                "",
                f"- Category: `{case['category']}`",
                f"- Second label required: {second_label}",
                f"- Curator-repaired: {'Yes' if repair else 'No'}",
                f"- Gold content profile: `{profiles.get(case['gold_memory_id'], 'unspecified')}`",
                f"- Secondary tags: {_code_list(case.get('secondary_tags', []))}",
                "",
                "### Situation",
                "",
                case["situation"],
                "",
                "### Query",
                "",
                f"> {case['query']}",
                "",
                "### Expected answer",
                "",
                f"> {case['answer_should']}",
                "",
                "### Gold memory",
                "",
                *_memory_lines(gold, profiles.get(gold["memory_id"])),
                "",
                "### Why this is intended as gold",
                "",
                case["why_gold"],
                "",
                "<details>",
                f"<summary>Closest distractors ({len(contexts)})</summary>",
                "",
            ]
        )
        for memory in contexts:
            lines.extend(
                [*_memory_lines(memory, profiles.get(memory["memory_id"])), ""]
            )
        lines.extend(["</details>", ""])

        if repair:
            lines.extend(
                [
                    "<details>",
                    "<summary>Original generated query</summary>",
                    "",
                    f"> {repair['original_query']}",
                    "",
                    "</details>",
                    "",
                ]
            )

        lines.extend(
            [
                "### Human decision",
                "",
                "- [ ] Query is natural and unambiguous",
                "- [ ] Gold alone supports the complete expected answer",
                "- [ ] No distractor is equally sufficient",
                "- [ ] Expected answer is correct and appropriately scoped",
                "- [ ] **Approve**",
                "- [ ] **Needs changes**",
                "",
                "Notes:",
                "",
                "<!-- Add specific concerns or proposed corrections here. -->",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _memory_lines(memory: dict[str, Any], profile: str | None = None) -> list[str]:
    lines = [
        f"**`{memory['memory_id']}` — {memory['description']}**",
        "",
        f"Category: `{memory['category']}`  ",
    ]
    if profile:
        lines.append(f"Content profile: `{profile}`  ")
    contents = memory["contents"].strip()
    lines.append(
        f"Contents: {contents}"
        if contents
        else "Contents: _Fact fully captured by description._"
    )
    return lines


def _code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "None"


def _label_rows(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for case in cases:
        rows.append(_blank_label_row(case, "primary"))
    for case_id in DOUBLE_LABEL_CASE_IDS:
        case = next(case for case in cases if case["case_id"] == case_id)
        rows.append(_blank_label_row(case, "calibration"))
    return rows


def _blank_label_row(case: dict[str, Any], review_pass: str) -> dict[str, str]:
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "review_pass": review_pass,
        "reviewer": "",
        "query_natural": "",
        "gold_supported": "",
        "no_equally_sufficient_alternative": "",
        "answer_should_correct": "",
        "overall_decision": "",
        "notes": "",
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    return parser.parse_args()


def main() -> None:
    args = _args()
    summary = prepare_human_review(args.curated, args.packet, args.labels)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

from shared_memory_eval.realistic_search_benchmark.prepare_human_review import (
    DOUBLE_LABEL_CASE_IDS,
    prepare_human_review,
)


_ARTIFACTS = (
    Path(__file__).parent
    / "shared_memory_eval"
    / "realistic_search_benchmark"
    / "artifacts"
)


def test_prepares_readable_packet_and_label_sheet(tmp_path):
    packet_path = tmp_path / "review.md"
    labels_path = tmp_path / "labels.csv"

    summary = prepare_human_review(
        _ARTIFACTS / "pilot_enriched.json",
        packet_path,
        labels_path,
    )

    packet = packet_path.read_text()
    with labels_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["cases"] == 25
    assert packet.count("\n## Case ") == 25
    assert "pilot_enriched_reviews.json" in packet
    assert "without opening" in packet
    assert "Closest distractors" in packet
    assert "Gold content profile: `episodic`" in packet
    assert "Benchmark as of: `2026-06-05`" in packet
    assert "Fact fully captured by description" in packet
    assert len(rows) == 30
    assert sum(row["review_pass"] == "primary" for row in rows) == 25
    assert sum(row["review_pass"] == "calibration" for row in rows) == 5
    assert {
        row["case_id"] for row in rows if row["review_pass"] == "calibration"
    } == set(DOUBLE_LABEL_CASE_IDS)
    assert all(not row["overall_decision"] for row in rows)

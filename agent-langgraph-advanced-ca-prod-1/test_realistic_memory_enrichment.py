from collections import Counter
import json
from pathlib import Path

from shared_memory_eval.realistic_search_benchmark.enrich_memories import (
    EPISODIC_MEMORY_IDS,
    MEDIUM_MEMORY_IDS,
    PROFILE_EPISODIC,
    PROFILE_MEDIUM,
    PROFILE_QUOTAS,
    PROFILE_SHORT,
    PILOT_AS_OF_DATE,
    SHORT_MEMORY_IDS,
    apply_rewrites,
    validate_enriched_artifact,
    validate_rewrite_collection,
)


_ARTIFACTS = (
    Path(__file__).parent
    / "shared_memory_eval"
    / "realistic_search_benchmark"
    / "artifacts"
)


def _inputs():
    curated = json.loads((_ARTIFACTS / "pilot_curated.json").read_text())
    world = json.loads((_ARTIFACTS / "pilot_world.json").read_text())
    rewrites = json.loads((_ARTIFACTS / "pilot_memory_rewrites.json").read_text())
    return curated, world, rewrites


def test_profile_partition_has_exact_pilot_quotas():
    assert not (SHORT_MEMORY_IDS & MEDIUM_MEMORY_IDS)
    assert not (SHORT_MEMORY_IDS & EPISODIC_MEMORY_IDS)
    assert not (MEDIUM_MEMORY_IDS & EPISODIC_MEMORY_IDS)
    assert Counter(
        {
            PROFILE_SHORT: len(SHORT_MEMORY_IDS),
            PROFILE_MEDIUM: len(MEDIUM_MEMORY_IDS),
            PROFILE_EPISODIC: len(EPISODIC_MEMORY_IDS),
        }
    ) == Counter(PROFILE_QUOTAS)


def test_applies_rewrites_to_corpus_and_every_bundle():
    curated, world, rewrites = _inputs()

    validate_rewrite_collection(rewrites["rewrites"], curated)
    enriched = apply_rewrites(curated, world, rewrites)
    validate_enriched_artifact(enriched)

    corpus_by_id = {memory["memory_id"]: memory for memory in enriched["corpus"]}
    assert enriched["benchmark_as_of_date"] == PILOT_AS_OF_DATE
    assert enriched["summary"]["episodic_gold_cases"] == 7
    assert all(
        not memory["contents"]
        for memory in enriched["corpus"]
        if memory["memory_id"] in SHORT_MEMORY_IDS
    )
    for bundle in enriched["bundles"]:
        for memory in bundle["memories"]:
            assert memory == corpus_by_id[memory["memory_id"]]


def test_episodic_gold_descriptions_do_not_reveal_complete_answers():
    enriched = json.loads((_ARTIFACTS / "pilot_enriched.json").read_text())
    corpus = {memory["memory_id"]: memory for memory in enriched["corpus"]}

    assert "hybrid" not in corpus["mem-0004"]["description"].casefold()
    assert "mara" not in corpus["mem-0015"]["description"].casefold()
    assert "brightgrove hq" not in corpus["mem-0042"]["description"].casefold()
    assert "pine notes" not in corpus["mem-0048"]["description"].casefold()

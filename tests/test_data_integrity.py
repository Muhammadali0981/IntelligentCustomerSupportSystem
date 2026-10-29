"""Leakage and consistency checks on prepared and synthetic data.

Run after scripts/prepare_data.py and scripts/generate_synthetic.py:  py -3.11 -m pytest tests -q
Tests that need generated files are skipped if those files do not exist yet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.retrieval.chunking import chunk_kb_dir, slugify

ROOT = Path(__file__).resolve().parent.parent
CLF = ROOT / "data" / "processed" / "classification"
SYN = ROOT / "data" / "synthetic"
TICKETS = SYN / "tickets"


def _tickets(split: str) -> list[dict]:
    path = TICKETS / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_slugify() -> None:
    assert slugify("EU and UK right of withdrawal") == "eu-and-uk-right-of-withdrawal"
    assert slugify("Matching a competitor's price") == "matching-a-competitor-s-price"


def test_kb_chunk_ids_unique() -> None:
    ids = [c.chunk_id for c in chunk_kb_dir(SYN / "kb")]
    assert len(ids) == len(set(ids))


def test_classification_splits_disjoint() -> None:
    if not CLF.exists():
        pytest.skip("classification data not prepared yet")
    splits = {s: pd.read_parquet(CLF / f"{s}.parquet") for s in ("train", "dev", "test")}
    texts = {s: set(df["text"]) for s, df in splits.items()}
    assert not texts["train"] & texts["test"]
    assert not texts["train"] & texts["dev"]
    assert not texts["dev"] & texts["test"]
    ids = pd.concat(splits.values())["id"]
    assert ids.is_unique


def test_tobi_input_has_no_answer_column() -> None:
    if not CLF.exists():
        pytest.skip("classification data not prepared yet")
    df = pd.read_parquet(CLF / "train.parquet")
    assert "answer" not in df.columns


def test_synthetic_scenarios_do_not_cross_splits() -> None:
    scen = {s: {t["scenario_id"] for t in _tickets(s)} for s in ("train", "dev", "test")}
    assert not scen["train"] & scen["test"]
    assert not scen["train"] & scen["dev"]
    assert not scen["dev"] & scen["test"]


def test_test_tickets_not_in_known_issues_or_other_splits() -> None:
    test_bodies = {t["body"] for t in _tickets("test")}
    other = {t["body"] for s in ("train", "dev") for t in _tickets(s)}
    assert not test_bodies & other
    known_text = (SYN / "known_issues.jsonl").read_text(encoding="utf-8")
    assert not any(b in known_text for b in test_bodies)


def test_relevant_chunks_exist() -> None:
    chunk_ids = {c.chunk_id for c in chunk_kb_dir(SYN / "kb")}
    for split in ("train", "dev", "test"):
        for t in _tickets(split):
            assert set(t["relevant_chunk_ids"]) <= chunk_ids, t["ticket_id"]


def test_frozen_test_hash_matches() -> None:
    path = TICKETS / "test.jsonl"
    if not path.exists():
        pytest.skip("not generated yet")
    expected = (TICKETS / "test.sha256").read_text().strip()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

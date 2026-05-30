"""
test_data.py — sanity checks on corpus loaders and Dataset classes.

Run with:  .venv/bin/python -m pytest models/epistemic/tests/ -v
"""

import pytest
import torch
from transformers import AutoTokenizer

from models.epistemic.data import (
    ASSERTED, HEDGED, SPECULATIVE,
    SentDataset, SentExample,
    TokenDataset, TokenExample,
    load_factbank_ldc,
    load_szeged_wiki,
)
from models.epistemic.train import doc_level_split

RAW = "/mldata/ece283-sentiment-analyzer/epistemic/raw"
WIKI_PATH = f"{RAW}/uncertainty/wiki.xml"
FB_LDC_ANN = f"{RAW}/factbank_ldc/data/annotation"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def wiki_data():
    return load_szeged_wiki(WIKI_PATH)


@pytest.fixture(scope="module")
def fb_ldc_data():
    return load_factbank_ldc(FB_LDC_ANN)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base")


# ── wiki.xml ──────────────────────────────────────────────────────────────────

def test_wiki_sentence_count(wiki_data):
    sent_ex, _ = wiki_data
    assert len(sent_ex) == pytest.approx(20748, abs=50), \
        "wiki.xml sentence count changed unexpectedly"


def test_wiki_label_distribution(wiki_data):
    sent_ex, _ = wiki_data
    from collections import Counter
    dist = Counter(e.label for e in sent_ex)
    total = len(sent_ex)
    assert dist[ASSERTED] / total > 0.85, "asserted should dominate (>85%)"
    assert dist[HEDGED] / total > 0.02,   "hedged should be >2% (modal filter active)"
    assert dist[SPECULATIVE] > 0,         "speculative class must not be empty"


def test_wiki_all_labels_valid(wiki_data):
    sent_ex, _ = wiki_data
    assert all(e.label in (ASSERTED, HEDGED, SPECULATIVE) for e in sent_ex)


def test_wiki_doc_ids_nonempty(wiki_data):
    sent_ex, tok_ex = wiki_data
    assert all(e.doc_id for e in sent_ex), "every SentExample must have a doc_id"
    assert all(e.doc_id for e in tok_ex),  "every TokenExample must have a doc_id"


def test_wiki_token_cue_spans_valid(wiki_data):
    _, tok_ex = wiki_data
    for ex in tok_ex:
        for start, end in ex.cue_spans:
            assert start >= 0
            assert end > start
            assert end <= len(ex.text), \
                f"cue span ({start},{end}) out of bounds for text len {len(ex.text)}"


# ── FactBank LDC ──────────────────────────────────────────────────────────────

def test_factbank_ldc_loads(fb_ldc_data):
    sent_ex, tok_ex = fb_ldc_data
    assert len(sent_ex) > 2000, "expected >2000 annotated sentences"
    assert len(tok_ex)  > 0,    "expected some hedge token examples"


def test_factbank_ldc_all_labels_valid(fb_ldc_data):
    sent_ex, _ = fb_ldc_data
    assert all(e.label in (ASSERTED, HEDGED, SPECULATIVE) for e in sent_ex)


def test_factbank_ldc_doc_ids_nonempty(fb_ldc_data):
    sent_ex, tok_ex = fb_ldc_data
    assert all(e.doc_id for e in sent_ex)
    assert all(e.doc_id for e in tok_ex)


# ── Document-level split ──────────────────────────────────────────────────────

def test_doc_level_split_no_leakage(wiki_data):
    sent_ex, _ = wiki_data
    train, val, test = doc_level_split(sent_ex, 0.80, 0.10, seed=42)

    train_docs = {e.doc_id for e in train}
    val_docs   = {e.doc_id for e in val}
    test_docs  = {e.doc_id for e in test}

    assert train_docs.isdisjoint(val_docs),  "train/val doc overlap"
    assert train_docs.isdisjoint(test_docs), "train/test doc overlap"
    assert val_docs.isdisjoint(test_docs),   "val/test doc overlap"


def test_doc_level_split_fractions(wiki_data):
    sent_ex, _ = wiki_data
    train, val, test = doc_level_split(sent_ex, 0.80, 0.10, seed=42)
    total = len(sent_ex)
    assert len(train) / total == pytest.approx(0.80, abs=0.05)
    assert len(val)   / total == pytest.approx(0.10, abs=0.05)


# ── Dataset classes ───────────────────────────────────────────────────────────

def test_sent_dataset_length(wiki_data, tokenizer):
    sent_ex, _ = wiki_data
    sample = sent_ex[:100]
    ds = SentDataset(sample, tokenizer, max_len=64)
    assert len(ds) == 100


def test_sent_dataset_item_shapes(wiki_data, tokenizer):
    sent_ex, _ = wiki_data
    ds = SentDataset(sent_ex[:10], tokenizer, max_len=64)
    item = ds[0]
    assert item["input_ids"].shape      == (64,)
    assert item["attention_mask"].shape == (64,)
    assert item["sent_label"].dtype     == torch.long


def test_sent_dataset_empty_guard(tokenizer):
    ds = SentDataset([], tokenizer, max_len=64)
    assert len(ds) == 0


def test_token_dataset_length(wiki_data, tokenizer):
    _, tok_ex = wiki_data
    sample = tok_ex[:50]
    ds = TokenDataset(sample, tokenizer, max_len=64)
    assert len(ds) == 50


def test_token_dataset_item_shapes(wiki_data, tokenizer):
    _, tok_ex = wiki_data
    ds   = TokenDataset(tok_ex[:10], tokenizer, max_len=64)
    item = ds[0]
    assert item["input_ids"].shape      == (64,)
    assert item["token_labels"].shape   == (64,)


def test_token_dataset_labels_valid(wiki_data, tokenizer):
    _, tok_ex = wiki_data
    ds = TokenDataset(tok_ex[:50], tokenizer, max_len=64)
    for i in range(len(ds)):
        lbls = ds[i]["token_labels"]
        # Each token label must be 0, 1, or -100 (ignore)
        assert torch.all((lbls == 0) | (lbls == 1) | (lbls == -100)), \
            f"unexpected token label value in example {i}"


def test_token_dataset_empty_guard(tokenizer):
    ds = TokenDataset([], tokenizer, max_len=64)
    assert len(ds) == 0

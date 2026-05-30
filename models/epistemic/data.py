"""
data.py — corpus loaders and Dataset classes for the epistemic certainty model.

Sentence head sources (news domain only):
    wiki.xml          Szeged XML   ~20k sentences
    factbank.xml      Szeged XML   ~3k sentences  (same NYT docs as LDC FactBank;
                                                   kept separate for ablation)
    factbank_ldc/     Pipe-delimited tables        ~2.8k annotated sentences

Token head sources:
    wiki.xml          Szeged XML   (news, full weight)
    factbank.xml      Szeged XML   (news, full weight)
    bio_bmc/fly/hbc   Szeged XML   (biomedical, down-weight in train.py)
    bioscope/         BioScope XML (biomedical, down-weight in train.py)
    factbank_ldc/     Pipe-delimited tables — PR+/PR- event tokens as cues

Label schema (sentence head):
    0 = asserted    no qualifying cue;  CT+/CT- in FactBank
    1 = hedged      speculation_modal_probable_ cue;  PR+/PR- in FactBank
    2 = speculative hypo_doxastic word from SPECULATIVE_WORDS;  PS+/PS-/Uu in FactBank

Aggregation (sentence label when multiple events present): most uncertain wins
    speculative (2) > hedged (1) > asserted (0)

Token labels (binary per subword token):
    1 = hedge cue token
    0 = not a hedge cue
    -100 = special / padding token (ignored by loss)

FactBank LDC token note: only sentences with ≥1 AUTHOR PR+/PR- event are included
as token examples. Their event word is marked as the cue; all other tokens are 0.
Sentences with only CT/PS/Uu events are excluded from token examples to avoid
labeling non-cue tokens as 0 when the cue might simply be unannotated.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ── Label constants ───────────────────────────────────────────────────────────

ASSERTED    = 0
HEDGED      = 1
SPECULATIVE = 2
LABEL_NAMES = ["asserted", "hedged", "speculative"]

# Confirmed salvageable subset of hypo_doxastic (LLM validation study)
SPECULATIVE_WORDS = frozenset({
    "allegedly", "reportedly", "supposedly",
    "ostensibly", "purportedly", "apparently",
})

# FactBank AUTHOR fact value → 3-class label
_FB_LABEL = {
    "CT+": ASSERTED, "CT-": ASSERTED,
    "PR+": HEDGED,   "PR-": HEDGED,
    "PS+": SPECULATIVE, "PS-": SPECULATIVE,
    "Uu":  SPECULATIVE,
    # NA and CTu → absent from dict; callers skip None results
}

_PRIORITY = {ASSERTED: 0, HEDGED: 1, SPECULATIVE: 2}


# ── Shared data structures ────────────────────────────────────────────────────

@dataclass
class SentExample:
    text:   str
    label:  int   # ASSERTED / HEDGED / SPECULATIVE
    source: str
    doc_id: str = ""  # document-level ID used for train/val/test splitting

@dataclass
class TokenExample:
    text:      str
    cue_spans: list   # [(char_start, char_end), ...] of hedge cue tokens
    source:    str
    doc_id: str = ""  # document-level ID used for train/val/test splitting


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_label(current: int, candidate: int) -> int:
    return candidate if _PRIORITY[candidate] > _PRIORITY[current] else current


# Counterfactual/epistemic modals that distinguish genuinely speculative
# hypo_condition sentences from causal-rule conditionals.
# "will" and "may" excluded: "will" is usually asserted prediction;
# "may" is often deontic permission ("may arrest if...").
# Validated against LLM agreement study: would/could/might recover ~7/8
# confirmed speculative conditionals while misclassifying 0 confirmed asserted ones.
_EPISTEMIC_MODAL_RE = re.compile(r"\b(would|could|might)\b", re.IGNORECASE)

# Bare modal auxiliaries tagged modal_probable are ambiguous: they can express
# deontic permission ("companies may merge"), generic capacity ("water can freeze"),
# or genuine epistemic hedging ("this may cause harm").  Require a corroborating
# epistemic frame word elsewhere in the sentence before accepting them as HEDGED.
_MODAL_AUXILIARIES = frozenset({
    "may", "can", "could", "might", "will", "would", "shall", "should",
})

# Epistemic frame words: if any appear in the sentence, a bare modal is likely epistemic.
# Deliberately excludes the modal auxiliaries themselves to avoid circular matching.
_EPISTEMIC_FRAME_RE = re.compile(
    r"\b(probably|possibly|perhaps|apparently|seemingly|likely|unlikely|"
    r"appear|appears|seem|seems|suggest|suggests|think|believe|estimate|"
    r"expect|uncertain|unclear|unsure|speculate|contend|allege|claim)\b",
    re.IGNORECASE,
)


def _szeged_cue_label(cue_type: str, cue_text: str, full_text: str = ""):
    """Map a Szeged <ccue type=...> to a label, or None if not in our schema."""
    t = cue_type.lower()
    if "modal_probable" in t:
        w = cue_text.strip().lower()
        if w in _MODAL_AUXILIARIES:
            # Bare modal: only hedged when an epistemic frame word co-occurs.
            # Without context it is more likely deontic/generic — skip the cue.
            return HEDGED if _EPISTEMIC_FRAME_RE.search(full_text) else None
        return HEDGED   # epistemic adverb/verb/noun → unambiguously hedged
    if "hypo_doxastic" in t and cue_text.strip().lower() in SPECULATIVE_WORDS:
        return SPECULATIVE
    # hypo_condition: only speculative when the sentence contains a counterfactual
    # or epistemic modal in the consequent clause.  Pure causal-rule conditionals
    # ("if a crime is committed", "if the salt is long enough") stay asserted.
    if "hypo_condition" in t and _EPISTEMIC_MODAL_RE.search(full_text):
        return SPECULATIVE
    return None


def _extract_szeged_sentence(elem):
    """
    Walk a Szeged <Sentence> element and return (text, label, cue_spans).

    Two-pass: first rebuild the full text, then score each ccue with
    access to that text (needed for the hypo_condition modal check).
    cue_spans will be empty for asserted sentences (no qualifying ccue).
    """
    # Pass 1 — reconstruct text and collect raw ccue positions
    parts = []
    pos   = 0
    ccues = []   # (char_start, char_end, cue_type, cue_text)

    if elem.text:
        parts.append(elem.text)
        pos += len(elem.text)

    for child in elem:
        cue_text = child.text or ""
        if child.tag == "ccue":
            ccues.append((pos, pos + len(cue_text), child.get("type", ""), cue_text))
        parts.append(cue_text)
        pos += len(cue_text)
        if child.tail:
            parts.append(child.tail)
            pos += len(child.tail)

    full_text = "".join(parts).strip()

    # Pass 2 — score each ccue now that full_text is available
    label     = ASSERTED
    cue_spans = []

    for start, end, cue_type, cue_text in ccues:
        cue_lbl = _szeged_cue_label(cue_type, cue_text, full_text)
        if cue_lbl is not None:
            cue_spans.append((start, end))
            label = _merge_label(label, cue_lbl)

    return full_text, label, cue_spans


def _walk_bioscope(node, pos, cue_spans, parts):
    """
    Recursively walk a BioScope element, collecting speculation cue spans.
    Returns updated pos. Handles nested <xcope> elements.
    """
    if node.text:
        parts.append(node.text)
        pos += len(node.text)
    for child in node:
        if child.tag == "cue" and child.get("type") == "speculation":
            cue_text = child.text or ""
            cue_spans.append((pos, pos + len(cue_text)))
            parts.append(cue_text)
            pos += len(cue_text)
            # cue elements have no children in practice; skip inner recursion
        else:
            pos = _walk_bioscope(child, pos, cue_spans, parts)
        if child.tail:
            parts.append(child.tail)
            pos += len(child.tail)
    return pos


def _parse_fb_row(line: str) -> list:
    """Split a FactBank pipe-delimited row; strip surrounding single quotes."""
    return [f.strip().strip("'") for f in line.strip().split("|||")]


# ── Szeged XML loaders ────────────────────────────────────────────────────────

def _parse_szeged(path, sent_head: bool, token_head: bool):
    """
    Internal parser for any Szeged-format XML file.
    Returns (sent_examples, token_examples).

    All sentences are included in token_examples (not just those with cues),
    so the token head sees genuine negative examples.
    """
    path   = Path(path)
    source = path.stem
    root   = ET.parse(path).getroot()

    sent_examples  = []
    token_examples = []

    for doc in root.iter("Document"):
        doc_id_elem = doc.find("DocID")
        doc_id = doc_id_elem.text.strip() if doc_id_elem is not None else source

        for sentence in doc.iter("Sentence"):
            text, label, cue_spans = _extract_szeged_sentence(sentence)
            if not text:
                continue
            if sent_head:
                sent_examples.append(
                    SentExample(text=text, label=label, source=source, doc_id=doc_id)
                )
            if token_head:
                token_examples.append(
                    TokenExample(text=text, cue_spans=cue_spans, source=source, doc_id=doc_id)
                )

    return sent_examples, token_examples


def load_szeged_wiki(path) -> tuple:
    """Returns (sent_examples, token_examples) from uncertainty/wiki.xml."""
    return _parse_szeged(path, sent_head=True, token_head=True)


def load_szeged_factbank(path) -> tuple:
    """Returns (sent_examples, token_examples) from uncertainty/factbank.xml."""
    return _parse_szeged(path, sent_head=True, token_head=True)


def load_szeged_bio(path) -> list:
    """
    Returns token_examples only from a Szeged bio XML (bio_bmc/fly/hbc.xml).
    Excluded from sentence head: biomedical text is out-of-domain for news.
    """
    _, token_examples = _parse_szeged(path, sent_head=False, token_head=True)
    return token_examples


# ── BioScope XML loader ───────────────────────────────────────────────────────

def load_bioscope(path) -> list:
    """
    Parse BioScope-format XML (bioscope/abstracts.xml, full_papers.xml).
    Returns token_examples only (biomedical; excluded from sentence head).

    BioScope uses <sentence> (lowercase), with hedge scopes marked as
    <xcope><cue type="speculation">word</cue> scope text</xcope>.
    All sentences included, including those without speculation cues.
    """
    path   = Path(path)
    source = path.stem
    root   = ET.parse(path).getroot()

    token_examples = []
    for doc in root.iter("Document"):
        doc_id_elem = doc.find("DocID")
        doc_id = doc_id_elem.text.strip() if doc_id_elem is not None else source

        for sentence in doc.iter("sentence"):   # lowercase tag in BioScope
            parts     = []
            cue_spans = []
            _walk_bioscope(sentence, 0, cue_spans, parts)
            text = "".join(parts).strip()
            if not text:
                continue
            token_examples.append(
                TokenExample(text=text, cue_spans=cue_spans, source=source, doc_id=doc_id)
            )

    return token_examples


# ── FactBank LDC loader ───────────────────────────────────────────────────────

def load_factbank_ldc(ann_dir) -> tuple:
    """
    Parse FactBank LDC pipe-delimited annotation tables.

    Format: fields separated by '|||', strings single-quoted.
    Tables used: sentences.txt, fb_factValue.txt, tokens_tml.txt

    Returns (sent_examples, token_examples).

    Sentence label: most uncertain AUTHOR (source_id='s0') event per sentence.
    Sentences with no AUTHOR annotations are dropped (no ground truth).

    Token examples: sentences with ≥1 AUTHOR PR+/PR- event. Only those event
    words are marked as cues (cue_spans); other tokens implicitly receive 0.
    Other-label sentences are excluded from token examples because their
    non-cue tokens cannot be reliably labeled negative.
    """
    ann_dir = Path(ann_dir)

    # ── sentences.txt: (fname, sent_idx) → text ──────────────────────────────
    sentences = {}
    with open(ann_dir / "sentences.txt", encoding="utf-8") as f:
        for line in f:
            row = _parse_fb_row(line)
            if len(row) < 3:
                continue
            fname, sent_idx, text = row[0], row[1], row[2]
            if sent_idx == "0":
                # sent_idx 0 holds the document ID string, not a real sentence
                continue
            sentences[(fname, sent_idx)] = text

    # ── fb_factValue.txt: aggregate AUTHOR labels per sentence ────────────────
    # sent_labels: most uncertain label seen so far
    # hedge_event_words: event words from PR+/PR- annotations (for token head)
    sent_labels       = {}   # key → int
    hedge_event_words = {}   # key → [str, ...]

    with open(ann_dir / "fb_factValue.txt", encoding="utf-8") as f:
        for line in f:
            row = _parse_fb_row(line)
            if len(row) < 9:
                continue
            fname, sent_idx = row[0], row[1]
            source_id, event_word, fact_val = row[5], row[6], row[8]

            if source_id != "s0":
                continue   # only AUTHOR perspective

            label = _FB_LABEL.get(fact_val)
            if label is None:
                continue   # NA or CTu — not useful

            key = (fname, sent_idx)
            sent_labels[key] = _merge_label(sent_labels.get(key, ASSERTED), label)

            if label == HEDGED:
                hedge_event_words.setdefault(key, []).append(event_word)

    # ── Build sentence examples ───────────────────────────────────────────────
    sent_examples = []
    for key, label in sent_labels.items():
        text = sentences.get(key)
        if not text:
            continue
        sent_examples.append(
            SentExample(text=text, label=label, source="factbank_ldc", doc_id=key[0])
        )

    # ── Build token examples from hedge events ────────────────────────────────
    # Locate each PR+/PR- event word in the sentence via word-boundary regex.
    # This is approximate (first occurrence wins) but acceptable given the
    # low count (~114 PR+/PR- events) and the supplementary role of this data.
    token_examples = []
    for key, event_words in hedge_event_words.items():
        text = sentences.get(key)
        if not text:
            continue
        cue_spans = []
        seen = set()
        for word in event_words:
            if word in seen:
                continue
            seen.add(word)
            m = re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE)
            if m:
                cue_spans.append((m.start(), m.end()))
        if cue_spans:
            token_examples.append(
                TokenExample(text=text, cue_spans=cue_spans, source="factbank_ldc", doc_id=key[0])
            )

    return sent_examples, token_examples


# ── PyTorch Datasets ──────────────────────────────────────────────────────────

class SentDataset(Dataset):
    """
    Sentence-level dataset for the sentence classification head.

    Tokenizes all examples at construction time (fast for ~25k sentences).
    Each item: {input_ids, attention_mask, sent_label}.
    """

    def __init__(self, examples: list, tokenizer, max_len: int = 128):
        if not examples:
            self.input_ids      = torch.zeros(0, max_len, dtype=torch.long)
            self.attention_mask = torch.zeros(0, max_len, dtype=torch.long)
            self.labels         = torch.zeros(0, dtype=torch.long)
            return

        texts  = [ex.text  for ex in examples]
        labels = [ex.label for ex in examples]

        enc = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "sent_label":     self.labels[idx],
        }


class TokenDataset(Dataset):
    """
    Token-level dataset for the hedge-cue detection head.

    Aligns character-span cue annotations to RoBERTa subword tokens via
    offset_mapping. Special tokens and padding receive label -100 (loss-ignored).
    Each item: {input_ids, attention_mask, token_labels}.
    """

    def __init__(self, examples: list, tokenizer, max_len: int = 128):
        if not examples:
            self.input_ids      = torch.zeros(0, max_len, dtype=torch.long)
            self.attention_mask = torch.zeros(0, max_len, dtype=torch.long)
            self.token_labels   = torch.zeros(0, max_len, dtype=torch.long)
            return

        texts      = [ex.text      for ex in examples]
        spans_list = [ex.cue_spans for ex in examples]

        enc = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        offset_mapping      = enc["offset_mapping"]   # (N, max_len, 2)

        N = len(texts)
        token_labels = torch.full((N, max_len), -100, dtype=torch.long)

        for i in range(N):
            spans = spans_list[i]
            for j in range(max_len):
                tok_start, tok_end = offset_mapping[i, j].tolist()
                if tok_start == 0 and tok_end == 0:
                    # Special token ([CLS], [SEP]) or padding — leave as -100
                    continue
                is_cue = any(
                    tok_start < s_end and tok_end > s_start
                    for s_start, s_end in spans
                )
                token_labels[i, j] = 1 if is_cue else 0

        self.token_labels = token_labels

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_labels":   self.token_labels[idx],
        }

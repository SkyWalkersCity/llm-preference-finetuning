"""Data loading, prompt construction and A/B-swap augmentation.

Task: given a user prompt and two chatbot responses, predict which one a human
preferred: class 0 = response A, 1 = response B, 2 = tie.

The Kaggle / Hugging Face CSVs store `prompt`, `response_a` and `response_b` as
JSON-encoded lists of turns (and a few cells contain `null`), so parsing is
defensive.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

LABEL_NAMES = ["a", "b", "tie"]
WINNER_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
SWAP_LABEL = {0: 1, 1: 0, 2: 2}

HEAD_TEXT = "Which response would a human prefer? Compare the two responses to the prompt.\n\n### Prompt:\n"
SEP_A_TEXT = "\n\n### Response A:\n"
SEP_B_TEXT = "\n\n### Response B:\n"
TAIL_TEXT = "\n\n### Verdict:"


def parse_turns(value) -> List[str]:
    """Turn a cell (JSON list string, list, str or NaN) into a list of strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return [""]
    if isinstance(value, list):
        return [t if isinstance(t, str) else "" for t in value] or [""]
    text = str(value)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return [text]
    if parsed is None:
        return [""]
    if isinstance(parsed, list):
        return [t if isinstance(t, str) else "" for t in parsed] or [""]
    if isinstance(parsed, str):
        return [parsed]
    return [text]


def join_turns(turns: Sequence[str]) -> str:
    return "\n".join(turns)


def load_frame(path: str) -> pd.DataFrame:
    """Read a CSV and add `prompt_text`, `resp_a_text`, `resp_b_text` and (if present) `label`."""
    df = pd.read_csv(path)
    return prepare_frame(df)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prompt_text"] = df["prompt"].map(lambda v: join_turns(parse_turns(v)))
    df["resp_a_text"] = df["response_a"].map(lambda v: join_turns(parse_turns(v)))
    df["resp_b_text"] = df["response_b"].map(lambda v: join_turns(parse_turns(v)))
    if all(c in df.columns for c in WINNER_COLS):
        winners = df[WINNER_COLS].to_numpy()
        if not (winners.sum(axis=1) == 1).all():
            raise ValueError("Each row must have exactly one winner_* column set to 1")
        df["label"] = winners.argmax(axis=1).astype(int)
    return df


def group_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    """Split into train/val so that rows sharing the same prompt never straddle the split."""
    keys = df["prompt_text"].map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    unique = np.array(sorted(keys.unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_keys = set(unique[:n_val])
    is_val = keys.isin(val_keys).to_numpy()
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def swap_label(label: int) -> int:
    return SWAP_LABEL[int(label)]


def _truncate(ids: List[int], budget: int, keep: str) -> List[int]:
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    return ids[-budget:] if keep == "tail" else ids[:budget]


def encode_example(tokenizer, prompt: str, resp_a: str, resp_b: str, max_len: int,
                   prompt_frac: float = 0.25) -> List[int]:
    """Build the model input: header + prompt + response A + response B + verdict marker.

    The token budget is split so that the three variable parts fit in `max_len`:
    the prompt gets at most `prompt_frac` of it (its most recent part is kept) and the
    two responses share the rest evenly, with unused space handed to the longer one
    (the start of each response is kept). Markers are never truncated, so the final
    token is always the verdict marker that the classification head reads.
    """
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    head, sep_a, sep_b, tail = enc(HEAD_TEXT), enc(SEP_A_TEXT), enc(SEP_B_TEXT), enc(TAIL_TEXT)
    fixed = len(head) + len(sep_a) + len(sep_b) + len(tail)
    body = max(max_len - fixed, 3)
    p, a, b = enc(prompt), enc(resp_a), enc(resp_b)

    p_budget = min(len(p), int(body * prompt_frac))
    rest = body - p_budget
    half = rest // 2
    a_budget, b_budget = min(len(a), half), min(len(b), half)
    spare = rest - a_budget - b_budget
    if len(a) > a_budget:
        extra = min(spare, len(a) - a_budget)
        a_budget += extra
        spare -= extra
    if len(b) > b_budget:
        b_budget += min(spare, len(b) - b_budget)

    ids = (head + _truncate(p, p_budget, "tail") + sep_a + _truncate(a, a_budget, "head")
           + sep_b + _truncate(b, b_budget, "head") + tail)
    return ids[-max_len:] if len(ids) > max_len else ids


class PreferenceDataset:
    """Torch-style dataset yielding tokenised examples (optionally with random A/B swapping)."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, swap_prob: float = 0.0,
                 seed: int = 0, deterministic_swap: bool = False):
        self.prompts = df["prompt_text"].tolist()
        self.a = df["resp_a_text"].tolist()
        self.b = df["resp_b_text"].tolist()
        self.labels = df["label"].tolist() if "label" in df.columns else None
        self.tokenizer, self.max_len = tokenizer, max_len
        self.swap_prob, self.deterministic_swap = swap_prob, deterministic_swap
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.prompts)

    def _do_swap(self, idx: int) -> bool:
        if self.swap_prob <= 0:
            return False
        if self.deterministic_swap:
            return random.Random(idx).random() < self.swap_prob
        return self._rng.random() < self.swap_prob

    def __getitem__(self, idx: int) -> dict:
        a, b = self.a[idx], self.b[idx]
        label = self.labels[idx] if self.labels is not None else None
        if self._do_swap(idx):
            a, b = b, a
            label = swap_label(label) if label is not None else None
        item = {"input_ids": encode_example(self.tokenizer, self.prompts[idx], a, b, self.max_len)}
        if label is not None:
            item["labels"] = int(label)
        return item


def make_collator(pad_token_id: int):
    """Right-pad a list of examples into tensors."""
    import torch

    def collate(batch: List[dict]) -> dict:
        longest = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
            mask[i, :n] = 1
        out = {"input_ids": ids, "attention_mask": mask}
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return out

    return collate

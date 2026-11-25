"""Cheap, honest baselines to put the LLM's numbers in context.

1. Class prior: always predict the training label frequencies.
2. Length/style logistic regression: hand-crafted features of each response (length, lists, code,
   headers, refusals ...) fed to a multinomial logistic regression.
3. Same features with gradient-boosted trees.

Human voters in Chatbot Arena are known to favour longer, more structured answers, so these
features alone beat the prior; a fine-tuned LLM should beat all of them.
"""
from __future__ import annotations

import re
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import metrics as M

_REFUSAL = re.compile(r"\b(i'm sorry|i am sorry|i cannot|i can't|as an ai|i apologize|i'm unable|i am unable)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_HEADER = re.compile(r"^\s*#{1,6}\s", re.M)


def response_features(text: str) -> np.ndarray:
    text = text or ""
    n_chars = len(text)
    n_words = len(text.split())
    return np.array([
        np.log1p(n_chars),
        np.log1p(n_words),
        np.log1p(text.count("\n")),
        len(_BULLET.findall(text)),
        len(_HEADER.findall(text)),
        text.count("```") // 2,
        text.count("**") // 2,
        text.count("?"),
        1.0 if _REFUSAL.search(text) else 0.0,
        1.0 if text.strip().endswith((".", "!", "?", "```", ")")) else 0.0,
    ], dtype=np.float64)


def pair_features(df: pd.DataFrame, swap: bool = False) -> np.ndarray:
    a_col, b_col = ("resp_b_text", "resp_a_text") if swap else ("resp_a_text", "resp_b_text")
    fa = np.stack([response_features(t) for t in df[a_col]])
    fb = np.stack([response_features(t) for t in df[b_col]])
    plen = np.log1p(df["prompt_text"].str.len().to_numpy())[:, None]
    return np.hstack([fa - fb, np.abs(fa - fb), fa + fb, plen])


def _swapped_labels(y: np.ndarray) -> np.ndarray:
    return np.array([1, 0, 2])[y]


def _fit_predict(model, train_df, val_df) -> np.ndarray:
    """Train on original + A/B-swapped rows; predict as the average of both orderings."""
    ytr = train_df["label"].to_numpy()
    X = np.vstack([pair_features(train_df), pair_features(train_df, swap=True)])
    y = np.concatenate([ytr, _swapped_labels(ytr)])
    model.fit(X, y)
    p = model.predict_proba(pair_features(val_df))
    ps = model.predict_proba(pair_features(val_df, swap=True))[:, [1, 0, 2]]
    return (p + ps) / 2


def run_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int = 0) -> Dict[str, dict]:
    yv = val_df["label"].to_numpy()
    prior = np.bincount(train_df["label"], minlength=3) / len(train_df)
    out = {"class_prior": M.summarize(yv, np.tile(prior, (len(yv), 1)))}
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    out["length_style_logreg"] = M.summarize(yv, _fit_predict(lr, train_df, val_df))
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    out["length_style_gbdt"] = M.summarize(yv, _fit_predict(gb, train_df, val_df))
    return out

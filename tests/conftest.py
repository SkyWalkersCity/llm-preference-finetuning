import os

# Some conda installs ship two OpenMP runtimes (torch + scikit-learn), which can segfault when both are
# loaded in one process. These settings only affect the test process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import random

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def tokenizer():
    """Qwen2.5 tokenizer (a few MB). Skips the dependent tests when it cannot be fetched."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # offline / hub unavailable
        pytest.skip(f"tokenizer unavailable: {exc}")


def make_raw_frame(n=60, seed=0):
    """Synthetic data in the exact Kaggle/HF CSV format (JSON-list strings, one-hot winners)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        la, lb = rng.randint(3, 60), rng.randint(3, 60)
        winner = 0 if la > lb + 5 else 1 if lb > la + 5 else 2
        rows.append({
            "id": i, "model_a": "m1", "model_b": "m2",
            "prompt": json.dumps([f"question number {i % 20}"]),
            "response_a": json.dumps(["word " * la]),
            "response_b": json.dumps(["word " * lb]),
            "winner_model_a": int(winner == 0), "winner_model_b": int(winner == 1), "winner_tie": int(winner == 2),
        })
    return pd.DataFrame(rows)

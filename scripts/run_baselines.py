"""Compute the non-LLM baselines on the same group-aware split used for LLM training.

    python scripts/run_baselines.py --data data/train.csv --out results/baselines.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lmsys_pref import data as D  # noqa: E402
from lmsys_pref.baselines import run_baselines  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val_frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df = D.load_frame(args.data)
train_df, val_df = D.group_split(df, args.val_frac, args.seed)
res = run_baselines(train_df, val_df, seed=args.seed)
res["_split"] = {"n_train": len(train_df), "n_val": len(val_df), "seed": args.seed, "val_frac": args.val_frac}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

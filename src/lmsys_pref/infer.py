"""Inference with a trained LoRA adapter, producing a Kaggle-format submission.

    python -m lmsys_pref.infer --adapter outputs/qwen1.5b/adapter --test test.csv --out submission.csv

The test CSV needs `id`, `prompt`, `response_a`, `response_b`. Probabilities are averaged over both
A/B orderings (test-time augmentation) because the true label is symmetric under swapping.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from . import data as D
from . import model as MD
from .train import predict_probs


def load_adapter_model(adapter_dir: str, base_model: str, dtype: torch.dtype, load_in_4bit: bool = False):
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = MD.load_base_model(base_model, dtype, load_in_4bit=load_in_4bit)
    MD.set_pad_token(base, tokenizer)
    model = PeftModel.from_pretrained(base, adapter_dir)
    if torch.cuda.is_available() and not load_in_4bit:
        model = model.cuda()
    return model, tokenizer


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--base_model", help="defaults to the base model recorded next to the adapter")
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--no_tta", action="store_true")
    args = p.parse_args(argv)

    base_model = args.base_model
    if base_model is None:
        with open(f"{args.adapter}/adapter_config.json") as f:
            base_model = json.load(f)["base_model_name_or_path"]
    model, tok = load_adapter_model(args.adapter, base_model, MD.pick_dtype(args.dtype))
    df = D.prepare_frame(pd.read_csv(args.test))
    probs = predict_probs(model, tok, df, args.max_len, args.batch_size, swap=False)
    if not args.no_tta:
        probs = (probs + predict_probs(model, tok, df, args.max_len, args.batch_size, swap=True)) / 2
    sub = pd.DataFrame({"id": df["id"], "winner_model_a": probs[:, 0],
                        "winner_model_b": probs[:, 1], "winner_tie": probs[:, 2]})
    sub.to_csv(args.out, index=False)
    return sub


if __name__ == "__main__":
    main()

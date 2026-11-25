"""LoRA fine-tuning entry point.

    python -m lmsys_pref.train --data data/train.csv --preset qwen1.5b --output outputs/qwen1.5b

It trains on a group-aware train split, evaluates on a held-out split (no prompt appears in both),
and writes the LoRA adapter, validation probabilities (with and without A/B-swap averaging) and a
metrics.json that records the exact configuration used.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from . import data as D
from . import metrics as M
from . import model as MD

PRESETS: Dict[str, dict] = {
    # model id, per-device batch size, grad-accum steps, max tokens per example, learning rate
    "qwen0.5b": dict(model_name="Qwen/Qwen2.5-0.5B-Instruct", batch_size=8, grad_accum=4, max_len=1024, lr=2e-4),
    "qwen1.5b": dict(model_name="Qwen/Qwen2.5-1.5B-Instruct", batch_size=4, grad_accum=8, max_len=1024, lr=1e-4),
    "qwen3b": dict(model_name="Qwen/Qwen2.5-3B-Instruct", batch_size=2, grad_accum=16, max_len=1024, lr=1e-4),
    "gemma2-9b-4bit": dict(model_name="google/gemma-2-9b-it", batch_size=1, grad_accum=32, max_len=1024, lr=5e-5,
                           load_in_4bit=True),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CSV with prompt/response_a/response_b/winner_* columns")
    p.add_argument("--output", required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="qwen1.5b")
    p.add_argument("--model_name", help="override the preset's model id or local path")
    p.add_argument("--max_len", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--grad_accum", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_train_samples", type=int, default=None, help="subsample the training split")
    p.add_argument("--max_val_samples", type=int, default=None, help="subsample the held-out split used for the final evaluation")
    p.add_argument("--train_eval_samples", type=int, default=1000, help="size of the val subset scored during training (checkpoint selection)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--no_swap_aug", action="store_true", help="disable random A/B swapping while training")
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=-1, help="stop early after N optimizer steps (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p


def resolve(args) -> dict:
    cfg = dict(PRESETS[args.preset])
    for key in ("model_name", "max_len", "batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.lr is not None:
        cfg["lr"] = args.lr
    return cfg


def predict_probs(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int = 8,
                  swap: bool = False) -> np.ndarray:
    """Class probabilities [P(A), P(B), P(tie)] for each row. With swap=True, A and B are exchanged
    in the input and the output columns are exchanged back, so the result is still in the original frame."""
    work = df.copy()
    if swap:
        work["resp_a_text"], work["resp_b_text"] = df["resp_b_text"].values, df["resp_a_text"].values
    ds = D.PreferenceDataset(work.drop(columns=["label"], errors="ignore"), tokenizer, max_len)
    lengths = [len(ds[i]["input_ids"]) for i in range(len(ds))]
    order = np.argsort(lengths)
    collate = D.make_collator(tokenizer.pad_token_id)
    device = next(model.parameters()).device
    out = np.zeros((len(ds), 3), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = collate([ds[int(i)] for i in idx])
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            out[idx] = M.softmax(logits.float().cpu().numpy())
    if swap:
        out = out[:, [1, 0, 2]]
    return out


def main(argv=None) -> dict:
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    args = build_parser().parse_args(argv)
    cfg = resolve(args)
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    df = D.load_frame(args.data)
    train_df, val_df = D.group_split(df, args.val_frac, args.seed)
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed).reset_index(drop=True)
    if args.max_val_samples:
        val_df = val_df.sample(n=min(args.max_val_samples, len(val_df)), random_state=args.seed).reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)} model={cfg['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dtype = MD.pick_dtype(args.dtype)
    base = MD.load_base_model(cfg["model_name"], dtype, load_in_4bit=cfg.get("load_in_4bit", False))
    MD.set_pad_token(base, tokenizer)
    model = MD.wrap_with_lora(base, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                              gradient_checkpointing=not args.no_grad_ckpt,
                              quantized=cfg.get("load_in_4bit", False))
    params = MD.count_parameters(model)
    print(f"parameters: {params}")

    train_ds = D.PreferenceDataset(train_df, tokenizer, cfg["max_len"],
                                   swap_prob=0.0 if args.no_swap_aug else 0.5, seed=args.seed)
    val_ds = D.PreferenceDataset(val_df.iloc[:args.train_eval_samples], tokenizer, cfg["max_len"])
    collate = D.make_collator(tokenizer.pad_token_id)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return M.summarize(labels, M.softmax(np.asarray(logits, dtype=np.float64)))

    use_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.output, per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2, gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], num_train_epochs=args.epochs, max_steps=args.max_steps,
        lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        bf16=use_cuda and dtype == torch.bfloat16, fp16=use_cuda and dtype == torch.float16,
        logging_steps=args.logging_steps, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_log_loss", greater_is_better=False,
        report_to="none", remove_unused_columns=False, label_names=["labels"], seed=args.seed,
        dataloader_num_workers=2 if use_cuda else 0, optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate, compute_metrics=compute_metrics)

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    model.save_pretrained(os.path.join(args.output, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.output, "adapter"))

    # Final evaluation of the best checkpoint: plain, and averaged over both A/B orderings.
    eval_bs = cfg["batch_size"] * 2
    p_plain = predict_probs(model, tokenizer, val_df, cfg["max_len"], eval_bs, swap=False)
    p_swap = predict_probs(model, tokenizer, val_df, cfg["max_len"], eval_bs, swap=True)
    p_tta = (p_plain + p_swap) / 2
    y = val_df["label"].to_numpy()
    results = {
        "plain": M.summarize(y, p_plain),
        "swap_tta": M.summarize(y, p_tta),
        "prior_baseline_log_loss": M.log_loss(y, np.tile(np.bincount(train_df["label"], minlength=3) / len(train_df), (len(y), 1))),
        "config": {**cfg, **{k: getattr(args, k) for k in (
            "epochs", "max_train_samples", "val_frac", "lora_r", "lora_alpha", "lora_dropout",
            "warmup_ratio", "weight_decay", "label_smoothing", "no_swap_aug", "seed", "dtype")}},
        "n_train": int(len(train_df)), "n_val": int(len(val_df)), "parameters": params,
        "train_seconds": round(train_seconds, 1),
        "hardware": (torch.cuda.get_device_name(0) if use_cuda else platform.processor() or platform.machine()),
        "torch": torch.__version__,
    }
    pd.DataFrame({"id": val_df["id"], "label": y, "p_a": p_tta[:, 0], "p_b": p_tta[:, 1], "p_tie": p_tta[:, 2]}) \
        .to_csv(os.path.join(args.output, "val_predictions.csv"), index=False)
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("plain", "swap_tta", "prior_baseline_log_loss")}, indent=2))
    return results


if __name__ == "__main__":
    main()

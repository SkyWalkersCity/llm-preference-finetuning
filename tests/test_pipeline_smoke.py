"""End-to-end smoke test: real Trainer + LoRA + evaluation code on a tiny randomly initialised Qwen2 model."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw_frame


@pytest.fixture()
def tiny_model_dir(tmp_path, tokenizer):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    cfg = transformers.Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                   max_position_embeddings=512, num_labels=3)
    model = transformers.Qwen2ForSequenceClassification(cfg)
    path = tmp_path / "tiny"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def test_train_end_to_end_and_infer(tmp_path, tiny_model_dir):
    from lmsys_pref import infer, train
    csv = tmp_path / "train.csv"
    make_raw_frame(120).to_csv(csv, index=False)
    out = tmp_path / "out"
    res = train.main(["--data", str(csv), "--output", str(out), "--model_name", tiny_model_dir,
                      "--preset", "qwen0.5b", "--max_len", "96", "--batch_size", "4", "--grad_accum", "1",
                      "--max_steps", "6", "--eval_steps", "3", "--save_steps", "3", "--logging_steps", "1",
                      "--train_eval_samples", "16", "--lora_r", "4", "--dtype", "fp32", "--no_grad_ckpt"])
    assert (out / "adapter" / "adapter_config.json").exists()
    assert os.path.exists(out / "metrics.json")
    val = pd.read_csv(out / "val_predictions.csv")
    assert np.allclose(val[["p_a", "p_b", "p_tie"]].sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(res["swap_tta"]["log_loss"]) and 0.0 <= res["swap_tta"]["accuracy"] <= 1.0
    assert res["parameters"]["trainable"] < res["parameters"]["total"]

    test_csv = tmp_path / "test.csv"
    make_raw_frame(10, seed=5).drop(columns=["winner_model_a", "winner_model_b", "winner_tie"]).to_csv(test_csv, index=False)
    sub = infer.main(["--adapter", str(out / "adapter"), "--base_model", tiny_model_dir, "--test", str(test_csv),
                      "--out", str(tmp_path / "sub.csv"), "--max_len", "96", "--batch_size", "4", "--dtype", "fp32"])
    assert list(sub.columns) == ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert np.allclose(sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1), 1.0, atol=1e-6)

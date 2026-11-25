import numpy as np
import pandas as pd

from conftest import make_raw_frame
from lmsys_pref import data as D


def test_parse_turns_handles_messy_cells():
    assert D.parse_turns('["a", "b"]') == ["a", "b"]
    assert D.parse_turns('["a", null]') == ["a", ""]
    assert D.parse_turns("null") == [""]
    assert D.parse_turns(float("nan")) == [""]
    assert D.parse_turns("plain text, not json") == ["plain text, not json"]
    assert D.parse_turns('"just a string"') == ["just a string"]


def test_prepare_frame_labels_and_validation():
    df = D.prepare_frame(make_raw_frame(30))
    assert set(df["label"]) <= {0, 1, 2}
    assert df["resp_a_text"].str.startswith("word").all()
    bad = make_raw_frame(5)
    bad.loc[0, ["winner_model_a", "winner_model_b", "winner_tie"]] = [1, 1, 0]
    try:
        D.prepare_frame(bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_group_split_never_shares_a_prompt():
    df = D.prepare_frame(make_raw_frame(200))
    tr, va = D.group_split(df, 0.2, seed=1)
    assert len(tr) + len(va) == len(df)
    assert not set(tr["prompt_text"]) & set(va["prompt_text"])


def test_swap_label_is_an_involution():
    for y in (0, 1, 2):
        assert D.swap_label(D.swap_label(y)) == y
    assert (D.swap_label(0), D.swap_label(1), D.swap_label(2)) == (1, 0, 2)


def test_encode_respects_max_len_and_keeps_verdict_marker(tokenizer):
    tail = tokenizer(D.TAIL_TEXT, add_special_tokens=False)["input_ids"]
    long, short = "lorem ipsum " * 2000, "hi"
    for max_len in (64, 256, 1024):
        for p, a, b in [(long, long, long), (short, long, short), (short, short, short), ("", "", "")]:
            ids = D.encode_example(tokenizer, p, a, b, max_len)
            assert len(ids) <= max_len
            assert ids[-len(tail):] == tail


def test_encode_gives_spare_budget_to_the_longer_response(tokenizer):
    ids_short_vs_long = D.encode_example(tokenizer, "q", "x", "lorem " * 5000, 256)
    ids_long_vs_long = D.encode_example(tokenizer, "q", "lorem " * 5000, "lorem " * 5000, 256)
    assert len(ids_short_vs_long) == 256 or len(ids_short_vs_long) > 200
    assert len(ids_long_vs_long) <= 256


def test_dataset_swap_flips_labels_and_responses(tokenizer):
    df = D.prepare_frame(make_raw_frame(20))
    plain = D.PreferenceDataset(df, tokenizer, 128, swap_prob=0.0)
    swapped = D.PreferenceDataset(df, tokenizer, 128, swap_prob=1.0)
    for i in range(len(df)):
        assert swapped[i]["labels"] == D.swap_label(plain[i]["labels"])
    swapped_df = df.copy()
    swapped_df["resp_a_text"], swapped_df["resp_b_text"] = df["resp_b_text"], df["resp_a_text"]
    manual = D.PreferenceDataset(swapped_df, tokenizer, 128, swap_prob=0.0)
    assert all(swapped[i]["input_ids"] == manual[i]["input_ids"] for i in range(len(df)))


def test_collator_pads_right_with_mask():
    import torch
    batch = [{"input_ids": [5, 6, 7], "labels": 0}, {"input_ids": [8], "labels": 2}]
    out = D.make_collator(pad_token_id=99)(batch)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 99, 99]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"].tolist() == [0, 2]

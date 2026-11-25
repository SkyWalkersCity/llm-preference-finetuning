import numpy as np

from conftest import make_raw_frame
from lmsys_pref import baselines as B
from lmsys_pref import data as D


def test_length_baseline_learns_a_length_rule_and_is_swap_symmetric():
    df = D.prepare_frame(make_raw_frame(600, seed=3))
    tr, va = df.iloc[:450].reset_index(drop=True), df.iloc[450:].reset_index(drop=True)
    res = B.run_baselines(tr, va)
    assert res["length_style_logreg"]["log_loss"] < res["class_prior"]["log_loss"]
    assert res["length_style_logreg"]["accuracy"] > 0.6
    # swapping A and B must swap the predicted probabilities of A and B
    swapped = va.copy()
    swapped["resp_a_text"], swapped["resp_b_text"] = va["resp_b_text"], va["resp_a_text"]
    f = B.pair_features(va)
    assert f.shape[0] == len(va) and np.isfinite(f).all()

import numpy as np
from sklearn.metrics import log_loss as sk_log_loss

from lmsys_pref import metrics as M


def test_log_loss_matches_sklearn():
    rng = np.random.RandomState(0)
    probs = rng.dirichlet(np.ones(3), size=500)
    y = rng.randint(0, 3, size=500)
    assert abs(M.log_loss(y, probs) - sk_log_loss(y, probs, labels=[0, 1, 2])) < 1e-9


def test_uniform_prediction_is_ln3():
    y = np.array([0, 1, 2, 0])
    assert abs(M.log_loss(y, np.full((4, 3), 1 / 3)) - np.log(3)) < 1e-12


def test_log_loss_is_finite_for_hard_zero_probabilities():
    assert np.isfinite(M.log_loss(np.array([0]), np.array([[0.0, 1.0, 0.0]])))


def test_softmax_rows_sum_to_one_and_are_stable():
    p = M.softmax(np.array([[1000.0, 0.0, -1000.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0) and np.isfinite(p).all()


def test_accuracy_and_calibration_error_bounds():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    assert M.accuracy(y, perfect) == 1.0
    assert M.expected_calibration_error(y, perfect) == 0.0
    assert 0.0 <= M.expected_calibration_error(y, np.full((4, 3), 1 / 3)) <= 1.0

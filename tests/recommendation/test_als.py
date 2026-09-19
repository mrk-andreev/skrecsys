import numpy as np
import pytest
from sklearn.utils import check_random_state

from skrecsys.recommendation import AlternatingLeastSquares


def _ratings(seed=0, n_users=12, n_items=9, density=0.5):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), np.arange(n_users) % n_items] = True
    mask[np.arange(n_items) % n_users, np.arange(n_items)] = True
    users, items = np.nonzero(mask)
    X = np.column_stack([users, items])
    y = rng.integers(1, 6, len(users)).astype(float)
    return X, y


def _libfm_als_reference(X, y, n_factors, n_iter, init_stdev, reg0, reg_w, reg_v, seed):
    """Readable libFM ALS sweep (fm_learn_mcmc_simultaneous, no sampling) on one-hot data."""
    n_users, n_items = X[:, 0].max() + 1, X[:, 1].max() + 1
    rows = [[int(u), n_users + int(i)] for u, i in X]
    n_features = n_users + n_items
    columns = [[c for c, row in enumerate(rows) if j in row] for j in range(n_features)]
    rng = check_random_state(seed)
    v = rng.normal(0.0, init_stdev, (n_factors, n_features)).tolist()
    w = rng.normal(0.0, init_stdev, n_features).tolist()
    e = [_predict(0.0, w, v, row) - y[c] for c, row in enumerate(rows)]

    w0, history = 0.0, []
    for _ in range(n_iter):
        new = -sum(e_c - w0 for e_c in e) / (reg0 + len(e))
        e = [e_c + (new - w0) for e_c in e]
        w0 = new
        for j in range(n_features):
            _update_w(w, j, columns[j], e, reg_w)
        for v_f in v:
            q = [sum(v_f[j] for j in row) for row in rows]
            for j in range(n_features):
                _update_v(v_f, j, columns[j], e, q, reg_v)
        history.append(np.sqrt(np.mean(np.square(e))))
    return w0, np.array(w), np.array(v), np.array(history)


def _predict(w0, w, v, row):
    pred = w0 + sum(w[j] for j in row)
    for v_f in v:
        s = sq = 0.0
        for j in row:
            s += v_f[j]
            sq += v_f[j] * v_f[j]
        pred += 0.5 * (s * s - sq)
    return pred


def _update_w(w, j, column, e, reg):
    mean = sigma = 0.0
    for c in column:
        mean += e[c] - w[j]
        sigma += 1.0
    new = -mean / (sigma + reg)
    for c in column:
        e[c] += new - w[j]
    w[j] = new


def _update_v(v_f, j, column, e, q, reg):
    old, mean, sigma = v_f[j], 0.0, 0.0
    for c in column:
        h = q[c] - old
        mean += h * e[c]
        sigma += h * h
    mean -= old * sigma
    new = -mean / (sigma + reg)
    for c in column:
        h = q[c] - old
        e[c] += h * (new - old)
        q[c] += new - old
    v_f[j] = new


def test_matches_libfm_reference():
    X, y = _ratings()
    n_factors, n_iter, init_stdev = 3, 7, 0.3
    reg_global, reg_bias, reg_factors = 0.5, 0.2, 0.4
    est = AlternatingLeastSquares(
        n_factors=n_factors,
        n_iter=n_iter,
        init_stdev=init_stdev,
        reg_global=reg_global,
        reg_bias=reg_bias,
        reg_factors=reg_factors,
        random_state=42,
    ).fit(X, y)
    w0, w, v, history = _libfm_als_reference(
        X,
        y,
        n_factors,
        n_iter,
        init_stdev,
        reg_global,
        reg_bias,
        reg_factors,
        seed=42,
    )
    n_users = est.n_users_
    np.testing.assert_allclose(est.global_bias_, w0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(est.user_bias_, w[:n_users], rtol=0, atol=1e-10)
    np.testing.assert_allclose(est.item_bias_, w[n_users:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(est.user_factors_, v[:, :n_users].T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(est.item_factors_, v[:, n_users:].T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(est.loss_curve_, history, rtol=0, atol=1e-10)


def test_biases_only_converges_to_ridge_solution():
    X, y = _ratings(seed=1)
    reg = 0.7
    est = AlternatingLeastSquares(n_factors=0, n_iter=3000, reg_global=reg, reg_bias=reg)
    est.fit(X, y)
    n_users, n_items = est.n_users_, est.n_items_
    design = np.zeros((len(y), 1 + n_users + n_items))
    design[:, 0] = 1
    design[np.arange(len(y)), 1 + X[:, 0]] = 1
    design[np.arange(len(y)), 1 + n_users + X[:, 1]] = 1
    theta = np.linalg.solve(design.T @ design + reg * np.eye(design.shape[1]), design.T @ y)
    fitted = np.concatenate([[est.global_bias_], est.user_bias_, est.item_bias_])
    np.testing.assert_allclose(fitted, theta, atol=1e-8)


def test_loss_decreases_without_regularization():
    X, y = _ratings(seed=2)
    est = AlternatingLeastSquares(n_factors=4, n_iter=30, reg_factors=0.0, random_state=0)
    est.fit(X, y)
    assert est.loss_curve_.shape == (30,)
    assert np.all(np.diff(est.loss_curve_) <= 1e-12)
    assert est.loss_curve_[-1] < np.std(y)


def test_predict_clips_to_training_range():
    X, y = _ratings(seed=3)
    est = AlternatingLeastSquares(n_factors=4, n_iter=20, random_state=0).fit(X, y)
    users = np.repeat(np.arange(est.n_users_), est.n_items_)
    items = np.tile(np.arange(est.n_items_), est.n_users_)
    pred = est.predict(np.column_stack([users, items]))
    assert pred.min() >= y.min()
    assert pred.max() <= y.max()
    raw = est._score_users(np.arange(est.n_users_), np.arange(est.n_items_)).ravel()
    np.testing.assert_allclose(pred, np.clip(raw, y.min(), y.max()))


def test_fits_training_ratings():
    X, y = _ratings(seed=4)
    est = AlternatingLeastSquares(n_factors=8, n_iter=50, reg_factors=0.01, random_state=0)
    pred = est.fit(X, y).predict(X)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.5 * np.std(y)


def test_random_state_reproducible():
    X, y = _ratings(seed=5)
    a = AlternatingLeastSquares(n_iter=5, random_state=3).fit(X, y)
    b = AlternatingLeastSquares(n_iter=5, random_state=3).fit(X, y)
    np.testing.assert_array_equal(a.user_factors_, b.user_factors_)
    np.testing.assert_array_equal(a.loss_curve_, b.loss_curve_)


@pytest.mark.parametrize(
    "params",
    [
        {"n_factors": -1},
        {"n_factors": 2.5},
        {"n_iter": -1},
        {"init_stdev": -0.1},
        {"reg_global": -1.0},
        {"reg_bias": float("nan")},
        {"reg_factors": "big"},
    ],
)
def test_invalid_params(params):
    X, y = _ratings()
    with pytest.raises(ValueError, match=next(iter(params))):
        AlternatingLeastSquares(**params).fit(X, y)


def test_partial_fit_resumes_rather_than_restarting():
    """A warm-started sweep must start where the last one stopped, not from a draw.

    The cheapest way to tell the two apart: a second call with no sweeps at all leaves
    the parameters exactly as they were, which a fit that redrew them could not do.
    """
    X, y = _ratings()
    est = AlternatingLeastSquares(n_factors=3, n_iter=8, random_state=0).fit(X, y)
    before = est.user_factors_.copy(), est.item_factors_.copy(), est.global_bias_
    est.set_params(n_iter_partial=0).partial_fit(X[:5], y[:5])
    np.testing.assert_array_equal(est.user_factors_, before[0])
    np.testing.assert_array_equal(est.item_factors_, before[1])
    assert est.global_bias_ == before[2]

    est.set_params(n_iter_partial=4).partial_fit(X[:5], y[:5])
    assert len(est.loss_curve_) == 8 + 0 + 4, "the loss curve is not the whole history."
    assert not np.array_equal(est.user_factors_, before[0])


def test_partial_fit_draws_factors_for_the_identifiers_it_adds():
    """A new user starts from the initialization, not from the zeros `_remap` left.

    Zero is a fixed point of the ALS factor update -- a row of zeros contributes
    nothing to any other parameter and is itself left at zero -- so a new user whose
    factors were not drawn would stay invisible to the model forever.
    """
    X, y = _ratings()
    est = AlternatingLeastSquares(n_factors=3, n_iter=5, random_state=0).fit(X, y)
    newcomer = int(X[:, 0].max()) + 7
    est.partial_fit(np.array([[newcomer, X[0, 1]], [newcomer, X[1, 1]]]), [4.0, 2.0])
    added = np.searchsorted(est.user_ids_, newcomer)
    assert np.any(est.user_factors_[added] != 0.0)
    assert est.target_range_[0] <= 2.0 and est.target_range_[1] >= 4.0

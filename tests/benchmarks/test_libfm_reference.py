"""Head-to-head benchmark of AlternatingLeastSquares against the libFM reference binary.

Both run ALS with the same hyperparameters on MovieLens 100K ``ua``. Initialization uses
different random generators, so results are compared within a tolerance, not exactly.

Build libFM from https://github.com/srendle/libfm (``make all``) and point ``LIBFM_BIN``
at ``bin/libFM``, or put ``libFM`` on ``PATH``. Without it the benchmark is skipped.
"""

import os
import re
import shutil
import subprocess
import time

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.recommendation import AlternatingLeastSquares

pytestmark = pytest.mark.benchmark

ESTIMATOR = AlternatingLeastSquares(
    n_factors=8, n_iter=100, init_stdev=0.1, reg_global=0.0, reg_bias=1.0, reg_factors=10.0,
    random_state=0,
)  # fmt: skip
TEST_RMSE_TOLERANCE = 0.005
TRAIN_RMSE_TOLERANCE = 0.002


@pytest.fixture(scope="module")
def libfm_bin():
    path = os.environ.get("LIBFM_BIN") or shutil.which("libFM")
    if not path or not os.access(path, os.X_OK):
        pytest.skip("libFM binary not found; set LIBFM_BIN or put libFM on PATH.")
    return path


def _write_libfm(path, est, X, y):
    users = np.searchsorted(est.user_ids_, X[:, 0])
    items = est.n_users_ + np.searchsorted(est.item_ids_, X[:, 1])
    path.write_text(
        "".join(f"{r:g} {u}:1 {i}:1\n" for r, u, i in zip(y, users, items, strict=True))
    )


def _rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def test_als_matches_libfm(libfm_bin, movielens_100k_ua, tmp_path, request):
    dataset = movielens_100k_ua
    X_train, y_train = dataset.data[dataset.train_indices], dataset.target[dataset.train_indices]
    X_test, y_test = dataset.data[dataset.test_indices], dataset.target[dataset.test_indices]

    start = time.perf_counter()
    est = clone(ESTIMATOR).fit(X_train, y_train)
    ours_seconds = time.perf_counter() - start

    # libFM has no embeddings for unseen ids either; compare on warm test pairs only.
    warm = np.isin(X_test[:, 0], est.user_ids_) & np.isin(X_test[:, 1], est.item_ids_)
    X_test, y_test = X_test[warm], y_test[warm]
    _write_libfm(tmp_path / "train.libfm", est, X_train, y_train)
    _write_libfm(tmp_path / "test.libfm", est, X_test, y_test)

    command = [
        libfm_bin, "-task", "r", "-method", "als",
        "-train", str(tmp_path / "train.libfm"), "-test", str(tmp_path / "test.libfm"),
        "-dim", f"1,1,{est.n_factors}", "-iter", str(est.n_iter),
        "-init_stdev", str(est.init_stdev),
        "-regular", f"{est.reg_global},{est.reg_bias},{est.reg_factors}",
        "-seed", "0", "-out", str(tmp_path / "pred.txt"),
    ]  # fmt: skip
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    libfm_seconds = time.perf_counter() - start

    # libFM's time includes reading the text files and predicting the test set.
    # libFM reports the clipped training RMSE after each sweep; -out holds the last
    # sweep's test predictions, clipped to the training target range.
    libfm_train = float(re.findall(r"Train=([\d.eE+-]+)", result.stdout)[-1])
    libfm_test = _rmse(np.loadtxt(tmp_path / "pred.txt"), y_test)
    ours_train = _rmse(est.predict(X_train), y_train)
    ours_test = _rmse(est.predict(X_test), y_test)

    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"\nALS vs libFM on ML-100k ua ({len(y_test)} warm test ratings)\n"
            f"  {'':8}{'train RMSE':>12}{'test RMSE':>12}{'seconds':>12}\n"
            f"  {'skrecsys':8}{ours_train:>12.4f}{ours_test:>12.4f}{ours_seconds:>12.2f}\n"
            f"  {'libFM':8}{libfm_train:>12.4f}{libfm_test:>12.4f}{libfm_seconds:>12.2f}"
        )

    assert abs(ours_train - libfm_train) <= TRAIN_RMSE_TOLERANCE, (ours_train, libfm_train)
    assert abs(ours_test - libfm_test) <= TEST_RMSE_TOLERANCE, (ours_test, libfm_test)

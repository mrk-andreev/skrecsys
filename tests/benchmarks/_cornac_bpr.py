"""Fit Cornac's BPR on a CSR interaction matrix and save its factors.

Run by ``test_cornac_reference.py`` through ``uv run --isolated --with cornac``: Cornac
pins an older numpy and pulls a deep-learning stack, so it never enters our lockfile.

Usage: ``python _cornac_bpr.py <input.npz> <output.npz> <k> <lr> <reg> <max_iter> <seed>``
"""

import sys
import time

import numpy as np
import scipy.sparse as sp
from cornac.data import Dataset
from cornac.models import BPR


def main(
    input_path: str,
    output_path: str,
    k: int,
    learning_rate: float,
    lambda_reg: float,
    max_iter: int,
    seed: int,
) -> None:
    loaded = np.load(input_path)
    interactions = sp.csr_matrix(
        (loaded["data"], loaded["indices"], loaded["indptr"]), shape=tuple(loaded["shape"])
    ).tocoo()
    n_users, n_items = interactions.shape
    triplets = list(
        zip(
            interactions.row.tolist(),
            interactions.col.tolist(),
            interactions.data.tolist(),
            strict=True,
        )
    )
    dataset = Dataset.from_uir(triplets)

    model = BPR(
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        use_bias=True,
        seed=seed,
        verbose=False,
    )
    start = time.perf_counter()
    model.fit(dataset)
    seconds = time.perf_counter() - start

    # Cornac numbers users and items in order of appearance; restore our own order.
    users = np.empty(n_users, dtype=np.int64)
    for raw, internal in dataset.uid_map.items():
        users[int(raw)] = internal
    items = np.empty(n_items, dtype=np.int64)
    for raw, internal in dataset.iid_map.items():
        items[int(raw)] = internal

    np.savez(
        output_path,
        user_factors=np.asarray(model.u_factors)[users],
        item_factors=np.asarray(model.i_factors)[items],
        item_bias=np.asarray(model.i_biases)[items],
        seconds=np.asarray(seconds),
    )


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        int(sys.argv[6]),
        int(sys.argv[7]),
    )

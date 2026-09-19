"""Fit the reference RP3beta on a CSR interaction matrix and save its similarity matrix.

Run by ``test_dacrema_reference.py`` through ``uv run --isolated``: the reference is a
research framework rather than a package, so the test clones it and passes its path
here. It is never installed into our own environment.

Usage: ``python _dacrema_rp3beta.py <repo> <input.npz> <output.npz> <alpha> <beta> <topk>``
"""

import importlib
import sys
import time

import numpy as np
import scipy.sparse as sp

# The framework predates the removal of the `np.int` alias in NumPy 1.24.
if not hasattr(np, "int"):
    np.int = int  # noqa: NPY001


def main(repo: str, input_path: str, output_path: str, alpha: float, beta: float, top_k: int):
    sys.path.insert(0, repo)
    # The clone's location is an argument, so it is imported once that is known.
    recommender = importlib.import_module("GraphBased.RP3betaRecommender")

    loaded = np.load(input_path)
    interactions = sp.csr_matrix(
        (loaded["data"], loaded["indices"], loaded["indptr"]), shape=tuple(loaded["shape"])
    )

    model = recommender.RP3betaRecommender(interactions, verbose=False)
    start = time.perf_counter()
    model.fit(alpha=alpha, beta=beta, topK=top_k, normalize_similarity=True)
    seconds = time.perf_counter() - start

    weights = sp.csr_matrix(model.W_sparse)
    np.savez(
        output_path,
        indptr=weights.indptr,
        indices=weights.indices,
        values=weights.data,
        shape=np.asarray(weights.shape),
        seconds=np.asarray(seconds),
    )


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        float(sys.argv[4]),
        float(sys.argv[5]),
        int(sys.argv[6]),
    )

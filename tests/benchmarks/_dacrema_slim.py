"""Fit the reference SLIM ElasticNet on a CSR interaction matrix and save its weights.

Run by ``test_dacrema_reference.py`` through ``uv run --isolated``: the reference is a
research framework rather than a package, so the test clones it and passes its path
here. It is never installed into our own environment.

Usage: ``python _dacrema_slim.py <repo> <input.npz> <output.npz> <alpha> <l1_ratio> <topk>``
"""

import sys
import time

import numpy as np
import scipy.sparse as sp

# The framework predates the removal of the `np.int` alias in NumPy 1.24.
if not hasattr(np, "int"):
    np.int = int  # noqa: NPY001


def main(repo: str, input_path: str, output_path: str, alpha: float, l1_ratio: float, top_k: int):
    sys.path.insert(0, repo)
    # Importable only once the clone is on the path, so not at the top of the file.
    from SLIM_ElasticNet.SLIMElasticNetRecommender import (  # noqa: PLC0415
        SLIMElasticNetRecommender,
    )

    loaded = np.load(input_path)
    interactions = sp.csr_matrix(
        (loaded["data"], loaded["indices"], loaded["indptr"]), shape=tuple(loaded["shape"])
    )

    model = SLIMElasticNetRecommender(interactions, verbose=False)
    start = time.perf_counter()
    model.fit(l1_ratio=l1_ratio, alpha=alpha, positive_only=True, topK=top_k)
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

"""Fit RecTools' EASE on a CSR interaction matrix and save its weights.

Run by ``test_rectools_reference.py`` through ``uv run --isolated --with rectools``:
RecTools needs ``pm-implicit`` and an older pandas, which would conflict with our own
``reference`` group, so it never enters our lockfile.

Usage: ``python _rectools_ease.py <input.npz> <output.npz> <l2_reg>``
"""

import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import EASEModel


def main(input_path: str, output_path: str, l2_reg: float) -> None:
    loaded = np.load(input_path)
    interactions = sp.csr_matrix(
        (loaded["data"], loaded["indices"], loaded["indptr"]), shape=tuple(loaded["shape"])
    ).tocoo()
    frame = pd.DataFrame(
        {
            Columns.User: interactions.row,
            Columns.Item: interactions.col,
            Columns.Weight: interactions.data,
            Columns.Datetime: np.zeros(interactions.nnz, dtype=np.int64),
        }
    )
    dataset = Dataset.construct(frame)

    model = EASEModel(regularization=l2_reg)
    start = time.perf_counter()
    model.fit(dataset)
    seconds = time.perf_counter() - start

    n_items = len(dataset.item_id_map.external_ids)
    np.savez(
        output_path,
        # Row/column i of `weight` is item `item_ids[i]` in our own item numbering.
        weight=np.asarray(model.weight),
        item_ids=np.asarray(dataset.item_id_map.convert_to_external(np.arange(n_items))),
        seconds=np.asarray(seconds),
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))

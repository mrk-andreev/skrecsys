import numpy as np
from numpy.typing import NDArray

def ease_weights(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    n_cols: int,
    l2_reg: float,
    n_threads: int,
) -> NDArray[np.float64]: ...
def fm_als_fit(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    y: NDArray[np.float64],
    group: NDArray[np.int64],
    w0: float,
    w: NDArray[np.float64],
    v: NDArray[np.float64],
    reg0: float,
    reg_w: NDArray[np.float64],
    reg_v: NDArray[np.float64],
    n_iter: int,
) -> tuple[float, list[float]]: ...
def item_knn_top_k(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    n_cols: int,
    k: int,
    n_threads: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]: ...
def rp3beta_similarity(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    n_cols: int,
    row_scale: NDArray[np.float64],
    col_scale: NDArray[np.float64],
    k: int,
    n_threads: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]: ...
def top_k_per_row(
    scores: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    k: int,
) -> NDArray[np.int64]: ...

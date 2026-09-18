import numpy as np
from numpy.typing import NDArray

__build_profile__: str

def bpr_fit(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    n_cols: int,
    user_factors: NDArray[np.float64],
    item_factors: NDArray[np.float64],
    item_bias: NDArray[np.float64],
    learning_rate: float,
    regularization: float,
    use_bias: bool,
    n_iter: int,
    seed: int,
    n_threads: int,
) -> list[float]: ...
def coo_to_csr(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    data: NDArray[np.float64],
    n_rows: int,
    n_cols: int,
    n_threads: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]: ...
def csr_top_k_per_row(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    n_cols: int,
    k: int,
    n_threads: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]: ...
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
def factorize(
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
def item_cosine_top_k(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    n_cols: int,
    norms: NDArray[np.float64],
    shrink: float,
    k: int,
    n_threads: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]: ...
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
def slim_elasticnet_weights(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    n_cols: int,
    alpha: float,
    l1_ratio: float,
    positive: bool,
    max_iter: int,
    tol: float,
    k: int,
    n_threads: int,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]], int]: ...
def recommend_from_similarity(
    users_indptr: NDArray[np.int64],
    users_indices: NDArray[np.int64],
    users_data: NDArray[np.float64],
    similarity_indptr: NDArray[np.int64],
    similarity_indices: NDArray[np.int64],
    similarity_data: NDArray[np.float64],
    n_items: int,
    candidates: NDArray[np.int64],
    excluded_indptr: NDArray[np.int64],
    excluded_indices: NDArray[np.int64],
    k: int,
    n_threads: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def top_k_per_row(
    scores: NDArray[np.float64],
    excluded_indptr: NDArray[np.int64],
    excluded_indices: NDArray[np.int64],
    k: int,
) -> NDArray[np.int64]: ...

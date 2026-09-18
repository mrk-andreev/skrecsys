//! Native kernels for skrecsys, exposed to Python as `skrecsys._core`.

pub mod bpr;
pub mod ease;
pub mod encode;
pub mod fm_als;
pub mod knn;
pub mod prune;
pub mod ranking;
pub mod recommend;
pub mod rp3beta;
pub mod slim;
pub mod sparse;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_usize(values: &[i64], name: &str, bound: usize) -> PyResult<Vec<usize>> {
    values
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .ok()
                .filter(|&u| u < bound)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("{name} value {v} out of range [0, {bound})."))
                })
        })
        .collect()
}

/// Validate index values in place, without the `nnz`-sized copy `to_usize` would make.
fn check_indices(values: &[i64], name: &str, bound: usize) -> PyResult<()> {
    let bound = i64::try_from(bound).unwrap_or(i64::MAX);
    match values.iter().find(|&&v| v < 0 || v >= bound) {
        Some(&v) => Err(PyValueError::new_err(format!(
            "{name} value {v} out of range [0, {bound})."
        ))),
        None => Ok(()),
    }
}

/// Validate a `(n_rows, n_cols)` exclusion pattern in CSR layout.
///
/// The selection kernels walk a row's excluded columns alongside the scores, so the
/// indices have to ascend within a row rather than merely be in range.
fn check_excluded(
    indptr: &[i64],
    indices: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Vec<usize>> {
    check_indices(indices, "excluded column", n_cols.max(1))?;
    let indptr = check_indptr(indptr, indices.len())?;
    if indptr.len() != n_rows + 1 {
        return Err(PyValueError::new_err(
            "excluded_indptr must have one entry per row of scores, plus one.",
        ));
    }
    if indptr
        .windows(2)
        .any(|p| indices[p[0]..p[1]].windows(2).any(|c| c[0] >= c[1]))
    {
        return Err(PyValueError::new_err(
            "excluded column indices must ascend within a row.",
        ));
    }
    Ok(indptr)
}

/// Validate a CSR index pointer against the number of stored values, as `usize`.
fn check_indptr(indptr: &[i64], nnz: usize) -> PyResult<Vec<usize>> {
    if indptr.is_empty() {
        return Err(PyValueError::new_err("indptr must not be empty."));
    }
    let indptr = to_usize(indptr, "indptr", nnz + 1)?;
    if indptr[0] != 0 || indptr.last() != Some(&nnz) || indptr.windows(2).any(|p| p[0] > p[1]) {
        return Err(PyValueError::new_err(
            "indptr is not a valid CSR index pointer.",
        ));
    }
    Ok(indptr)
}

/// Fit a factorization machine by alternating least squares, libFM style.
///
/// `w` (n_features,) and `v` (n_factors, n_features) are updated in place.
/// Returns the fitted global bias and the training RMSE after each sweep.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fm_als_fit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    group: PyReadonlyArray1<'py, i64>,
    w0: f64,
    mut w: PyReadwriteArray1<'py, f64>,
    mut v: PyReadwriteArray2<'py, f64>,
    reg0: f64,
    reg_w: PyReadonlyArray1<'py, f64>,
    reg_v: PyReadonlyArray1<'py, f64>,
    n_iter: usize,
) -> PyResult<(f64, Vec<f64>)> {
    let y = y.as_slice()?;
    let data = data.as_slice()?;
    let reg_w = reg_w.as_slice()?;
    let reg_v = reg_v.as_slice()?;
    let v_cols = v.shape()[1];
    let w = w.as_slice_mut()?;
    let v = v.as_slice_mut()?;
    let n_rows = y.len();
    let n_features = w.len();
    if indptr.as_slice()?.len() != n_rows + 1 {
        return Err(PyValueError::new_err(
            "indptr must have len(y) + 1 entries.",
        ));
    }
    if indices.as_slice()?.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    if v_cols != n_features || group.as_slice()?.len() != n_features {
        return Err(PyValueError::new_err(
            "v columns and group must match len(w).",
        ));
    }
    if reg_w.len() != reg_v.len() {
        return Err(PyValueError::new_err(
            "reg_w and reg_v must have the same length.",
        ));
    }
    let indptr = check_indptr(indptr.as_slice()?, data.len())?;
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_features)?;
    let group = to_usize(group.as_slice()?, "group", reg_w.len())?;

    Ok(py.detach(|| {
        let x = sparse::Csr {
            n_rows,
            n_cols: n_features,
            indptr: &indptr,
            indices,
            data,
        };
        let mut model = fm_als::Model {
            w0,
            w,
            v,
            group: &group,
            reg0,
            reg_w,
            reg_v,
        };
        let history = fm_als::fit(&x, y, &mut model, n_iter);
        (model.w0, history)
    }))
}

/// Fit a BPR matrix factorization by stochastic gradient ascent over sampled triplets.
///
/// `r` is the users x items CSR interaction structure, whose values are ignored and whose
/// column indices must be sorted. `user_factors` (n_users, n_factors), `item_factors`
/// (n_items, n_factors) and `item_bias` (n_items,) are updated in place. Returns the
/// share of correctly ranked triplets in each epoch. `n_threads == 0` uses all cores.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bpr_fit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    n_cols: usize,
    mut user_factors: PyReadwriteArray2<'py, f64>,
    mut item_factors: PyReadwriteArray2<'py, f64>,
    mut item_bias: PyReadwriteArray1<'py, f64>,
    learning_rate: f64,
    regularization: f64,
    use_bias: bool,
    n_iter: usize,
    seed: u64,
    n_threads: usize,
) -> PyResult<Vec<f64>> {
    let n_factors = user_factors.shape()[1];
    if item_factors.shape()[1] != n_factors {
        return Err(PyValueError::new_err(
            "user_factors and item_factors must have the same number of columns.",
        ));
    }
    if item_factors.shape()[0] != n_cols || item_bias.as_slice()?.len() != n_cols {
        return Err(PyValueError::new_err(
            "item_factors and item_bias must have n_cols rows.",
        ));
    }
    let nnz = indices.as_slice()?.len();
    let indptr = check_indptr(indptr.as_slice()?, nnz)?;
    if indptr.len() != user_factors.shape()[0] + 1 {
        return Err(PyValueError::new_err(
            "indptr must have one entry per user, plus one.",
        ));
    }
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_cols.max(1))?;
    if indptr
        .windows(2)
        .any(|p| indices[p[0]..p[1]].windows(2).any(|c| c[0] >= c[1]))
    {
        return Err(PyValueError::new_err(
            "indices must be sorted and unique within each row.",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let user_factors = user_factors.as_slice_mut()?;
    let item_factors = item_factors.as_slice_mut()?;
    let item_bias = item_bias.as_slice_mut()?;
    Ok(py.detach(|| {
        let r = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data: &[],
        };
        let mut model = bpr::Model {
            user_factors,
            item_factors,
            item_bias,
        };
        let hyper = bpr::Hyper {
            n_factors,
            learning_rate,
            regularization,
            use_bias,
            n_iter,
            seed,
        };
        pool.install(|| bpr::fit(&r, &mut model, &hyper))
    }))
}

type CsrArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Top `k` entries of every row of `W^T W` for a users x items CSR matrix `W`.
///
/// Returns the item-item similarity as CSR `(indptr, indices, data)` with sorted
/// column indices. `n_threads == 0` uses all cores.
#[pyfunction]
fn item_knn_top_k<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    k: usize,
    n_threads: usize,
) -> PyResult<CsrArrays<'py>> {
    let data = data.as_slice()?;
    let indptr = check_indptr(indptr.as_slice()?, data.len())?;
    if indices.as_slice()?.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_cols)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = py.detach(|| {
        let w = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| knn::all_pairs_top_k(&w, k))
    });
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

/// Top `k` cosine neighbours of every item for a users x items CSR matrix `W`.
///
/// `norms` holds the Euclidean norm of each column of `W`; entry `(i, j)` of the result
/// is `<W_i, W_j> / (norms[i] * norms[j] + shrink)`. Returns the item-item similarity as
/// CSR `(indptr, indices, data)` with sorted column indices and a zero diagonal.
/// `n_threads == 0` uses all cores.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn item_cosine_top_k<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    norms: PyReadonlyArray1<'py, f64>,
    shrink: f64,
    k: usize,
    n_threads: usize,
) -> PyResult<CsrArrays<'py>> {
    let data = data.as_slice()?;
    let indptr = check_indptr(indptr.as_slice()?, data.len())?;
    if indices.as_slice()?.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_cols)?;
    let norms = norms.as_slice()?;
    if norms.len() != n_cols {
        return Err(PyValueError::new_err("norms must have n_cols entries."));
    }
    if !shrink.is_finite() || shrink < 0.0 {
        return Err(PyValueError::new_err("shrink must be finite and >= 0."));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = py.detach(|| {
        let w = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| knn::cosine_top_k(&w, norms, shrink, k))
    });
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

/// EASE item-item weights for a users x items CSR matrix `R`.
///
/// Returns the dense `n_cols x n_cols` weight matrix with a zero diagonal.
/// `n_threads == 0` uses all cores.
#[pyfunction]
fn ease_weights<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    l2_reg: f64,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data = data.as_slice()?;
    let indptr = check_indptr(indptr.as_slice()?, data.len())?;
    if indices.as_slice()?.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_cols)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let weights = py.detach(|| {
        let r = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| {
            let gram = ease::gram(&r);
            ease::weights(&gram, n_cols, l2_reg, faer::Par::rayon(0))
        })
    });
    let weights = weights.map_err(|_| {
        PyValueError::new_err(format!(
            "The item Gram matrix is not positive definite with l2_reg={l2_reg}; \
             use a larger l2_reg."
        ))
    })?;
    Array2::from_shape_vec((n_cols, n_cols), weights)
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .map(|a| a.into_pyarray(py))
}

/// Top `k` entries of every row of the RP3beta walk matrix for a users x items CSR `Pui`.
///
/// `Pui` must already carry the row normalization and the `alpha` power; `row_scale` and
/// `col_scale` hold the walk and popularity damping per item. Returns the item-item
/// similarity as CSR `(indptr, indices, data)` with sorted column indices and a zero
/// diagonal. `n_threads == 0` uses all cores.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rp3beta_similarity<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    row_scale: PyReadonlyArray1<'py, f64>,
    col_scale: PyReadonlyArray1<'py, f64>,
    k: usize,
    n_threads: usize,
) -> PyResult<CsrArrays<'py>> {
    let data = data.as_slice()?;
    let indptr = check_indptr(indptr.as_slice()?, data.len())?;
    if indices.as_slice()?.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_cols)?;
    let (row_scale, col_scale) = (row_scale.as_slice()?, col_scale.as_slice()?);
    if row_scale.len() != n_cols || col_scale.len() != n_cols {
        return Err(PyValueError::new_err(
            "row_scale and col_scale must have n_cols entries.",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = py.detach(|| {
        let pui = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| rp3beta::similarity(&pui, row_scale, col_scale, k))
    });
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

/// Per-item elastic-net (SLIM) weights for a users x items CSR matrix `R`.
///
/// Returns the transpose of the item-item weight matrix as CSR `(indptr, indices, data)`
/// with sorted column indices and a zero diagonal, so row `j` holds the `k` largest
/// weights of the regression that predicts item `j`, together with the number of columns
/// that hit `max_iter` without converging. `n_threads == 0` uses all cores.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn slim_elasticnet_weights<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    alpha: f64,
    l1_ratio: f64,
    positive: bool,
    max_iter: usize,
    tol: f64,
    k: usize,
    n_threads: usize,
) -> PyResult<(CsrArrays<'py>, usize)> {
    let data = data.as_slice()?;
    let indptr = check_indptr(indptr.as_slice()?, data.len())?;
    if indices.as_slice()?.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_cols)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let (result, unconverged) = py.detach(|| {
        let r = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        let enet = slim::ElasticNet {
            alpha,
            l1_ratio,
            positive,
            max_iter,
            tol,
        };
        pool.install(|| slim::similarity(&r, &enet, k))
    });
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        (
            to_i64(result.indptr).into_pyarray(py),
            to_i64(result.indices).into_pyarray(py),
            result.data.into_pyarray(py),
        ),
        unconverged,
    ))
}

/// Keep the `k` largest entries of each row of a CSR matrix.
///
/// Returns CSR `(indptr, indices, data)` with sorted column indices. Ties keep the entry
/// that comes first in its row, which is what a stable sort by descending value gives.
/// Explicit zeros are ordinary candidates. `n_threads == 0` uses all cores.
#[pyfunction]
fn csr_top_k_per_row<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    k: usize,
    n_threads: usize,
) -> PyResult<CsrArrays<'py>> {
    let data = data.as_slice()?;
    let indptr = check_indptr(indptr.as_slice()?, data.len())?;
    if indices.as_slice()?.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    let indices = indices.as_slice()?;
    check_indices(indices, "indices", n_cols)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = py.detach(|| {
        let m = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| prune::top_k_per_row(&m, k))
    });
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

/// Column indices of the `k` best entries of each row of `scores`, minus the excluded.
///
/// The exclusions are the CSR index pointer and column indices of a
/// `(n_rows, n_cols)` sparse pattern, whose indices must ascend within a row; passing
/// them in this form keeps `recommend` from building an `n_rows * n_cols` mask.
///
/// Ranks by descending score, breaking ties by ascending column index. Raises
/// `ValueError` if any row has fewer than `k` entries left.
#[pyfunction]
fn top_k_per_row<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray2<'py, f64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let shape = scores.shape().to_vec();
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let scores = scores.as_slice()?;

    let excluded_indices = excluded_indices.as_slice()?;
    let indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_indices,
        n_rows,
        n_cols,
    )?;
    let excluded = ranking::Excluded {
        indptr: &indptr,
        indices: excluded_indices,
    };

    let selected = py
        .detach(|| ranking::top_k_per_row(scores, &excluded, n_cols, k))
        .map_err(|e| {
            PyValueError::new_err(format!(
                "Cannot recommend {k} items: query {} has only {} eligible items.",
                e.row, e.found
            ))
        })?;
    let selected = selected.into_iter().map(|i| i as i64).collect();
    Array2::from_shape_vec((n_rows, k), selected)
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .map(|a| a.into_pyarray(py))
}

/// Borrow a CSR matrix given as three numpy arrays, validating it as `usize` indices.
fn borrow_csr(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    n_cols: usize,
) -> PyResult<Vec<usize>> {
    if indices.len() != data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length.",
        ));
    }
    check_indices(indices, "indices", n_cols)?;
    check_indptr(indptr, data.len())
}

type Ranked<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>);

/// The `k` best candidates of each row of `users * similarity`, and their scores.
///
/// Scores a query row into a reusable accumulator, drops the excluded candidates and
/// reduces it to its `k` best before moving on, so no dense `(n_queries, n_items)`
/// matrix is ever built. `candidates` holds the item index of each candidate position,
/// ascending; a candidate the product never reaches scores zero, exactly as it did when
/// the dense matrix was ranked. `n_threads == 0` uses all cores.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn recommend_from_similarity<'py>(
    py: Python<'py>,
    users_indptr: PyReadonlyArray1<'py, i64>,
    users_indices: PyReadonlyArray1<'py, i64>,
    users_data: PyReadonlyArray1<'py, f64>,
    similarity_indptr: PyReadonlyArray1<'py, i64>,
    similarity_indices: PyReadonlyArray1<'py, i64>,
    similarity_data: PyReadonlyArray1<'py, f64>,
    n_items: usize,
    candidates: PyReadonlyArray1<'py, i64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
    n_threads: usize,
) -> PyResult<Ranked<'py>> {
    let users_data = users_data.as_slice()?;
    let users_cols = users_indices.as_slice()?;
    let users_indptr = borrow_csr(users_indptr.as_slice()?, users_cols, users_data, n_items)?;
    let similarity_data = similarity_data.as_slice()?;
    let similarity_cols = similarity_indices.as_slice()?;
    let similarity_indptr = borrow_csr(
        similarity_indptr.as_slice()?,
        similarity_cols,
        similarity_data,
        n_items,
    )?;
    if similarity_indptr.len() != n_items + 1 {
        return Err(PyValueError::new_err(
            "similarity must have one row per item.",
        ));
    }

    let candidates = to_usize(candidates.as_slice()?, "candidate", n_items.max(1))?;
    if candidates.windows(2).any(|c| c[0] >= c[1]) {
        return Err(PyValueError::new_err("candidates must ascend."));
    }
    let mut position = vec![-1i64; n_items];
    for (p, &j) in candidates.iter().enumerate() {
        position[j] = p as i64;
    }

    let n_rows = users_indptr.len() - 1;
    let excluded_cols = excluded_indices.as_slice()?;
    let excluded_indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_cols,
        n_rows,
        candidates.len(),
    )?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let (order, scores) = py
        .detach(|| {
            let users = sparse::Csr {
                n_rows,
                n_cols: n_items,
                indptr: &users_indptr,
                indices: users_cols,
                data: users_data,
            };
            let similarity = sparse::Csr {
                n_rows: n_items,
                n_cols: n_items,
                indptr: &similarity_indptr,
                indices: similarity_cols,
                data: similarity_data,
            };
            let excluded = ranking::Excluded {
                indptr: &excluded_indptr,
                indices: excluded_cols,
            };
            pool.install(|| {
                recommend::top_k_from_similarity(
                    &users,
                    &similarity,
                    &candidates,
                    &position,
                    &excluded,
                    k,
                )
            })
        })
        .map_err(|e| {
            PyValueError::new_err(format!(
                "Cannot recommend {k} items: query {} has only {} eligible items.",
                e.row, e.found
            ))
        })?;

    let order = order.into_iter().map(|i| i as i64).collect();
    Ok((
        Array2::from_shape_vec((n_rows, k), order)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py),
        Array2::from_shape_vec((n_rows, k), scores)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py),
    ))
}

type Codes<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// The sorted distinct values of `values`, and the position of each value among them.
///
/// `np.unique(values, return_inverse=True)` without sorting every row: the distinct
/// values are found in one pass and only those are sorted.
#[pyfunction]
fn factorize<'py>(py: Python<'py>, values: PyReadonlyArray1<'py, i64>) -> PyResult<Codes<'py>> {
    let values = values.as_slice()?;
    let (uniques, codes) = py.detach(|| encode::factorize(values));
    Ok((uniques.into_pyarray(py), codes.into_pyarray(py)))
}

/// A CSR matrix built from coordinate triples, with duplicate pairs summed.
///
/// The rows come out with ascending column indices and one entry per pair, which is
/// scipy's canonical form; explicit zeros are kept. `n_threads == 0` uses all cores.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn coo_to_csr<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_rows: usize,
    n_cols: usize,
    n_threads: usize,
) -> PyResult<CsrArrays<'py>> {
    let (rows, cols, data) = (rows.as_slice()?, cols.as_slice()?, data.as_slice()?);
    if rows.len() != data.len() || cols.len() != data.len() {
        return Err(PyValueError::new_err(
            "rows, cols and data must have the same length.",
        ));
    }
    check_indices(rows, "row", n_rows.max(1))?;
    check_indices(cols, "column", n_cols.max(1))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result =
        py.detach(|| pool.install(|| encode::coo_to_csr(rows, cols, data, n_rows, n_cols)));
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

/// Whether this extension was compiled with optimizations, for benchmarks to report.
///
/// Inferred from `debug_assertions`, which Cargo disables for `--release` and enables
/// otherwise. A profile that overrides `debug-assertions` would defeat it, but the
/// profiles this crate ships do not.
const BUILD_PROFILE: &str = if cfg!(debug_assertions) {
    "debug"
} else {
    "release"
};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__build_profile__", BUILD_PROFILE)?;
    m.add_function(wrap_pyfunction!(bpr_fit, m)?)?;
    m.add_function(wrap_pyfunction!(coo_to_csr, m)?)?;
    m.add_function(wrap_pyfunction!(csr_top_k_per_row, m)?)?;
    m.add_function(wrap_pyfunction!(ease_weights, m)?)?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(fm_als_fit, m)?)?;
    m.add_function(wrap_pyfunction!(item_cosine_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(item_knn_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(recommend_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(rp3beta_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(slim_elasticnet_weights, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_per_row, m)?)
}

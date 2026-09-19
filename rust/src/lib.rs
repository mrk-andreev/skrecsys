//! The `skrecsys._core` extension module: validation, conversion and GIL handling
//! around the kernels of the `skrecsys-kernels` crate.
//!
//! Every function here turns numpy arrays into the borrowed slices a kernel takes,
//! checks whatever the kernel assumes -- index bounds, CSR structure, matching shapes --
//! and releases the GIL for the call. The kernels themselves know nothing about Python,
//! which is what lets `rust/kernels/benches` measure them directly.

#![forbid(unsafe_code)]

use skrecsys_kernels::{
    bpr, ease, encode, fm_als, hnsw, knn, prune, quantized, ranking, recommend, rp3beta, slim,
    sparse, vectors,
};

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod pools;

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

/// Validate a row subset for the restricted similarity kernels.
///
/// Sorted and distinct, because the caller splices the result back into a stored matrix
/// row by row and the kernels hand the rows back in the order they were asked for.
fn check_rows(
    rows: Option<&PyReadonlyArray1<'_, i64>>,
    n_rows: usize,
) -> PyResult<Option<Vec<usize>>> {
    let Some(rows) = rows else { return Ok(None) };
    let rows = to_usize(rows.as_slice()?, "rows", n_rows)?;
    if rows.windows(2).any(|p| p[0] >= p[1]) {
        return Err(PyValueError::new_err("rows must be sorted and distinct."));
    }
    Ok(Some(rows))
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
///
/// `positives`, when given, holds offsets into `indices` and restricts an epoch to
/// drawing its positive from those entries. Negatives are still rejected against the
/// user's whole row, so this trains on a subset of the interactions while knowing all
/// of them.
#[pyfunction]
#[pyo3(signature = (
    indptr,
    indices,
    n_cols,
    user_factors,
    item_factors,
    item_bias,
    learning_rate,
    regularization,
    use_bias,
    n_iter,
    seed,
    n_threads,
    positives = None,
))]
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
    positives: Option<PyReadonlyArray1<'py, i64>>,
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
    // Offsets into `indices`, so `nnz` is the bound; the array is released before the
    // GIL is, because the epoch loop only needs the converted copy.
    let positives = match positives {
        Some(array) => Some(to_usize(array.as_slice()?, "positives", nnz)?),
        None => None,
    };
    let pool = pools::pool(n_threads)?;

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
        pool.install(|| bpr::fit(&r, positives.as_deref(), &mut model, &hyper))
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
#[pyo3(signature = (indptr, indices, data, n_cols, k, n_threads, rows = None))]
#[allow(clippy::too_many_arguments)]
fn item_knn_top_k<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    k: usize,
    n_threads: usize,
    rows: Option<PyReadonlyArray1<'py, i64>>,
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
    let rows = check_rows(rows.as_ref(), n_cols)?;
    let pool = pools::pool(n_threads)?;

    let result = py.detach(|| {
        let w = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| knn::all_pairs_top_k_rows(&w, k, rows.as_deref()))
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
#[pyo3(signature = (indptr, indices, data, n_cols, norms, shrink, k, n_threads, rows = None))]
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
    rows: Option<PyReadonlyArray1<'py, i64>>,
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
    let rows = check_rows(rows.as_ref(), n_cols)?;
    let pool = pools::pool(n_threads)?;

    let result = py.detach(|| {
        let w = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| knn::cosine_top_k_rows(&w, norms, shrink, k, rows.as_deref()))
    });
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

/// `(G + l2_reg * I)^-1` and the EASE weights derived from it, for a CSR matrix `R`.
///
/// One body behind `ease_weights` and `ease_inverse_gram`, which differ only in which of
/// the two they hand back.
#[allow(clippy::too_many_arguments)]
fn ease_inverse_and_weights<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    l2_reg: f64,
    n_threads: usize,
    want_weights: bool,
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
    let pool = pools::pool(n_threads)?;

    let result = py.detach(|| {
        let r = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| {
            let gram = ease::gram(&r);
            let p = ease::inverse(&gram, n_cols, l2_reg, faer::Par::rayon(0))?;
            Ok(if want_weights {
                ease::weights_from_inverse(&p, n_cols)
            } else {
                p
            })
        })
    });
    let result = result.map_err(|_: ease::NotPositiveDefinite| {
        PyValueError::new_err(format!(
            "The item Gram matrix is not positive definite with l2_reg={l2_reg}; \
             use a larger l2_reg."
        ))
    })?;
    Array2::from_shape_vec((n_cols, n_cols), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .map(|a| a.into_pyarray(py))
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
    ease_inverse_and_weights(py, indptr, indices, data, n_cols, l2_reg, n_threads, true)
}

/// `(G + l2_reg * I)^-1` for a users x items CSR matrix `R`, where `G = R^T R`.
///
/// Dense, `n_cols x n_cols`, symmetric and fully populated. This is what EASE factors on
/// the way to its weights; an incremental fit keeps it so that adding interactions is a
/// low-rank update rather than another factorization. `n_threads == 0` uses all cores.
#[pyfunction]
fn ease_inverse_gram<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    l2_reg: f64,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ease_inverse_and_weights(py, indptr, indices, data, n_cols, l2_reg, n_threads, false)
}

/// EASE item-item weights from an inverse Gram matrix that is already in hand.
///
/// `p` is the dense, symmetric `n x n` matrix `ease_inverse_gram` returns, or that an
/// incremental update produced from one. `n_threads == 0` uses all cores.
#[pyfunction]
fn ease_weights_from_inverse<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = p.shape()[0];
    if p.shape()[1] != n {
        return Err(PyValueError::new_err("p must be square."));
    }
    let p = p.as_slice()?;
    let pool = pools::pool(n_threads)?;
    let weights = py.detach(|| pool.install(|| ease::weights_from_inverse(p, n)));
    Array2::from_shape_vec((n, n), weights)
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
#[pyo3(signature = (indptr, indices, data, n_cols, row_scale, col_scale, k, n_threads, rows = None))]
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
    rows: Option<PyReadonlyArray1<'py, i64>>,
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
    let rows = check_rows(rows.as_ref(), n_cols)?;
    let pool = pools::pool(n_threads)?;

    let result = py.detach(|| {
        let pui = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| rp3beta::similarity_rows(&pui, row_scale, col_scale, k, rows.as_deref()))
    });
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

/// The item Gram matrix `R^T R` for a users x items CSR matrix `R`.
///
/// Dense row-major `n_cols x n_cols`. Both EASE and SLIM are functions of it alone, and
/// an incremental fit keeps it so that a batch is a low-rank update rather than another
/// pass over the interactions. `n_threads == 0` uses all cores.
#[pyfunction]
fn item_gram<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
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
    let pool = pools::pool(n_threads)?;
    let gram = py.detach(|| {
        let r = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices,
            data,
        };
        pool.install(|| ease::gram(&r))
    });
    Array2::from_shape_vec((n_cols, n_cols), gram)
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .map(|a| a.into_pyarray(py))
}

/// SLIM elastic-net weights from an item Gram matrix that is already in hand.
///
/// `gram` is the dense `n x n` matrix `item_gram` returns and `n_rows` the number of
/// users, which the penalties scale by. `columns` restricts the solve to those targets,
/// which must be sorted and distinct, and the result then has one row per entry of it.
/// `warm_*` gives each target a starting point for the coordinate descent, as a CSR with
/// one row per target. `n_threads == 0` uses all cores.
#[pyfunction]
#[pyo3(signature = (
    gram,
    n_rows,
    alpha,
    l1_ratio,
    positive,
    max_iter,
    tol,
    k,
    n_threads,
    columns = None,
    warm_indptr = None,
    warm_indices = None,
    warm_data = None,
))]
#[allow(clippy::too_many_arguments)]
fn slim_elasticnet_weights_from_gram<'py>(
    py: Python<'py>,
    gram: PyReadonlyArray2<'py, f64>,
    n_rows: f64,
    alpha: f64,
    l1_ratio: f64,
    positive: bool,
    max_iter: usize,
    tol: f64,
    k: usize,
    n_threads: usize,
    columns: Option<PyReadonlyArray1<'py, i64>>,
    warm_indptr: Option<PyReadonlyArray1<'py, i64>>,
    warm_indices: Option<PyReadonlyArray1<'py, i64>>,
    warm_data: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(CsrArrays<'py>, usize)> {
    let n = gram.shape()[0];
    if gram.shape()[1] != n {
        return Err(PyValueError::new_err("gram must be square."));
    }
    let gram = gram.as_slice()?;
    let columns = check_rows(columns.as_ref(), n)?;
    let n_out = columns.as_ref().map_or(n, Vec::len);

    // The arrays outlive the slices borrowed from them, so they are bound here rather
    // than inside the match that validates them.
    let warm_arrays = match (warm_indptr, warm_indices, warm_data) {
        (None, None, None) => None,
        (Some(indptr), Some(indices), Some(data)) => Some((indptr, indices, data)),
        _ => {
            return Err(PyValueError::new_err(
                "warm_indptr, warm_indices and warm_data must be given together.",
            ));
        }
    };
    let warm = match &warm_arrays {
        None => None,
        Some((indptr, indices, data)) => {
            let data = data.as_slice()?;
            let indptr = check_indptr(indptr.as_slice()?, data.len())?;
            if indptr.len() != n_out + 1 {
                return Err(PyValueError::new_err(
                    "warm_indptr must have one entry per solved column, plus one.",
                ));
            }
            let indices = indices.as_slice()?;
            if indices.len() != data.len() {
                return Err(PyValueError::new_err(
                    "warm_indices and warm_data must have the same length.",
                ));
            }
            check_indices(indices, "warm_indices", n.max(1))?;
            Some((indptr, indices, data))
        }
    };
    let pool = pools::pool(n_threads)?;

    let (result, unconverged) = py.detach(|| {
        let warm = warm.as_ref().map(|(indptr, indices, data)| sparse::Csr {
            n_rows: n_out,
            n_cols: n,
            indptr,
            indices,
            data,
        });
        let enet = slim::ElasticNet {
            alpha,
            l1_ratio,
            positive,
            max_iter,
            tol,
        };
        pool.install(|| {
            slim::similarity_from_gram(gram, n, n_rows, columns.as_deref(), warm.as_ref(), &enet, k)
        })
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
    let pool = pools::pool(n_threads)?;

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
    let pool = pools::pool(n_threads)?;

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

/// A fitted CSR matrix borrowed for a recommend call, checked in constant time.
///
/// The index pointer stays `int64` -- the kernels read it through
/// [`recommend::Offset`] -- because converting it would be a pass over every row on
/// every request, where the call itself only touches the rows it asked for. For the
/// same reason only the shape is checked here. Validating every stored index, as `borrow_csr` does,
/// is a pass over `nnz` -- the whole interaction matrix, on every request -- so the
/// kernels read these matrices through bounds-checked slices instead, and a malformed
/// one stops with a panic that [`recommend_call`] turns into a `ValueError`. Nothing
/// can read out of bounds either way; the difference is only which error is raised.
fn fitted_csr<'a>(
    name: &str,
    indptr: &'a [i64],
    indices: &'a [i64],
    data: &'a [f64],
    n_cols: usize,
) -> PyResult<recommend::CsrRows<'a, i64>> {
    let n_rows = indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| PyValueError::new_err(format!("{name} indptr must not be empty.")))?;
    if indices.len() != data.len()
        || indptr[0] != 0
        || usize::try_from(indptr[n_rows]).ok() != Some(indices.len())
    {
        return Err(PyValueError::new_err(format!(
            "{name} is not a valid CSR matrix."
        )));
    }
    Ok(recommend::CsrRows {
        n_cols,
        indptr,
        indices,
        data,
    })
}

/// The candidate subset of a recommend call: `None` for every item, else the item at
/// each position, ascending, and the inverse mapping over the catalog.
fn candidate_subset(
    candidates: Option<&[i64]>,
    n_items: usize,
) -> PyResult<Option<(Vec<usize>, Vec<i64>)>> {
    let Some(candidates) = candidates else {
        return Ok(None);
    };
    let items = to_usize(candidates, "candidate", n_items.max(1))?;
    if items.windows(2).any(|c| c[0] >= c[1]) {
        return Err(PyValueError::new_err("candidates must ascend."));
    }
    let mut position = vec![-1i64; n_items];
    for (p, &j) in items.iter().enumerate() {
        position[j] = p as i64;
    }
    Ok(Some((items, position)))
}

fn candidates_of(
    subset: &Option<(Vec<usize>, Vec<i64>)>,
    n_items: usize,
) -> recommend::Candidates<'_> {
    match subset {
        None => recommend::Candidates::All(n_items),
        Some((items, position)) => recommend::Candidates::Subset { items, position },
    }
}

/// Validate the per-query exclusions a recommend call may be handed on top of the seen
/// items: candidate positions, one CSR row per query, ascending within each row.
fn extra_positions(
    indptr: Option<&[i64]>,
    indices: Option<&[i64]>,
    n_queries: usize,
    n_candidates: usize,
) -> PyResult<Option<Vec<usize>>> {
    match (indptr, indices) {
        (Some(indptr), Some(indices)) => {
            check_excluded(indptr, indices, n_queries, n_candidates).map(Some)
        }
        (None, None) => Ok(None),
        _ => Err(PyValueError::new_err(
            "excluded_indptr and excluded_indices go together.",
        )),
    }
}

/// The exclusions of a recommend call: the seen items of each query's own row, the
/// positions the caller added, both, or neither.
fn exclusions_of<'a, P: recommend::Offset>(
    seen: Option<(&'a [P], &'a [i64])>,
    extra: Option<(&'a [usize], &'a [i64])>,
) -> recommend::Exclusions<'a, P> {
    let extra = extra.map(|(indptr, indices)| ranking::Excluded { indptr, indices });
    match (seen, extra) {
        (None, None) => recommend::Exclusions::Nothing,
        (Some((indptr, indices)), None) => recommend::Exclusions::Items { indptr, indices },
        (None, Some(extra)) => recommend::Exclusions::Positions(extra),
        (Some((indptr, indices)), Some(extra)) => recommend::Exclusions::ItemsAndPositions {
            indptr,
            indices,
            extra,
        },
    }
}

/// Run a recommend kernel without the GIL, on the cached pool when `parallel` says the
/// kernel can use one (else on this thread, with no pool to wake), and turn what it
/// returns into numpy arrays.
///
/// `first_query` is the position of this batch's first query in the caller's whole
/// request, so that an error names the query the user actually passed.
fn recommend_call<'py>(
    py: Python<'py>,
    n_rows: usize,
    parallel: bool,
    k: usize,
    n_threads: usize,
    first_query: usize,
    kernel: impl FnOnce() -> Result<(Vec<usize>, Vec<f64>), ranking::TooFewEligible> + Send,
) -> PyResult<Ranked<'py>> {
    let pool = if parallel {
        Some(pools::pool(n_threads)?)
    } else {
        None
    };
    let outcome = py.detach(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match pool {
            Some(pool) => pool.install(kernel),
            None => kernel(),
        }))
    });
    let (order, scores) = outcome
        .map_err(|_| {
            PyValueError::new_err(
                "The fitted matrices are malformed: an index is out of range or an \
                 index pointer is not monotone.",
            )
        })?
        .map_err(|e| {
            PyValueError::new_err(format!(
                "Cannot recommend {k} items: query {} has only {} eligible items.",
                e.row + first_query,
                e.found
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

/// The `k` best candidates of `users[rows] * similarity`, and their scores.
///
/// `users` is the whole fitted interaction matrix and `rows` the users being queried,
/// so nothing is sliced on the Python side; with `exclude_seen` each query also skips
/// the items stored in its own row. Every query is scored into a reusable accumulator
/// and reduced to its `k` best before the next, so no dense `(n_queries, n_items)`
/// matrix is ever built. `candidates` holds the item index of each candidate position,
/// ascending, or is `None` for every item; a candidate the product never reaches scores
/// zero, exactly as it did when the dense matrix was ranked. Returns candidate
/// positions. `n_threads == 0` uses all cores.
#[pyfunction]
#[pyo3(signature = (
    users_indptr, users_indices, users_data, rows,
    similarity_indptr, similarity_indices, similarity_data,
    candidates, exclude_seen, k, n_threads, first_query=0,
    excluded_indptr=None, excluded_indices=None,
))]
#[allow(clippy::too_many_arguments)]
fn recommend_from_similarity<'py>(
    py: Python<'py>,
    users_indptr: PyReadonlyArray1<'py, i64>,
    users_indices: PyReadonlyArray1<'py, i64>,
    users_data: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i64>,
    similarity_indptr: PyReadonlyArray1<'py, i64>,
    similarity_indices: PyReadonlyArray1<'py, i64>,
    similarity_data: PyReadonlyArray1<'py, f64>,
    candidates: Option<PyReadonlyArray1<'py, i64>>,
    exclude_seen: bool,
    k: usize,
    n_threads: usize,
    first_query: usize,
    excluded_indptr: Option<PyReadonlyArray1<'py, i64>>,
    excluded_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Ranked<'py>> {
    let similarity = fitted_csr(
        "similarity",
        similarity_indptr.as_slice()?,
        similarity_indices.as_slice()?,
        similarity_data.as_slice()?,
        0,
    )?;
    let n_items = similarity.n_rows();
    let similarity = recommend::CsrRows {
        n_cols: n_items,
        ..similarity
    };
    let users = fitted_csr(
        "users",
        users_indptr.as_slice()?,
        users_indices.as_slice()?,
        users_data.as_slice()?,
        n_items,
    )?;
    let rows = to_usize(rows.as_slice()?, "rows", users.n_rows().max(1))?;
    let subset = candidate_subset(
        candidates.as_ref().map(|c| c.as_slice()).transpose()?,
        n_items,
    )?;
    let candidates = candidates_of(&subset, n_items);
    let extra_indices = excluded_indices
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()?;
    let extra_indptr = extra_positions(
        excluded_indptr.as_ref().map(|a| a.as_slice()).transpose()?,
        extra_indices,
        rows.len(),
        candidates.len(),
    )?;
    let exclusions = exclusions_of(
        exclude_seen.then_some((users.indptr, users.indices)),
        extra_indptr.as_deref().zip(extra_indices),
    );
    let parallel = rows.len() > recommend::SERIAL_QUERIES;
    recommend_call(py, rows.len(), parallel, k, n_threads, first_query, || {
        recommend::top_k_from_similarity(&users, &rows, &similarity, candidates, &exclusions, k)
    })
}

/// The `k` best candidates of `users[rows] @ weights` for a dense `(n_items, n_items)`
/// item-item matrix -- EASE's -- and their scores.
///
/// Arguments and result as for `recommend_from_similarity`, with `weights` C-contiguous.
#[pyfunction]
#[pyo3(signature = (
    users_indptr, users_indices, users_data, rows, weights,
    candidates, exclude_seen, k, n_threads, first_query=0,
    excluded_indptr=None, excluded_indices=None,
))]
#[allow(clippy::too_many_arguments)]
fn recommend_from_dense_rows<'py>(
    py: Python<'py>,
    users_indptr: PyReadonlyArray1<'py, i64>,
    users_indices: PyReadonlyArray1<'py, i64>,
    users_data: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray2<'py, f64>,
    candidates: Option<PyReadonlyArray1<'py, i64>>,
    exclude_seen: bool,
    k: usize,
    n_threads: usize,
    first_query: usize,
    excluded_indptr: Option<PyReadonlyArray1<'py, i64>>,
    excluded_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Ranked<'py>> {
    let [n_items, n_cols] = *weights.shape() else {
        unreachable!("a two-dimensional array has two dimensions")
    };
    if n_items != n_cols {
        return Err(PyValueError::new_err("weights must be square."));
    }
    let weights = weights.as_slice()?;
    let users = fitted_csr(
        "users",
        users_indptr.as_slice()?,
        users_indices.as_slice()?,
        users_data.as_slice()?,
        n_items,
    )?;
    let rows = to_usize(rows.as_slice()?, "rows", users.n_rows().max(1))?;
    let subset = candidate_subset(
        candidates.as_ref().map(|c| c.as_slice()).transpose()?,
        n_items,
    )?;
    let candidates = candidates_of(&subset, n_items);
    let extra_indices = excluded_indices
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()?;
    let extra_indptr = extra_positions(
        excluded_indptr.as_ref().map(|a| a.as_slice()).transpose()?,
        extra_indices,
        rows.len(),
        candidates.len(),
    )?;
    let exclusions = exclusions_of(
        exclude_seen.then_some((users.indptr, users.indices)),
        extra_indptr.as_deref().zip(extra_indices),
    );
    let parallel = rows.len() > recommend::SERIAL_QUERIES;
    recommend_call(py, rows.len(), parallel, k, n_threads, first_query, || {
        recommend::top_k_from_dense_rows(&users, &rows, weights, candidates, &exclusions, k)
    })
}

/// The `k` best candidates of a latent-factor model for users `rows`, and their scores.
///
/// `score(u, j) = item_bias[j] + <user_factors[u], item_factors[j]> + user_offset[u]`,
/// either bias optional; a model with no factors passes zero-width factor matrices. The
/// seen matrix, when given, is the fitted interaction matrix whose row `u` lists what
/// user `u` must not be recommended. Result as for `recommend_from_similarity`.
#[pyfunction]
#[pyo3(signature = (
    user_factors, item_factors, item_bias, user_offset, rows,
    seen_indptr, seen_indices, candidates, k, n_threads, first_query=0,
    excluded_indptr=None, excluded_indices=None,
))]
#[allow(clippy::too_many_arguments)]
fn recommend_from_factors<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray2<'py, f64>,
    item_factors: PyReadonlyArray2<'py, f64>,
    item_bias: Option<PyReadonlyArray1<'py, f64>>,
    user_offset: Option<PyReadonlyArray1<'py, f64>>,
    rows: PyReadonlyArray1<'py, i64>,
    seen_indptr: Option<PyReadonlyArray1<'py, i64>>,
    seen_indices: Option<PyReadonlyArray1<'py, i64>>,
    candidates: Option<PyReadonlyArray1<'py, i64>>,
    k: usize,
    n_threads: usize,
    first_query: usize,
    excluded_indptr: Option<PyReadonlyArray1<'py, i64>>,
    excluded_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Ranked<'py>> {
    let [n_users, dim] = *user_factors.shape() else {
        unreachable!("a two-dimensional array has two dimensions")
    };
    let [n_items, item_dim] = *item_factors.shape() else {
        unreachable!("a two-dimensional array has two dimensions")
    };
    if dim != item_dim {
        return Err(PyValueError::new_err(
            "user and item factors must have the same width.",
        ));
    }
    let item_bias = item_bias.as_ref().map(|b| b.as_slice()).transpose()?;
    let user_offset = user_offset.as_ref().map(|o| o.as_slice()).transpose()?;
    if item_bias.is_some_and(|b| b.len() != n_items) {
        return Err(PyValueError::new_err(
            "item_bias must have one value per item.",
        ));
    }
    if user_offset.is_some_and(|o| o.len() != n_users) {
        return Err(PyValueError::new_err(
            "user_offset must have one value per user.",
        ));
    }
    let factors = recommend::Factors {
        users: user_factors.as_slice()?,
        items: item_factors.as_slice()?,
        dim,
        item_bias,
        user_offset,
    };
    let rows = to_usize(rows.as_slice()?, "rows", n_users.max(1))?;
    let subset = candidate_subset(
        candidates.as_ref().map(|c| c.as_slice()).transpose()?,
        n_items,
    )?;
    let candidates = candidates_of(&subset, n_items);
    let seen = match (&seen_indptr, &seen_indices) {
        (Some(indptr), Some(indices)) => {
            let indptr = indptr.as_slice()?;
            if indptr.len() != n_users + 1 {
                return Err(PyValueError::new_err(
                    "seen_indptr must have one entry per user, plus one.",
                ));
            }
            Some((indptr, indices.as_slice()?))
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "seen_indptr and seen_indices go together.",
            ));
        }
    };
    let extra_indices = excluded_indices
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()?;
    let extra_indptr = extra_positions(
        excluded_indptr.as_ref().map(|a| a.as_slice()).transpose()?,
        extra_indices,
        rows.len(),
        candidates.len(),
    )?;
    let exclusions = exclusions_of(seen, extra_indptr.as_deref().zip(extra_indices));
    let parallel = recommend::factors_want_threads(rows.len(), candidates.len(), dim);
    recommend_call(py, rows.len(), parallel, k, n_threads, first_query, || {
        recommend::top_k_from_factors(&factors, &rows, candidates, &exclusions, k)
    })
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

    let pool = pools::pool(n_threads)?;
    let result =
        py.detach(|| pool.install(|| encode::coo_to_csr(rows, cols, data, n_rows, n_cols)));
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    Ok((
        to_i64(result.indptr).into_pyarray(py),
        to_i64(result.indices).into_pyarray(py),
        result.data.into_pyarray(py),
    ))
}

type HnswGraph<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    i64,
);

/// Validate the construction parameters every `hnsw_build_*` shares.
fn hnsw_params(m: usize, ef_construction: usize, seed: u64) -> PyResult<hnsw::Params> {
    if m == 0 {
        return Err(PyValueError::new_err("m must be >= 1."));
    }
    if ef_construction == 0 {
        return Err(PyValueError::new_err("ef_construction must be >= 1."));
    }
    Ok(hnsw::Params {
        m,
        ef_construction,
        seed,
    })
}

/// Build a graph on a pool of `n_threads`, and hand back its flat arrays.
fn hnsw_built<'py>(
    py: Python<'py>,
    items: &impl hnsw::Items,
    params: &hnsw::Params,
    n_threads: usize,
) -> PyResult<HnswGraph<'py>> {
    let pool = pools::pool(n_threads)?;
    let graph = py.detach(|| pool.install(|| hnsw::build(items, params)));
    Ok((
        graph.node_level.into_pyarray(py),
        graph.links_indptr.into_pyarray(py),
        graph.links_indices.into_pyarray(py),
        graph.entry_point,
    ))
}

/// The candidate item indices, ascending and in range, as every index takes them.
fn index_candidates(candidates: &[i64], n_items: usize) -> PyResult<Vec<usize>> {
    let candidates = to_usize(candidates, "candidate", n_items.max(1))?;
    if candidates.windows(2).any(|c| c[0] >= c[1]) {
        return Err(PyValueError::new_err("candidates must ascend."));
    }
    Ok(candidates)
}

/// The candidates, plus the table mapping an item back to its candidate position.
///
/// Only the graph needs the inverse: a walk arrives at items and has to look up where
/// each one sits in the answer. A scan enumerates the positions itself, so it takes the
/// candidates alone and never pays for this `n_items`-sized allocation.
fn graph_candidates(candidates: &[i64], n_items: usize) -> PyResult<(Vec<usize>, Vec<i64>)> {
    let candidates = index_candidates(candidates, n_items)?;
    let mut position = vec![-1i64; n_items];
    for (p, &j) in candidates.iter().enumerate() {
        position[j] = p as i64;
    }
    Ok((candidates, position))
}

/// Borrow a graph's arrays, rejecting anything that does not describe one.
///
/// A fitted index pickles as these arrays, so they can come back altered or truncated;
/// every way of being wrong has to raise rather than panic or read out of bounds.
fn hnsw_view<'a>(
    node_level: &'a [i32],
    links_indptr: &'a [i64],
    links_indices: &'a [i32],
    entry_point: i64,
    n_items: usize,
) -> PyResult<hnsw::GraphView<'a>> {
    if node_level.len() != n_items {
        return Err(PyValueError::new_err(
            "node_level must have one entry per item.",
        ));
    }
    hnsw::GraphView::new(node_level, links_indptr, links_indices, entry_point)
        .ok_or_else(|| PyValueError::new_err("the index arrays do not describe a graph."))
}

/// Run a batch search and shape its result the way `recommend` expects.
#[allow(clippy::too_many_arguments)]
fn hnsw_ranked<'py>(
    py: Python<'py>,
    items: &impl hnsw::Items,
    view: &hnsw::GraphView<'_>,
    queries: &hnsw::Queries<'_>,
    candidates: &[usize],
    position: &[i64],
    excluded_indptr: &[usize],
    excluded_indices: &[i64],
    k: usize,
    ef_search: usize,
    n_threads: usize,
) -> PyResult<Ranked<'py>> {
    if ef_search == 0 {
        return Err(PyValueError::new_err("ef_search must be >= 1."));
    }
    let n_rows = queries.len();
    let pool = pools::pool(n_threads)?;
    let ranked = py.detach(|| {
        let excluded = ranking::Excluded {
            indptr: excluded_indptr,
            indices: excluded_indices,
        };
        pool.install(|| {
            hnsw::top_k(
                items, view, queries, candidates, position, &excluded, k, ef_search,
            )
        })
    });
    ranked_arrays(py, ranked, n_rows, k)
}

/// Shape a kernel's `(positions, scores)` into the pair `recommend` expects.
///
/// Both indexes return the same thing and fail the same way, so the conversion and the
/// too-few-eligible message live here rather than once per search function.
fn ranked_arrays(
    py: Python<'_>,
    ranked: Result<(Vec<usize>, Vec<f64>), ranking::TooFewEligible>,
    n_rows: usize,
    k: usize,
) -> PyResult<Ranked<'_>> {
    let (order, scores) = ranked.map_err(|e| {
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

/// Build a hierarchical navigable small-world graph over dense item vectors.
///
/// `vectors` is `(n_items, dim)` row-major, and row `j` is the vector whose dot product
/// with a query is item `j`'s score. `m` caps the neighbours a node keeps above level 0
/// and `2 * m` caps them at level 0; `ef_construction` bounds the candidate list an
/// insertion keeps. Levels are drawn from `seed` alone, so they do not depend on the
/// order nodes are inserted in.
///
/// Returns the layered adjacency as `(node_level, links_indptr, links_indices,
/// entry_point)`: everything a search needs and nothing else, so a fitted index is
/// numpy arrays all the way down. `n_threads == 0` uses all cores; a build on more than
/// one thread is not reproducible, because concurrent insertions see different partial
/// graphs.
#[pyfunction]
fn hnsw_build_dense<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f64>,
    m: usize,
    ef_construction: usize,
    seed: u64,
    n_threads: usize,
) -> PyResult<HnswGraph<'py>> {
    let params = hnsw_params(m, ef_construction, seed)?;
    let dim = vectors.shape()[1];
    let items = hnsw::DenseItems {
        vectors: vectors.as_slice()?,
        dim,
    };
    hnsw_built(py, &items, &params, n_threads)
}

/// Build a hierarchical navigable small-world graph over sparse item vectors.
///
/// Row `j` of the CSR is item `j`'s vector, columns ascending, as Qdrant indexes a
/// sparse collection. `dim` is the number of columns, which for the item-item models is
/// the size of the catalog. See [`hnsw_build_dense`] for the rest.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hnsw_build_sparse<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    dim: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
    n_threads: usize,
) -> PyResult<HnswGraph<'py>> {
    let params = hnsw_params(m, ef_construction, seed)?;
    let data = data.as_slice()?;
    let cols = indices.as_slice()?;
    let indptr = borrow_csr(indptr.as_slice()?, cols, data, dim)?;
    let items = hnsw::SparseItems {
        matrix: sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols: dim,
            indptr: &indptr,
            indices: cols,
            data,
        },
    };
    hnsw_built(py, &items, &params, n_threads)
}

/// The `k` best candidates of each dense query under the graph, and their scores.
///
/// `candidates` holds the item index of each candidate position, ascending, and the
/// traversal passes *through* the items that are not candidates or are excluded rather
/// than stopping at them, so a filter costs recall rather than connectivity. Excluded
/// columns are candidate positions, as in `top_k_per_row`. A query the walk leaves short
/// is completed by an exact scan of its eligible items, so the result is always `k`
/// items the caller may show. `n_threads == 0` uses all cores.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hnsw_search_dense<'py>(
    py: Python<'py>,
    node_level: PyReadonlyArray1<'py, i32>,
    links_indptr: PyReadonlyArray1<'py, i64>,
    links_indices: PyReadonlyArray1<'py, i32>,
    entry_point: i64,
    vectors: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    candidates: PyReadonlyArray1<'py, i64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
    ef_search: usize,
    n_threads: usize,
) -> PyResult<Ranked<'py>> {
    let dim = vectors.shape()[1];
    if queries.shape()[1] != dim {
        return Err(PyValueError::new_err(
            "queries must have the same width as the indexed vectors.",
        ));
    }
    let items = hnsw::DenseItems {
        vectors: vectors.as_slice()?,
        dim,
    };
    let n_items = vectors.shape()[0];
    let view = hnsw_view(
        node_level.as_slice()?,
        links_indptr.as_slice()?,
        links_indices.as_slice()?,
        entry_point,
        n_items,
    )?;
    let (candidates, position) = graph_candidates(candidates.as_slice()?, n_items)?;
    let queries = hnsw::Queries::Dense {
        data: queries.as_slice()?,
        dim,
    };
    let excluded_cols = excluded_indices.as_slice()?;
    let excluded_indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_cols,
        queries.len(),
        candidates.len(),
    )?;
    hnsw_ranked(
        py,
        &items,
        &view,
        &queries,
        &candidates,
        &position,
        &excluded_indptr,
        excluded_cols,
        k,
        ef_search,
        n_threads,
    )
}

/// The `k` best candidates of each sparse query under a graph over sparse item vectors.
///
/// The item-item path. A query row is scattered once into a buffer of `dim` values, so
/// each distance is a gather over an item vector's stored entries rather than a merge of
/// two sorted runs. See [`hnsw_search_dense`] for the filtering rules.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hnsw_search_sparse<'py>(
    py: Python<'py>,
    node_level: PyReadonlyArray1<'py, i32>,
    links_indptr: PyReadonlyArray1<'py, i64>,
    links_indices: PyReadonlyArray1<'py, i32>,
    entry_point: i64,
    vectors_indptr: PyReadonlyArray1<'py, i64>,
    vectors_indices: PyReadonlyArray1<'py, i64>,
    vectors_data: PyReadonlyArray1<'py, f64>,
    dim: usize,
    queries_indptr: PyReadonlyArray1<'py, i64>,
    queries_indices: PyReadonlyArray1<'py, i64>,
    queries_data: PyReadonlyArray1<'py, f64>,
    candidates: PyReadonlyArray1<'py, i64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
    ef_search: usize,
    n_threads: usize,
) -> PyResult<Ranked<'py>> {
    let vectors_data = vectors_data.as_slice()?;
    let vectors_cols = vectors_indices.as_slice()?;
    let vectors_indptr = borrow_csr(vectors_indptr.as_slice()?, vectors_cols, vectors_data, dim)?;
    let n_items = vectors_indptr.len() - 1;
    let items = hnsw::SparseItems {
        matrix: sparse::Csr {
            n_rows: n_items,
            n_cols: dim,
            indptr: &vectors_indptr,
            indices: vectors_cols,
            data: vectors_data,
        },
    };
    let view = hnsw_view(
        node_level.as_slice()?,
        links_indptr.as_slice()?,
        links_indices.as_slice()?,
        entry_point,
        n_items,
    )?;
    let (candidates, position) = graph_candidates(candidates.as_slice()?, n_items)?;

    let queries_data = queries_data.as_slice()?;
    let queries_cols = queries_indices.as_slice()?;
    let queries_indptr = borrow_csr(queries_indptr.as_slice()?, queries_cols, queries_data, dim)?;
    let queries = hnsw::Queries::Scattered(sparse::Csr {
        n_rows: queries_indptr.len() - 1,
        n_cols: dim,
        indptr: &queries_indptr,
        indices: queries_cols,
        data: queries_data,
    });
    let excluded_cols = excluded_indices.as_slice()?;
    let excluded_indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_cols,
        queries.len(),
        candidates.len(),
    )?;
    hnsw_ranked(
        py,
        &items,
        &view,
        &queries,
        &candidates,
        &position,
        &excluded_indptr,
        excluded_cols,
        k,
        ef_search,
        n_threads,
    )
}

/// The `k` best candidates of each sparse query under a graph over dense item vectors.
///
/// EASE's path: its weight matrix is dense but catalog-wide, so scattering the query
/// would cost a full pass per distance. The query stays sparse and the gather runs over
/// its handful of nonzeros instead. See [`hnsw_search_dense`] for the filtering rules.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hnsw_search_sparse_dense<'py>(
    py: Python<'py>,
    node_level: PyReadonlyArray1<'py, i32>,
    links_indptr: PyReadonlyArray1<'py, i64>,
    links_indices: PyReadonlyArray1<'py, i32>,
    entry_point: i64,
    vectors: PyReadonlyArray2<'py, f64>,
    queries_indptr: PyReadonlyArray1<'py, i64>,
    queries_indices: PyReadonlyArray1<'py, i64>,
    queries_data: PyReadonlyArray1<'py, f64>,
    candidates: PyReadonlyArray1<'py, i64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
    ef_search: usize,
    n_threads: usize,
) -> PyResult<Ranked<'py>> {
    let dim = vectors.shape()[1];
    let n_items = vectors.shape()[0];
    let items = hnsw::DenseItems {
        vectors: vectors.as_slice()?,
        dim,
    };
    let view = hnsw_view(
        node_level.as_slice()?,
        links_indptr.as_slice()?,
        links_indices.as_slice()?,
        entry_point,
        n_items,
    )?;
    let (candidates, position) = graph_candidates(candidates.as_slice()?, n_items)?;

    let queries_data = queries_data.as_slice()?;
    let queries_cols = queries_indices.as_slice()?;
    let queries_indptr = borrow_csr(queries_indptr.as_slice()?, queries_cols, queries_data, dim)?;
    let queries = hnsw::Queries::Sparse(sparse::Csr {
        n_rows: queries_indptr.len() - 1,
        n_cols: dim,
        indptr: &queries_indptr,
        indices: queries_cols,
        data: queries_data,
    });
    let excluded_cols = excluded_indices.as_slice()?;
    let excluded_indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_cols,
        queries.len(),
        candidates.len(),
    )?;
    hnsw_ranked(
        py,
        &items,
        &view,
        &queries,
        &candidates,
        &position,
        &excluded_indptr,
        excluded_cols,
        k,
        ef_search,
        n_threads,
    )
}

/// Validate a quantized store and the shortlist width a search will use.
///
/// A fitted index pickles as plain numpy arrays, so every one of them can come back
/// altered or truncated; nothing below may panic or read out of bounds on a store that
/// does not describe itself.
fn quantized_checked(
    store: &quantized::Store<'_>,
    n_items: usize,
    dim: usize,
    bits: u32,
    oversample: usize,
) -> PyResult<()> {
    if !quantized::supported_bits(bits) {
        return Err(PyValueError::new_err(format!(
            "bits must be one of {:?}, got {bits}.",
            quantized::SUPPORTED_BITS
        )));
    }
    if oversample == 0 {
        return Err(PyValueError::new_err("oversample must be >= 1."));
    }
    if !store.is_well_formed(n_items, dim) {
        return Err(PyValueError::new_err(
            "the code arrays do not describe a quantized store.",
        ));
    }
    Ok(())
}

/// Borrow a dense store's codes, with the row stride the packing implies.
fn dense_store<'a>(
    codes: &'a [u8],
    bits: u32,
    dim: usize,
    scale: &'a [f64],
    offset: &'a [f64],
) -> quantized::Store<'a> {
    quantized::Store::Dense {
        codes: quantized::Codes {
            data: codes,
            bits,
            stride: quantized::row_bytes(dim, bits),
        },
        dim,
        scale,
        offset,
    }
}

/// Run a quantized scan and shape its result the way `recommend` expects.
///
/// No thread count and no pool: this index is sequential over queries by construction,
/// which is the property it is chosen for. The GIL is still released, because the scan
/// touches nothing Python owns.
#[allow(clippy::too_many_arguments)]
fn quantized_ranked<'py>(
    py: Python<'py>,
    store: &quantized::Store<'_>,
    exact: &impl vectors::Items,
    queries: &vectors::Queries<'_>,
    candidates: &[usize],
    excluded_indptr: &[usize],
    excluded_indices: &[i64],
    k: usize,
    oversample: usize,
) -> PyResult<Ranked<'py>> {
    let n_rows = queries.len();
    let ranked = py.detach(|| {
        let excluded = ranking::Excluded {
            indptr: excluded_indptr,
            indices: excluded_indices,
        };
        quantized::top_k(store, exact, queries, candidates, &excluded, k, oversample)
    });
    ranked_arrays(py, ranked, n_rows, k)
}

/// The `k` best candidates of each dense query under a quantized flat scan.
///
/// `codes` holds `dim` codes of `bits` each per item, packed `8 / bits` to a byte with
/// the low-order code first and every item's run starting on a byte boundary. `scale`
/// and `offset` are per dimension, and `vectors` is the exact matrix the shortlist is
/// reranked against -- the same vectors the codes were made from, in the same order.
///
/// `candidates` holds the item index of each candidate position, ascending; excluded
/// columns are candidate positions, as in `top_k_per_row`. The scan sees every eligible
/// candidate, so a query short of `k` is genuinely short and raises rather than falling
/// back. `oversample * k` candidates are shortlisted by their codes and then rescored
/// exactly, so the scores returned are always the exact ones.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quantized_search_dense<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u8>,
    bits: u32,
    scale: PyReadonlyArray1<'py, f64>,
    offset: PyReadonlyArray1<'py, f64>,
    vectors: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    candidates: PyReadonlyArray1<'py, i64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
    oversample: usize,
) -> PyResult<Ranked<'py>> {
    let dim = vectors.shape()[1];
    if queries.shape()[1] != dim {
        return Err(PyValueError::new_err(
            "queries must have the same width as the indexed vectors.",
        ));
    }
    let n_items = vectors.shape()[0];
    let store = dense_store(
        codes.as_slice()?,
        bits,
        dim,
        scale.as_slice()?,
        offset.as_slice()?,
    );
    quantized_checked(&store, n_items, dim, bits, oversample)?;
    let exact = vectors::DenseItems {
        vectors: vectors.as_slice()?,
        dim,
    };
    let candidates = index_candidates(candidates.as_slice()?, n_items)?;
    let queries = vectors::Queries::Dense {
        data: queries.as_slice()?,
        dim,
    };
    let excluded_cols = excluded_indices.as_slice()?;
    let excluded_indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_cols,
        queries.len(),
        candidates.len(),
    )?;
    quantized_ranked(
        py,
        &store,
        &exact,
        &queries,
        &candidates,
        &excluded_indptr,
        excluded_cols,
        k,
        oversample,
    )
}

/// The `k` best candidates of each sparse query over quantized *dense* item vectors.
///
/// EASE's shape: catalog-wide dense item rows and a query of a handful of nonzeros, so
/// the coarse pass gathers over the query's own columns rather than the item's width.
/// See [`quantized_search_dense`] for the packing and the filtering rules.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quantized_search_sparse_dense<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u8>,
    bits: u32,
    scale: PyReadonlyArray1<'py, f64>,
    offset: PyReadonlyArray1<'py, f64>,
    vectors: PyReadonlyArray2<'py, f64>,
    queries_indptr: PyReadonlyArray1<'py, i64>,
    queries_indices: PyReadonlyArray1<'py, i64>,
    queries_data: PyReadonlyArray1<'py, f64>,
    candidates: PyReadonlyArray1<'py, i64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
    oversample: usize,
) -> PyResult<Ranked<'py>> {
    let dim = vectors.shape()[1];
    let n_items = vectors.shape()[0];
    let store = dense_store(
        codes.as_slice()?,
        bits,
        dim,
        scale.as_slice()?,
        offset.as_slice()?,
    );
    quantized_checked(&store, n_items, dim, bits, oversample)?;
    let exact = vectors::DenseItems {
        vectors: vectors.as_slice()?,
        dim,
    };
    let candidates = index_candidates(candidates.as_slice()?, n_items)?;

    let queries_data = queries_data.as_slice()?;
    let queries_cols = queries_indices.as_slice()?;
    let queries_indptr = borrow_csr(queries_indptr.as_slice()?, queries_cols, queries_data, dim)?;
    let queries = vectors::Queries::Sparse(sparse::Csr {
        n_rows: queries_indptr.len() - 1,
        n_cols: dim,
        indptr: &queries_indptr,
        indices: queries_cols,
        data: queries_data,
    });
    let excluded_cols = excluded_indices.as_slice()?;
    let excluded_indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_cols,
        queries.len(),
        candidates.len(),
    )?;
    quantized_ranked(
        py,
        &store,
        &exact,
        &queries,
        &candidates,
        &excluded_indptr,
        excluded_cols,
        k,
        oversample,
    )
}

/// The `k` best candidates of each sparse query over quantized sparse item vectors.
///
/// The item-item path. One code per stored value, sharing the rows' own
/// `indptr`/`indices`, with a single `scale` and `offset` over the whole matrix -- a
/// catalog-sized column dimension has no per-column statistics worth keeping. The query
/// is scattered once into a buffer of `dim` values and every candidate gathers its own
/// stored columns out of it. See [`quantized_search_dense`] for the filtering rules.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quantized_search_sparse<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u8>,
    bits: u32,
    scale: f64,
    offset: f64,
    vectors_indptr: PyReadonlyArray1<'py, i64>,
    vectors_indices: PyReadonlyArray1<'py, i64>,
    vectors_data: PyReadonlyArray1<'py, f64>,
    dim: usize,
    queries_indptr: PyReadonlyArray1<'py, i64>,
    queries_indices: PyReadonlyArray1<'py, i64>,
    queries_data: PyReadonlyArray1<'py, f64>,
    candidates: PyReadonlyArray1<'py, i64>,
    excluded_indptr: PyReadonlyArray1<'py, i64>,
    excluded_indices: PyReadonlyArray1<'py, i64>,
    k: usize,
    oversample: usize,
) -> PyResult<Ranked<'py>> {
    let vectors_data = vectors_data.as_slice()?;
    let vectors_cols = vectors_indices.as_slice()?;
    let vectors_indptr = borrow_csr(vectors_indptr.as_slice()?, vectors_cols, vectors_data, dim)?;
    let n_items = vectors_indptr.len() - 1;
    let matrix = sparse::Csr {
        n_rows: n_items,
        n_cols: dim,
        indptr: &vectors_indptr,
        indices: vectors_cols,
        data: vectors_data,
    };
    let store = quantized::Store::Sparse {
        codes: quantized::Codes {
            data: codes.as_slice()?,
            bits,
            stride: 1,
        },
        matrix: sparse::Csr { ..matrix },
        scale,
        offset,
    };
    quantized_checked(&store, n_items, dim, bits, oversample)?;
    let exact = vectors::SparseItems { matrix };
    let candidates = index_candidates(candidates.as_slice()?, n_items)?;

    let queries_data = queries_data.as_slice()?;
    let queries_cols = queries_indices.as_slice()?;
    let queries_indptr = borrow_csr(queries_indptr.as_slice()?, queries_cols, queries_data, dim)?;
    let queries = vectors::Queries::Scattered(sparse::Csr {
        n_rows: queries_indptr.len() - 1,
        n_cols: dim,
        indptr: &queries_indptr,
        indices: queries_cols,
        data: queries_data,
    });
    let excluded_cols = excluded_indices.as_slice()?;
    let excluded_indptr = check_excluded(
        excluded_indptr.as_slice()?,
        excluded_cols,
        queries.len(),
        candidates.len(),
    )?;
    quantized_ranked(
        py,
        &store,
        &exact,
        &queries,
        &candidates,
        &excluded_indptr,
        excluded_cols,
        k,
        oversample,
    )
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
    m.add_function(wrap_pyfunction!(ease_inverse_gram, m)?)?;
    m.add_function(wrap_pyfunction!(ease_weights, m)?)?;
    m.add_function(wrap_pyfunction!(ease_weights_from_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(fm_als_fit, m)?)?;
    m.add_function(wrap_pyfunction!(hnsw_build_dense, m)?)?;
    m.add_function(wrap_pyfunction!(hnsw_build_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(hnsw_search_dense, m)?)?;
    m.add_function(wrap_pyfunction!(hnsw_search_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(hnsw_search_sparse_dense, m)?)?;
    m.add_function(wrap_pyfunction!(item_cosine_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_search_dense, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_search_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_search_sparse_dense, m)?)?;
    m.add_function(wrap_pyfunction!(item_gram, m)?)?;
    m.add_function(wrap_pyfunction!(item_knn_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(recommend_from_dense_rows, m)?)?;
    m.add_function(wrap_pyfunction!(recommend_from_factors, m)?)?;
    m.add_function(wrap_pyfunction!(recommend_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(rp3beta_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(slim_elasticnet_weights, m)?)?;
    m.add_function(wrap_pyfunction!(slim_elasticnet_weights_from_gram, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_per_row, m)?)
}

//! Native kernels for skrecsys, exposed to Python as `skrecsys._core`.

pub mod ease;
pub mod fm_als;
pub mod knn;
pub mod ranking;
pub mod rp3beta;
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
    let indices = to_usize(indices.as_slice()?, "indices", n_features)?;
    let group = to_usize(group.as_slice()?, "group", reg_w.len())?;

    Ok(py.detach(|| {
        let x = sparse::Csr {
            n_rows,
            n_cols: n_features,
            indptr: &indptr,
            indices: &indices,
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
    let indices = to_usize(indices.as_slice()?, "indices", n_cols)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = py.detach(|| {
        let w = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices: &indices,
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
    let indices = to_usize(indices.as_slice()?, "indices", n_cols)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let weights = py.detach(|| {
        let r = sparse::Csr {
            n_rows: indptr.len() - 1,
            n_cols,
            indptr: &indptr,
            indices: &indices,
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
    let indices = to_usize(indices.as_slice()?, "indices", n_cols)?;
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
            indices: &indices,
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

/// Column indices of the `k` best eligible entries of each row of `scores`.
///
/// Ranks by descending score, breaking ties by ascending column index. Raises
/// `ValueError` if any row has fewer than `k` eligible entries.
#[pyfunction]
fn top_k_per_row<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray2<'py, f64>,
    eligible: PyReadonlyArray2<'py, bool>,
    k: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let shape = scores.shape().to_vec();
    if eligible.shape() != shape.as_slice() {
        return Err(PyValueError::new_err(
            "scores and eligible must have the same shape.",
        ));
    }
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let scores = scores.as_slice()?;
    let eligible = eligible.as_slice()?;

    let selected = py
        .detach(|| ranking::top_k_per_row(scores, eligible, n_cols, k))
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ease_weights, m)?)?;
    m.add_function(wrap_pyfunction!(fm_als_fit, m)?)?;
    m.add_function(wrap_pyfunction!(item_knn_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(rp3beta_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_per_row, m)?)
}

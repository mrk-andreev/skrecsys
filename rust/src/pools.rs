//! Thread pools the kernels run in, built once per thread count and reused.
//!
//! Every entry point used to build a fresh `rayon::ThreadPool` and drop it on return,
//! which spawns and joins one OS thread per core on every call. For a `recommend` of a
//! single user that was two hundred microseconds of a call whose arithmetic takes one,
//! so the pools are kept here instead, keyed by the `n_threads` the caller asked for.
//!
//! A forked child inherits the map but none of the threads behind it, and a pool whose
//! workers do not exist deadlocks the first `install`. The cache therefore remembers the
//! process it was filled in and starts over when that is no longer the current one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use rayon::ThreadPool;

struct Cache {
    pid: u32,
    pools: HashMap<usize, Arc<ThreadPool>>,
}

static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

/// The pool with `n_threads` workers, `0` meaning one per core, as rayon reads it.
pub fn pool(n_threads: usize) -> PyResult<Arc<ThreadPool>> {
    let pid = std::process::id();
    let cache = CACHE.get_or_init(|| {
        Mutex::new(Cache {
            pid,
            pools: HashMap::new(),
        })
    });
    // A panic while the lock was held cannot leave the map half-updated: the only
    // mutation is one insert of a fully built pool.
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cache.pid != pid {
        // Forked: the workers behind the inherited pools belong to the parent. The
        // handles are leaked rather than dropped, since dropping one would try to
        // signal threads that are not there.
        for (_, stale) in cache.pools.drain() {
            std::mem::forget(stale);
        }
        cache.pid = pid;
    }
    if let Some(pool) = cache.pools.get(&n_threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    );
    cache.pools.insert(n_threads, Arc::clone(&pool));
    Ok(pool)
}

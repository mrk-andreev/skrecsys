//! The native kernels behind skrecsys, as a plain Rust library.
//!
//! They live in a crate of their own so that two targets can link them: the
//! `skrecsys` package one directory up, which wraps them as the `skrecsys._core`
//! extension module, and the criterion benches in `benches/`. Keeping them here is what
//! lets the extension stay a `cdylib` and keep its fat LTO -- a target that also emits
//! an `rlib` has LTO dropped silently -- while the benches still link the same code.
//!
//! Nothing in this crate knows about Python: the modules take borrowed slices and the
//! sparse types in [`sparse`], and all validation, conversion and GIL handling belongs
//! to the wrapper.

#![forbid(unsafe_code)]

pub mod bpr;
pub mod ease;
pub mod encode;
pub mod fm_als;
pub mod hnsw;
pub mod knn;
pub mod prune;
pub mod quantized;
pub mod ranking;
pub mod recommend;
pub mod rp3beta;
pub mod slim;
pub mod sparse;
pub mod vectors;

//! The engine's Rust CUDA kernels: layer ops, greedy sampling, and the small
//! MoE, Gated DeltaNet, K2, attention and glue kernels.
//!
//! This is the only source for the kernels the engine launches from the
//! embedded PTX; there is no CUDA C++ twin. Raw pointers keep the cudarc
//! launch ABI, including in-place output. The CPU references are in
//! `llmcuda-kernels`, and the launchers and their differential tests in
//! `llmcuda-cuda` and `llmcuda-engine`.
//!
//! The PTX is built without implicit multiply-add fusion (`regenerate.py`),
//! so each `mul_add` here is one rounding and every other product is rounded
//! before it is added. Several kernels are compared bit for bit with fused
//! CUDA C++ kernels that share their arithmetic; do not reorder a sum or
//! change which products fuse without changing those too.
//!
//! One module per kernel family; `common` holds what more than one uses.

mod attention;
mod common;
mod decision;
mod dense;
mod gdn;
mod gemm;
mod glue;
mod inkling;
mod inkling_decode;
mod k2;
mod layer_ops;
mod moe;
mod quant;
mod sampling;
mod vision;

fn main() {}

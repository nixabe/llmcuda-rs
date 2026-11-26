#!/usr/bin/env python3
"""Rebuild the pinned Rust kernel and refresh the engine's embedded PTX."""
import hashlib
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src/main.rs"
OUTPUT = HERE.parents[1] / "crates/llmcuda-cuda/src/kernels/rust/tensor_add.ptx"
REVISION = "f35d6ccc2e03a54e73e3796f415f11313e439d50"

source_before = SOURCE.read_bytes()
subprocess.run(
    ["cargo", "oxide", "build", "--arch", "sm_75", "--", "--release", "--locked"],
    cwd=HERE,
    # No implicit mul+add fusion: a kernel fuses where the source says
    # `mul_add` and nowhere else, so its rounding is readable from the source.
    env={**os.environ, "CUDA_OXIDE_NO_FMA": "1"},
    check=True,
)
if SOURCE.read_bytes() != source_before:
    raise SystemExit("Rust source changed during compilation; refusing to publish PTX")
ptx = (HERE / "llmcuda_rust_device.ptx").read_text()
if ".target sm_75\n" not in ptx or any(
    f".visible .entry {entry}(" not in ptx
    for entry in (
        "tensor_add", "swiglu_mul", "sigmoid_gate_mul",
        "rms_norm_rows", "rms_norm_swiglu_rows", "softplus_elementwise", "rope_partial",
        "attn_rope_partial_neox", "attn_split_query_gate", "attn_row_positions",
        "rms_norm_swiglu_q8_rows", "conv1d_causal_depthwise", "conv1d_step",
        "conv1d_step_batch", "conv1d_update_state",
        "moe_swiglu", "moe_reduce", "gdn_silu", "gdn_split_qkv", "gdn_gates",
        "gdn_scan_decay", "k2_combine", "k2_combine_residual", "k2_values", "k2_gate",
        "attn_flash_decode_combine",
    )
):
    raise SystemExit("compiler output has the wrong target or entry point")
header = (
    "// Generated from experiments/cuda-oxide/src/main.rs; do not edit.\n"
    f"// cuda-oxide (NVIDIA/cuda-rust): {REVISION}; Rust 1.99.1; sm_75\n"
    f"// Rust source SHA-256: {hashlib.sha256(source_before).hexdigest()}\n"
)
# libdevice inline PTX can carry trailing spaces; normalize only whitespace.
OUTPUT.write_text(header + "\n".join(line.rstrip() for line in ptx.splitlines()) + "\n")

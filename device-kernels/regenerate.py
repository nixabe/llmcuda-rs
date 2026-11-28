#!/usr/bin/env python3
"""Rebuild the pinned Rust kernel and refresh the engine's embedded PTX."""
import hashlib
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCES = sorted((HERE / "src").glob("*.rs"))
OUTPUT = HERE.parent / "crates/llmcuda-cuda/src/kernels/rust/tensor_add.ptx"
REVISION = "f35d6ccc2e03a54e73e3796f415f11313e439d50"


def read_sources():
    """Every source file's name and bytes, in one digestible stream."""
    return b"".join(
        path.name.encode() + b"\0" + path.read_bytes() + b"\0" for path in SOURCES
    )


source_before = read_sources()
subprocess.run(
    ["cargo", "oxide", "build", "--arch", "sm_75", "--", "--release", "--locked"],
    cwd=HERE,
    # No implicit mul+add fusion: a kernel fuses where the source says
    # `mul_add` and nowhere else, so its rounding is readable from the source.
    env={**os.environ, "CUDA_OXIDE_NO_FMA": "1"},
    check=True,
)
if read_sources() != source_before or SOURCES != sorted((HERE / "src").glob("*.rs")):
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
        "argmax_partial", "argmax_final", "hg_splitk_reduce", "k2_dispatch_count",
        "k2_dispatch_prefix", "k2_dispatch_gather", "k2_norm", "dense_ffn_add_residual",
        "mtp_concat_eh", "dflash_tap", "moe_align_count",
        "hg_to_half", "vision_gelu", "vision_gelu_erf", "vision_add_f16", "vision_half_to_float",
        "attn_kv_append", "dense_swiglu_narrow", "dec_add_owner_rows", "dec_span_mean",
        "dec_attention_combine", "k2_rotary_apply",
        "attn_flash_decode_combine_k2_batch", "embed_q8_0", "dequantize_q8_0",
        "gdn_silu_split_qkv", "gdn_conv_silu_split_step_batch", "gdn_normalize_qk",
        "gdn_chunk_normalize_qk", "dense_rms_norm_narrow", "dec_layer_norm",
        "dequantize_q6_k", "vision_softmax_rows", "vision_rope_qk",
        "k2_rotary_prepare", "moe_block_gate_and_combine",
        "gdn_proj_f32", "dec_final_scores", "gdn_norm_alpha_beta_gates",
        "gdn_recurrent_step", "argmax_prob",
        "moe_route", "moe_dispatch_t1", "moe_route_dispatch_t1", "moe_align_block_size",
        "attn_rope_partial_imrope", "attn_kv_append_kv8", "attn_decode_rope_append_batch",
        "vision_layer_norm", "hg_q8_to_half_w", "dec_lexical", "dec_question_fields",
        "dec_features", "fwd_embed_quant",
        "k2_rotary_append", "k2_rotary_append_kv8",
        "mma_quantize_rows_q8", "mma_repack_q8_0_h",
        "ink_embed", "ink_linear", "ink_mix", "ink_ctrl_pair", "ink_ctrl_apply",
        "ink_conv", "ink_clip", "ink_rms_groups", "ink_rel_table", "ink_attn",
        "ink_gather_half", "ink_situ_half", "ink_moe_combine", "ink_softcap",
        "ink_dec_inputs", "ink_dec_kv", "ink_dec_attn", "ink_dec_attn_merge", "ink_dec_commit", "ink_dec_ffn_in", "ink_dec_ffn_out",
        "ink_dec_gate", "ink_dec_ctrl", "ink_dec_norm", "ink_dec_router", "ink_dec_route",
    )
):
    raise SystemExit("compiler output has the wrong target or entry point")
header = (
    "// Generated from device-kernels/src/*.rs; do not edit.\n"
    f"// cuda-oxide (NVIDIA/cuda-rust): {REVISION}; Rust 1.99.1; sm_75\n"
    f"// Rust source SHA-256 (each file's name and bytes, NUL-terminated, by name): {hashlib.sha256(source_before).hexdigest()}\n"
)
# libdevice inline PTX can carry trailing spaces; normalize only whitespace.
OUTPUT.write_text(header + "\n".join(line.rstrip() for line in ptx.splitlines()) + "\n")

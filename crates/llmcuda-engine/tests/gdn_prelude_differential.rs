//! `gdn_scan_prelude` and `gdn_scan_prefill_fold` against the launches they
//! replace.
//!
//! A batched prefill ran the causal convolution once a sequence, then
//! `gdn_silu_split_qkv`, then `gdn_chunk_normalize_qk`, then
//! `gdn_scan_prefill`. The fused prelude does the first three in one launch
//! and the folded scan does the fourth in fewer instructions, and both
//! promise the same floats: every operation on the same operands in the same
//! order. So the activated `v`, the normalized `q` and `k`, the decay, the
//! scan's outputs and the final recurrent states must be **bit-identical**
//! across the two forms, and so must the convolution caches once advanced.
//!
//! Synthetic weights, caches and states, each sequence its own, so a wrong
//! slot or stride has something to trip over; no model file. The lengths
//! cover a chunk shorter than the convolution's carry (every tap of the
//! first token reads the cache), one ragged against every block size, and
//! the 512 the scan's own differential test uses.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_cuda::kernels::gdn_chunked::GdnChunkedKernels;
use llmcuda_engine::block::gdn::{GdnBlock, GdnGeometry};
use llmcuda_kernels::rng::Xorshift64Star;
use llmcuda_model::config::ModelConfig;

fn setup() -> Option<Arc<CudaContext>> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return None;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    Some(ctx)
}

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

/// What a scan and everything upstream of it leaves behind.
struct Outputs {
    v: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out: Vec<f32>,
    states: Vec<Vec<f32>>,
    caches: Vec<Vec<f32>>,
}

/// Which launches produce the scan and its inputs.
#[derive(Clone, Copy, Debug)]
enum Form {
    /// Convolution, SiLU and split, normalize, unfolded scan: the reference.
    Separate,
    /// The same three upstream launches, then the folded scan.
    FoldedScan,
    /// The fused prelude, then the folded scan.
    Fused,
}

fn differing(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn run_case(ctx: &Arc<CudaContext>, seed: u64, n: usize, seq_len: usize) {
    let stream = ctx.default_stream();
    let config = ModelConfig::qwen3_6_35b_a3b();
    let total = n * seq_len;
    let geometry = GdnGeometry::from_config(&config, total, 1e-6);
    let block = GdnBlock::new(ctx, geometry).expect("kernels compile");
    let (head_dim, value_heads, qk_heads) =
        (geometry.head_dim, geometry.value_heads, geometry.qk_heads);
    let conv_dim = geometry.conv_dim();
    let key_dim = geometry.key_dim();
    let value_dim = geometry.value_dim();
    let conv_kernel = geometry.conv_kernel;

    let mut rng = Xorshift64Star::new(seed);
    let x = stream
        .clone_htod(&rng.vec_f32(total * conv_dim, -2.0, 2.0))
        .expect("upload x");
    let weight = stream
        .clone_htod(&rng.vec_f32(conv_dim * conv_kernel, -0.5, 0.5))
        .expect("upload conv weight");
    let caches_host: Vec<Vec<f32>> = (0..n)
        .map(|_| rng.vec_f32(geometry.conv_state_len(), -2.0, 2.0))
        .collect();
    let states_host: Vec<Vec<f32>> = (0..n)
        .map(|_| rng.vec_f32(geometry.recurrent_state_len(), -0.1, 0.1))
        .collect();
    let log_decay = stream
        .clone_htod(&rng.vec_f32(total * value_heads, -0.3, 0.0))
        .expect("upload log_decay");
    let beta = stream
        .clone_htod(&rng.vec_f32(total * value_heads, 0.0, 1.0))
        .expect("upload beta");

    let run = |form: Form| -> Outputs {
        let mut kernels =
            GdnChunkedKernels::new(ctx, head_dim, value_heads, qk_heads, geometry.chunk_len)
                .expect("scan kernels compile");
        kernels.set_scan_fold(!matches!(form, Form::Separate));
        let mut scratch = kernels.scratch(&stream, total).expect("scratch");
        let mut caches: Vec<CudaSlice<f32>> = caches_host
            .iter()
            .map(|c| stream.clone_htod(c).expect("upload cache"))
            .collect();
        let states: Vec<CudaSlice<f32>> = states_host
            .iter()
            .map(|s| stream.clone_htod(s).expect("upload state"))
            .collect();
        let state_ptrs: Vec<u64> = states.iter().map(|s| s.device_ptr(&stream).0).collect();
        let mut v = stream.alloc_zeros::<f32>(total * value_dim).expect("v");
        let mut out = stream.alloc_zeros::<f32>(total * value_dim).expect("out");

        if matches!(form, Form::Fused) {
            let cache_ptrs: Vec<u64> = caches.iter().map(|c| c.device_ptr(&stream).0).collect();
            // SAFETY: the states and caches are live, distinct allocations of
            // the geometry's lengths, held for the call; every flattened
            // buffer was allocated above at `[n][seq_len][width]`.
            unsafe {
                kernels
                    .prelude_scan_batch_raw(
                        &stream,
                        &mut scratch,
                        &state_ptrs,
                        &cache_ptrs,
                        &x,
                        &weight,
                        conv_kernel,
                        &mut v,
                        &log_decay,
                        &beta,
                        &mut out,
                        seq_len,
                    )
                    .expect("fused prelude and scan");
            }
            for (seq, cache) in caches.iter_mut().enumerate() {
                let xs = x.slice(seq * seq_len * conv_dim..(seq + 1) * seq_len * conv_dim);
                // The advance takes an owned-slice reference; copy the view.
                let mut owned = stream
                    .alloc_zeros::<f32>(seq_len * conv_dim)
                    .expect("x seq");
                stream.memcpy_dtod(&xs, &mut owned).expect("copy x seq");
                block
                    .layer_ops()
                    .conv1d_advance_state(&stream, &owned, cache, seq_len, conv_dim, conv_kernel)
                    .expect("cache advance");
            }
        } else {
            let mut conv_raw = stream
                .alloc_zeros::<f32>(total * conv_dim)
                .expect("conv_raw");
            for (seq, cache) in caches.iter_mut().enumerate() {
                let range = seq * seq_len * conv_dim..(seq + 1) * seq_len * conv_dim;
                let mut xs = stream
                    .alloc_zeros::<f32>(seq_len * conv_dim)
                    .expect("x seq");
                stream
                    .memcpy_dtod(&x.slice(range.clone()), &mut xs)
                    .expect("copy x seq");
                let mut ys = stream
                    .alloc_zeros::<f32>(seq_len * conv_dim)
                    .expect("y seq");
                block
                    .layer_ops()
                    .conv1d(
                        &stream,
                        &xs,
                        &weight,
                        cache,
                        &mut ys,
                        seq_len,
                        conv_dim,
                        conv_kernel,
                    )
                    .expect("convolution");
                stream
                    .memcpy_dtod(&ys, &mut conv_raw.slice_mut(range))
                    .expect("copy conv seq");
            }
            let mut silu = stream.alloc_zeros::<f32>(total * conv_dim).expect("silu");
            let mut q = stream.alloc_zeros::<f32>(total * key_dim).expect("q");
            let mut k = stream.alloc_zeros::<f32>(total * key_dim).expect("k");
            block
                .silu_split_qkv(&stream, &conv_raw, &mut silu, &mut q, &mut k, &mut v, total)
                .expect("silu and split");
            // SAFETY: as above.
            unsafe {
                kernels
                    .scan_batch_raw(
                        &stream,
                        &mut scratch,
                        &state_ptrs,
                        &q,
                        &k,
                        &v,
                        &log_decay,
                        &beta,
                        &mut out,
                        seq_len,
                    )
                    .expect("normalize and scan");
            }
        }

        let (q_norm, k_norm) = scratch.normalized();
        Outputs {
            v: dtoh(&stream, &v),
            q_norm: dtoh(&stream, q_norm),
            k_norm: dtoh(&stream, k_norm),
            out: dtoh(&stream, &out),
            states: states.iter().map(|s| dtoh(&stream, s)).collect(),
            caches: caches.iter().map(|c| dtoh(&stream, c)).collect(),
        }
    };

    let reference = run(Form::Separate);
    assert!(
        reference.out.iter().any(|&o| o != 0.0),
        "the reference scan wrote nothing"
    );
    for form in [Form::FoldedScan, Form::Fused] {
        let got = run(form);
        let mut wrong = Vec::new();
        let mut note = |what: &str, a: &[f32], b: &[f32]| {
            let d = differing(a, b);
            if d != 0 {
                wrong.push(format!("{what}: {d} of {}", a.len()));
            }
        };
        note("v", &got.v, &reference.v);
        note("q_norm", &got.q_norm, &reference.q_norm);
        note("k_norm", &got.k_norm, &reference.k_norm);
        note("out", &got.out, &reference.out);
        for seq in 0..n {
            note(
                &format!("state {seq}"),
                &got.states[seq],
                &reference.states[seq],
            );
            note(
                &format!("conv cache {seq}"),
                &got.caches[seq],
                &reference.caches[seq],
            );
        }
        assert!(
            wrong.is_empty(),
            "{form:?} at {n} x {seq_len} tokens differs from the separate launches in bits: \
             {wrong:?}"
        );
        println!(
            "{form:?} at {n} x {seq_len} tokens: {} outputs, {} normalized q/k and {n} final \
             states bit-identical",
            got.out.len(),
            got.q_norm.len() * 2,
        );
    }
}

#[test]
fn a_chunk_shorter_than_the_convolutions_carry_is_bit_identical() {
    let Some(ctx) = setup() else { return };
    run_case(&ctx, 0x_C0DE_5EED_0101, 3, 2);
}

#[test]
fn three_ragged_sequences_are_bit_identical() {
    let Some(ctx) = setup() else { return };
    run_case(&ctx, 0x_C0DE_5EED_0102, 3, 37);
}

#[test]
fn one_512_token_sequence_is_bit_identical() {
    let Some(ctx) = setup() else { return };
    run_case(&ctx, 0x_C0DE_5EED_0103, 1, 512);
}

#[test]
fn every_pointer_slot_is_bit_identical() {
    let Some(ctx) = setup() else { return };
    run_case(&ctx, 0x_C0DE_5EED_0104, 8, 19);
}

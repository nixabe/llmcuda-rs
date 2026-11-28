//! P1–P5: the checks GATE 0 requires to pass on a real GPU.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use half::f16;
use llmcuda_cuda::CheckOutcome;
use llmcuda_kernels::compare::compare;
use llmcuda_kernels::norm::residual_add;
use llmcuda_kernels::rng::Xorshift64Star;
use llmcuda_trt::plugin::Field;
use llmcuda_trt::{
    AllocationStrategy, Builder, BuilderConfig, DataType, ElementWise, GpuAllocator, HostMemory,
    MatrixOp, Network, Runtime,
};
use tracing::info;

use crate::env::{CountingAllocator, Env, Error, Unpooled, free_bytes};
use crate::plugins::{self, CTX_SEEN, NAMESPACE};

/// Fresh builder objects with no auxiliary streams, so everything runs on
/// the stream given to `enqueueV3`.
pub fn config<'b>(builder: &'b Builder) -> Result<BuilderConfig<'b>, Error> {
    let config = builder.config()?;
    config.set_max_aux_streams(0)?;
    Ok(config)
}

pub fn build(
    builder: &Builder,
    network: &Network<'_>,
    config: &BuilderConfig<'_>,
) -> Result<HostMemory, Error> {
    let started = Instant::now();
    let plan = builder.build(network, config)?;
    info!(
        "  built {} layers into a {:.1} KiB plan in {:.1} s",
        network.nb_layers(),
        plan.len() as f64 / 1024.0,
        started.elapsed().as_secs_f64()
    );
    Ok(plan)
}

/// Bit-level comparison: `(differing elements, first index)`.
pub fn bit_diff(a: &[f32], b: &[f32]) -> (usize, Option<usize>) {
    let mut first = None;
    let n = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(i, (x, y))| {
            let differ = x.to_bits() != y.to_bits();
            if differ && first.is_none() {
                first = Some(*i);
            }
            differ
        })
        .count();
    (n + a.len().abs_diff(b.len()), first)
}

/// Operands that stress exactness: signed zero, subnormals, cancellation, a
/// rounding tie, and |a| >> |b|. The first entries are the edges
/// the residual add is gated on.
pub fn exact_operands(n: usize, seed: u64, edges: bool) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let mut a = rng.vec_f32(n, -100.0, 100.0);
    let mut b = rng.vec_f32(n, -1e-3, 1e-3);
    if edges {
        let edge = [
            (-0.0, -0.0),
            (0.0, -0.0),
            (f32::from_bits(1), f32::from_bits(1)),
            (f32::MIN_POSITIVE, -f32::from_bits(1)),
            (f32::from_bits(0x0040_0000), f32::from_bits(0x0000_0003)),
            (f32::MAX, -f32::MAX),
            (1.0, f32::EPSILON / 2.0),
            (-1.0, -f32::EPSILON / 2.0),
        ];
        for (i, &(x, y)) in edge.iter().take(n).enumerate() {
            a[i] = x;
            b[i] = y;
        }
    }
    (a, b)
}

/// A labelled check.
pub type Check = (&'static str, fn(&Env) -> Result<CheckOutcome, Error>);

pub fn run_all(env: &Env, selected: &dyn Fn(&str) -> bool) -> Vec<(&'static str, CheckOutcome)> {
    let checks: [Check; 5] = [
        ("P1 library coexistence, one context", p1),
        ("P2 enqueueV3 on cudarc's stream", p2),
        ("P3 Rust plugin trampoline", p3),
        ("P4 CUDA graph capture of enqueueV3", p4),
        ("P5 no allocation across 100 enqueues", p5),
    ];
    checks
        .into_iter()
        .filter(|(label, _)| selected(label))
        .map(|(label, check)| {
            info!("{label}");
            let outcome = check(env).unwrap_or_else(|e| CheckOutcome::Fail(format!("error: {e}")));
            info!("  -> {outcome}");
            (label, outcome)
        })
        .collect()
}

/// P1: TensorRT and cudarc in one process, one context.
///
/// A native add feeds `ProbeCtx`, so the plugin's input is activation
/// memory TensorRT allocated itself (default strategy, default allocator).
/// The plugin records the current context, the stream's context and that
/// buffer's context; all three must be cudarc's, which must be the primary.
fn p1(env: &Env) -> Result<CheckOutcome, Error> {
    let lib = llmcuda_trt::library_version();
    let hdr = llmcuda_trt::header_version();
    if lib != hdr {
        return Ok(CheckOutcome::Fail(format!(
            "libnvinfer {lib:?} != headers {hdr:?}"
        )));
    }
    const N: i64 = 4096;
    let builder = Builder::new()?;
    let network = builder.network()?;
    let a = network.input("a", DataType::Float, [N])?;
    let b = network.input("b", DataType::Float, [N])?;
    let t = network.elementwise(a, b, ElementWise::Sum)?;
    let out = network.plugin(&[t], "ProbeCtx", "1", NAMESPACE, &[], 1)?[0];
    out.set_name("out")?;
    network.mark_output(out)?;
    let plan = build(&builder, &network, &config(&builder)?)?;

    let runtime = Runtime::new()?;
    let engine = runtime.deserialize(&plan)?;
    let context = engine.context(AllocationStrategy::Static)?;
    let (ha, hb) = exact_operands(N as usize, 0x5101, false);
    let (da, db) = (env.upload(&ha)?, env.upload(&hb)?);
    let dout = env.zeros::<f32>(N as usize)?;
    *CTX_SEEN.lock().unwrap() = None;
    // SAFETY: buffers hold N floats and outlive the synchronize below.
    unsafe {
        context.set_tensor_address("a", env.ptr(&da))?;
        context.set_tensor_address("b", env.ptr(&db))?;
        context.set_tensor_address("out", env.ptr(&dout))?;
        context.enqueue(env.raw_stream())?;
    }
    let got = env.download(&dout)?;
    let (bad, _) = bit_diff(&got, &residual_add(&ha, &hb));

    let primary = env.primary_context()?;
    let ours = env.ctx.cu_ctx();
    let Some(seen) = *CTX_SEEN.lock().unwrap() else {
        return Ok(CheckOutcome::Fail("ProbeCtx never ran".into()));
    };
    let fields = [
        ("cudarc context is the primary context", ours == primary),
        ("current context inside the plugin", seen.current == ours),
        ("context of the plugin's stream", seen.of_stream == ours),
        (
            "context of TensorRT's activation buffer",
            seen.of_input == ours,
        ),
        (
            "plugin stream is the enqueueV3 stream",
            seen.stream == env.raw_stream(),
        ),
        ("output exact", bad == 0),
    ];
    let failed: Vec<&str> = fields.iter().filter(|f| !f.1).map(|f| f.0).collect();
    let summary = format!(
        "TensorRT {}.{}.{} (headers {}.{}.{}, ONNX parser {}); sm_75 plan built and run; \
         context {ours:?} = primary {primary:?}; plugin saw current {:?}, stream ctx {:?}, \
         activation 0x{:x} in ctx {:?}",
        lib.0,
        lib.1,
        lib.2,
        hdr.0,
        hdr.1,
        hdr.2,
        llmcuda_trt::onnx_parser_version(),
        seen.current,
        seen.of_stream,
        seen.input,
        seen.of_input,
    );
    Ok(if failed.is_empty() {
        CheckOutcome::Pass(summary)
    } else {
        CheckOutcome::Fail(format!(
            "{summary}; mismatched: {failed:?}; {bad} wrong outputs"
        ))
    })
}

/// P2: a native-only network on cudarc's stream, cudarc's buffers.
///
/// Uploads, enqueue and download are all queued on the one stream without
/// an intervening synchronize, so work TensorRT put on any other stream
/// would race and show up as wrong output. Then the same network on
/// subnormal and signed-zero operands, reported but not gated: it says
/// whether TensorRT's own elementwise kernels flush denormals.
fn p2(env: &Env) -> Result<CheckOutcome, Error> {
    const N: i64 = (1 << 20) + 3;
    let builder = Builder::new()?;
    let network = builder.network()?;
    let a = network.input("a", DataType::Float, [N])?;
    let b = network.input("b", DataType::Float, [N])?;
    let out = network.elementwise(a, b, ElementWise::Sum)?;
    out.set_name("out")?;
    network.mark_output(out)?;
    let plan = build(&builder, &network, &config(&builder)?)?;
    let runtime = Runtime::new()?;
    let engine = runtime.deserialize(&plan)?;
    let context = engine.context(AllocationStrategy::Static)?;

    let mut results = Vec::new();
    for edges in [false, true] {
        let (ha, hb) = exact_operands(N as usize, 0x5202, edges);
        let da = env.upload(&ha)?;
        let db = env.upload(&hb)?;
        let dout = env.zeros::<f32>(N as usize)?;
        // SAFETY: buffers hold N floats and outlive the download's sync.
        unsafe {
            context.set_tensor_address("a", env.ptr(&da))?;
            context.set_tensor_address("b", env.ptr(&db))?;
            context.set_tensor_address("out", env.ptr(&dout))?;
            context.enqueue(env.raw_stream())?;
        }
        let got = env.download(&dout)?;
        results.push((bit_diff(&got, &residual_add(&ha, &hb)), got, ha, hb));
    }
    let ((bad, first), _, _, _) = &results[0];
    let ((edge_bad, edge_first), edge_got, ea, eb) = &results[1];
    let edge_note = match edge_first {
        None => "native add also exact on signed zero and subnormals".to_owned(),
        Some(i) => format!(
            "native add differs on {edge_bad} edge elements, first [{i}]: {:e} + {:e} -> {:e} \
             (bits {:#010x}), CPU {:e}",
            ea[*i],
            eb[*i],
            edge_got[*i],
            edge_got[*i].to_bits(),
            ea[*i] + eb[*i]
        ),
    };
    Ok(if *bad == 0 {
        CheckOutcome::Pass(format!(
            "{N} elements bit-identical to residual_add, queued without sync; {edge_note}"
        ))
    } else {
        CheckOutcome::Fail(format!(
            "{bad} of {N} differ, first at {first:?}; {edge_note}"
        ))
    })
}

/// P3: a network that is only a Rust plugin launching the engine's NVRTC
/// `tensor_add`, bit-exact against `residual_add` at ragged sizes with the
/// edge operands. The engine is serialized and deserialized, so the plugin
/// also went through field serialization and runtime creation.
fn p3(env: &Env) -> Result<CheckOutcome, Error> {
    const MAX: i64 = 1 << 20;
    let builder = Builder::new()?;
    let network = builder.network()?;
    let a = network.input("a", DataType::Float, [-1])?;
    let b = network.input("b", DataType::Float, [-1])?;
    let out = network.plugin(
        &[a, b],
        "ProbeAdd",
        "1",
        NAMESPACE,
        &[Field::i64s("probe_tag", &[3])],
        1,
    )?[0];
    out.set_name("out")?;
    network.mark_output(out)?;
    let config = config(&builder)?;
    let (min, opt, max) = ([1i64].into(), [65536i64].into(), [MAX].into());
    config.add_profile(&[("a", min, opt, max), ("b", min, opt, max)])?;
    let plan = build(&builder, &network, &config)?;
    let runtime = Runtime::new()?;
    let engine = runtime.deserialize(&plan)?;
    let context = engine.context(AllocationStrategy::Static)?;

    let sizes = [1usize, 3, 5, 255, 256, 257, 2048, 6151, 16385, MAX as usize];
    for &n in &sizes {
        let (ha, hb) = exact_operands(n, 0x5303 ^ n as u64, true);
        let (da, db) = (env.upload(&ha)?, env.upload(&hb)?);
        let dout = env.zeros::<f32>(n)?;
        context.set_input_shape("a", [n as i64])?;
        context.set_input_shape("b", [n as i64])?;
        // SAFETY: buffers hold n floats and outlive the download's sync.
        unsafe {
            context.set_tensor_address("a", env.ptr(&da))?;
            context.set_tensor_address("b", env.ptr(&db))?;
            context.set_tensor_address("out", env.ptr(&dout))?;
            context.enqueue(env.raw_stream())?;
        }
        let got = env.download(&dout)?;
        let (bad, first) = bit_diff(&got, &residual_add(&ha, &hb));
        if bad != 0 {
            let i = first.unwrap_or(0);
            return Ok(CheckOutcome::Fail(format!(
                "n={n}: {bad} elements differ; first [{i}] {:e} + {:e} -> {:e}, CPU {:e}",
                ha[i],
                hb[i],
                got[i],
                ha[i] + hb[i]
            )));
        }
    }
    Ok(CheckOutcome::Pass(format!(
        "plugin plan serialized and deserialized; bit-identical to residual_add at n = {sizes:?}, \
         with signed zero, subnormals and cancellation; {} plugin enqueues, {} without bindings",
        plugins::ENQUEUES.load(Ordering::Relaxed),
        plugins::UNBOUND_ENQUEUES.load(Ordering::Relaxed),
    )))
}

/// P4: enqueue once, capture one enqueue into a CUDA graph, replay it five
/// times. The plugin increments an accumulator through a device pointer
/// table, so the count proves each replay ran the plugin exactly once and
/// the capture itself ran nothing.
fn p4(env: &Env) -> Result<CheckOutcome, Error> {
    let builder = Builder::new()?;
    let network = builder.network()?;
    let table = network.input("table", DataType::Int64, [1])?;
    let out = network.plugin(&[table], "ProbeAccumulate", "1", NAMESPACE, &[], 1)?[0];
    out.set_name("out")?;
    network.mark_output(out)?;
    let plan = build(&builder, &network, &config(&builder)?)?;
    let runtime = Runtime::new()?;
    let engine = runtime.deserialize(&plan)?;
    let context = engine.context(AllocationStrategy::Static)?;

    let acc = env.zeros::<f32>(1)?;
    let dtable = env.upload(&[env.ptr(&acc) as i64])?;
    let dout = env.zeros::<f32>(1)?;
    // SAFETY: all three buffers live until the end of this function, after
    // the last synchronize.
    unsafe {
        context.set_tensor_address("table", env.ptr(&dtable))?;
        context.set_tensor_address("out", env.ptr(&dout))?;
        context.enqueue(env.raw_stream())?;
    }
    let after_warmup = env.download(&acc)?[0];

    env.stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    // SAFETY: as above; during capture this records rather than runs.
    let captured = unsafe { context.enqueue(env.raw_stream()) };
    let graph = env
        .stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
    captured?;
    let Some(graph) = graph else {
        return Ok(CheckOutcome::Fail("capture produced an empty graph".into()));
    };
    let after_capture = env.download(&acc)?[0];
    for _ in 0..5 {
        graph.launch()?;
    }
    let after_replays = env.download(&acc)?[0];
    let last_out = env.download(&dout)?[0];
    let summary = format!(
        "accumulator {after_warmup} after warm-up enqueue, {after_capture} after capture, \
         {after_replays} after 5 replays (plugin output {last_out})"
    );
    Ok(
        if after_warmup == 1.0 && after_capture == 1.0 && after_replays == 6.0 && last_out == 6.0 {
            CheckOutcome::Pass(summary)
        } else {
            CheckOutcome::Fail(summary)
        },
    )
}

/// P5: a user-managed context bound to one unpooled `cuMemAlloc` block,
/// every TensorRT allocation counted, then 100 enqueues of an fp16 GEMM
/// followed by a plugin that uses workspace. Zero allocations and zero frees
/// across the 100 — counted by the allocator and confirmed by the driver's
/// free-memory figure, which also catches allocations that bypass it.
fn p5(env: &Env) -> Result<CheckOutcome, Error> {
    const M: i64 = 512;
    const K: i64 = 2048;
    const N: i64 = 1024;
    let mut rng = Xorshift64Star::new(0x5505);
    let w: Vec<f16> = (0..K * N)
        .map(|_| f16::from_f32(rng.next_f32_range(-0.05, 0.05)))
        .collect();
    let x: Vec<f32> = rng.vec_f32((M * K) as usize, -1.0, 1.0);

    let builder = Builder::new()?;
    let network = builder.network()?;
    let input = network.input("x", DataType::Float, [M, K])?;
    let x16 = network.cast(input, DataType::Half)?;
    let wt = network.constant([K, N], DataType::Half, half_bytes(&w), None)?;
    let y16 = network.matmul(x16, MatrixOp::None, wt, MatrixOp::None)?;
    let y = network.cast(y16, DataType::Float)?;
    let out = network.plugin(&[y], "ProbeWorkspace", "1", NAMESPACE, &[], 1)?[0];
    out.set_name("out")?;
    network.mark_output(out)?;
    let plan = build(&builder, &network, &config(&builder)?)?;

    let counter = Arc::new(CountingAllocator::default());
    let allocator = GpuAllocator::new(counter.clone())?;
    let mut runtime = Runtime::new()?;
    runtime.set_allocator(allocator.clone());
    let engine = runtime.deserialize(&plan)?;
    let mut context = engine.context(AllocationStrategy::UserManaged)?;
    context.set_temporary_allocator(allocator)?;
    let size = engine.device_memory_size(None);
    let block = Unpooled::new(size as usize)?;
    let at_load = counter.snapshot();

    let dx = env.upload(&x)?;
    let dout = env.zeros::<f32>((M * N) as usize)?;
    // SAFETY: the block and buffers outlive every enqueue below.
    unsafe {
        context.set_device_memory(block.ptr, size)?;
        context.set_tensor_address("x", env.ptr(&dx))?;
        context.set_tensor_address("out", env.ptr(&dout))?;
        context.enqueue(env.raw_stream())?;
    }
    env.sync()?;
    let after_warmup = counter.snapshot();
    let free_before = free_bytes()?;
    for _ in 0..100 {
        // SAFETY: as above.
        unsafe { context.enqueue(env.raw_stream())? };
    }
    env.sync()?;
    let after = counter.snapshot();
    let free_after = free_bytes()?;
    let got = env.download(&dout)?;

    // Sanity on the arithmetic, not a gate: fp16-rounded operands, f64 sum,
    // then the plugin's 2y + 1, over the first 8 rows.
    let rows = 8usize;
    let mut reference = Vec::with_capacity(rows * N as usize);
    for r in 0..rows {
        for c in 0..N as usize {
            let s: f64 = (0..K as usize)
                .map(|k| {
                    f64::from(f16::from_f32(x[r * K as usize + k]).to_f32())
                        * f64::from(w[k * N as usize + c].to_f32())
                })
                .sum();
            reference.push((s as f32) * 2.0 + 1.0);
        }
    }
    let cmp = compare(&got[..rows * N as usize], &reference);

    let enqueue_allocs = after.0 - after_warmup.0;
    let enqueue_frees = after.1 - after_warmup.1;
    let summary = format!(
        "device memory {size} B from one cuMemAlloc block; allocator saw {} allocations \
         ({:.1} MiB) at load, {} more on the warm-up enqueue, {enqueue_allocs} allocations and \
         {enqueue_frees} frees across 100 enqueues; driver free memory {free_before} -> \
         {free_after}; output vs fp16-rounded CPU GEMM (8 rows): max_abs {:.3e}, cosine {:.7}",
        at_load.0,
        at_load.2 as f64 / (1024.0 * 1024.0),
        after_warmup.0 - at_load.0,
        cmp.max_abs_error,
        cmp.cosine_similarity,
    );
    Ok(
        if enqueue_allocs == 0
            && enqueue_frees == 0
            && free_before == free_after
            && cmp.cosine_similarity > 0.9999
        {
            CheckOutcome::Pass(summary)
        } else {
            CheckOutcome::Fail(summary)
        },
    )
}

pub fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|h| h.to_bits().to_ne_bytes()).collect()
}

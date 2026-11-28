//! TensorRT toolchain gate.
//!
//! ```sh
//! export TENSORRT_ROOT=/path/to/tensorrt-11 CUDA_ROOT=/path/to/cuda-12
//! export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
//! CUDA_VISIBLE_DEVICES=<idle card> \
//!     cargo run --release -p llmcuda-trt --features tensorrt --bin trt_probe
//! ```
//!
//! P1–P5 gate: each must pass on a real GPU, and the process exits non-zero
//! otherwise. P6 records answers to the capability questions (U2–U8 and
//! IAttention); a "no" there is a finding, not a failure. Every check
//! verifies numbers, not just that a call returned.

mod capability;
mod env;
mod gate;
mod onnx;
mod plugins;

use std::process::ExitCode;

use llmcuda_cuda::{check_gate, device};
use llmcuda_trt::bindings;
use tracing::{error, info};

use crate::env::Env;
use crate::plugins::ProbeBindings;

fn main() -> ExitCode {
    // Remaining arguments select checks by label prefix, e.g. `U6 P5`.
    let only = llmcuda_log::init_from_args();
    info!("llmcuda-rs TensorRT probe\n");

    if !device::driver_available() {
        error!(
            "No CUDA driver reachable; every check would be skipped, and skipped is not passed."
        );
        return ExitCode::FAILURE;
    }
    let devices = match device::probe_all() {
        Ok(d) if !d.is_empty() => d,
        Ok(_) => {
            error!("Driver present, but no CUDA devices are visible.");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            error!("Failed to probe devices: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = check_gate(&devices) {
        error!("Capability gate: FAIL — {e}");
        return ExitCode::FAILURE;
    }
    let d = &devices[0];
    info!(
        "Device 0 (of CUDA_VISIBLE_DEVICES={}): {} compute {}, {:.1} GiB",
        std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()),
        d.name,
        d.compute_capability,
        d.total_memory as f64 / (1u64 << 30) as f64
    );

    let env = match Env::new(0) {
        Ok(e) => e,
        Err(e) => {
            error!("CUDA setup failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = plugins::register_all() {
        error!("Plugin registration failed: {e}");
        return ExitCode::FAILURE;
    }
    let probe_bindings = ProbeBindings {
        kernels: env.kernels,
    };
    let _bound = bindings::install(&probe_bindings);

    info!("\nGATE 0 checks (must pass)\n");
    let selected =
        |label: &str| only.is_empty() || only.iter().any(|p| label.starts_with(p.as_str()));
    let gate = gate::run_all(&env, &selected);
    info!("\nP6 capability answers (recorded)\n");
    let answers = capability::run_all(&env, &selected);

    info!("\nSummary");
    for (label, outcome) in gate.iter().chain(answers.iter()) {
        info!("  {label:<40} {outcome}");
    }
    info!(
        "  plugin enqueues {}, of which without bindings {}",
        plugins::ENQUEUES.load(std::sync::atomic::Ordering::Relaxed),
        plugins::UNBOUND_ENQUEUES.load(std::sync::atomic::Ordering::Relaxed)
    );

    if !only.is_empty() {
        info!("\nOnly {only:?} ran; GATE 0 needs a full run.");
        return ExitCode::from(u8::from(!gate.iter().all(|(_, o)| o.is_pass())));
    }
    if gate.len() == 5 && gate.iter().all(|(_, o)| o.is_pass()) {
        info!("\nGATE 0: P1–P5 passed on this device.");
        ExitCode::SUCCESS
    } else {
        error!("\nGATE 0: at least one of P1–P5 did not pass.");
        ExitCode::FAILURE
    }
}

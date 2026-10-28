//! Compiles the C ABI shim over TensorRT, only with feature `tensorrt`.
//!
//! Without the feature this does nothing, so `cargo build --workspace` needs
//! neither TensorRT nor a CUDA toolkit. With it, two roots are read from the
//! environment and nothing is guessed beyond the conventional CUDA prefix:
//!
//! - `TENSORRT_ROOT`: `include/NvInfer.h` and `lib/libnvinfer.so`. Required.
//! - `CUDA_ROOT` (or `CUDA_PATH`, `CUDA_HOME`): only for
//!   `include/cuda_runtime_api.h`, which TensorRT's headers include for
//!   `cudaStream_t`. The shim calls no CUDA runtime or driver function, so no
//!   CUDA library is linked here; libnvinfer carries its own runtime.
//!
//! The `println!`s are Cargo's build-script directive channel, not console
//! output; Cargo reads build-script stdout and never shows it.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    for var in ["TENSORRT_ROOT", "CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if env::var_os("CARGO_FEATURE_TENSORRT").is_none() {
        return;
    }

    let trt = PathBuf::from(env::var_os("TENSORRT_ROOT").unwrap_or_else(|| {
        panic!(
            "feature `tensorrt` needs TENSORRT_ROOT pointing at a TensorRT 11 install \
             (include/NvInfer.h, lib/libnvinfer.so); see docs/TOOLCHAIN.md"
        )
    }));
    let trt_include = trt.join("include");
    let trt_lib = trt.join("lib");
    assert!(
        trt_include.join("NvInfer.h").is_file(),
        "{} has no NvInfer.h",
        trt_include.display()
    );

    let cuda = ["CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"]
        .iter()
        .find_map(env::var_os)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let cuda_include = cuda.join("include");
    assert!(
        cuda_include.join("cuda_runtime_api.h").is_file(),
        "{} has no cuda_runtime_api.h; set CUDA_ROOT to a CUDA 12 toolkit",
        cuda_include.display()
    );

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/shim.cpp")
        .include("cpp")
        .include(&trt_include)
        .include(&cuda_include)
        // TensorRT 11 still declares the deprecated v1 allocator entry points
        // that IGpuAsyncAllocator overrides; that is its header, not this file.
        .flag_if_supported("-Wno-deprecated-declarations")
        .warnings(true)
        .compile("llmcuda_trt_shim");

    println!("cargo:rustc-link-search=native={}", trt_lib.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvonnxparser");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

# llmcuda-rs

A single-process CUDA inference engine with cross-GPU prefix-cache sharing,
optimized for Turing (sm_75) and limited models.

> **Experimental.** This project was built entirely by AI agents. It is not
> production-ready and has been tested on one machine only. Expect bugs.

## Status

- **Faster than llama.cpp at its own best settings** in every measured
  Qwen3.6 and K2-Horizon cell. Qwen3.6 was measured on one card with three
  sequences: prefill 512–128K, decode 2K and 32K. Qwen3.8-27B is ahead or
  level. Numbers are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- **Accuracy is a hard gate.** Output matches llama.cpp's argmax on captured
  golden data, and a sequence decodes bit-identically batched or alone. No
  speedup was bought with a looser tolerance.
- **Serves OpenAI and Anthropic APIs:** completions, chat completions,
  responses and messages. Streaming, sampling, tool calls and an optional API
  key work in every dialect, and images with `--mmproj`.

## Models

The architecture comes from the file's `general.architecture`; nothing is
inferred from tensor shapes.

| Model | Architecture | Notes |
| --- | --- | --- |
| [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | `qwen35moe` | The tuned target, `unsloth` `UD-Q6_K_XL` |
| [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `qwen35` | Dense sibling |
| [K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-GGUF) | `k2-horizon` | Q4_K_M and Q6_K; see [MODEL.md](docs/MODEL.md#k2-horizon) |
| [Clef-Flash](https://huggingface.co/ggml-org/Clef-Flash-GGUF) | `clef` | Decision model, served on `/v1/systemone` only |

## Why one process

Three `llama-server` processes, one per card, each keep a private prefix
cache. A shared system prompt is cached three times, and a request sent to the
wrong replica pays full prefill. In one address space the shared cache is a
radix tree behind a lock. Resubmitting a 25,136-token prompt takes 0.3 s warm
against 9.7 s cold, 32× faster to first token.

## Architecture

```
+-- single process, one address space -------------------------+
|   HTTP / admission queue                                     |
|            |                                                 |
|   cache-aware router  -- score(w) = a*prefix_match(w)        |
|            |                       - b*queued_tokens(w)      |
|            |                       - c*kv_utilization(w)     |
|      +-----+-----+                                           |
|      v     v     v                                           |
|   worker0 worker1 worker2   each: own CUDA context, stream,  |
|    GPU 0   GPU 1   GPU 2    two-group paged pool, captured   |
|      |     |     |          graph, chunked-prefill scheduler |
|      +-----+-----+                                           |
|            v                                                 |
|   shared prefix cache -- radix tree in host RAM              |
|   (pinned arena, cross-worker, no serialization)             |
+--------------------------------------------------------------+
```

Each worker holds a full copy of the model; the three are replicas sharing a
host-side prefix cache, not a tensor-parallel split. Workers never touch each
other's device memory: no P2P, no NCCL, nothing crosses PCIe on the decode
path.

## Hardware

| | |
| --- | --- |
| GPUs | 3× Quadro RTX 8000: 48 GB, sm_75 (Turing), 672 GB/s |
| Host | 125 GB RAM, 20 vCPU |
| CUDA | 12.4 |

Turing has no `cp.async`, so double-buffering is done by hand. Its tensor
cores offer fp16 `m16n8k8` and int8 `m8n8k16` MMA. Why the model's shape
drives the design is in [docs/MODEL.md](docs/MODEL.md);
`cargo run -p llmcuda-model --example budget` prints VRAM and bandwidth
tables for any context length.

## Build and run

```sh
cargo build --release --workspace
cargo test --workspace --release                        # device tests skip without a GPU, and say so
cargo run --release -p llmcuda-cuda --bin probe         # device inventory and sm_75 gate
cargo run --release -p llmcuda-server -- --model path/to/model.gguf   # serves on 127.0.0.1:8000
docker compose up --build                               # container: models mounted, images enabled
```

The workspace builds and tests without a GPU; that is what CI runs. Tests
that need the real model read `$LLMCUDA_MODEL`. Every binary takes
`--log-level info | debug | trace`. Options are documented in
[docs/CLI.md](docs/CLI.md), endpoints in [docs/API.md](docs/API.md) and the
container in [docs/DOCKER.md](docs/DOCKER.md).

## Documentation

| | |
| --- | --- |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | **Start here for performance:** the current standing, and why / why not |
| [AGENTS.md](AGENTS.md) | Instructions for AI agents and the binding design rules |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, workflow, logging, commit conventions |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component design and scope |
| [docs/MODEL.md](docs/MODEL.md) | Model structure, VRAM and bandwidth analysis |
| [docs/CACHE.md](docs/CACHE.md) | The two-group cache and prefix sharing |
| [docs/SCHEDULER.md](docs/SCHEDULER.md) | Chunked prefill, admission, preemption |
| [docs/KERNELS.md](docs/KERNELS.md) | Kernel inventory and porting notes |
| [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) | The performance model and the remaining work |
| [docs/OPTIMIZATION_CAMPAIGN.md](docs/OPTIMIZATION_CAMPAIGN.md) | The finished optimization campaign, against the original |
| [docs/TESTING.md](docs/TESTING.md) | Differential harness and numerics thresholds |
| [docs/ORACLE.md](docs/ORACLE.md) | Capturing llama.cpp's logits as a test oracle |
| [docs/API.md](docs/API.md) | HTTP endpoints, streaming, authentication |
| [docs/CLI.md](docs/CLI.md) | Server command-line options |
| [docs/DOCKER.md](docs/DOCKER.md) | Running the container image |
| [docs/MCP.md](docs/MCP.md) | Calling tools on configured MCP servers |
| [docs/SANDBOX.md](docs/SANDBOX.md) | Sandboxed tool calls for Rust harnesses |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Environment, reference checkouts, the llama.cpp baseline |
| [docs/MILESTONES.md](docs/MILESTONES.md) | Milestone plan and status |
| [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) | Toolchain gate and the cudarc decision |
| [docs/HARDCODING.md](docs/HARDCODING.md) | Audit of model-dependent constants |

## License

Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and
[docs/LICENSE.md](docs/LICENSE.md). Model weights are not distributed here
and keep their publishers' terms.

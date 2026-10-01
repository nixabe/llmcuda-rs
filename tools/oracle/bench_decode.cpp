// Backend-greedy decode baseline for llmcuda-engine's bench_decode_batch.
// Build as capture.cpp (tools/oracle/Makefile), against the model publisher's
// llama.cpp fork. Usage: bench_decode model.gguf context steps ubatch.
// Synthetic prompts match bench_decode_batch::synthetic_prompt. Both engines
// ignore EOS, use f16 KV, discard four decode steps and include token readback.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using clock_type = std::chrono::steady_clock;

static void decode(llama_context * ctx, llama_batch & batch) {
    if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
    llama_synchronize(ctx);
}
static void add(llama_batch & batch, llama_token token, int pos, int seq, bool output) {
    int i = batch.n_tokens++;
    batch.token[i] = token; batch.pos[i] = pos;
    batch.n_seq_id[i] = 1; batch.seq_id[i][0] = seq; batch.logits[i] = output;
}
int main(int argc, char ** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: bench_decode model context steps ubatch\n"); return 1; }
    const int context = std::atoi(argv[2]), steps = std::atoi(argv[3]), ubatch = std::atoi(argv[4]);
    if (context <= 0 || steps <= 0 || ubatch <= 0) return 1;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99; mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int sequences : {1, 3}) {
        std::vector<llama_sampler *> samplers;
        std::vector<llama_sampler_seq_config> configs;
        for (int s = 0; s < sequences; ++s) {
            auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            samplers.push_back(chain); configs.push_back({s, chain});
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = sequences * ((context + steps + 4 + 255) / 256 * 256);
        cp.n_seq_max = sequences; cp.n_batch = std::max(4096, context); cp.n_ubatch = ubatch;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.n_threads = 12; cp.n_threads_batch = 12;
        cp.samplers = configs.data(); cp.n_samplers = configs.size();
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) return 1;
        auto batch = llama_batch_init(std::max(context, sequences), 0, 1);
        std::vector<llama_token> next(sequences);
        for (int s = 0; s < sequences; ++s) {
            batch.n_tokens = 0;
            for (int i = 0; i < context; ++i) {
                add(batch, ((s + 1) * 104729LL + i * 7919LL + 1234) % vocab, i, s, i == context - 1);
            }
            decode(ctx, batch);
            next[s] = llama_get_sampled_token_ith(ctx, context - 1);
            if (next[s] == LLAMA_TOKEN_NULL) throw std::runtime_error("backend sampling unavailable");
        }
        std::vector<double> samples;
        for (int step = 0; step < steps + 4; ++step) {
            auto start = clock_type::now();
            batch.n_tokens = 0;
            for (int s = 0; s < sequences; ++s) add(batch, next[s], context + step, s, true);
            decode(ctx, batch);
            for (int s = 0; s < sequences; ++s) {
                next[s] = llama_get_sampled_token_ith(ctx, s);
                if (next[s] == LLAMA_TOKEN_NULL) throw std::runtime_error("backend sampling unavailable");
                llama_sampler_accept(samplers[s], next[s]);
            }
            double ms = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
            if (step >= 4) samples.push_back(ms);
        }
        double mean = 0; for (double ms : samples) mean += ms / steps;
        std::sort(samples.begin(), samples.end());
        std::printf("{\"sequences\":%d,\"context\":%d,\"steps\":%d,\"ubatch\":%d,\"mean_ms\":%.6f,\"p50_ms\":%.6f,\"p95_ms\":%.6f,\"tokens_s\":%.6f}\n",
            sequences, context, steps, ubatch, mean, samples[steps/2], samples[std::min(steps-1, steps*95/100)], sequences*1000/mean);
        llama_batch_free(batch); llama_free(ctx);
        for (auto * s : samplers) llama_sampler_free(s);
    }
    llama_model_free(model); llama_backend_free();
}

// Golden-oracle capture tool for llmcuda-rs.
//
// Runs one llama.cpp forward pass over a fixed prompt and writes the full,
// untruncated contents of selected intermediate graph tensors plus the final
// logits to a single self-describing binary container.
//
// This exists because `llama-eval-callback` / `llama-debug --verbose` only
// print the first and last three elements of each dimension, which is not
// enough to gate a differential test on. The callback API is the same one
// those tools use (`llama_context_params::cb_eval`); only the sink differs.
//
// Build (out of tree, no llama.cpp source is modified):
//   g++ -O2 -std=c++17 capture.cpp -o capture \
//     -I$LLAMA/include -I$LLAMA/ggml/include \
//     -L$LLAMA/build/bin -lllama -lggml-base -lggml \
//     -Wl,-rpath,$LLAMA/build/bin
//
// Usage:
//   capture <model.gguf> <out.bin> <prompt> <regex>[,<regex>...]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <regex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Container format. Little-endian throughout; the capture host and the test
// host are the same x86-64 machine, and the file is regenerated rather than
// shipped, so no byte-swapping path is warranted.
//
//   magic   "LLMCGOLD"                       8 bytes
//   u32     version                          = 1
//   u32     n_records
//   record*:
//     u32   name_len
//     u8    name[name_len]                   (no NUL)
//     u32   dtype                            0 = f32, 1 = i32
//     i64   ne[4]                            ggml order, ne[0] fastest-varying
//     u64   n_elements                       = ne0*ne1*ne2*ne3
//     u8    payload[n_elements * 4]          contiguous, ne[0]-major
//
// Every float payload is written in ggml logical order: index
// i0 + ne0*(i1 + ne1*(i2 + ne2*i3)). Strided views are de-strided on write via
// nb[], so a reader never has to know ggml's stride rules.
// ---------------------------------------------------------------------------

static const char MAGIC[8] = { 'L','L','M','C','G','O','L','D' };

struct writer {
    FILE * f = nullptr;
    uint32_t n_records = 0;

    void open(const char * path) {
        f = fopen(path, "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
        fwrite(MAGIC, 1, 8, f);
        uint32_t v = 1;
        fwrite(&v, 4, 1, f);
        fwrite(&n_records, 4, 1, f); // patched on close
    }

    void record(const std::string & name, uint32_t dtype,
                const int64_t ne[4], const void * data, uint64_t n_elem) {
        uint32_t nl = (uint32_t) name.size();
        fwrite(&nl, 4, 1, f);
        fwrite(name.data(), 1, nl, f);
        fwrite(&dtype, 4, 1, f);
        fwrite(ne, 8, 4, f);
        fwrite(&n_elem, 8, 1, f);
        fwrite(data, 4, n_elem, f);
        n_records++;
    }

    void close() {
        fseek(f, 12, SEEK_SET);
        fwrite(&n_records, 4, 1, f);
        fclose(f);
        f = nullptr;
    }
};

struct cb_state {
    writer * w = nullptr;
    std::vector<std::regex> filters;
    std::vector<uint8_t> raw;
    std::vector<float>   flat;
    int matched = 0;
    int n_tokens = 0;      // set before llama_decode
};

static float read_scalar(const uint8_t * d, ggml_type type, size_t off) {
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) (d + off);
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (d + off));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (d + off));
        case GGML_TYPE_I32:  return (float) *(const int32_t *) (d + off);
        case GGML_TYPE_I16:  return (float) *(const int16_t *) (d + off);
        case GGML_TYPE_I8:   return (float) *(const int8_t  *) (d + off);
        default: fprintf(stderr, "unhandled type %s\n", ggml_type_name(type)); exit(1);
    }
}

// Flatten `t` into `st->flat` in ne[0]-fastest order and return its element
// count.
static uint64_t flatten(cb_state * st, const struct ggml_tensor * t) {
    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * src;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        src = (const uint8_t *) t->data;
    } else {
        st->raw.resize(nbytes);
        ggml_backend_tensor_get(t, st->raw.data(), 0, nbytes);
        src = st->raw.data();
    }

    const int64_t * ne = t->ne;
    const size_t  * nb = t->nb;
    const uint64_t n_elem = (uint64_t) ne[0] * ne[1] * ne[2] * ne[3];
    st->flat.resize(n_elem);

    uint64_t o = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++)
    for (int64_t i2 = 0; i2 < ne[2]; i2++)
    for (int64_t i1 = 0; i1 < ne[1]; i1++)
    for (int64_t i0 = 0; i0 < ne[0]; i0++) {
        st->flat[o++] = read_scalar(src, t->type, i3*nb[3] + i2*nb[2] + i1*nb[1] + i0*nb[0]);
    }
    return n_elem;
}

static void emit(cb_state * st, const std::string & name, const char * type_name,
                 const int64_t ne[4], const float * data, uint64_t n_elem) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n_elem; i++) sum += data[i];
    st->w->record(name, 0, ne, data, n_elem);
    printf("captured %-28s type=%-5s ne=[%lld,%lld,%lld,%lld] n=%llu sum=%.6f\n",
           name.c_str(), type_name,
           (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
           (unsigned long long) n_elem, sum);
    fflush(stdout);
}

// The gated-norm output that feeds a GDN block's output projection.
//
// llama.cpp named this node `final_output-N` until it stopped calling its
// callback on it; the tensor is still there, as the activation operand of the
// MUL_MAT that produces `linear_attn_out-N`. It is read while that node's
// callback runs, which is before the scheduler can hand its buffer to anything
// else: the node that consumed it has only just finished.
static const struct ggml_tensor * projection_input(const struct ggml_tensor * t) {
    const struct ggml_tensor * mm = t;
    for (int hop = 0; hop < 2 && mm && mm->op != GGML_OP_MUL_MAT; hop++) {
        mm = mm->src[0];
    }
    if (!mm || mm->op != GGML_OP_MUL_MAT) return nullptr;
    return mm->src[1];
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (cb_state *) user_data;

    bool match = false;
    for (const auto & re : st->filters) {
        if (std::regex_match(t->name, re)) { match = true; break; }
    }

    if (ask) {
        // Returning false means the scheduler will not call back with the data.
        return match;
    }
    if (!match) {
        return true;
    }
    if (ggml_is_quantized(t->type)) {
        fprintf(stderr, "skipping quantized tensor %s (%s)\n", t->name, ggml_type_name(t->type));
        return true;
    }

    const std::string name = t->name;
    const int64_t * ne = t->ne;

    // `final_output-N` first, so the container keeps the graph's own order.
    if (name.rfind("linear_attn_out-", 0) == 0) {
        const struct ggml_tensor * in = projection_input(t);
        if (!in || ggml_is_quantized(in->type)) {
            fprintf(stderr, "%s: no MUL_MAT activation operand to capture as final_output\n", t->name);
            exit(1);
        }
        const uint64_t n_in = flatten(st, in);
        emit(st, "final_output-" + name.substr(strlen("linear_attn_out-")),
             ggml_type_name(in->type), in->ne, st->flat.data(), n_in);
        st->matched++;
    }

    const uint64_t n_elem = flatten(st, t);

    // Every position is an output position in this capture (see main), so the
    // final norm and the logits arrive as `[n, n_tokens]`. The container has
    // always carried them for the last position alone, with the full final
    // norm under `h_nextn`; llama.cpp names one tensor both `h_nextn` and
    // `result_norm` when nothing is cropped between them, and the second name
    // wins, so both records are cut from it here.
    const bool per_token = st->n_tokens > 1 && ne[1] == st->n_tokens && ne[2] == 1 && ne[3] == 1;
    if (per_token && (name == "result_norm" || name == "result_output")) {
        if (name == "result_norm") {
            emit(st, "h_nextn", ggml_type_name(t->type), ne, st->flat.data(), n_elem);
            st->matched++;
        }
        const int64_t last[4] = { ne[0], 1, 1, 1 };
        emit(st, name, ggml_type_name(t->type), last,
             st->flat.data() + (uint64_t) ne[0] * (st->n_tokens - 1), (uint64_t) ne[0]);
        st->matched++;
        return true;
    }

    emit(st, name, ggml_type_name(t->type), ne, st->flat.data(), n_elem);
    st->matched++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> <prompt> <regex>[,<regex>...]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    const char * prompt     = argv[3];

    cb_state st;
    writer w;
    w.open(out_path);
    st.w = &w;
    {
        std::string spec = argv[4];
        size_t p = 0;
        while (p <= spec.size()) {
            size_t q = spec.find(',', p);
            if (q == std::string::npos) q = spec.size();
            std::string pat = spec.substr(p, q - p);
            if (!pat.empty()) st.filters.emplace_back(pat, std::regex::optimize);
            p = q + 1;
        }
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu     = 0;   // relative to CUDA_VISIBLE_DEVICES

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 4096;
    cp.n_batch           = 4096;
    cp.n_ubatch          = 4096;
    cp.n_seq_max         = 1;
    cp.n_threads         = 4;
    cp.n_threads_batch   = 4;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k            = GGML_TYPE_F16;
    cp.type_v            = GGML_TYPE_F16;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = &st;
    cp.no_perf           = false;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // Tokenize exactly as llama-cli/llama-debug do: add_special = model default.
    std::vector<llama_token> tokens(512);
    int32_t n = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt),
                               tokens.data(), (int32_t) tokens.size(),
                               /*add_special =*/ true, /*parse_special =*/ true);
    if (n < 0) { fprintf(stderr, "tokenize failed (%d)\n", n); return 1; }
    tokens.resize(n);

    printf("add_bos=%d n_tokens=%d\n", (int) llama_vocab_get_add_bos(vocab), n);
    for (int i = 0; i < n; i++) {
        char piece[256];
        int m = llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece), 0, true);
        printf("  [%2d] %6d  '%.*s'\n", i, tokens[i], m > 0 ? m : 0, piece);
    }
    fflush(stdout);

    // Every position is requested as an output. With only the last one
    // requested, llama.cpp crops the last block to that row before its
    // residual add, and `attn_residual-39` onward would hold one position
    // instead of the prompt.
    st.n_tokens = n;
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    llama_batch_free(batch);

    // Final logits for the last prompt position, straight from the public API.
    // Captured independently of the graph callback so the two can be
    // cross-checked against each other.
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits  = llama_get_logits_ith(ctx, n - 1);
    if (!logits) { fprintf(stderr, "no logits\n"); return 1; }
    {
        const int64_t ne[4] = { n_vocab, 1, 1, 1 };
        w.record("api.logits", 0, ne, logits, (uint64_t) n_vocab);
    }
    {
        std::vector<int32_t> tk(tokens.begin(), tokens.end());
        const int64_t ne[4] = { n, 1, 1, 1 };
        w.record("api.tokens", 1, ne, tk.data(), (uint64_t) n);
    }

    int best = 0;
    for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
    char piece[256];
    int m = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
    printf("argmax token = %d '%.*s' logit = %.6f\n", best, m > 0 ? m : 0, piece, logits[best]);
    printf("records = %u (callback matched %d)\n", w.n_records, st.matched);

    w.close();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// Golden-oracle capture for the Inkling 2 Mini vision tower (`inkling2v`).
//
// Runs llama.cpp's own preprocessing and vision graph over a fixed set of
// synthetic images and writes, per image, the input pixels, the preprocessed
// graph input, selected intermediate graph tensors and the final embeddings to
// one binary container. `crates/llmcuda-kernels/tests/inkling_vision_golden.rs`
// holds the CPU reference to these files.
//
// The images are generated here, from integer arithmetic only, and stored in
// the container, so the test needs no generator of its own and nothing is
// downloaded. A second output, `sizes.txt`, lists the token grid llama.cpp
// assigns to a sweep of image sizes (no encode — tokenization alone decides
// it), which is what the reference's `token_count` is held to.
//
// Everything goes through libmtmd's public API (`mtmd_tokenize`,
// `mtmd_encode_chunk`); intermediates come from the `cb_eval` hook that
// `mtmd_context_params` forwards to the graph scheduler, the same hook
// `llama-mtmd-debug` prints from. No llama.cpp source is modified.
//
// The language model is opened `vocab_only`: mtmd needs its vocabulary to
// tokenize the text around the image and nothing else, so its weights are
// never read. The tower runs on the CPU backend (`use_gpu = false`); run with
// `CUDA_VISIBLE_DEVICES=""` to keep the CUDA backend from touching a device at
// load.
//
// Build:   make capture_inkling2v      (see the Makefile for the link line)
// Usage:   capture_inkling2v <model.gguf> <mmproj.gguf> <out-dir> [threads]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Container: the `LLMCGOLD` format of capture.cpp, version 1. Little-endian.
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
//     u8    payload[n_elements * 4]
//
// Records, in order:
//   image_rgb        i32 [3, width, height]   the input bytes, one per element
//   image_max_tokens i32 [1]                  the ceiling the context ran with
//   flash_attn       i32 [1]                  0 = off, 1 = on
//   inp_raw          f32 [nx, ny, 3]          the preprocessed graph input
//   <graph nodes>    f32                      by llama.cpp node name
//   output           f32 [n_embd, n_tokens]   mtmd_get_output_embd
// ---------------------------------------------------------------------------

static const char MAGIC[8] = { 'L','L','M','C','G','O','L','D' };

struct writer {
    FILE * f = nullptr;
    uint32_t n_records = 0;

    void open(const std::string & path) {
        f = fopen(path.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
        n_records = 0;
        fwrite(MAGIC, 1, 8, f);
        uint32_t v = 1;
        fwrite(&v, 4, 1, f);
        fwrite(&n_records, 4, 1, f); // patched on close
    }

    void record(const std::string & name, uint32_t dtype,
                const int64_t ne[4], const void * data) {
        const uint64_t n_elem = (uint64_t) ne[0] * ne[1] * ne[2] * ne[3];
        uint32_t nl = (uint32_t) name.size();
        fwrite(&nl, 4, 1, f);
        fwrite(name.data(), 1, nl, f);
        fwrite(&dtype, 4, 1, f);
        fwrite(ne, 8, 4, f);
        fwrite(&n_elem, 8, 1, f);
        fwrite(data, 4, n_elem, f);
        n_records++;
    }

    void scalar(const std::string & name, int32_t value) {
        const int64_t ne[4] = { 1, 1, 1, 1 };
        record(name, 1, ne, &value);
    }

    void close() {
        fseek(f, 12, SEEK_SET);
        fwrite(&n_records, 4, 1, f);
        fclose(f);
        f = nullptr;
    }
};

// ---------------------------------------------------------------------------
// Which graph nodes are kept. Whole-sequence tensors are megabytes each and
// there are fourteen named nodes per block, so only the first `detail` blocks
// are kept in full; every block's output is always kept.
// ---------------------------------------------------------------------------

static const char * const GLOBAL_NODES[] = {
    "patch_embd", "embeddings", "pre_ln", "encoder_out", "pixel_shuffle", "projected",
};
static const char * const BLOCK_NODES[] = {
    "layer_inp_normed", "Qcur", "Kcur", "Vcur", "Qcur_pos", "Kcur_pos", "kqv_out",
    "attn_out", "attn_out_scaled", "ffn_inp", "ffn_inp_normed", "ffn_out", "ffn_out_scaled",
};

struct cb_state {
    writer * w = nullptr; // null: not capturing (model load, the size sweep)
    int detail = 0;
    bool saw_input = false;
    int64_t n_embd = 0; // width of the last `projected` node seen
    std::vector<uint8_t> raw;
};

static bool wanted(const cb_state * st, const char * name) {
    for (const char * g : GLOBAL_NODES) {
        if (strcmp(name, g) == 0) return true;
    }
    const char * dash = strrchr(name, '-');
    if (!dash) return false;
    // The suffix must be a bare block index: ggml derives names such as
    // "Vcur-0 (permuted) (cont)" for the copies it inserts.
    if (dash[1] == '\0' || strspn(dash + 1, "0123456789") != strlen(dash + 1)) return false;
    const std::string base(name, dash - name);
    const int il = atoi(dash + 1);
    if (base == "layer_out") return true;
    if (il >= st->detail) return false;
    for (const char * b : BLOCK_NODES) {
        if (base == b) return true;
    }
    return false;
}

static void dump(cb_state * st, const char * name, const struct ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "skipping %s: type %s, contiguous %d\n", name, ggml_type_name(t->type),
                (int) ggml_is_contiguous(t));
        return;
    }
    const size_t nbytes = ggml_nbytes(t);
    st->raw.resize(nbytes);
    ggml_backend_tensor_get(t, st->raw.data(), 0, nbytes);
    st->w->record(name, 0, t->ne, st->raw.data());
    if (strcmp(name, "projected") == 0) {
        st->n_embd = t->ne[0];
    }
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    cb_state * st = (cb_state *) user_data;
    if (!st->w) {
        return ask ? false : true;
    }
    // The graph input is not a node, so it is taken from the first operator
    // that reads it: the im2col of the patch convolution.
    const bool is_input_reader = t->op == GGML_OP_IM2COL && !st->saw_input;
    if (ask) {
        return is_input_reader || wanted(st, t->name);
    }
    if (is_input_reader) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = t->src[i];
            if (src && strcmp(src->name, "inp_raw") == 0) {
                dump(st, "inp_raw", src);
                st->saw_input = true;
            }
        }
    }
    if (wanted(st, t->name)) {
        dump(st, t->name, t);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Synthetic images: integer arithmetic only, so they are the same bytes on
// every host and compiler.
// ---------------------------------------------------------------------------

static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static uint8_t noise(int x, int y, int c) {
    return (uint8_t) (mix((uint32_t) x * 73856093U ^ (uint32_t) y * 19349663U ^ (uint32_t) c * 83492791U) >> 24);
}

enum pattern { CHECKER, GRADIENT, NOISE };

static std::vector<uint8_t> make_image(pattern p, int w, int h) {
    std::vector<uint8_t> rgb((size_t) w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t * px = &rgb[((size_t) y * w + x) * 3];
            switch (p) {
                case CHECKER: {
                    // 8-pixel squares in two saturated colours.
                    const bool on = ((x / 8) + (y / 8)) & 1;
                    px[0] = on ? 255 : 10;
                    px[1] = on ? 230 : 40;
                    px[2] = on ? 20 : 200;
                } break;
                case GRADIENT: {
                    // Smooth ramps on each axis with four bits of texture.
                    px[0] = (uint8_t) ((x * 255) / (w > 1 ? w - 1 : 1));
                    px[1] = (uint8_t) ((y * 255) / (h > 1 ? h - 1 : 1));
                    px[2] = (uint8_t) ((((x + y) * 255) / (w + h - 1)) ^ (noise(x, y, 0) & 15));
                } break;
                case NOISE: {
                    px[0] = noise(x, y, 0);
                    px[1] = noise(x, y, 1);
                    px[2] = noise(x, y, 2);
                } break;
            }
        }
    }
    return rgb;
}

// ---------------------------------------------------------------------------

struct context_config {
    int max_tokens;  // 0: llama.cpp's default for the projector (4096)
    bool flash_attn;
};

struct image_case {
    const char * name;
    pattern pat;
    int w, h;
    int detail; // leading blocks captured node by node
};

static mtmd_context * open_context(const char * mmproj, const llama_model * model,
                                   const context_config & cfg, int threads, cb_state * st) {
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu          = false;
    mp.print_timings    = false;
    mp.n_threads        = threads;
    mp.warmup           = false;
    mp.flash_attn_type  = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mp.image_max_tokens = cfg.max_tokens > 0 ? cfg.max_tokens : -1;
    mp.cb_eval           = cb_eval;
    mp.cb_eval_user_data = st;
    mtmd_context * ctx = mtmd_init_from_file(mmproj, model, mp);
    if (!ctx) { fprintf(stderr, "failed to load %s\n", mmproj); exit(1); }
    return ctx;
}

// Tokenize `rgb` and return the image chunk, or null when llama.cpp refuses
// the image. The chunk is owned by `chunks`.
static const mtmd_input_chunk * tokenize(mtmd_context * ctx, mtmd_input_chunks * chunks,
                                         const std::vector<uint8_t> & rgb, int w, int h) {
    mtmd_bitmap * bmp = mtmd_bitmap_init((uint32_t) w, (uint32_t) h, rgb.data());
    mtmd_input_text text;
    text.text          = mtmd_default_marker();
    text.add_special   = false;
    text.parse_special = true;
    const mtmd_bitmap * bitmaps[1] = { bmp };
    int32_t rc = 2;
    try {
        rc = mtmd_tokenize(ctx, chunks, &text, bitmaps, 1);
    } catch (const std::exception & e) {
        fprintf(stderr, "tokenize %dx%d threw: %s\n", w, h, e.what());
    }
    mtmd_bitmap_free(bmp);
    if (rc != 0) return nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) return c;
    }
    return nullptr;
}

static void capture(mtmd_context * ctx, cb_state * st, const context_config & cfg,
                    const image_case & ic, const std::string & out_dir) {
    const std::vector<uint8_t> rgb = make_image(ic.pat, ic.w, ic.h);
    std::string path = out_dir + "/" + ic.name;
    if (cfg.max_tokens > 0) path += "-max" + std::to_string(cfg.max_tokens);
    if (cfg.flash_attn) path += "-fa";
    path += ".bin";

    writer w;
    w.open(path);
    {
        std::vector<int32_t> wide(rgb.begin(), rgb.end());
        const int64_t ne[4] = { 3, ic.w, ic.h, 1 };
        w.record("image_rgb", 1, ne, wide.data());
    }
    w.scalar("image_max_tokens", cfg.max_tokens > 0 ? cfg.max_tokens : 4096);
    w.scalar("flash_attn", cfg.flash_attn ? 1 : 0);

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_input_chunk * chunk = tokenize(ctx, chunks, rgb, ic.w, ic.h);
    if (!chunk) { fprintf(stderr, "%s: tokenize failed\n", ic.name); exit(1); }
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);

    st->w = &w;
    st->detail = ic.detail;
    st->saw_input = false;
    st->n_embd = 0;
    if (mtmd_encode_chunk(ctx, chunk) != 0) { fprintf(stderr, "%s: encode failed\n", ic.name); exit(1); }
    st->w = nullptr;
    if (!st->saw_input || st->n_embd == 0) {
        fprintf(stderr, "%s: graph input or `projected` was not seen\n", ic.name);
        exit(1);
    }

    // What the language model is handed. The vocab-only language model cannot
    // say how wide a row is, so the width is the `projected` node's.
    const int64_t ne[4] = { st->n_embd, (int64_t) n_tokens, 1, 1 };
    w.record("output", 0, ne, mtmd_get_output_embd(ctx));
    fprintf(stderr, "%s: %zu tokens x %lld, %u records\n", path.c_str(), n_tokens,
            (long long) st->n_embd, w.n_records);
    w.close();
    mtmd_input_chunks_free(chunks);
}

// One line per size: `width height max_tokens grid_x grid_y n_tokens`, the
// grid in tokens. A size llama.cpp refuses is written with zeros.
static void sweep(mtmd_context * ctx, const context_config & cfg, FILE * out,
                  const std::vector<std::pair<int, int>> & sizes) {
    for (const auto & s : sizes) {
        const std::vector<uint8_t> rgb((size_t) s.first * s.second * 3, 0);
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_input_chunk * chunk = tokenize(ctx, chunks, rgb, s.first, s.second);
        size_t nx = 0, ny = 0, n = 0;
        if (chunk) {
            const mtmd_image_tokens * tokens = mtmd_input_chunk_get_tokens_image(chunk);
            nx = mtmd_image_tokens_get_nx(tokens);
            ny = mtmd_image_tokens_get_ny(tokens);
            n  = mtmd_input_chunk_get_n_tokens(chunk);
        }
        fprintf(out, "%d %d %d %zu %zu %zu\n", s.first, s.second,
                cfg.max_tokens > 0 ? cfg.max_tokens : 4096, nx, ny, n);
        mtmd_input_chunks_free(chunks);
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <mmproj.gguf> <out-dir> [threads]\n", argv[0]);
        return 1;
    }
    const char * model_path  = argv[1];
    const char * mmproj_path = argv[2];
    const std::string out_dir = argv[3];
    const int threads = argc > 4 ? atoi(argv[4]) : 8;

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only   = true;
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }

    cb_state st;
    FILE * sizes = fopen((out_dir + "/sizes.txt").c_str(), "w");
    if (!sizes) { fprintf(stderr, "cannot write to %s\n", out_dir.c_str()); return 1; }

    // Default ceiling, flash attention off: the fp32 graph, which is what the
    // reference is held to most tightly.
    {
        const context_config cfg = { 0, false };
        mtmd_context * ctx = open_context(mmproj_path, model, cfg, threads, &st);
        const image_case cases[] = {
            { "checker-64x48",    CHECKER,   64,  48, 23 }, // pads to 80x80: 4 tokens
            { "gradient-333x207", GRADIENT, 333, 207,  2 }, // pads to 360x240: 54 tokens
            { "noise-640x480",    NOISE,    640, 480,  1 }, // no padding: 192 tokens
        };
        for (const image_case & ic : cases) {
            capture(ctx, &st, cfg, ic, out_dir);
        }
        // Around the padding boundary, at and past the ceiling, both
        // orientations, the rounding case and a thin image llama.cpp refuses.
        sweep(ctx, cfg, sizes, {
            { 1, 1 }, { 39, 41 }, { 40, 40 }, { 41, 40 }, { 64, 48 }, { 333, 207 },
            { 640, 480 }, { 640, 640 }, { 1280, 720 }, { 1920, 1080 }, { 2560, 2560 },
            { 2561, 2560 }, { 2560, 2561 }, { 2010, 3250 }, { 3250, 2010 }, { 2020, 3260 },
            { 4000, 3000 }, { 3000, 4000 }, { 3840, 2160 }, { 5000, 1300 }, { 1300, 5000 },
            { 8000, 500 }, { 500, 8000 }, { 6553, 1001 }, { 7001, 4999 }, { 100000, 30 },
            { 200000, 30 },
        });
        mtmd_free(ctx);
    }
    // Flash attention on, which llama.cpp picks by default when the backend
    // supports it: keys and values are cast to F16.
    {
        const context_config cfg = { 0, true };
        mtmd_context * ctx = open_context(mmproj_path, model, cfg, threads, &st);
        capture(ctx, &st, cfg, { "gradient-333x207", GRADIENT, 333, 207, 1 }, out_dir);
        mtmd_free(ctx);
    }
    // A low ceiling (`--image-max-tokens 20`) so that an ordinary image takes
    // the resize path.
    {
        const context_config cfg = { 20, false };
        mtmd_context * ctx = open_context(mmproj_path, model, cfg, threads, &st);
        capture(ctx, &st, cfg, { "gradient-500x380", GRADIENT, 500, 380, 1 }, out_dir);
        capture(ctx, &st, cfg, { "noise-97x611", NOISE, 97, 611, 0 }, out_dir);
        sweep(ctx, cfg, sizes, {
            { 40, 40 }, { 200, 160 }, { 201, 160 }, { 500, 380 }, { 380, 500 }, { 97, 611 },
            { 1000, 30 }, { 30, 1000 }, { 179, 181 }, { 1024, 768 }, { 4000, 3000 },
        });
        mtmd_free(ctx);
    }

    fclose(sizes);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

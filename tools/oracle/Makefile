# Builds the llama.cpp reference-tensor capture program.
#
# This is the generator for the golden data that `crates/llmcuda-engine/tests/`
# gates the forward pass against. It builds OUT OF TREE against llama.cpp's
# installed headers and shared objects — no llama.cpp source is modified, and
# it registers the same public `llama_context_params::cb_eval` hook that
# `llama-eval-callback` uses, with a binary sink instead of a printer.
#
# See ../../docs/ORACLE.md for the full procedure, the prompt, and the
# fidelity controls.

LLAMA ?= /home/nixabe/llama.cpp
LLAMA_BUILD ?= $(LLAMA)/build
OUT   ?= capture
BENCH_OUT ?= bench_decode
VISION_OUT ?= capture_inkling2v

$(OUT): capture.cpp
	g++ -O2 -std=c++17 capture.cpp -o $(OUT) \
	    -I$(LLAMA)/include -I$(LLAMA)/ggml/include \
	    -L$(LLAMA_BUILD)/bin -lllama -lggml-base -lggml \
	    -Wl,-rpath,$(LLAMA_BUILD)/bin

$(BENCH_OUT): bench_decode.cpp
	g++ -O2 -std=c++17 bench_decode.cpp -o $(BENCH_OUT) \
	    -I$(LLAMA)/include -I$(LLAMA)/ggml/include \
	    -L$(LLAMA_BUILD)/bin -lllama -lggml-base -lggml \
	    -Wl,-rpath,$(LLAMA_BUILD)/bin

# The vision-tower capture for Inkling 2 Mini (`inkling2v`). It goes through
# libmtmd's public API, so it needs that library and its header. libggml pulls
# in the CUDA backend's shared objects at link time; export
# LD_LIBRARY_PATH to wherever those live (e.g. ~/cuda-12.8/lib) before `make`.
$(VISION_OUT): capture_inkling2v.cpp
	g++ -O2 -std=c++17 -Wno-deprecated-declarations capture_inkling2v.cpp -o $(VISION_OUT) \
	    -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/tools/mtmd \
	    -L$(LLAMA_BUILD)/bin -lmtmd -lllama -lggml-base -lggml \
	    -Wl,-rpath,$(LLAMA_BUILD)/bin

clean:
	rm -f $(OUT) $(BENCH_OUT) $(VISION_OUT)

.PHONY: clean

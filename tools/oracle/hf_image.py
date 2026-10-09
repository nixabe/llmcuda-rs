#!/usr/bin/env python3
"""Fixtures for `llmcuda_kernels::vision::preprocess`'s Hugging Face path.

Writes, into the one directory given:

- `smart_resize.txt`: transformers' own `smart_resize` (the Qwen2-VL image
  processor's) over sizes chosen to hit its rounding, growth, shrink and
  aspect-ratio branches. One case per line, `w h min max W H`, or
  `w h min max error`.
- `resize-NNNN.bin`: PyTorch's CPU uint8 antialiased bicubic
  (`F.interpolate(mode="bicubic", antialias=True)`), the kernel torchvision's
  resize hands a uint8 CPU bicubic to. Little-endian u32 `w h tw th`, then
  the `w*h*3` source bytes and the `tw*th*3` result, both interleaved RGB.

Needs torch, numpy and transformers; torchvision is not needed. Captured
fixtures are never committed (AGENTS.md): point
`LLMCUDA_HF_IMAGE_GOLDEN_DIR` at the directory and run
`cargo test --release -p llmcuda-kernels --test hf_image_golden`.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
except ImportError:  # the torchvision backend's module needs torchvision
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize

FACTOR = 32  # patch 16 x merge 2
BOUNDS = [(65536, 16777216), (65536, 1024 * 1024), (8192, 4096 * 1024), (65536, 64 * 1024)]


def smart_resize_cases(rng):
    edges = [1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 80, 112, 144, 176, 200, 255, 256, 257,
             300, 500, 512, 640, 1000, 1080, 1920, 3000, 4000, 6000]
    sizes = [(w, h) for w in edges for h in edges]
    sizes += [tuple(int(v) for v in rng.integers(1, 5000, 2)) for _ in range(400)]
    sizes += [(1, 250), (250, 1), (2, 401), (8000, 39)]
    lines = []
    for w, h in sizes:
        for lo, hi in BOUNDS:
            try:
                rh, rw = smart_resize(h, w, factor=FACTOR, min_pixels=lo, max_pixels=hi)
                lines.append(f"{w} {h} {lo} {hi} {rw} {rh}")
            except ValueError:
                lines.append(f"{w} {h} {lo} {hi} error")
    return lines


def image(rng, w, h, kind):
    if kind == 0:
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x * 7 + y * 3) % 256], -1)
    return a.astype(np.uint8)


def resize(a, th, tw):
    t = torch.from_numpy(a).permute(2, 0, 1)[None].contiguous()
    out = F.interpolate(t, size=(th, tw), mode="bicubic", align_corners=False, antialias=True)
    return out[0].permute(1, 2, 0).contiguous().numpy()


def resize_cases(rng):
    cases = []
    # What serving asks for: smart_resize targets of random sources.
    for _ in range(40):
        w, h = (int(v) for v in rng.integers(8, 900, 2))
        if max(w, h) / min(w, h) > 200:
            continue
        lo, hi = BOUNDS[int(rng.integers(0, len(BOUNDS)))]
        th, tw = smart_resize(h, w, factor=FACTOR, min_pixels=lo, max_pixels=hi)
        if tw * th <= 1536 * 1024:
            cases.append((w, h, tw, th))
    # Arbitrary shapes, one axis held, and a one-pixel source edge; never an
    # unresized one-pixel axis, where PyTorch repeats the first row.
    for _ in range(40):
        w, h = (int(v) for v in rng.integers(1, 400, 2))
        tw, th = (int(v) for v in rng.integers(2, 400, 2))
        if rng.integers(0, 4) == 0:
            tw = w
        if rng.integers(0, 4) == 0:
            th = h
        if (tw == w == 1) or (th == h == 1):
            continue
        cases.append((w, h, tw, th))
    cases += [(1920, 1080, 1312, 736), (500, 300, 512, 288), (1, 150, 32, 3136), (4000, 37, 1024, 32)]
    return cases


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261009)
    (out / "smart_resize.txt").write_text("\n".join(smart_resize_cases(rng)) + "\n")
    for n, (w, h, tw, th) in enumerate(resize_cases(rng)):
        a = image(rng, w, h, n % 2)
        b = resize(a, th, tw)
        (out / f"resize-{n:04d}.bin").write_bytes(struct.pack("<4I", w, h, tw, th) + a.tobytes() + b.tobytes())
    print(f"wrote {len(list(out.glob('resize-*.bin')))} resize cases and smart_resize.txt to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))

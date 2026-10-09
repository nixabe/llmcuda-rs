//! Image preprocessing: smart-resize, bilinear rescale, normalize,
//! patchify.
//!
//! Ported from llama.cpp `tools/mtmd/mtmd-image.cpp`:
//! `img_tool::calc_size_preserved_ratio` (the HF `smart_resize`),
//! `img_tool::resize` with `PAD_CEIL` (centered black padding), and
//! `resize_bilinear` (align-corners on u8 with truncation). Normalization
//! order is `tools/mtmd/clip-impl.h`: scale to `[0,1]` first, then
//! `(x - mean) / std` per channel.
//!
//! [`preprocess_hf`] is the second pipeline: the model's Hugging Face
//! processor (transformers' `Qwen2VLImageProcessor`), which stretches to its
//! own `smart_resize` target with PyTorch's antialiased bicubic instead of
//! fitting and padding: the path for a model whose reference *is* that
//! processor, as Clef-Flash's `joint_schema_model.py` is.

use crate::vision::cell_order_index;
use llmcuda_model::VisionConfig;

/// Pixel-count bounds, in *output tokens* worth of pixels.
///
/// llama.cpp sets `set_limit_image_tokens(8, 4096)` for `qwen3vl_merger`
/// (`tools/mtmd/clip.cpp`): one output token covers
/// `patch^2 * merge^2 = 1024` pixels, so images are scaled into
/// `[8*1024, 4096*1024]` pixels before patch alignment.
pub const MIN_IMAGE_TOKENS: u32 = 8;
pub const MAX_IMAGE_TOKENS: u32 = 4096;

/// The multiple every image edge is aligned to: `patch * merge` = 32 px.
pub fn align_edge(cfg: &VisionConfig) -> u32 {
    cfg.patch_size * cfg.spatial_merge
}

fn round_by(v: f64, f: f64) -> u32 {
    ((v / f).round() * f) as u32
}
fn ceil_by(v: f64, f: f64) -> u32 {
    ((v / f).ceil() * f) as u32
}
fn floor_by(v: f64, f: f64) -> u32 {
    ((v / f).floor() * f) as u32
}

/// Target size for a `w × h` input: aspect-preserving, aligned to 32,
/// clamped into the pixel budget. Both beta branches divide the
/// **original** dimensions, matching upstream exactly.
pub fn smart_resize(cfg: &VisionConfig, w: u32, h: u32) -> (u32, u32) {
    smart_resize_bounded(cfg, w, h, MAX_IMAGE_TOKENS)
}

/// [`smart_resize`] with the token ceiling lowered below the model's
/// [`MAX_IMAGE_TOKENS`] — how a serving limit like `--image-max-tokens`
/// reaches the resize. `max_tokens` is clamped into the model's own budget.
pub fn smart_resize_bounded(cfg: &VisionConfig, w: u32, h: u32, max_tokens: u32) -> (u32, u32) {
    let f = f64::from(align_edge(cfg));
    let token_px = f64::from(align_edge(cfg) * align_edge(cfg));
    let min_px = f64::from(MIN_IMAGE_TOKENS) * token_px;
    let max_px = f64::from(max_tokens.clamp(MIN_IMAGE_TOKENS, MAX_IMAGE_TOKENS)) * token_px;
    let (wf, hf) = (f64::from(w), f64::from(h));

    let mut w_bar = round_by(wf, f).max(align_edge(cfg));
    let mut h_bar = round_by(hf, f).max(align_edge(cfg));
    let area = f64::from(w_bar) * f64::from(h_bar);
    if area > max_px {
        let beta = (wf * hf / max_px).sqrt();
        h_bar = floor_by(hf / beta, f).max(align_edge(cfg));
        w_bar = floor_by(wf / beta, f).max(align_edge(cfg));
    } else if area < min_px {
        let beta = (min_px / (wf * hf)).sqrt();
        h_bar = ceil_by(hf * beta, f);
        w_bar = ceil_by(wf * beta, f);
    }
    (w_bar, h_bar)
}

/// Bilinear resize of interleaved RGB u8, align-corners, truncating to u8 —
/// `img_tool::resize_bilinear` in `tools/mtmd/mtmd-image.cpp`.
fn resize_bilinear(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * 3) as usize];
    let x_ratio = if dw > 1 {
        (sw - 1) as f32 / (dw - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if dh > 1 {
        (sh - 1) as f32 / (dh - 1) as f32
    } else {
        0.0
    };
    for y in 0..dh {
        let py = y as f32 * y_ratio;
        let y0 = py.floor() as u32;
        let y1 = (y0 + 1).min(sh - 1);
        let yf = py - y0 as f32;
        for x in 0..dw {
            let px = x as f32 * x_ratio;
            let x0 = px.floor() as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let xf = px - x0 as f32;
            for c in 0..3u32 {
                let at = |xx: u32, yy: u32| src[((yy * sw + xx) * 3 + c) as usize] as f32;
                let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * xf;
                let bot = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * xf;
                out[((y * dw + x) * 3 + c) as usize] = (top + (bot - top) * yf) as u8;
            }
        }
    }
    out
}

/// A preprocessed image, ready for the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedImage {
    /// Flattened normalized patches in cell order, each
    /// `3 * patch^2` elements laid out `x + patch*y + patch^2*c`.
    pub patches: Vec<f32>,
    /// Patch-grid height (rows of patches).
    pub grid_h: u32,
    /// Patch-grid width.
    pub grid_w: u32,
}

impl PreprocessedImage {
    /// Language-model tokens this image expands to after the 2×2 merge.
    pub fn output_tokens(&self, cfg: &VisionConfig) -> u32 {
        cfg.output_tokens(self.grid_h, self.grid_w)
    }
}

/// Preprocess an interleaved RGB8 image of size `w × h`.
///
/// Pipeline (each step cited in the module docs): smart-resize target,
/// aspect-preserving bilinear rescale, centered black padding to the
/// target, `[0,1]` scaling, mean/std normalization, patchify into cell
/// order.
pub fn preprocess(cfg: &VisionConfig, rgb: &[u8], w: u32, h: u32) -> PreprocessedImage {
    preprocess_bounded(cfg, rgb, w, h, MAX_IMAGE_TOKENS)
}

/// [`preprocess`] under a lowered token ceiling (see
/// [`smart_resize_bounded`]).
pub fn preprocess_bounded(
    cfg: &VisionConfig,
    rgb: &[u8],
    w: u32,
    h: u32,
    max_tokens: u32,
) -> PreprocessedImage {
    assert_eq!(rgb.len(), (w * h * 3) as usize, "interleaved RGB8 expected");
    assert!(w > 0 && h > 0);
    let (tw, th) = smart_resize_bounded(cfg, w, h, max_tokens);

    // Aspect-preserving fit, then centered pad with black — the PAD_CEIL
    // branch of img_tool::resize.
    let scale = (f64::from(tw) / f64::from(w)).min(f64::from(th) / f64::from(h));
    let new_w = ((f64::from(w) * scale).ceil() as u32).min(tw);
    let new_h = ((f64::from(h) * scale).ceil() as u32).min(th);
    let scaled = if (new_w, new_h) == (w, h) {
        rgb.to_vec()
    } else {
        resize_bilinear(rgb, w, h, new_w, new_h)
    };
    let (off_x, off_y) = ((tw - new_w) / 2, (th - new_h) / 2);
    let mut canvas = vec![0u8; (tw * th * 3) as usize];
    for y in 0..new_h {
        let src = ((y * new_w) * 3) as usize;
        let dst = (((y + off_y) * tw + off_x) * 3) as usize;
        canvas[dst..dst + (new_w * 3) as usize]
            .copy_from_slice(&scaled[src..src + (new_w * 3) as usize]);
    }

    patchify(cfg, &canvas, tw, th)
}

/// Scale to `[0, 1]`, normalize per channel, and patchify a `tw × th`
/// interleaved RGB8 canvas into cell order.
fn patchify(cfg: &VisionConfig, canvas: &[u8], tw: u32, th: u32) -> PreprocessedImage {
    let patch = cfg.patch_size;
    let grid_w = tw / patch;
    let grid_h = th / patch;
    let patch_len = (3 * patch * patch) as usize;
    let mut patches = vec![0.0f32; (grid_w * grid_h) as usize * patch_len];
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let base = cell_order_index(gx, gy, grid_w) * patch_len;
            for c in 0..3u32 {
                let mean = cfg.image_mean[c as usize];
                let std = cfg.image_std[c as usize];
                for py in 0..patch {
                    for px in 0..patch {
                        let sx = gx * patch + px;
                        let sy = gy * patch + py;
                        let v = canvas[((sy * tw + sx) * 3 + c) as usize] as f32 / 255.0;
                        patches[base + (px + patch * py + patch * patch * c) as usize] =
                            (v - mean) / std;
                    }
                }
            }
        }
    }

    PreprocessedImage {
        patches,
        grid_h,
        grid_w,
    }
}

/// The pixel bounds of the Hugging Face processor: `size.shortest_edge` and
/// `size.longest_edge` in Cloudflare/clef-flash's `processor_config.json`,
/// the Qwen3-VL processor's own defaults. No GGUF carries them.
pub const HF_MIN_PIXELS: u64 = 65_536;
pub const HF_MAX_PIXELS: u64 = 16_777_216;

/// transformers' `smart_resize` (`models/qwen2_vl/image_processing_qwen2_vl.py`),
/// operation for operation: `round` is Python's, ties to even, so 80 px at a
/// 32 px factor is 64, not 96; the shrink keeps one factor per edge and the
/// growth needs none; an aspect ratio past 200 is an error, as there.
/// Returns `(width, height)`.
pub fn smart_resize_hf(
    cfg: &VisionConfig,
    w: u32,
    h: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32), String> {
    let (wf, hf) = (f64::from(w), f64::from(h));
    let ratio = wf.max(hf) / wf.min(hf);
    if ratio > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {ratio}"
        ));
    }
    let factor = u64::from(align_edge(cfg));
    let f = factor as f64;
    let mut h_bar = (hf / f).round_ties_even() as u64 * factor;
    let mut w_bar = (wf / f).round_ties_even() as u64 * factor;
    let area = (u64::from(h) * u64::from(w)) as f64;
    if h_bar * w_bar > max_pixels {
        let beta = (area / max_pixels as f64).sqrt();
        h_bar = factor.max((hf / beta / f).floor() as u64 * factor);
        w_bar = factor.max((wf / beta / f).floor() as u64 * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / area).sqrt();
        h_bar = (hf * beta / f).ceil() as u64 * factor;
        w_bar = (wf * beta / f).ceil() as u64 * factor;
    }
    let edge =
        |v: u64| u32::try_from(v).map_err(|_| format!("the image resizes to an edge of {v} px"));
    Ok((edge(w_bar)?, edge(h_bar)?))
}

/// The antialiased cubic kernel, `a = -0.5` as PIL uses it, in PyTorch's
/// evaluation order (`HelperInterpCubic::aa_filter`).
fn aa_cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// One axis of the resample: per output index, the first source index, how
/// many taps it reads, and those taps in fixed point (rows `taps` wide), at
/// a precision shared by the axis.
struct AaAxis {
    first: Vec<usize>,
    len: Vec<usize>,
    weights: Vec<i32>,
    taps: usize,
    precision: u32,
}

/// `_compute_weights_aa` and `_compute_index_ranges_int16_weights`
/// (aten/src/ATen/native/cpu/UpSampleKernel.cpp): PIL's support and
/// centering, weights normalized in f64, then rounded to int16 at the
/// largest precision under which the axis's largest weight still fits.
fn aa_axis(input: usize, output: usize) -> AaAxis {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let taps = support.ceil() as usize * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut first = Vec::with_capacity(output);
    let mut len = Vec::with_capacity(output);
    let mut real = vec![0.0f64; output * taps];
    let mut wt_max = 0.0f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        // `as i64` truncates toward zero, as the C++ casts do.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xsize = (((center + support + 0.5) as i64).min(input as i64) - xmin)
            .clamp(0, taps as i64) as usize;
        let row = &mut real[i * taps..i * taps + xsize];
        let mut total = 0.0f64;
        for (j, w) in row.iter_mut().enumerate() {
            *w = aa_cubic(((j as i64 + xmin) as f64 - center + 0.5) * invscale);
            total += *w;
        }
        if total != 0.0 {
            for w in row.iter_mut() {
                *w /= total;
                wt_max = wt_max.max(*w);
            }
        }
        first.push(xmin as usize);
        len.push(xsize);
    }
    let mut precision = 0u32;
    while precision < 22 {
        if (0.5 + wt_max * f64::from(1u32 << (precision + 1))) as i32 >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let one = f64::from(1u32 << precision);
    let weights = real
        .iter()
        .map(|&w| {
            let v = w * one;
            i32::from((if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i16)
        })
        .collect();
    AaAxis {
        first,
        len,
        weights,
        taps,
        precision,
    }
}

/// Resample interleaved RGB8 along x (`horizontal`) or y, rounding back to
/// u8 with the kernel's half-up offset and saturation.
fn aa_pass(src: &[u8], w: usize, h: usize, axis: &AaAxis, horizontal: bool) -> Vec<u8> {
    let (out_w, out_h) = if horizontal {
        (axis.first.len(), h)
    } else {
        (w, axis.first.len())
    };
    let mut out = vec![0u8; out_w * out_h * 3];
    let half = (1i32 << axis.precision) >> 1;
    for y in 0..out_h {
        for x in 0..out_w {
            let (i, at) = if horizontal { (x, y * w) } else { (y, x) };
            let taps = &axis.weights[i * axis.taps..i * axis.taps + axis.len[i]];
            for c in 0..3 {
                let mut acc = half;
                for (j, &wt) in taps.iter().enumerate() {
                    let s = if horizontal {
                        at + axis.first[i] + j
                    } else {
                        (axis.first[i] + j) * w + at
                    };
                    acc += i32::from(src[s * 3 + c]) * wt;
                }
                out[(y * out_w + x) * 3 + c] = (acc >> axis.precision).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// Antialiased bicubic resize of interleaved RGB8, bit-exact with PyTorch's
/// CPU uint8 kernel (`F.interpolate(mode="bicubic", antialias=True)`): x
/// first, then y, through a u8 intermediate, each axis only if it changes.
///
/// PyTorch leaves one quirk out of this port: an unresized axis one pixel
/// wide makes it repeat the first row of the other. [`smart_resize_hf`]
/// never asks for that shape — both target edges are multiples of 32.
pub fn resize_bicubic_aa(src: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    let (w, h, tw, th) = (w as usize, h as usize, tw as usize, th as usize);
    assert_eq!(src.len(), w * h * 3, "interleaved RGB8 expected");
    assert!(tw > 0 && th > 0, "empty target");
    let wide = if tw != w {
        aa_pass(src, w, h, &aa_axis(w, tw), true)
    } else {
        src.to_vec()
    };
    if th != h {
        aa_pass(&wide, tw, h, &aa_axis(h, th), false)
    } else {
        wide
    }
}

/// Preprocess as the model's Hugging Face processor does: [`smart_resize_hf`]
/// within [`HF_MIN_PIXELS`] and the lesser of [`HF_MAX_PIXELS`] and
/// `max_tokens`, a stretch with [`resize_bicubic_aa`] (no padding), then the
/// same scaling, normalization and patch order as [`preprocess`].
///
/// transformers' default backend resizes uint8 tensors with torchvision,
/// whose CPU bicubic is the PyTorch kernel ported here; its PIL backend
/// differs from that kernel by at most 2 levels on about 0.35% of values
/// (1.3% at worst, on noise). Above
/// `max_tokens` the processor would keep more pixels than the tower was
/// allocated for, so the image is shrunk to the ceiling instead; a growth
/// that would overshoot it is refused.
pub fn preprocess_hf(
    cfg: &VisionConfig,
    rgb: &[u8],
    w: u32,
    h: u32,
    max_tokens: u32,
) -> Result<PreprocessedImage, String> {
    assert_eq!(rgb.len(), (w * h * 3) as usize, "interleaved RGB8 expected");
    assert!(w > 0 && h > 0);
    let token_px = u64::from(align_edge(cfg)).pow(2);
    let max_px = HF_MAX_PIXELS
        .min(u64::from(max_tokens.clamp(MIN_IMAGE_TOKENS, MAX_IMAGE_TOKENS)) * token_px);
    let (tw, th) = smart_resize_hf(cfg, w, h, HF_MIN_PIXELS.min(max_px), max_px)?;
    if u64::from(tw) * u64::from(th) > max_px {
        return Err(format!(
            "the image resizes to {tw}x{th}, {} tokens, over the {} the server allows \
             (--image-max-tokens)",
            u64::from(tw) * u64::from(th) / token_px,
            max_px / token_px
        ));
    }
    Ok(if (tw, th) == (w, h) {
        patchify(cfg, rgb, tw, th)
    } else {
        patchify(cfg, &resize_bicubic_aa(rgb, w, h, tw, th), tw, th)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        VisionConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn already_aligned_sizes_pass_through() {
        assert_eq!(smart_resize(&cfg(), 768, 768), (768, 768));
        assert_eq!(smart_resize(&cfg(), 96, 128), (96, 128));
    }

    #[test]
    fn sizes_round_to_the_nearest_multiple_of_32() {
        assert_eq!(smart_resize(&cfg(), 100, 100), (96, 96));
        assert_eq!(smart_resize(&cfg(), 113, 97), (128, 96));
    }

    #[test]
    fn images_below_the_pixel_floor_scale_up() {
        // 64x64 = 4096 px < 8192 px minimum: beta = sqrt(2), 64*sqrt(2) ≈
        // 90.5, ceil to 96 on both edges.
        assert_eq!(smart_resize(&cfg(), 64, 64), (96, 96));
    }

    #[test]
    fn tiny_images_scale_up_to_the_minimum_budget() {
        // 10x10 = 100 px < 8192 px minimum; beta = sqrt(8192/100) ≈ 9.05,
        // 10*9.05 = 90.5 -> ceil to 96. 96*96 = 9216 >= 8192.
        let (w, h) = smart_resize(&cfg(), 10, 10);
        assert_eq!((w, h), (96, 96));
        assert!(w * h >= MIN_IMAGE_TOKENS * 1024);
    }

    #[test]
    fn huge_images_scale_down_to_the_maximum_budget() {
        let (w, h) = smart_resize(&cfg(), 10_000, 10_000);
        assert!(w * h <= MAX_IMAGE_TOKENS * 1024);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        // Aspect preserved: square in, square out.
        assert_eq!(w, h);
    }

    #[test]
    fn extreme_aspect_ratios_keep_the_minimum_edge() {
        let (w, h) = smart_resize(&cfg(), 5000, 40);
        assert!(w >= 32 && h >= 32);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn identity_resize_needs_no_interpolation() {
        let cfg = cfg();
        // 96x96 solid color (9216 px, above the floor): every normalized
        // value is exact.
        let rgb = vec![128u8; 96 * 96 * 3];
        let img = preprocess(&cfg, &rgb, 96, 96);
        assert_eq!((img.grid_w, img.grid_h), (6, 6));
        assert_eq!(img.output_tokens(&cfg), 9);
        let expect = (128.0 / 255.0 - 0.5) / 0.5;
        assert!(img.patches.iter().all(|&v| (v - expect).abs() < 1e-6));
    }

    #[test]
    fn channels_are_planar_within_a_patch() {
        let cfg = cfg();
        // Pure red 96x96: R channel = (1.0-0.5)/0.5 = 1, G/B = -1.
        let mut rgb = vec![0u8; 96 * 96 * 3];
        for px in rgb.as_chunks_mut::<3>().0 {
            px[0] = 255;
        }
        let img = preprocess(&cfg, &rgb, 96, 96);
        let plane = (cfg.patch_size * cfg.patch_size) as usize;
        for p in 0..36 {
            let patch = &img.patches[p * 3 * plane..(p + 1) * 3 * plane];
            assert!(patch[..plane].iter().all(|&v| (v - 1.0).abs() < 1e-6));
            assert!(patch[plane..].iter().all(|&v| (v + 1.0).abs() < 1e-6));
        }
    }

    #[test]
    fn patchify_places_pixels_in_cell_order() {
        let cfg = cfg();
        // 96x96, unique value per 16x16 patch: patch (gx,gy) is filled with
        // 24*gy+4*gx. After preprocessing, sequence slot
        // cell_order_index(gx,gy) must hold that value.
        let mut rgb = vec![0u8; 96 * 96 * 3];
        for y in 0..96u32 {
            for x in 0..96u32 {
                let v = ((y / 16) * 24 + (x / 16) * 4) as u8;
                let i = ((y * 96 + x) * 3) as usize;
                rgb[i] = v;
                rgb[i + 1] = v;
                rgb[i + 2] = v;
            }
        }
        let img = preprocess(&cfg, &rgb, 96, 96);
        let patch_len = (3 * cfg.patch_size * cfg.patch_size) as usize;
        for gy in 0..6u32 {
            for gx in 0..6u32 {
                let slot = cell_order_index(gx, gy, 6);
                let got = img.patches[slot * patch_len];
                let raw = (gy * 24 + gx * 4) as f32;
                let expect = (raw / 255.0 - 0.5) / 0.5;
                assert!(
                    (got - expect).abs() < 1e-6,
                    "patch ({gx},{gy}) at slot {slot}: got {got}, expected {expect}"
                );
            }
        }
    }

    #[test]
    fn a_lowered_ceiling_shrinks_large_images_and_spares_small_ones() {
        let cfg = cfg();
        // 1920x1080 lands over 256 tokens at the model budget; a 256-token
        // ceiling must pull it under while keeping alignment and aspect.
        let (tw, th) = smart_resize_bounded(&cfg, 1920, 1080, 256);
        assert!((tw / 32) * (th / 32) <= 256, "{tw}x{th}");
        assert_eq!(tw % 32, 0);
        assert_eq!(th % 32, 0);
        // A small image is untouched by the ceiling.
        assert_eq!(smart_resize_bounded(&cfg, 96, 96, 256), (96, 96));
        // A ceiling below the model's floor clamps to the floor, and the
        // shrink's floor-rounding still bounds the result by it.
        let (tw, th) = smart_resize_bounded(&cfg, 96, 96, 1);
        assert!((tw / 32) * (th / 32) <= MIN_IMAGE_TOKENS);
        assert!(tw >= 32 && th >= 32);
    }

    #[test]
    fn hf_smart_resize_rounds_ties_to_even_like_python() {
        // 80/32 = 2.5 and 48/32 = 1.5 both round to 2; half-away-from-zero
        // would make the first 96.
        assert_eq!(smart_resize_hf(&cfg(), 80, 48, 0, u64::MAX), Ok((64, 64)));
    }

    #[test]
    fn hf_smart_resize_grows_to_the_processor_floor_and_shrinks_to_the_ceiling() {
        // 100x100: beta = sqrt(65536 / 10000) = 2.56, and 256 is aligned.
        assert_eq!(
            smart_resize_hf(&cfg(), 100, 100, HF_MIN_PIXELS, HF_MAX_PIXELS),
            Ok((256, 256))
        );
        // 4000x3000 under 1 MiP: beta = sqrt(12e6 / 2^20) ~ 3.383, so
        // floor(36.95) and floor(27.71) blocks of 32.
        assert_eq!(
            smart_resize_hf(&cfg(), 4000, 3000, HF_MIN_PIXELS, 1 << 20),
            Ok((1152, 864))
        );
        assert!(smart_resize_hf(&cfg(), 200, 1, 0, u64::MAX).is_ok());
        assert!(smart_resize_hf(&cfg(), 201, 1, 0, u64::MAX).is_err());
    }

    #[test]
    fn an_unchanged_size_resizes_to_the_same_bytes() {
        let src: Vec<u8> = (0..32 * 64 * 3).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(resize_bicubic_aa(&src, 32, 64, 32, 64), src);
    }

    #[test]
    fn hf_preprocessing_of_an_aligned_in_range_image_is_the_llama_cpp_one() {
        // 512x384 is aligned and 192 tokens: neither pipeline resizes, so
        // the two agree exactly — which is why such images served the same
        // pixels before this path existed.
        let cfg = cfg();
        let rgb: Vec<u8> = (0..512 * 384 * 3).map(|i| (i * 31 % 253) as u8).collect();
        let hf = preprocess_hf(&cfg, &rgb, 512, 384, 1024).expect("in range");
        assert_eq!(hf, preprocess_bounded(&cfg, &rgb, 512, 384, 1024));
    }

    #[test]
    fn hf_preprocessing_grows_small_images_and_refuses_what_overflows_the_tower() {
        let cfg = cfg();
        let img = preprocess_hf(&cfg, &[128u8; 96 * 96 * 3], 96, 96, 1024).unwrap();
        assert_eq!(
            (img.grid_w, img.grid_h),
            (16, 16),
            "64 tokens, the processor's floor"
        );
        // 1x150 grows to 32x3136 = 98 tokens; a 64-token ceiling cannot
        // hold it, and shrinking it would no longer be the processor's image.
        let thin = vec![0u8; 150 * 3];
        assert!(preprocess_hf(&cfg, &thin, 1, 150, 1024).is_ok());
        assert!(preprocess_hf(&cfg, &thin, 1, 150, 64).is_err());
    }

    #[test]
    fn output_token_count_stays_within_the_declared_budget() {
        let cfg = cfg();
        for (w, h) in [(48, 48), (640, 480), (1920, 1080), (4096, 4096), (33, 5000)] {
            let (tw, th) = smart_resize(&cfg, w, h);
            let tokens = (tw / 32) * (th / 32);
            assert!(
                (MIN_IMAGE_TOKENS..=MAX_IMAGE_TOKENS).contains(&tokens),
                "{w}x{h} -> {tw}x{th} = {tokens} tokens"
            );
        }
    }
}

//! Image preprocessing for the `inkling2v` tower: pad, normalize, patchify —
//! and, only for an image over the token ceiling, shrink first.
//!
//! Ported from llama.cpp `tools/mtmd/mtmd-image.cpp` at ae6a7ad2c:
//! `mtmd_image_preprocessor_inkling2v::preprocess`,
//! `img_tool::calc_size_preserved_ratio` and `img_tool::resize_pillow`
//! (itself adapted from Pillow's `Resample.c`), plus
//! `mtmd_image_f32::normalize` in `mtmd-impl.h`.
//!
//! The common path touches no pixel value: the image keeps its size, is
//! padded on the right and bottom to a multiple of
//! `patch_size * spatial_merge` (40 px) and normalized. There is no
//! upscaling, so a tiny image becomes a single 40×40 block and one token.

use llmcuda_model::InklingVisionConfig;

/// The raw value the padding is filled with *before* normalization.
///
/// llama.cpp fills the padded canvas with `-1.0f` on the 0–255 byte scale
/// and then normalizes all of it, so a padded element is
/// `(-1 - 127.5) / 127.5`, slightly below the value of a black pixel.
pub const PAD_RAW_VALUE: f32 = -1.0;

/// The largest edge `img_tool::resize_pillow` accepts as a resize target;
/// llama.cpp throws above it.
pub const MAX_RESIZE_EDGE: u32 = 65_536;

/// Why an image cannot be preprocessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InklingImageError {
    /// A zero-sized image.
    #[error("image is {width}x{height}: both edges must be at least one pixel")]
    Empty {
        /// Stated width.
        width: u32,
        /// Stated height.
        height: u32,
    },
    /// The pixel buffer is not `width * height * 3` bytes.
    #[error("RGB8 buffer holds {found} bytes, but {expected} are needed")]
    BufferSize {
        /// `width * height * 3`.
        expected: u64,
        /// Bytes supplied.
        found: u64,
    },
    /// A token ceiling of zero.
    #[error("the image token ceiling must be at least one")]
    NoTokenBudget,
    /// The shrink target has an edge llama.cpp's resizer refuses.
    ///
    /// Reachable only by an extremely thin image: the size computation
    /// keeps the short edge at one block and can then *grow* the long one.
    #[error("resize target {width}x{height} exceeds the {MAX_RESIZE_EDGE}-pixel edge limit")]
    ResizeOutOfRange {
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
    },
}

/// What preprocessing will do to an image of a given size, decided without
/// touching a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InklingImagePlan {
    /// The size the image is resampled to first, `(width, height)`; `None`
    /// on the common path, where the pixels are used as they are.
    pub resize: Option<(u32, u32)>,
    /// Width of the padded canvas in pixels, a multiple of the token edge.
    pub width: u32,
    /// Height of the padded canvas in pixels, a multiple of the token edge.
    pub height: u32,
    /// Patch columns: `width / patch_size`.
    pub grid_w: u32,
    /// Patch rows: `height / patch_size`.
    pub grid_h: u32,
    /// Columns of language-model tokens: `grid_w / spatial_merge`.
    pub merged_w: u32,
    /// Rows of language-model tokens: `grid_h / spatial_merge`.
    pub merged_h: u32,
}

impl InklingImagePlan {
    /// Language-model tokens the image becomes: `merged_w * merged_h`,
    /// row-major, with no separator or end-of-row tokens.
    pub const fn tokens(&self) -> usize {
        self.merged_w as usize * self.merged_h as usize
    }
}

/// `MTMD_ALIGN`: round `x` up to a multiple of `align`.
fn align_up(x: u32, align: u32) -> u64 {
    u64::from(x).div_ceil(u64::from(align)) * u64::from(align)
}

/// `img_tool::calc_size_preserved_ratio` with `align_size = align` and
/// `max_pixels`, every other option at its default — the call
/// `mtmd_image_preprocessor_inkling2v::preprocess` makes. Returns
/// `(width, height)`.
///
/// Operation for operation in fp32, as upstream: each edge is first
/// **rounded** (half away from zero, not ceiled) to the alignment and kept
/// at one block or more; if that area is over `max_pixels`, both edges are
/// scaled by `1 / sqrt(w * h / max_pixels)` computed from the *original*
/// size and floored to the alignment, again kept at one block or more.
///
/// That last clamp means the result is not always within `max_pixels`: an
/// edge that floors to zero is held at one block while the other edge is
/// scaled as if it had shrunk.
pub fn calc_size_preserved_ratio(
    width: u32,
    height: u32,
    align: u32,
    max_pixels: u64,
) -> (u32, u32) {
    let f = align as f32;
    let round_by_factor = |x: f32| (x / f).round() as i64 * i64::from(align);
    let floor_by_factor = |x: f32| (x / f).floor() as i64 * i64::from(align);
    let floor_at_one_block = |v: i64| v.max(i64::from(align));

    let mut w_bar = floor_at_one_block(round_by_factor(width as f32));
    let mut h_bar = floor_at_one_block(round_by_factor(height as f32));
    if h_bar * w_bar > max_pixels as i64 {
        let beta = (height as f32 * width as f32 / max_pixels as f32).sqrt();
        h_bar = floor_at_one_block(floor_by_factor(height as f32 / beta));
        w_bar = floor_at_one_block(floor_by_factor(width as f32 / beta));
    }
    (w_bar as u32, h_bar as u32)
}

/// Decide the canvas and token grid for a `width × height` image under a
/// ceiling of `max_tokens` language-model tokens.
///
/// `max_tokens` is llama.cpp's `--image-max-tokens`; pass
/// [`InklingVisionConfig::max_image_tokens`] for its default. One token
/// covers `token_edge^2` pixels, so the ceiling is `max_tokens * 1600`
/// pixels of padded canvas. An image whose *padded* area is within it is
/// only padded; a larger one is resized to
/// [`calc_size_preserved_ratio`]'s target, which is already aligned.
///
/// The ceiling is not a guarantee: for a very thin image the shrink target
/// keeps one full block on the short edge and the token count can exceed
/// `max_tokens` (1000×30 under 16 tokens yields 23). Count with
/// [`token_count`]; do not assume.
pub fn plan(
    cfg: &InklingVisionConfig,
    width: u32,
    height: u32,
    max_tokens: u32,
) -> Result<InklingImagePlan, InklingImageError> {
    if width == 0 || height == 0 {
        return Err(InklingImageError::Empty { width, height });
    }
    if max_tokens == 0 {
        return Err(InklingImageError::NoTokenBudget);
    }
    let align = cfg.token_edge();
    let max_pixels = u64::from(max_tokens) * u64::from(align) * u64::from(align);

    let too_large = align_up(width, align) * align_up(height, align) > max_pixels;
    let (resize, content_w, content_h) = if too_large {
        let (w, h) = calc_size_preserved_ratio(width, height, align, max_pixels);
        if w > MAX_RESIZE_EDGE || h > MAX_RESIZE_EDGE {
            return Err(InklingImageError::ResizeOutOfRange {
                width: w,
                height: h,
            });
        }
        (Some((w, h)), w, h)
    } else {
        (None, width, height)
    };

    let width = align_up(content_w, align) as u32;
    let height = align_up(content_h, align) as u32;
    let (grid_w, grid_h) = (width / cfg.patch_size, height / cfg.patch_size);
    Ok(InklingImagePlan {
        resize,
        width,
        height,
        grid_w,
        grid_h,
        merged_w: grid_w / cfg.spatial_merge,
        merged_h: grid_h / cfg.spatial_merge,
    })
}

/// Language-model tokens a `width × height` image occupies in the prompt,
/// without touching pixels. See [`plan`] for the grid and the caveat on
/// `max_tokens`.
pub fn token_count(
    cfg: &InklingVisionConfig,
    width: u32,
    height: u32,
    max_tokens: u32,
) -> Result<usize, InklingImageError> {
    plan(cfg, width, height, max_tokens).map(|p| p.tokens())
}

/// Fractional bits of the resampler's fixed-point weights: an `i32`
/// accumulator, 8-bit pixels, two bits of headroom.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Pillow's bicubic kernel (`a = -0.5`), as `resample_filter` in
/// `img_tool::resize_pillow` evaluates it.
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// One axis of the resample: per output index, the first source index and
/// tap count, and the taps in 22-bit fixed point (`ksize` per output).
struct AxisWeights {
    ksize: usize,
    /// `(first source index, taps)` per output index.
    bounds: Vec<(usize, usize)>,
    weights: Vec<i32>,
}

/// `precompute_weights` in `img_tool::resize_pillow`, without the
/// torchvision int16 variant this tower does not use.
fn precompute_weights(in_size: u32, out_size: u32) -> AxisWeights {
    const FILTER_SUPPORT: f64 = 2.0;
    let scale = f64::from(in_size) / f64::from(out_size);
    // Widen the kernel when shrinking (antialiasing), never narrow it.
    let filterscale = scale.max(1.0);
    let support = FILTER_SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let ss = 1.0 / filterscale;

    let mut bounds = Vec::with_capacity(out_size as usize);
    let mut weights = vec![0i32; out_size as usize * ksize];
    let mut taps = vec![0.0f64; ksize];
    for xx in 0..out_size as usize {
        let center = (xx as f64 + 0.5) * scale;
        // Truncating casts, as upstream's `static_cast<int>`.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(i64::from(in_size));
        let count = (xmax - xmin) as usize;

        let mut total = 0.0f64;
        for (x, tap) in taps[..count].iter_mut().enumerate() {
            *tap = bicubic_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
            total += *tap;
        }
        for (tap, out) in taps[..count].iter().zip(&mut weights[xx * ksize..]) {
            let w = if total != 0.0 { tap / total } else { *tap };
            // Pillow adds +/- 0.5 and truncates toward zero.
            let fixed = w * f64::from(1u32 << PRECISION_BITS) + if w < 0.0 { -0.5 } else { 0.5 };
            *out = fixed as i32;
        }
        bounds.push((xmin as usize, count));
    }
    AxisWeights {
        ksize,
        bounds,
        weights,
    }
}

/// Resample interleaved RGB8 along x (`horizontal`) or y to `out` samples:
/// `resample_horizontal` / `resample_vertical` in `img_tool::resize_pillow`.
fn resample_axis(src: &[u8], in_w: u32, in_h: u32, out: u32, horizontal: bool) -> Vec<u8> {
    let (in_w, in_h, out) = (in_w as usize, in_h as usize, out as usize);
    let axis = precompute_weights(if horizontal { in_w } else { in_h } as u32, out as u32);
    let (out_w, out_h) = if horizontal { (out, in_h) } else { (in_w, out) };
    let mut dst = vec![0u8; out_w * out_h * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            let (first, count) = axis.bounds[if horizontal { x } else { y }];
            let k = &axis.weights[if horizontal { x } else { y } * axis.ksize..][..count];
            for c in 0..3 {
                // Rounding bias of one half, then an arithmetic shift.
                let mut acc: i32 = 1 << (PRECISION_BITS - 1);
                for (i, &w) in k.iter().enumerate() {
                    let (sx, sy) = if horizontal {
                        (first + i, y)
                    } else {
                        (x, first + i)
                    };
                    acc += i32::from(src[(sy * in_w + sx) * 3 + c]) * w;
                }
                dst[(y * out_w + x) * 3 + c] = (acc >> PRECISION_BITS).clamp(0, 255) as u8;
            }
        }
    }
    dst
}

/// Antialiased bicubic resize of interleaved RGB8 to `tw × th`, the
/// integer arithmetic of Pillow's `Image.resize(..., BICUBIC)`:
/// `img_tool::resize_pillow` with `RESIZE_ALGO_BICUBIC`.
///
/// Separable, x first and then y through a u8 intermediate, each axis only
/// if it changes. Kernel `a = -0.5`, support widened by the shrink factor,
/// weights normalized in f64 and rounded to 22 fractional bits, an `i32`
/// accumulator with a half-unit bias, arithmetic shift, clamp to a byte.
///
/// This is **not** [`crate::vision::resize_bicubic_aa`]: that one is
/// PyTorch's kernel, which shares the filter and centering but rounds its
/// weights to int16 at a per-axis precision, and so differs from this one
/// by a level or two on a small fraction of values.
///
/// # Panics
///
/// If `src` is not `w * h * 3` bytes or any edge is zero.
pub fn resize_bicubic_pillow(src: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    assert!(w > 0 && h > 0 && tw > 0 && th > 0, "empty image");
    assert_eq!(src.len(), w as usize * h as usize * 3, "RGB8 buffer size");
    let mut cur = src.to_vec();
    if tw != w {
        cur = resample_axis(&cur, w, h, tw, true);
    }
    if th != h {
        cur = resample_axis(&cur, tw, h, th, false);
    }
    cur
}

/// Normalize one channel value on the 0–255 byte scale.
///
/// llama.cpp fuses the rescale into the normalization rather than dividing
/// by 255 first: `(v - mean * 255) / (std * 255)`, in fp32. The fused form
/// is what makes the padding value (see [`PAD_RAW_VALUE`]) well defined.
pub fn normalize(cfg: &InklingVisionConfig, raw: f32, channel: usize) -> f32 {
    (raw - cfg.image_mean[channel] * 255.0) / (cfg.image_std[channel] * 255.0)
}

/// A preprocessed image, ready for the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedInklingImage {
    /// `[grid_h * grid_w][3 * patch * patch]` normalized values. Patches
    /// are in raster order, `p = y * grid_w + x`; inside a patch the layout
    /// is `c * patch^2 + ky * patch + kx`, the row layout of the patch conv
    /// weight, so patch embedding is one GEMM over this buffer.
    pub patches: Vec<f32>,
    /// Patch columns.
    pub grid_w: u32,
    /// Patch rows.
    pub grid_h: u32,
}

impl PreprocessedInklingImage {
    /// Patches in the image: `grid_w * grid_h`.
    pub const fn n_patches(&self) -> usize {
        self.grid_w as usize * self.grid_h as usize
    }

    /// Language-model tokens the tower will return for this image.
    pub fn tokens(&self, cfg: &InklingVisionConfig) -> usize {
        cfg.output_tokens(self.grid_h, self.grid_w) as usize
    }

    /// The same values as a planar image, `[3][height][width]` with
    /// `width = grid_w * patch` — llama.cpp's `inp_raw` graph input, and the
    /// form a strided convolution wants.
    pub fn planar(&self, cfg: &InklingVisionConfig) -> Vec<f32> {
        let patch = cfg.patch_size as usize;
        let (gw, gh) = (self.grid_w as usize, self.grid_h as usize);
        let (w, h) = (gw * patch, gh * patch);
        let mut out = vec![0.0f32; 3 * w * h];
        for (p, values) in self.patches.chunks_exact(3 * patch * patch).enumerate() {
            let (px, py) = (p % gw, p / gw);
            for c in 0..3 {
                for ky in 0..patch {
                    let row = &values[(c * patch + ky) * patch..][..patch];
                    let at = (c * h + py * patch + ky) * w + px * patch;
                    out[at..at + patch].copy_from_slice(row);
                }
            }
        }
        out
    }

    /// The inverse of [`Self::planar`]: patchify a planar
    /// `[3][height][width]` image whose edges are multiples of the patch.
    ///
    /// # Panics
    ///
    /// If `planar` is not `3 * grid_w * grid_h * patch^2` values.
    pub fn from_planar(
        cfg: &InklingVisionConfig,
        planar: &[f32],
        grid_w: u32,
        grid_h: u32,
    ) -> Self {
        let patch = cfg.patch_size as usize;
        let (gw, gh) = (grid_w as usize, grid_h as usize);
        let (w, h) = (gw * patch, gh * patch);
        assert_eq!(planar.len(), 3 * w * h, "planar image size");
        let mut patches = Vec::with_capacity(planar.len());
        for py in 0..gh {
            for px in 0..gw {
                for c in 0..3 {
                    for ky in 0..patch {
                        let at = (c * h + py * patch + ky) * w + px * patch;
                        patches.extend_from_slice(&planar[at..at + patch]);
                    }
                }
            }
        }
        Self {
            patches,
            grid_w,
            grid_h,
        }
    }
}

/// Preprocess an interleaved RGB8 image of `width × height` pixels under a
/// ceiling of `max_tokens` tokens (see [`plan`]).
///
/// The image is anchored at the top left of the padded canvas. Every
/// element of the canvas is normalized with [`normalize`], the padding
/// from [`PAD_RAW_VALUE`]. Only when the plan calls for it is the image
/// first resized, with [`resize_bicubic_pillow`]; that target is already
/// aligned, so a resized image carries no padding.
pub fn preprocess(
    cfg: &InklingVisionConfig,
    rgb: &[u8],
    width: u32,
    height: u32,
    max_tokens: u32,
) -> Result<PreprocessedInklingImage, InklingImageError> {
    let plan = plan(cfg, width, height, max_tokens)?;
    let expected = u64::from(width) * u64::from(height) * 3;
    if rgb.len() as u64 != expected {
        return Err(InklingImageError::BufferSize {
            expected,
            found: rgb.len() as u64,
        });
    }

    let resized;
    let (pixels, content_w, content_h) = match plan.resize {
        Some((w, h)) => {
            resized = resize_bicubic_pillow(rgb, width, height, w, h);
            (resized.as_slice(), w as usize, h as usize)
        }
        None => (rgb, width as usize, height as usize),
    };

    let patch = cfg.patch_size as usize;
    let (gw, gh) = (plan.grid_w as usize, plan.grid_h as usize);
    let mut patches = Vec::with_capacity(gw * gh * 3 * patch * patch);
    for py in 0..gh {
        for px in 0..gw {
            for c in 0..3 {
                for ky in 0..patch {
                    for kx in 0..patch {
                        let (x, y) = (px * patch + kx, py * patch + ky);
                        let raw = if x < content_w && y < content_h {
                            f32::from(pixels[(y * content_w + x) * 3 + c])
                        } else {
                            PAD_RAW_VALUE
                        };
                        patches.push(normalize(cfg, raw, c));
                    }
                }
            }
        }
    }
    Ok(PreprocessedInklingImage {
        patches,
        grid_w: plan.grid_w,
        grid_h: plan.grid_h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> InklingVisionConfig {
        InklingVisionConfig::inkling2_mini()
    }

    const MAX: u32 = 4096;

    #[test]
    fn an_aligned_image_keeps_its_size_and_a_ragged_one_pads_up() {
        let c = cfg();
        let p = plan(&c, 640, 480, MAX).unwrap();
        assert_eq!((p.resize, p.width, p.height), (None, 640, 480));
        assert_eq!(
            (p.grid_w, p.grid_h, p.merged_w, p.merged_h),
            (32, 24, 16, 12)
        );
        assert_eq!(p.tokens(), 192);

        let p = plan(&c, 333, 207, MAX).unwrap();
        assert_eq!((p.resize, p.width, p.height), (None, 360, 240));
        assert_eq!(p.tokens(), 9 * 6);
    }

    #[test]
    fn a_tiny_image_is_one_token_not_an_upscale() {
        let p = plan(&cfg(), 1, 1, MAX).unwrap();
        assert_eq!((p.resize, p.width, p.height, p.tokens()), (None, 40, 40, 1));
        assert_eq!(token_count(&cfg(), 40, 41, MAX).unwrap(), 2);
    }

    #[test]
    fn an_image_at_the_ceiling_is_not_resized_and_one_past_it_is() {
        let c = cfg();
        // 2560 * 2560 = 4096 tokens exactly.
        let p = plan(&c, 2560, 2560, MAX).unwrap();
        assert_eq!((p.resize, p.tokens()), (None, 4096));
        // One more pixel of width pads to 2600 * 2560, over the ceiling.
        // round(2561 / 40) = 64 blocks, so the rounded size is back within
        // it: a resize, but by one pixel.
        let p = plan(&c, 2561, 2560, MAX).unwrap();
        assert_eq!((p.resize, p.tokens()), (Some((2560, 2560)), 4096));
    }

    #[test]
    fn the_shrink_floors_both_edges_from_the_original_size() {
        // 4000 x 3000: beta = sqrt(12e6 / 6553600) = 1.35317; 3000 / beta =
        // 2217.0 -> 2200, 4000 / beta = 2956.0 -> 2920.
        let p = plan(&cfg(), 4000, 3000, MAX).unwrap();
        assert_eq!(p.resize, Some((2920, 2200)));
        assert_eq!((p.width, p.height), (2920, 2200));
        assert_eq!(p.tokens(), 73 * 55);
        assert!(p.tokens() <= MAX as usize);
    }

    #[test]
    fn rounding_not_ceiling_picks_the_resize_target() {
        // Padded, 2040 * 3280 is over the ceiling; rounded, 2000 * 3240 is
        // within it, so the image is squashed slightly rather than scaled.
        let p = plan(&cfg(), 2010, 3250, MAX).unwrap();
        assert_eq!(p.resize, Some((2000, 3240)));
        assert_eq!(p.tokens(), 50 * 81);
    }

    #[test]
    fn a_thin_image_can_exceed_the_ceiling() {
        // The short edge floors to zero blocks and is held at one, while
        // the long edge is scaled as if it had shrunk: 920 x 40.
        let p = plan(&cfg(), 1000, 30, 16).unwrap();
        assert_eq!(p.resize, Some((920, 40)));
        assert_eq!(p.tokens(), 23);
    }

    #[test]
    fn bad_requests_are_errors_not_panics() {
        let c = cfg();
        assert_eq!(
            plan(&c, 0, 10, MAX),
            Err(InklingImageError::Empty {
                width: 0,
                height: 10
            })
        );
        assert_eq!(plan(&c, 10, 10, 0), Err(InklingImageError::NoTokenBudget));
        assert_eq!(
            preprocess(&c, &[0; 10], 2, 2, MAX),
            Err(InklingImageError::BufferSize {
                expected: 12,
                found: 10
            })
        );
        assert!(matches!(
            plan(&c, 200_000, 30, MAX),
            Err(InklingImageError::ResizeOutOfRange { .. })
        ));
    }

    #[test]
    fn padding_is_below_black_and_anchored_bottom_right() {
        let c = cfg();
        // 41 x 39 white pixels: canvas 80 x 40, two patches wide, two tall.
        let img = preprocess(&c, &vec![255u8; 41 * 39 * 3], 41, 39, MAX).unwrap();
        assert_eq!((img.grid_w, img.grid_h), (4, 2));
        let planar = img.planar(&c);
        let at = |ch: usize, x: usize, y: usize| planar[(ch * 40 + y) * 80 + x];
        let pad = (-1.0f32 - 127.5) / 127.5;
        assert!(pad < normalize(&c, 0.0, 0));
        for ch in 0..3 {
            assert_eq!(at(ch, 0, 0), 1.0);
            assert_eq!(at(ch, 40, 38), 1.0);
            assert_eq!(at(ch, 41, 38), pad);
            assert_eq!(at(ch, 40, 39), pad);
            assert_eq!(at(ch, 79, 39), pad);
        }
    }

    #[test]
    fn patches_follow_the_conv_weight_layout() {
        let c = cfg();
        // Each pixel encodes its own coordinates and channel.
        let (w, h) = (80u32, 40u32);
        let mut rgb = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgb.extend([x as u8, y as u8, (x + y) as u8]);
            }
        }
        let img = preprocess(&c, &rgb, w, h, MAX).unwrap();
        assert_eq!(img.n_patches(), 8);
        assert_eq!(img.tokens(&c), 2);
        // Patch (x=3, y=1), kernel offset (kx=7, ky=11).
        let patch = &img.patches[(4 + 3) * 1200..][..1200];
        let (x, y) = (3 * 20 + 7, 20 + 11);
        assert_eq!(patch[11 * 20 + 7], normalize(&c, x as f32, 0));
        assert_eq!(patch[400 + 11 * 20 + 7], normalize(&c, y as f32, 1));
        assert_eq!(patch[800 + 11 * 20 + 7], normalize(&c, (x + y) as f32, 2));
        // Planar and back is the identity.
        let again = PreprocessedInklingImage::from_planar(&c, &img.planar(&c), 4, 2);
        assert_eq!(again, img);
    }

    #[test]
    fn resize_leaves_a_flat_image_flat_and_an_unchanged_axis_alone() {
        let flat = vec![93u8; 50 * 30 * 3];
        assert_eq!(
            resize_bicubic_pillow(&flat, 50, 30, 20, 13),
            vec![93u8; 20 * 13 * 3]
        );
        let ramp: Vec<u8> = (0..7 * 5 * 3).map(|i| (i * 2) as u8).collect();
        assert_eq!(resize_bicubic_pillow(&ramp, 7, 5, 7, 5), ramp);
    }

    #[test]
    fn halving_a_step_edge_matches_pillow_by_hand() {
        // One row, 8 -> 4, a step from 0 to 200 between source pixels 3 and
        // 4. Pillow's bicubic at scale 2 has support 4 and 9 taps; for the
        // output centred at source 3.0 the normalized taps over sources
        // 0..=6 are the kernel at -1.25, -0.75, -0.25, 0.25, 0.75, 1.25
        // (and 1.75 for source 6 — the centre sits between pixels 2 and 3).
        let k = |x: f64| bicubic_filter(x);
        let taps: Vec<f64> = (0..7).map(|s| k((s as f64 - 3.0 + 0.5) * 0.5)).collect();
        let total: f64 = taps.iter().sum();
        let want = (200.0 * taps[4..].iter().sum::<f64>() / total).round() as u8;
        let mut row = vec![0u8; 8 * 3];
        row[12..].fill(200);
        let out = resize_bicubic_pillow(&row, 8, 1, 4, 1);
        assert_eq!(out[3], want);
        // Symmetric about the step: the two middle outputs sum to 200.
        assert_eq!(u32::from(out[3]) + u32::from(out[6]), 200);
        // Overshoot on the far sides is clipped into a byte, not wrapped.
        assert!(out[0] <= 2 && out[9] >= 198);
    }
}

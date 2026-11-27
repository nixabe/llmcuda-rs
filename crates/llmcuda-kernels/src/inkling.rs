//! CPU reference for the `inkling2_loop` architecture (Inkling 2 Mini).
//!
//! A looped mixture-of-experts transformer: the middle layers run several
//! times over shared weights, each pass after the first steered by messages
//! mixed from the pass before it. There is no rotary embedding; positions
//! enter through a learned relative-position bias on the attention logits
//! and, on the full-attention layers, a logarithmic scale on the queries.
//!
//! Ported from llama.cpp `src/models/inkling2.cpp` and
//! `src/models/inkling2-loop.cpp` (`llama_model_inkling2::graph::build_layer`,
//! `mix`, `convolve`, `control`, `clip`, `build_output`) and the `LLM_FFN_SITU`
//! arm of `llm_graph_context::build_moe_ffn` in `src/llama-graph.cpp`, at
//! ae6a7ad2c. Everything is fp32 over whole sequences from position zero:
//! this is the oracle, not a serving path.
//!
//! Activations are row-major `[tokens][channels]`. Weights are in GGUF order,
//! `[in, out]` with `in` fastest.

use std::sync::Arc;

/// Hyperparameters, as the file states them.
#[derive(Debug, Clone, PartialEq)]
pub struct Hparams {
    /// Residual width.
    pub n_embd: usize,
    /// Layers of weights (not layer applications).
    pub n_layer: usize,
    /// Query heads.
    pub n_head: usize,
    /// Key/value heads; each serves `n_head / n_head_kv` query heads.
    pub n_head_kv: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Relative-position features per head.
    pub n_rel: usize,
    /// Distances the bias table of a full-attention layer covers.
    pub rel_extent: usize,
    /// Sliding window, and the extent of a sliding layer's bias table.
    pub n_swa: usize,
    /// Per layer: sliding-window attention rather than full.
    pub swa: Vec<bool>,
    /// Taps of the short convolutions.
    pub n_conv: usize,
    /// Layers per residual block.
    pub block: usize,
    /// First looped layer.
    pub loop_start: usize,
    /// One past the last looped layer.
    pub loop_end: usize,
    /// Passes through the looped layers.
    pub loop_count: usize,
    /// Experts per layer.
    pub n_expert: usize,
    /// Experts a token is routed to.
    pub n_expert_used: usize,
    /// Expert hidden width.
    pub n_ff_exp: usize,
    /// Scale on the normalized router weights.
    pub expert_weights_scale: f32,
    /// SiTU gate saturation.
    pub situ_beta: f32,
    /// SiTU linear-branch saturation.
    pub situ_linear_beta: f32,
    /// RMSNorm epsilon of the weighted norms and the value norm.
    pub rms_eps: f32,
    /// L2 bound on a query or key head.
    pub qk_clip: f32,
    /// L2 bound on a head's relative-position features.
    pub rel_clip: f32,
    /// Position below which the logarithmic scale is one.
    pub log_scale_floor: f32,
    /// Slope of the logarithmic scale.
    pub log_scale_alpha: f32,
    /// Final logit soft cap; zero disables it.
    pub softcap: f32,
    /// Vocabulary rows that are real tokens.
    pub vocab: usize,
}

impl Hparams {
    /// The reference's view of a loaded `inkling2_loop` configuration.
    ///
    /// # Panics
    ///
    /// If `config` is another architecture.
    pub fn from_config(config: &llmcuda_model::ModelConfig) -> Self {
        let k = config.inkling.expect("an inkling2_loop configuration");
        let moe = config.moe().expect("a routed feed-forward block");
        let f = f32::from_bits;
        Self {
            n_embd: config.hidden_size as usize,
            n_layer: config.num_layers as usize,
            n_head: config.attention.q_heads as usize,
            n_head_kv: config.attention.kv_heads as usize,
            head_dim: config.attention.head_dim as usize,
            n_rel: k.relative_length as usize,
            rel_extent: k.relative_extent as usize,
            n_swa: k.sliding_window as usize,
            swa: (0..config.num_layers).map(|l| k.is_sliding(l)).collect(),
            n_conv: k.conv_taps as usize,
            block: k.res_block as usize,
            loop_start: k.loop_start as usize,
            loop_end: k.loop_end as usize,
            loop_count: k.loop_count as usize,
            n_expert: moe.num_experts as usize,
            n_expert_used: moe.experts_per_token as usize,
            n_ff_exp: moe.expert_intermediate as usize,
            expert_weights_scale: f(k.expert_weights_scale_bits),
            situ_beta: f(k.situ_beta_bits),
            situ_linear_beta: f(k.situ_linear_beta_bits),
            rms_eps: f(k.rms_eps_bits),
            qk_clip: f(k.qk_clip_bits),
            rel_clip: f(k.relative_clip_bits),
            log_scale_floor: k.log_scale_floor as f32,
            log_scale_alpha: f(k.log_scale_alpha_bits),
            softcap: f(k.softcap_bits),
            vocab: config.vocab_size as usize,
        }
    }

    /// Cache slots: one per layer application.
    pub fn slots(&self) -> usize {
        self.n_layer + (self.loop_count - 1) * (self.loop_end - self.loop_start)
    }
}

/// Where the reference reads weights from.
pub trait Weights {
    /// A whole tensor, widened to fp32. `None` when the file has none.
    fn tensor(&self, name: &str) -> Option<Arc<Vec<f32>>>;
    /// Elements `[index * len, (index + 1) * len)` of a tensor, widened.
    fn part(&self, name: &str, index: usize, len: usize) -> Arc<Vec<f32>>;
}

const MIX_EPS: f32 = 1e-6;

fn rms_plain(x: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter().map(|v| v * scale).collect()
}

fn rms_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    rms_plain(x, eps)
        .iter()
        .zip(w)
        .map(|(v, w)| v * w)
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `y[o] = sum_i w[o * n_in + i] * x[i]` for every row of `x`.
pub fn project(w: &[f32], n_in: usize, x: &[f32]) -> Vec<f32> {
    let n_out = w.len() / n_in;
    let rows = x.len() / n_in;
    let mut out = vec![0.0f32; rows * n_out];
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (xs, ys) in x.chunks(per * n_in).zip(out.chunks_mut(per * n_out)) {
            scope.spawn(move || {
                for (x, y) in xs.chunks(n_in).zip(ys.chunks_mut(n_out)) {
                    for (y, row) in y.iter_mut().zip(w.chunks(n_in)) {
                        *y = dot(row, x);
                    }
                }
            });
        }
    });
    out
}

/// Scale each `width`-long group to an L2 norm of at most `threshold`,
/// after an optional per-channel scale (`scale[head][channel]`).
pub fn clip(x: &mut [f32], width: usize, scale: Option<&[f32]>, heads: usize, threshold: f32) {
    for (g, group) in x.chunks_mut(width).enumerate() {
        if let Some(scale) = scale {
            let h = g % heads;
            for (v, s) in group.iter_mut().zip(&scale[h * width..(h + 1) * width]) {
                *v *= s;
            }
        }
        let norm = group.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
        let factor = (threshold / norm).clamp(0.0, 1.0);
        for v in group.iter_mut() {
            *v *= factor;
        }
    }
}

/// Attention over the depth: mix `cur` and the earlier block outputs with
/// learned queries. Returns `[queries][n_embd]` for one token.
pub fn mix(blocks: &[&[f32]], cur: &[f32], query: &[f32]) -> Vec<Vec<f32>> {
    let n = cur.len();
    let states: Vec<Vec<f32>> = blocks
        .iter()
        .copied()
        .chain(std::iter::once(cur))
        .map(|s| rms_plain(s, MIX_EPS))
        .collect();
    let scale = 1.0 / (n as f32).sqrt();
    query
        .chunks(n)
        .map(|q| {
            let scores: Vec<f32> = states.iter().map(|s| dot(s, q) * scale).collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = e.iter().sum();
            let mut out = vec![0.0f32; n];
            for (s, e) in states.iter().zip(&e) {
                let p = e / sum;
                for (o, v) in out.iter_mut().zip(s) {
                    *o += p * v;
                }
            }
            out
        })
        .collect()
}

/// Causal short convolution with a residual.
///
/// `x` is `[tokens][channels]`, `weight` is `[channels][n_conv]` with the
/// last tap on the current token, and the history before the sequence is
/// zero. `dynamic`, when present, is `[tokens][heads][n_conv]`: input
/// dependent taps per head, tap 0 on the current token.
pub fn convolve(
    x: &[f32],
    channels: usize,
    weight: &[f32],
    n_conv: usize,
    dynamic: Option<(&[f32], usize)>,
) -> Vec<f32> {
    let tokens = x.len() / channels;
    let at = |t: isize, c: usize| {
        if t < 0 {
            0.0
        } else {
            x[t as usize * channels + c]
        }
    };
    let norm = 1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * n_conv + 1) as f32).sqrt();
    let mut out = vec![0.0f32; x.len()];
    for t in 0..tokens {
        for c in 0..channels {
            let mut sum = 0.0f32;
            for i in 0..n_conv {
                sum +=
                    at(t as isize + i as isize - (n_conv as isize - 1), c) * weight[c * n_conv + i];
            }
            if let Some((taps, heads)) = dynamic {
                let head = c / (channels / heads);
                for lag in 0..n_conv {
                    sum +=
                        at(t as isize - lag as isize, c) * taps[(t * heads + head) * n_conv + lag];
                }
            }
            out[t * channels + c] = (x[t * channels + c] + sum) * norm;
        }
    }
    out
}

/// One looped layer's controller weights for one input.
pub struct Controller {
    /// `[2 * n_embd, rank]`.
    pub down: Arc<Vec<f32>>,
    /// `[rank, n_embd]`.
    pub gain: Arc<Vec<f32>>,
    /// `[rank, n_embd]`.
    pub forget: Arc<Vec<f32>>,
}

/// Mix a cross-pass message into an input:
/// `(1 - forget) * cur + gain * message`, both gates read from the pair.
pub fn control(cur: &[f32], message: &[f32], c: &Controller) -> Vec<f32> {
    let n = cur.len();
    let mut pair = rms_plain(cur, MIX_EPS);
    pair.extend(rms_plain(message, MIX_EPS));
    let gate = project(&c.down, 2 * n, &pair);
    let rank = gate.len();
    let gain = project(&c.gain, rank, &gate);
    let forget = project(&c.forget, rank, &gate);
    (0..n)
        .map(|i| cur[i] - forget[i].tanh() * cur[i] + gain[i].tanh() * message[i])
        .collect()
}

/// The SiTU activation: `beta * tanh(gate / beta) * sigmoid(gate)` times
/// `lb * tanh(up / lb)`.
pub fn situ(gate: f32, up: f32, beta: f32, linear_beta: f32) -> f32 {
    let act = beta * (gate / beta).tanh() * sigmoid(gate);
    let up = if linear_beta > 0.0 {
        linear_beta * (up / linear_beta).tanh()
    } else {
        up
    };
    act * up
}

/// The router's choice for one token: expert ids, best first, and their
/// normalized, scaled weights.
pub fn route(h: &Hparams, logits: &[f32], bias: Option<&[f32]>) -> (Vec<usize>, Vec<f32>) {
    let probs: Vec<f32> = logits.iter().map(|l| sigmoid(*l)).collect();
    let mut order: Vec<usize> = (0..probs.len()).collect();
    let key = |e: usize| probs[e] + bias.map_or(0.0, |b| b[e]);
    // Descending, ties to the lower id.
    order.sort_by(|a, b| key(*b).total_cmp(&key(*a)).then(a.cmp(b)));
    order.truncate(h.n_expert_used);
    let sum: f32 = order.iter().map(|e| probs[*e]).sum::<f32>() + 1e-20;
    let weights = order
        .iter()
        .map(|e| probs[*e] / sum * h.expert_weights_scale)
        .collect();
    (order, weights)
}

/// Names a tap: the graph node and the cache slot it belongs to.
pub type Tap<'a> = &'a mut dyn FnMut(&str, usize, &[f32]);

struct Run<'a> {
    h: &'a Hparams,
    w: &'a dyn Weights,
    tokens: usize,
    tau: Vec<f32>,
}

impl Run<'_> {
    fn need(&self, name: &str) -> Arc<Vec<f32>> {
        self.w
            .tensor(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    fn rows<'b>(&self, x: &'b [f32]) -> impl Iterator<Item = &'b [f32]> {
        x.chunks(x.len() / self.tokens)
    }

    fn mix_rows(&self, blocks: &[Vec<f32>], cur: &[f32], query: &[f32]) -> Vec<Vec<f32>> {
        let n = self.h.n_embd;
        let queries = query.len() / n;
        let mut out = vec![Vec::with_capacity(cur.len()); queries];
        for t in 0..self.tokens {
            let b: Vec<&[f32]> = blocks.iter().map(|b| &b[t * n..(t + 1) * n]).collect();
            for (o, m) in out.iter_mut().zip(mix(&b, &cur[t * n..(t + 1) * n], query)) {
                o.extend(m);
            }
        }
        out
    }

    fn control_rows(&self, cur: &[f32], message: &[f32], c: &Controller) -> Vec<f32> {
        self.rows(cur)
            .zip(self.rows(message))
            .flat_map(|(c0, m)| control(c0, m, c))
            .collect()
    }

    fn norm_rows(&self, x: &[f32], weight: &str) -> Vec<f32> {
        let w = self.need(weight);
        self.rows(x)
            .flat_map(|r| rms_weighted(r, &w, self.h.rms_eps))
            .collect()
    }

    fn controller(&self, layer: usize, which: &str) -> Controller {
        let name = |part: &str| format!("blk.{layer}.loop_{which}_ctrl_{part}.weight");
        Controller {
            down: self.need(&name("down")),
            gain: self.need(&name("gain")),
            forget: self.need(&name("forget")),
        }
    }

    /// One layer application. `messages` is `[q, k, v, ffn][tokens][n_embd]`.
    fn layer(
        &self,
        slot: usize,
        layer: usize,
        cur: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<&[Vec<f32>]>,
        tap: Tap<'_>,
    ) -> Vec<f32> {
        let h = self.h;
        let n = h.n_embd;
        let name = |s: &str| format!("blk.{layer}.{s}");
        let swa = h.swa[layer];
        let kv_width = h.head_dim * h.n_head_kv;
        let input = cur;

        let mut inputs: Vec<Vec<f32>> = match self.w.tensor(&name("attn_res_query.weight")) {
            Some(q) => {
                let mixed = self.mix_rows(blocks, &input, &q);
                (0..3)
                    .map(|j| mixed[if swa { 0 } else { j }].clone())
                    .collect()
            }
            None => vec![input.clone(); 3],
        };
        if let Some(m) = messages {
            for (j, which) in ["q", "k", "v"].iter().enumerate() {
                inputs[j] = self.control_rows(&inputs[j], &m[j], &self.controller(layer, which));
            }
        }
        let norms = if swa {
            ["attn_norm.weight"; 3]
        } else {
            [
                "attn_q_inp_norm.weight",
                "attn_k_inp_norm.weight",
                "attn_v_inp_norm.weight",
            ]
        };
        for (x, norm) in inputs.iter_mut().zip(norms) {
            *x = self.norm_rows(x, &name(norm));
        }

        let lin = |weight: &str, x: &[f32]| project(&self.need(&name(weight)), n, x);
        let conv = |x: &[f32], channels: usize, weight: &str, dynamic: Option<(&[f32], usize)>| {
            convolve(x, channels, &self.need(&name(weight)), h.n_conv, dynamic)
        };
        let dyn_k = (!swa).then(|| lin("attn_k_conv_proj.weight", &inputs[1]));
        let dyn_v = (!swa).then(|| lin("attn_v_conv_proj.weight", &inputs[2]));
        let mut q = lin("attn_q.weight", &inputs[0]);
        let mut k = conv(
            &lin("attn_k.weight", &inputs[1]),
            kv_width,
            "attn_k_conv.weight",
            dyn_k.as_deref().map(|d| (d, h.n_head_kv)),
        );
        let v = conv(
            &lin("attn_v.weight", &inputs[2]),
            kv_width,
            "attn_v_conv.weight",
            dyn_v.as_deref().map(|d| (d, h.n_head_kv)),
        );
        let mut r = lin("attn_rel.weight", &inputs[0]);

        clip(
            &mut q,
            h.head_dim,
            Some(&self.need(&name("attn_q_clip.weight"))),
            h.n_head,
            h.qk_clip,
        );
        clip(&mut k, h.head_dim, None, h.n_head_kv, h.qk_clip);
        let v: Vec<f32> = v
            .chunks(h.head_dim)
            .flat_map(|g| rms_plain(g, h.rms_eps))
            .collect();
        clip(
            &mut r,
            h.n_rel,
            Some(&self.need(&name("attn_rel_clip.weight"))),
            h.n_head,
            h.rel_clip,
        );
        if !swa {
            for t in 0..self.tokens {
                for x in &mut q[t * h.n_head * h.head_dim..(t + 1) * h.n_head * h.head_dim] {
                    *x *= self.tau[t];
                }
                for x in &mut r[t * h.n_head * h.n_rel..(t + 1) * h.n_head * h.n_rel] {
                    *x *= self.tau[t];
                }
            }
        }
        tap("Qcur", slot, &q);
        tap("Kcur", slot, &k);
        tap("Vcur", slot, &v);
        tap("Rcur", slot, &r);

        let sinks = lin("attn_sink_proj.weight", &inputs[0]);
        let bias = self.need(&name("attn_rel_bias.weight"));
        let extent = if swa { h.n_swa } else { h.rel_extent };
        let ratio = h.n_head / h.n_head_kv;
        let scale = 1.0 / (h.head_dim as f32).sqrt();
        let mut attended = vec![0.0f32; self.tokens * h.n_head * h.head_dim];
        for t in 0..self.tokens {
            let first = if swa {
                (t + 1).saturating_sub(h.n_swa)
            } else {
                0
            };
            for head in 0..h.n_head {
                let kv = head / ratio;
                let qh = &q[(t * h.n_head + head) * h.head_dim..][..h.head_dim];
                let rh = &r[(t * h.n_head + head) * h.n_rel..][..h.n_rel];
                let sink = sinks[t * h.n_head + head];
                let scores: Vec<f32> = (first..=t)
                    .map(|j| {
                        let kj = &k[(j * h.n_head_kv + kv) * h.head_dim..][..h.head_dim];
                        let distance = t - j;
                        let b = if distance < extent {
                            dot(&bias[distance * h.n_rel..][..h.n_rel], rh)
                        } else {
                            0.0
                        };
                        // The bias rides in with the mask: it is not scaled.
                        dot(qh, kj) * scale + b - sink
                    })
                    .collect();
                // A zero-logit sink joins the softmax and attends to nothing.
                let max = scores.iter().copied().fold(0.0f32, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = e.iter().sum::<f32>() + (-max).exp();
                let out = &mut attended[(t * h.n_head + head) * h.head_dim..][..h.head_dim];
                for (j, e) in (first..=t).zip(&e) {
                    let p = e / sum;
                    let vj = &v[(j * h.n_head_kv + kv) * h.head_dim..][..h.head_dim];
                    for (o, v) in out.iter_mut().zip(vj) {
                        *o += p * v;
                    }
                }
            }
        }
        tap("kqv_out", slot, &attended);
        let gate = lin("attn_gate.weight", &inputs[0]);
        for (a, g) in attended.iter_mut().zip(&gate) {
            *a *= sigmoid(*g);
        }
        let out = project(
            &self.need(&name("attn_output.weight")),
            h.n_head * h.head_dim,
            &attended,
        );
        let mut cur = conv(&out, n, "attn_post_conv.weight", None);
        tap("attn_out", slot, &cur);

        // The first layer of a block starts a new residual stream; the one
        // before it joins the mixed blocks.
        if layer.is_multiple_of(h.block) {
            blocks.push(input);
        } else {
            for (c, i) in cur.iter_mut().zip(&input) {
                *c += i;
            }
        }
        let ffn_inp = cur;
        tap("ffn_inp", slot, &ffn_inp);

        let mut x = self
            .mix_rows(blocks, &ffn_inp, &self.need(&name("ffn_res_query.weight")))
            .remove(0);
        if let Some(m) = messages {
            x = self.control_rows(&x, &m[3], &self.controller(layer, "ffn"));
        }
        let x = self.norm_rows(&x, &name("ffn_norm.weight"));

        let logits = lin("ffn_gate_inp.weight", &x);
        tap("ffn_moe_logits", slot, &logits);
        let route_bias = self.w.tensor(&name("exp_probs_b.bias"));
        let gate_up_scale = self.w.tensor(&name("ffn_gate_up_exps.scale"));
        let down_scale = self.w.tensor(&name("ffn_down_exps.scale"));
        let mut moe = vec![0.0f32; self.tokens * n];
        let mut topk = Vec::with_capacity(self.tokens * h.n_expert_used);
        for t in 0..self.tokens {
            let (experts, weights) = route(
                h,
                &logits[t * h.n_expert..(t + 1) * h.n_expert],
                route_bias.as_deref().map(Vec::as_slice),
            );
            topk.extend(experts.iter().map(|e| *e as f32));
            for (e, weight) in experts.into_iter().zip(weights) {
                let gate_up = self
                    .w
                    .part(&name("ffn_gate_up_exps.weight"), e, n * 2 * h.n_ff_exp);
                let down = self
                    .w
                    .part(&name("ffn_down_exps.weight"), e, h.n_ff_exp * n);
                let gu = project(&gate_up, n, &x[t * n..(t + 1) * n]);
                let s = gate_up_scale.as_ref().map_or(1.0, |s| s[e]);
                let hidden: Vec<f32> = (0..h.n_ff_exp)
                    .map(|i| {
                        situ(
                            gu[i] * s,
                            gu[h.n_ff_exp + i] * s,
                            h.situ_beta,
                            h.situ_linear_beta,
                        )
                    })
                    .collect();
                let y = project(&down, h.n_ff_exp, &hidden);
                let s = down_scale.as_ref().map_or(1.0, |s| s[e]) * weight;
                for (o, y) in moe[t * n..(t + 1) * n].iter_mut().zip(&y) {
                    *o += y * s;
                }
            }
        }
        tap("ffn_moe_topk", slot, &topk);
        tap("ffn_moe_out", slot, &moe);
        let mut cur = conv(&moe, n, "ffn_post_conv.weight", None);
        tap("ffn_out", slot, &cur);
        for (c, i) in cur.iter_mut().zip(&ffn_inp) {
            *c += i;
        }
        tap("l_out", slot, &cur);
        cur
    }
}

/// Run `embeddings` (`[tokens][n_embd]`, the raw embedding rows) from
/// position zero and return the final normalized hidden states
/// (`[tokens][n_embd]`).
///
/// `tap` sees the same nodes llama.cpp names in its graph, by cache slot.
pub fn hidden_states(h: &Hparams, w: &dyn Weights, embeddings: &[f32], tap: Tap<'_>) -> Vec<f32> {
    let n = h.n_embd;
    let tokens = embeddings.len() / n;
    let tau = (0..tokens)
        .map(|p| 1.0 + h.log_scale_alpha * ((p + 1) as f32 / h.log_scale_floor).max(1.0).ln())
        .collect();
    let run = Run { h, w, tokens, tau };
    let mut cur = run.norm_rows(embeddings, "token_embd_norm.weight");
    let mut blocks: Vec<Vec<f32>> = Vec::new();

    for layer in 0..h.loop_start {
        cur = run.layer(layer, layer, cur, &mut blocks, None, tap);
    }
    // Every pass starts from the prelude output.
    let prelude = cur.clone();
    let prelude_blocks = blocks.clone();
    let core = h.loop_end - h.loop_start;
    for pass in 0..h.loop_count {
        if pass == 0 {
            for layer in h.loop_start..h.loop_end {
                cur = run.layer(layer, layer, cur, &mut blocks, None, tap);
            }
            continue;
        }
        // The bank of the previous pass: its blocks after the first core
        // block, and its output.
        let bank: Vec<Vec<f32>> = blocks[prelude_blocks.len() + 1..].to_vec();
        let bank_out = std::mem::replace(&mut cur, prelude.clone());
        blocks = prelude_blocks.clone();
        for layer in h.loop_start..h.loop_end {
            let slot = h.n_layer + (pass - 1) * core + (layer - h.loop_start);
            let query = run.need(&format!("blk.{layer}.loop_msg_query.weight"));
            let messages = run.mix_rows(&bank, &bank_out, &query);
            cur = run.layer(slot, layer, cur, &mut blocks, Some(&messages), tap);
        }
    }
    for layer in h.loop_end..h.n_layer {
        cur = run.layer(layer, layer, cur, &mut blocks, None, tap);
    }

    let mixed = run
        .mix_rows(&blocks, &cur, &run.need("output_res_query.weight"))
        .remove(0);
    run.norm_rows(&mixed, "output_norm.weight")
}

/// Logits of one normalized hidden state over the real vocabulary.
pub fn logits(h: &Hparams, w: &dyn Weights, hidden: &[f32]) -> Vec<f32> {
    let head = w
        .tensor("output.weight")
        .expect("missing tensor output.weight");
    let mut out = project(&head[..h.vocab * h.n_embd], h.n_embd, hidden);
    if h.softcap != 0.0 {
        for l in &mut out {
            *l = h.softcap * (*l / h.softcap).tanh();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_bounds_the_norm_and_leaves_short_vectors_alone() {
        let mut x = vec![3.0, 4.0, 0.3, 0.4];
        clip(&mut x, 2, None, 2, 1.0);
        assert!((x[0] - 0.6).abs() < 1e-5 && (x[1] - 0.8).abs() < 1e-5);
        assert_eq!(&x[2..], &[0.3, 0.4]);
        let mut y = vec![1.0, 0.0, 1.0, 0.0];
        clip(&mut y, 2, Some(&[10.0, 1.0, 0.5, 1.0]), 2, 2.0);
        assert!((y[0] - 2.0).abs() < 1e-5);
        assert_eq!(y[2], 0.5);
    }

    #[test]
    fn mix_of_one_state_is_that_state_normalized() {
        let cur = [3.0f32, 4.0];
        let out = mix(&[], &cur, &[1.0, 1.0]);
        let rms = (12.5f32 + MIX_EPS).sqrt();
        assert!((out[0][0] - 3.0 / rms).abs() < 1e-6);
        assert!((out[0][1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn convolve_reads_history_as_zero_and_orders_taps() {
        // One channel, taps [oldest .. current] = [1, 10, 100, 1000].
        let x = [1.0f32, 2.0, 3.0];
        let w = [1.0f32, 10.0, 100.0, 1000.0];
        let y = convolve(&x, 1, &w, 4, None);
        let norm = 1.0 / 5f32.sqrt();
        assert!((y[0] - (1.0 + 1000.0) * norm).abs() < 1e-3);
        assert!((y[2] - (3.0 + 3000.0 + 200.0 + 10.0) * norm).abs() < 1e-3);
        // Dynamic tap 0 is the current token, tap 1 the one before.
        let taps = [
            0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 7.0, 0.0, 0.0,
        ];
        let z = convolve(&x, 1, &[0.0; 4], 4, Some((&taps, 1)));
        assert!((z[2] - (3.0 + 15.0 + 14.0) / 3.0).abs() < 1e-5);
    }

    #[test]
    fn route_normalizes_selected_probabilities_and_selects_with_the_bias() {
        let h = Hparams {
            n_embd: 1,
            n_layer: 1,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 1,
            n_rel: 1,
            rel_extent: 1,
            n_swa: 1,
            swa: vec![true],
            n_conv: 4,
            block: 1,
            loop_start: 0,
            loop_end: 1,
            loop_count: 1,
            n_expert: 3,
            n_expert_used: 2,
            n_ff_exp: 1,
            expert_weights_scale: 8.0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            rms_eps: 1e-6,
            qk_clip: 1.0,
            rel_clip: 1.0,
            log_scale_floor: 1.0,
            log_scale_alpha: 0.0,
            softcap: 0.0,
            vocab: 1,
        };
        // The bias picks expert 2 over expert 1; the weights ignore it.
        let (ids, weights) = route(&h, &[0.0, 0.0, -1.0], Some(&[0.0, -1.0, 0.5]));
        assert_eq!(ids, vec![2, 0]);
        let p2 = sigmoid(-1.0);
        assert!((weights[0] - 8.0 * p2 / (p2 + 0.5)).abs() < 1e-5);
        assert!((weights.iter().sum::<f32>() - 8.0).abs() < 1e-4);
    }
}

# Hybrid cache

Implemented in [`llmcuda-cache`](../crates/llmcuda-cache). This document explains
the two design rules the crate exists to enforce, and why each has a
regression test rather than a comment.

## Two geometries, never unified

Qwen3.6 has two kinds of layer, and their cache behaviour has nothing in
common:

| Group | Layers | Natural page | Scaling |
| --- | --- | --- | --- |
| Attention | 10 | 5 MiB (256 tokens × 20,480 B) | **O(n)** — grows with position |
| Gated DeltaNet | 30 | 62.8125 MiB (full recurrent state) | **O(1)** — constant per sequence |

The attention page is 256 tokens of KV across all ten attention layers. The
GDN page is one complete recurrent-state snapshot across all thirty GDN
layers, including the convolution history as well as the 32 × 128 × 128 fp32
recurrent matrix per layer.

They differ by roughly 12.6×, and that difference is the whole point.

### Rule 1: allocate at natural page sizes

vLLM's allocator originally padded every cache group's page size up to the
maximum across groups. On Qwen3.5 — the same architecture family, 24 GDN and 8
attention layers of 32 — this produced roughly **7× KV memory
overestimation**, because every attention page was inflated to match a GDN
block. Both the reported token capacity and the actual allocation were wrong.

`CacheConfig` therefore computes `attention_page_bytes()` and
`gdn_page_bytes()` independently, and `BlockPool` is instantiated per group
with its own page size and free list.

**Capacity is reported per group and never summed.** `CapacityReport` has no
field that combines the two, and `attention_token_capacity()` is documented as
the only token figure the crate produces. Only attention grows with position;
the GDN group is a fixed per-slot cost, so adding them yields a number that
means nothing.

Regression tests:

- `attention_and_gdn_page_sizes_are_never_unified_to_a_shared_max_regression`
  — asserts the two page sizes differ by more than 5×. If a future change
  padded one to the other, the ratio would collapse toward 1.0 and this fails.
- `capacity_report_keeps_attention_and_gdn_capacity_independent_regression`

## Rule 2: retention interval is independent of block size

A GDN prefix snapshot is the full recurrent and convolution state — 62.8125
MiB — regardless of how much context it summarizes. Retaining one at every
attention block boundary is therefore catastrophic at small block sizes.

vLLM PR #45845 measured exactly that. At block size 128, **snapshots occupied
roughly 80% of the KV pool**, leaving no uncached headroom and forcing the
allocator to evict live attention prefixes:

| Metric | Effect |
| --- | --- |
| Prefix hit rate | ~85% → ~75% |
| Throughput | −18% |
| p99 latency | ~3× worse |

Block sizes of 256 and 512 were unaffected — which is precisely why coupling
the two parameters is a trap. It looks fine until someone tunes block size
down for hit-rate granularity and silently destroys throughput.

So the snapshot retention interval `R` is a separate parameter. Defaults:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DEFAULT_ATTENTION_BLOCK_SIZE` | 256 tokens | Attention page granularity |
| `DEFAULT_GDN_RETENTION_INTERVAL` | 2048 tokens | One snapshot per 8 attention blocks |

`R` is validated at construction to be a multiple of the block size, so
truncation to a snapshot boundary is plain integer division.

### The ratio to instrument

`CacheConfig::snapshot_to_kv_ratio()` reports snapshot bytes against the
attention KV accumulated over one retention interval. At the defaults:

```
62.8125 MiB snapshot / (20,480 B/token × 2048 tokens = 40 MiB KV) = 1.5703125
```

A ratio near or above 1 means snapshot overhead is comparable to the KV growth
it sits alongside. 1.5703125 is worth watching but is far from the pathological
case that produced the numbers above.

**Tune `R` against measured hit rate, not intuition.** Lowering it buys finer
GDN reuse granularity and pays in snapshot memory; this ratio is what
quantifies the trade.

Regression test:
`tiny_retention_interval_would_make_snapshots_dominate_the_pool_regression`.

## The exact-match constraint

This is the subtle one, and it is why `R` is a first-order parameter rather
than a tuning detail.

Linear-attention state at position *p* summarizes `[0, p)` **completely**.
That is excellent for resumption — one 62.8125 MiB blob restores thirty layers
of history. But it means the state cannot be sliced: you cannot reconstruct a
mid-prefix state from a longer one the way you can simply drop KV blocks past a
cut point.

**Reuse is possible only at points where a snapshot was actually retained.**

So a prefix match must be truncated back to the largest retained snapshot
boundary at or below the matched length. `CacheConfig::retention_floor()` does
that, and `RadixTree::match_prefix` applies it.

A 5,000-token prefix match with `R = 2048` yields a usable GDN resumption point
at token 4,096 — the remaining 904 tokens must be recomputed. That granularity
applies across **75% of the model's layers**, which is the entire reason `R`
matters more than it looks like it should.

Regression test:
`gdn_reuse_truncates_to_retained_snapshot_boundary_not_full_match_length_regression`.

### A state is kept where a request can resume, and nowhere else

The two halves of a snapshot are read at different densities. Restoring a
prefix copies the attention rows of *every* snapshot behind the resume point
— each holds only its own retention interval — and the recurrent state of the
resume point alone. So an attention interval has to stay wherever a later
resume point depends on it, while a recurrent state is read only by a request
that resumes exactly there.

The arena therefore holds them in two pools at their natural sizes (rule 1):
one attention interval per `R` tokens of restorable context, and a much
smaller number of recurrent states, `SNAPSHOT_STATES_PER_SEQUENCE` (8) per
serving sequence. A snapshot is still taken at every boundary, with a state.
What changes is which states survive:

| State | Why a request resumes there | Kept |
| --- | --- | --- |
| A sequence's newest | The sequence itself, continued verbatim | held |
| Last boundary inside the prompt | The conversation's next turn forks here, because the template re-renders the reply (see Known limitations) | held |
| Where the prompt left a prefix the tree had and no state survived for | Two requests reaching one boundary is a fork point: a system prompt, a shared document | held |
| Anything resumed from | It was a fork point once | held |
| Every other boundary a sequence has moved past | Only a request diverging inside the next `R` tokens | yielded |

A *yielded* state is still restored from while it lasts; it is the first to be
handed to a new snapshot when the pool has no free one, least recently used
first. A *held* state is given up only when its entry leaves the shared cache
(next section), and never under a request that has been told it can resume
there and is still waiting in its worker's queue (`ResumeHold`).

The cost is a recomputation the dense arena did not have: the *second*
request to share a prefix that no earlier request forked at finds attention
rows and no state, prefills through the shared part, and holds the state it
takes at the boundary where the match ended. The third hits it. vLLM
(`enable_mamba_shared_prefix_checkpoint`, `MambaManager`) and SGLang
(`unified_cache/components/mamba.py`, `branching_checkpoint_depth`) make the
same trade; unlike theirs it is paid here only once the state pool is full,
because yielding is lazy.

A match can now be deeper than its deepest surviving state, so the engine
walks the matched boundaries back to the deepest one that still holds a state
(`Engine::resume_candidates`) rather than trusting the tree's deepest marker.

Tests: `a_yielded_state_is_reassigned_only_when_none_is_free`,
`a_resumed_state_outlives_a_later_yield`,
`a_state_released_under_a_waiting_request_goes_when_it_settles`,
`a_budget_buys_states_for_its_sequences_and_intervals_with_the_rest`.

## The prefix tree

A radix tree over token-block hashes, held in host RAM behind an `RwLock` and
shared by all three workers. Structure mirrors vLLM's block-hash-chain design:
each node is keyed by a hash chaining its parent's hash with its own token
block, so identical prefixes converge on identical node chains regardless of
which worker inserted them.

| Operation | Purpose |
| --- | --- |
| `insert` | Publish a computed prefix |
| `match_prefix` | Longest-prefix lookup, truncated per rule 2 |
| `incr_ref_chain` / `decr_ref_chain` | Reference counting along a chain |
| `pin_chain` / `unpin_chain` | Eviction exemption |
| `evict_unreferenced` | LRU reclamation of unreferenced leaves |

**Pinning** exists for the platform system prompt: prefilled once per worker at
boot into an eviction-exempt slot. Three prefills at startup, zero at steady
state.

### What names a sequence's blocks

`llmcuda-engine`'s `SequenceChain` owns the hashing, and it hashes nothing but the
sequence's own tokens in order: the prompt the request was admitted with, then
every token the runtime reports generating, appended as it is reported. No
caller supplies hashes, because a chain that disagreed with the tokens would
file a snapshot under a prefix it does not describe, and the next request to
match that hash would resume from a prompt it never sent — fluently, with
nothing in the output to say so.

Two rules fall out of that:

- **Only complete blocks are hashed.** A partial trailing block has no stable
  name, since the same tokens hash differently once the block fills, and
  `RadixTree::insert` charges every entry a full `block_size` of positions —
  one partial entry would misplace every block after it. A prompt shorter than
  one block names nothing, and cannot.
- **Generated tokens are folded in before that step's snapshots are
  published.** Within a step the runtime interleaves the two — a speculative
  round retains at position `p`, then emits the token at `p` — so taking the
  generated tokens first is what guarantees every chain has reached the
  deepest position its step retained.

The engine also cross-checks the two records it has of the same fact: the
snapshot's `next_token`, recorded by the runtime, against the chain's own
record of the token that followed that position. They are built independently,
so a disagreement means the chain has drifted, and a drifted chain declines to
publish rather than share a prefix it cannot vouch for.

### Why the shared snapshot cache yields

A published snapshot pins at least one of its worker's attention intervals —
often several, since a snapshot holds an `Arc` to its parent for everything
before its own interval — and a recurrent state for as long as it holds one.
Those are the same pools live sequences snapshot into, and
running out is not a soft failure: the runtime answers `SnapshotArenaExhausted`
by switching that sequence's retention off *permanently*. Holding a snapshot
against a future hit therefore costs a present one.

So before a worker publishes, the cache drops that worker's least recently used
entries until its arena has one free interval and one obtainable state per
concurrent sequence — enough that every live sequence can still take its next
snapshot — and declines to publish if it cannot get there. A dropped entry
gives its recurrent state back at once, even while a later snapshot of the
same sequence keeps its attention rows alive. Recency counts actual reuse rather than insertion,
because a shared system prompt is old by construction and is exactly the entry
worth keeping.

### Why one process wins

The alternative was `--slot-save-path` on tmpfs as an L2 tier behind each
`llama-server` process's private cache. That design carried six risks — blob
corruption from parameter drift, weak-hash privacy leakage on a multi-user
platform, tmpfs competing with the page cache, torn reads needing atomic
rename, no reaper, and restore latency blocking sibling slots.

All six are artifacts of the replicas being separate OS processes. See
[ARCHITECTURE.md](ARCHITECTURE.md#why-one-process) for the mapping.

## Migration on cold hit

When the best-matching worker is saturated, the prefix is looked up in the host
tree, copied to a less-loaded worker, and only the tail is prefilled.

Staging goes through a **pre-allocated pinned arena** — pageable
host-to-device transfers roughly halve effective bandwidth. The arena keeps
the two natural geometries in separate pools: 40 MiB of attention KV per
2,048-token interval, and 62.8125 MiB per recurrent state.

`--cache-ram` sets the budget across all workers, and preflight prints the
intervals and states it bought; see [CLI.md](CLI.md#--cache-ram). Without it
a worker's arena covers its whole context, capped at a quarter of host
`MemAvailable` and floored at 24 intervals with a state each (2.41 GiB).

A snapshot holds its interval for as long as it or any child snapshot remains
reachable. If every interval is retained, or every state is held, the request
that reaches the next boundary stops publishing further snapshots but
continues inference. It does not later publish a delta spanning the missed
boundary.

Note what `--cache-ram` does *not* buy. It sets how many snapshots are kept,
never how far apart they are taken: retention granularity is `R`, so a match
is always truncated down to a multiple of it no matter how much memory is
available. Rule 2 above is why `R` is not a knob.

Snapshot copies currently run on and synchronize the serving stream at each
retention boundary. Moving them to a dedicated copy stream remains future
work; the arena removes hot-path allocation, not the copy stall.

## Known limitations

Stated rather than discovered later:

- `evict_unreferenced` is an O(n) linear scan per evicted node, finding the
  minimum-`last_used` leaf by scanning all nodes. Adequate for a maintenance
  operation, not benchmarked at scale. A production version wants an LRU heap
  or intrusive list, as vLLM uses in `FreeKVCacheBlockQueue`.
- `llmcuda-cache` itself does not touch device memory. `llmcuda-engine` accounts the
  per-worker device pools during admission and owns the pinned migration arena.
  The cache crate's tests remain host-side logic.
- **A chat turn's reply is not on the next turn's path**, and no amount of
  prefix caching changes that. It is a property of this model's chat template,
  not of this cache. Turn one's sequence runs
  `…<|im_start|>assistant\n<think>\n` and then its reply; turn two re-renders
  that same assistant turn as `…<|im_start|>assistant\n{content}<|im_end|>\n`,
  dropping the `<think>` block the generation prompt opened. The two token
  sequences diverge immediately after the assistant marker, so turn two's
  longest shared prefix ends at the end of turn one's *prompt* — which turn
  one's own prefill already published.

  Measured on one RTX 8000: a three-turn conversation over a ~3850-token
  system prompt reused 2048 tokens on turns two and three, and reused the same
  2048 before generated tokens were ever hashed. Do not expect this cache to
  earn its keep on chat beyond the shared prompt prefix; where it earns it is
  a prompt that genuinely continues an earlier reply — a raw completion picking
  up where it left off, or an agent replaying its own output verbatim. There,
  reuse went from 0 to 2048 tokens, worth 2.1× on time to first token over
  four interleaved pairs.

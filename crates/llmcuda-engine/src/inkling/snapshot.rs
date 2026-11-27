//! A sequence's resumable prefix: [`InklingState`] to and from the engine's
//! pinned snapshot arena.
//!
//! What a later row can still read is all a snapshot has to hold. A
//! full-attention slot reads every earlier position, so its keys and values
//! go in the arena's per-interval buffers, `[start, position)` at a time,
//! chained to the snapshot before. A sliding slot reads `window` positions
//! back, so its last `window` rows of keys and of values are fixed-size
//! state, beside every slot's four convolution carries. Rows are stored by
//! position, not by cache row: the sequence that resumes may hold rings of
//! another size, or none.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, HostSlice, PinnedHostSlice};

use super::{InklingModel, InklingState};
use crate::state::{SequenceSnapshot, SnapshotArena, StateError};

/// Rows `[first, end)` by position of a cache of `kv`-wide rows that holds
/// position `j` at row `j % ring`, as runs of contiguous cache rows:
/// `(cache row, position, rows)`.
fn runs(
    ring: Option<usize>,
    first: usize,
    end: usize,
) -> impl Iterator<Item = (usize, usize, usize)> {
    let mut at = first;
    std::iter::from_fn(move || {
        if at >= end {
            return None;
        }
        let (row, len) = match ring {
            Some(ring) => (at % ring, (end - at).min(ring - at % ring)),
            None => (at, end - at),
        };
        let run = (row, at, len);
        at += len;
        Some(run)
    })
}

/// The binary16 words a pinned f32 buffer is used as.
///
/// # Safety
/// As [`PinnedHostSlice::stream_synced_mut_slice`]; the returned slice is
/// twice the length and the same bytes.
unsafe fn words_mut<'a>(
    host: &'a mut PinnedHostSlice<f32>,
    stream: &'a Arc<CudaStream>,
) -> (&'a mut [u16], impl Drop + 'a) {
    // SAFETY: the caller's; f32 is aligned for u16 and every bit pattern
    // of either is valid.
    let (floats, record) = unsafe { host.stream_synced_mut_slice(stream) };
    let words = unsafe {
        std::slice::from_raw_parts_mut(floats.as_mut_ptr().cast::<u16>(), floats.len() * 2)
    };
    (words, record)
}

/// As [`words_mut`], to read.
///
/// # Safety
/// As [`PinnedHostSlice::stream_synced_slice`].
unsafe fn words<'a>(
    host: &'a PinnedHostSlice<f32>,
    stream: &'a Arc<CudaStream>,
) -> (&'a [u16], impl Drop + 'a) {
    // SAFETY: as `words_mut`.
    let (floats, record) = unsafe { host.stream_synced_slice(stream) };
    let words =
        unsafe { std::slice::from_raw_parts(floats.as_ptr().cast::<u16>(), floats.len() * 2) };
    (words, record)
}

impl InklingState {
    /// The carries of one slot, in the order a snapshot stores them.
    fn carries(slot: &super::SlotState) -> [&CudaSlice<f32>; 4] {
        [
            &slot.k_carry,
            &slot.v_carry,
            &slot.attn_carry,
            &slot.ffn_carry,
        ]
    }

    /// Copy what a resume at this position needs into page-locked host
    /// memory, chained to `parent` (an earlier snapshot of this same
    /// prefix) for the full slots' rows before it.
    pub(crate) fn snapshot(
        &self,
        stream: &Arc<CudaStream>,
        model: &InklingModel,
        arena: &SnapshotArena,
        parent: Option<Arc<SequenceSnapshot>>,
    ) -> Result<SequenceSnapshot, StateError> {
        let kv = model.kv_width();
        let window = model.k.sliding_window as usize;
        let layers = model.config.num_layers;
        let position = self.position;
        arena.capture(position, parent, |start, intervals, states| {
            if states.len() != self.slots.len() {
                return Err(StateError::SnapshotShape);
            }
            let mut full = intervals.iter_mut();
            for (index, (slot, host)) in self.slots.iter().zip(states.iter_mut()).enumerate() {
                let carries = Self::carries(slot);
                if host.conv.len() != carries.iter().map(|c| c.len()).sum::<usize>() {
                    return Err(StateError::SnapshotShape);
                }
                {
                    // SAFETY: used on this stream only; the guard outlives
                    // the enqueue of every copy into the buffer.
                    let (conv, _record) = unsafe { host.conv.stream_synced_mut_slice(stream) };
                    let mut at = 0;
                    for carry in carries {
                        stream.memcpy_dtoh(carry, &mut conv[at..at + carry.len()])?;
                        at += carry.len();
                    }
                }
                if model.k.is_sliding(model.k.slot_layer(layers, index as u32)) {
                    if host.recurrent.len() != window * kv {
                        return Err(StateError::SnapshotShape);
                    }
                    let first = position.saturating_sub(window);
                    // SAFETY: as above.
                    let (rows, _record) = unsafe { words_mut(&mut host.recurrent, stream) };
                    for (half, cache) in [&slot.k, &slot.v].into_iter().enumerate() {
                        for (row, at, len) in runs(slot.ring, first, position) {
                            let to = (half * window + at - first) * kv;
                            stream.memcpy_dtoh(
                                &cache.slice(row * kv..(row + len) * kv),
                                &mut rows[to..to + len * kv],
                            )?;
                        }
                    }
                } else {
                    let prefix = full.next().ok_or(StateError::SnapshotShape)?;
                    let elements = (position - start) * kv;
                    for (cache, host) in [(&slot.k, &mut prefix.k), (&slot.v, &mut prefix.v)] {
                        if elements > host.len() {
                            return Err(StateError::SnapshotShape);
                        }
                        // SAFETY: as above.
                        let (rows, _record) = unsafe { host.stream_synced_mut_slice(stream) };
                        stream.memcpy_dtoh(
                            &cache.slice(start * kv..start * kv + elements),
                            &mut rows[..elements],
                        )?;
                    }
                }
            }
            if full.next().is_some() {
                return Err(StateError::SnapshotShape);
            }
            stream.synchronize()?;
            Ok(())
        })
    }

    /// Become the prefix `snapshot` holds. It may come from another worker's
    /// device, and from a sequence whose rings were another size.
    pub(crate) fn restore(
        &mut self,
        stream: &Arc<CudaStream>,
        model: &InklingModel,
        snapshot: &SequenceSnapshot,
    ) -> Result<(), StateError> {
        let kv = model.kv_width();
        let window = model.k.sliding_window as usize;
        let layers = model.config.num_layers;
        let position = snapshot.position();
        if position > self.capacity {
            return Err(StateError::SnapshotShape);
        }
        let sliding = |index: usize| model.k.is_sliding(model.k.slot_layer(layers, index as u32));
        // Two closures write the slots, one after the other.
        let slots = std::cell::RefCell::new(&mut self.slots);
        snapshot.read(
            &mut |start, end, intervals| {
                let mut full = intervals.iter();
                for (index, slot) in slots.borrow_mut().iter_mut().enumerate() {
                    if sliding(index) {
                        continue;
                    }
                    let prefix = full.next().ok_or(StateError::SnapshotShape)?;
                    let elements = (end - start) * kv;
                    for (cache, host) in [(&mut slot.k, &prefix.k), (&mut slot.v, &prefix.v)] {
                        if elements > host.len() || end * kv > cache.len() {
                            return Err(StateError::SnapshotShape);
                        }
                        // SAFETY: capture initialized this prefix; the
                        // pinned event orders the read after it.
                        let (rows, _record) = unsafe { host.stream_synced_slice(stream) };
                        stream.memcpy_htod(
                            &rows[..elements],
                            &mut cache.slice_mut(start * kv..start * kv + elements),
                        )?;
                    }
                }
                if full.next().is_some() {
                    return Err(StateError::SnapshotShape);
                }
                Ok(())
            },
            |states| {
                let mut slots = slots.borrow_mut();
                if states.len() != slots.len() {
                    return Err(StateError::SnapshotShape);
                }
                for (index, (slot, host)) in slots.iter_mut().zip(states).enumerate() {
                    {
                        // SAFETY: as above.
                        let (conv, _record) = unsafe { host.conv.stream_synced_slice(stream) };
                        let mut at = 0;
                        for carry in [
                            &mut slot.k_carry,
                            &mut slot.v_carry,
                            &mut slot.attn_carry,
                            &mut slot.ffn_carry,
                        ] {
                            if at + carry.len() > conv.len() {
                                return Err(StateError::SnapshotShape);
                            }
                            stream.memcpy_htod(&conv[at..at + carry.len()], carry)?;
                            at += carry.len();
                        }
                    }
                    if !sliding(index) {
                        continue;
                    }
                    if host.recurrent.len() != window * kv {
                        return Err(StateError::SnapshotShape);
                    }
                    let first = position.saturating_sub(window);
                    // SAFETY: as above.
                    let (rows, _record) = unsafe { words(&host.recurrent, stream) };
                    let ring = slot.ring;
                    for (half, cache) in [&mut slot.k, &mut slot.v].into_iter().enumerate() {
                        for (row, at, len) in runs(ring, first, position) {
                            let from = (half * window + at - first) * kv;
                            stream.memcpy_htod(
                                &rows[from..from + len * kv],
                                &mut cache.slice_mut(row * kv..(row + len) * kv),
                            )?;
                        }
                    }
                }
                stream.synchronize()?;
                Ok(())
            },
        )?;
        self.position = position;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::runs;

    #[test]
    fn a_window_is_at_most_two_runs_of_a_ring() {
        // No ring: one run, by position.
        assert_eq!(runs(None, 5, 9).collect::<Vec<_>>(), [(5, 5, 4)]);
        // Inside the ring, and wrapping its end.
        assert_eq!(runs(Some(8), 9, 13).collect::<Vec<_>>(), [(1, 9, 4)]);
        assert_eq!(
            runs(Some(8), 13, 19).collect::<Vec<_>>(),
            [(5, 13, 3), (0, 16, 3)]
        );
        assert_eq!(runs(Some(8), 4, 4).count(), 0);
    }
}

//! Experimental resident continuation, confined to the test harness.
//!
//! This is not the production router. Slots and histories have fixed capacity;
//! a mutable lease prevents concurrent ownership, and only a successful commit
//! makes its consumed prefix reusable. A failed/abandoned pass invalidates it.

mod conversations;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Scope {
    model: usize,
    tokenizer: usize,
    rope: i32,
    session: u64,
}

struct Slot<T> {
    state: T,
    scope: Option<Scope>,
    consumed: Vec<i32>,
    valid: bool,
    owner: Option<u64>,
    used: u64,
}

struct ResidentSlots<T> {
    slots: Vec<Slot<T>>,
    capacity: usize,
    clock: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum Refused {
    Capacity,
    Busy,
}

struct Lease<'a, T> {
    slot: &'a mut Slot<T>,
    capacity: usize,
    reused: usize,
    prepared: bool,
    scope: Scope,
}

impl<T> ResidentSlots<T> {
    fn new(states: Vec<T>, capacity: usize) -> Self {
        assert!(!states.is_empty() && capacity > 0);
        Self {
            slots: states
                .into_iter()
                .map(|state| Slot {
                    state,
                    scope: None,
                    consumed: Vec::with_capacity(capacity),
                    valid: false,
                    owner: None,
                    used: 0,
                })
                .collect(),
            capacity,
            clock: 0,
        }
    }

    fn begin(
        &mut self,
        scope: Scope,
        prompt: &[i32],
        required: usize,
        owner: u64,
        resident: bool,
    ) -> Result<Lease<'_, T>, Refused> {
        if required < prompt.len() || required > self.capacity {
            return Err(Refused::Capacity);
        }
        let hit = resident.then(|| {
            self.slots.iter().position(|slot| {
                slot.owner.is_none()
                    && slot.valid
                    && slot.scope == Some(scope)
                    && !slot.consumed.is_empty()
                    && prompt.len() > slot.consumed.len()
                    && prompt.starts_with(&slot.consumed)
            })
        });
        let hit = hit.flatten();
        let index = hit
            .or_else(|| {
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.owner.is_none())
                    .min_by_key(|(_, slot)| slot.used)
                    .map(|(index, _)| index)
            })
            .ok_or(Refused::Busy)?;
        self.clock = self.clock.checked_add(1).expect("slot clock exhausted");
        let slot = &mut self.slots[index];
        let reused = hit.map_or(0, |_| slot.consumed.len());
        slot.owner = Some(owner);
        slot.used = self.clock;
        // Even a hit is invalid until the new pass succeeds. A partial update
        // followed by an error must never retain the previous prefix claim.
        slot.valid = false;
        Ok(Lease {
            slot,
            capacity: self.capacity,
            reused,
            prepared: reused != 0,
            scope,
        })
    }

    fn evict(&mut self, index: usize) -> bool {
        let slot = &mut self.slots[index];
        if slot.owner.is_some() {
            return false;
        }
        slot.valid = false;
        slot.scope = None;
        slot.consumed.clear();
        true
    }
}

impl<T> Lease<'_, T> {
    /// Misses must be reset or restored before any forward can access state.
    fn prepare(&mut self, fallback: impl FnOnce(&mut T)) {
        if !self.prepared {
            fallback(&mut self.slot.state);
            self.prepared = true;
        }
    }

    fn state(&mut self) -> &mut T {
        assert!(self.prepared, "a missed slot needs restoration or reset");
        &mut self.slot.state
    }

    fn commit(self, consumed: &[i32], position: usize) {
        assert!(self.prepared);
        assert_eq!(
            consumed.len(),
            position,
            "only consumed tokens are resident"
        );
        assert!(position <= self.capacity);
        self.slot.consumed.clear();
        self.slot.consumed.extend_from_slice(consumed);
        self.slot.scope = Some(self.scope);
        self.slot.valid = true;
    }
}

impl<T> Drop for Lease<'_, T> {
    fn drop(&mut self) {
        self.slot.owner = None;
    }
}

const TEST_SCOPE: Scope = Scope {
    model: 1,
    tokenizer: 2,
    rope: 0,
    session: 3,
};

#[test]
#[should_panic(expected = "a missed slot needs restoration or reset")]
fn a_miss_cannot_expose_a_previous_owners_state() {
    let mut slots = ResidentSlots::new(vec![99usize], 8);
    let mut lease = slots.begin(TEST_SCOPE, &[1, 2], 4, 1, true).unwrap();
    let _ = lease.state();
}

#[test]
#[should_panic(expected = "only consumed tokens are resident")]
fn an_emitted_but_unconsumed_token_cannot_enter_the_resident_prefix() {
    let mut slots = ResidentSlots::new(vec![0usize], 8);
    let mut lease = slots.begin(TEST_SCOPE, &[1], 4, 1, true).unwrap();
    lease.prepare(|position| *position = 1);
    lease.commit(&[1, 2], 1);
}

#[test]
fn reuse_requires_the_entire_consumed_prefix_and_execution_scope() {
    let mut slots = ResidentSlots::new(vec![0usize], 16);
    let mut lease = slots.begin(TEST_SCOPE, &[4, 5], 8, 1, true).unwrap();
    lease.prepare(|state| *state = 2);
    lease.commit(&[4, 5], 2);
    let mut lease = slots.begin(TEST_SCOPE, &[4, 5, 6], 8, 2, true).unwrap();
    assert_eq!(lease.reused, 2);
    lease.prepare(|_| panic!("a hit must not restore"));
    assert_eq!(*lease.state(), 2);
    lease.commit(&[4, 5], 2);
    for (scope, prompt) in [
        (TEST_SCOPE, vec![4]),
        (TEST_SCOPE, vec![4, 5]),
        (TEST_SCOPE, vec![4, 9, 6]),
        (
            Scope {
                model: 9,
                ..TEST_SCOPE
            },
            vec![4, 5, 6],
        ),
        (
            Scope {
                tokenizer: 9,
                ..TEST_SCOPE
            },
            vec![4, 5, 6],
        ),
        (
            Scope {
                rope: 9,
                ..TEST_SCOPE
            },
            vec![4, 5, 6],
        ),
        (
            Scope {
                session: 9,
                ..TEST_SCOPE
            },
            vec![4, 5, 6],
        ),
    ] {
        let mut lease = slots.begin(scope, &prompt, 8, 3, true).unwrap();
        assert_eq!(lease.reused, 0);
        lease.prepare(|state| *state = 0);
        assert_eq!(*lease.state(), 0);
        drop(lease);
        // Reinstall the original fixture for the next independent rejection.
        let mut fixture = slots.begin(TEST_SCOPE, &[4, 5], 8, 4, false).unwrap();
        fixture.prepare(|state| *state = 2);
        fixture.commit(&[4, 5], 2);
    }
}

#[test]
fn capacity_ownership_eviction_and_abandoned_passes_preserve_fallback() {
    let mut slots = ResidentSlots::new(vec![7usize, 8], 8);
    let pointers: Vec<_> = slots
        .slots
        .iter()
        .map(|slot| slot.consumed.as_ptr())
        .collect();
    assert!(matches!(
        slots.begin(TEST_SCOPE, &[1], 9, 1, true),
        Err(Refused::Capacity)
    ));
    assert!(matches!(
        slots.begin(TEST_SCOPE, &[1, 2], 1, 1, true),
        Err(Refused::Capacity)
    ));
    slots.slots[0].owner = Some(10);
    assert!(!slots.evict(0));
    let mut lease = slots.begin(TEST_SCOPE, &[1], 2, 2, true).unwrap();
    lease.prepare(|state| *state = 42);
    assert_eq!(*lease.state(), 42);
    lease.commit(&[1], 1);
    slots.slots[1].owner = Some(11);
    assert!(matches!(
        slots.begin(TEST_SCOPE, &[1], 2, 3, true),
        Err(Refused::Busy)
    ));
    slots.slots[0].owner = None;
    slots.slots[1].owner = None;
    {
        let mut lease = slots.begin(TEST_SCOPE, &[1, 2], 3, 4, true).unwrap();
        assert_eq!(lease.reused, 1);
        *lease.state() = 99;
        // An error/early return drops the lease without committing new tokens.
    }
    assert!(slots.slots.iter().all(|slot| !slot.valid));
    let mut lease = slots.begin(TEST_SCOPE, &[1, 2], 3, 5, true).unwrap();
    assert_eq!(lease.reused, 0);
    lease.prepare(|state| *state = 12); // ordinary verified snapshot restoration
    lease.commit(&[1, 2], 2);
    let at = slots.slots.iter().position(|slot| slot.valid).unwrap();
    assert!(slots.evict(at));
    assert_eq!(
        slots
            .begin(TEST_SCOPE, &[1, 2, 3], 4, 6, true)
            .unwrap()
            .reused,
        0
    );
    assert_eq!(
        pointers,
        slots
            .slots
            .iter()
            .map(|slot| slot.consumed.as_ptr())
            .collect::<Vec<_>>()
    );
}

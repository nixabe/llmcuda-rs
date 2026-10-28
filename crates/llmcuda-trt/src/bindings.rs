//! Per-enqueue runtime resources for plugins, through a thread-local.
//!
//! A plan stores plugin geometry only, so a deserialized plugin knows *what*
//! it computes but not *where*: weight pointers, kernel handles and the
//! worker's stream belong to the process that runs it. The caller installs
//! them with [`install`] around `enqueueV3`; a plugin reads them with
//! [`with`]. TensorRT calls plugin `enqueue` synchronously on the enqueueing
//! thread (with no auxiliary streams), so a thread-local reaches exactly the
//! enqueue it was installed for and no other worker's.
//!
//! A plugin that finds nothing installed, or a binding of another type,
//! must fail the launch with an error status — never guess.

use std::any::TypeId;
use std::cell::Cell;
use std::marker::PhantomData;

thread_local! {
    static CURRENT: Cell<Option<(TypeId, *const ())>> = const { Cell::new(None) };
}

/// Restores the previous binding on drop, so installs nest.
#[must_use = "the binding is removed when the guard drops"]
pub struct Guard<'a, T> {
    previous: Option<(TypeId, *const ())>,
    _bound: PhantomData<&'a T>,
    /// Not `Send`: the binding belongs to this thread.
    _thread: PhantomData<*const ()>,
}

/// Make `bindings` visible to plugins enqueued on this thread until the
/// guard drops.
pub fn install<T: 'static>(bindings: &T) -> Guard<'_, T> {
    let previous =
        CURRENT.with(|c| c.replace(Some((TypeId::of::<T>(), bindings as *const T as *const ()))));
    Guard {
        previous,
        _bound: PhantomData,
        _thread: PhantomData,
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.previous));
    }
}

/// Run `f` on the installed bindings of type `T`, or return `None` when
/// none (or another type) is installed.
pub fn with<T: 'static, R>(f: impl FnOnce(&T) -> R) -> Option<R> {
    let (id, ptr) = CURRENT.with(Cell::get)?;
    if id != TypeId::of::<T>() {
        return None;
    }
    // SAFETY: `ptr` came from a `&T` whose guard is still alive on this
    // thread (the guard clears it on drop), and the type was checked.
    Some(f(unsafe { &*(ptr as *const T) }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_are_scoped_typed_and_nested() {
        assert!(with::<u32, _>(|_| ()).is_none(), "nothing installed yet");
        let outer = 7u32;
        let g = install(&outer);
        assert_eq!(with::<u32, _>(|v| *v), Some(7));
        assert!(with::<i64, _>(|_| ()).is_none(), "wrong type is not found");
        {
            let inner = 9u32;
            let _g2 = install(&inner);
            assert_eq!(with::<u32, _>(|v| *v), Some(9));
        }
        assert_eq!(
            with::<u32, _>(|v| *v),
            Some(7),
            "inner guard restored outer"
        );
        drop(g);
        assert!(with::<u32, _>(|_| ()).is_none());
        std::thread::spawn(|| assert!(with::<u32, _>(|_| ()).is_none()))
            .join()
            .unwrap();
    }
}

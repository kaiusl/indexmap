#![allow(unsafe_code)]

use ::core::iter::FusedIterator;
use ::core::ptr::{self, NonNull};
use ::core::{fmt, mem, ops, slice};

use super::indices::UniqueSortedIter;
use super::{equivalent, update_index_last, IndexMultimapCore, IndicesBucket, IndicesTable};
use crate::map::Slice;
use crate::util::{debug_iter_as_list, simplify_range, DebugIterAsNumberedCompactList};
use crate::{Bucket, Equivalent, HashValue};

use super::Indices;

/// An iterator that shift removes pairs from [`IndexMultimap`].
///
/// This struct is created by [`IndexMultimap::shift_remove`] and
/// [`OccupiedEntry::shift_remove`], see their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::shift_remove`]: crate::IndexMultimap::shift_remove
pub struct ShiftRemove<'a, K, V>
where
    // Needed by map.debug_assert_invariants if cfg!(more_debug_assertions) in Drop impl.
    // It's a bit annoying but since one cannot reasonably use a map without
    // `K: Eq`, then I don't mind too much about having this bound here.
    K: Eq,
{
    /* ---
    Inspired by std's `drain_filter` and `retain` implementations for vec.

    Notable differences are:
    * we already have exact indices which need to be removed,
      we don't need to walk every item,
    * and our predicate cannot panic (since there is none),
      this simplifies the impl a bit as we don't need to consider a case
      where predicate panics.
    * our vec cannot contain ZSTs

    We take ownership of map's indices table.
    This ensures that an empty table is left if this struct is leaked
    without dropping. At construction map.pairs length is set to zero.
    These two things together ensure that in the case this struct is leaked
    map itself will be left in a valid and safe (but empty) state.

    Correct state will be restored in `Drop` implementation.
    `Drop` will also remove and drop any remaining items that should be removed
    but haven't yet.

    # Safety

    * indices must be already removed from map.indices
    * indices must be valid to index into map.pairs
    * on construction `map.pairs.len` must be set to 0
    * on construction we must take ownership of `map.indices`
      and leave an empty table in it's place
    --- */
    map: &'a mut IndexMultimapCore<K, V>,
    /// map.indices we took ownership of
    indices_table: IndicesTable,
    /// The index of pair that was removed previously.
    prev_removed_idx: Option<usize>,
    /// The number of items that have been removed thus far.
    del: usize,
    /// The original length of `vec` prior to removing.
    orig_len: usize,
    /// The indices that will be removed.
    indices_to_remove: UniqueSortedIter<alloc::vec::IntoIter<usize>>,
}

impl<'a, K: 'a, V: 'a> ShiftRemove<'a, K, V>
where
    K: Eq,
{
    pub(super) fn new<Q>(
        map: &'a mut IndexMultimapCore<K, V>,
        hash: HashValue,
        key: &Q,
    ) -> Option<Self>
    where
        Q: ?Sized + Equivalent<K>,
        K: Eq,
    {
        let eq = equivalent(key, &map.pairs);
        match map.indices.remove_entry(hash.get(), eq) {
            Some(indices) => Some(unsafe { Self::new_unchecked(map, indices) }),
            None => None,
        }
    }

    /// # Safety
    ///
    /// * `bucket` must be alive and from the `map`
    pub(super) unsafe fn new_raw(
        map: &'a mut IndexMultimapCore<K, V>,
        bucket: IndicesBucket,
    ) -> Self {
        unsafe {
            let (indices, _) = map.indices.remove(bucket);
            Self::new_unchecked(map, indices)
        }
    }

    /// # Safety
    ///
    /// See the comment in the type definition for safety.
    unsafe fn new_unchecked(map: &'a mut IndexMultimapCore<K, V>, indices: Indices) -> Self {
        debug_assert!(indices.is_empty() || *indices.last().unwrap() < map.pairs.len());

        unsafe { map.decrement_indices_batched(&indices) };
        let old_len = map.pairs.len();
        unsafe { map.pairs.set_len(0) };
        let indices_table = mem::take(&mut map.indices);
        Self {
            map,
            indices_table,
            prev_removed_idx: None,
            del: 0,
            orig_len: old_len,
            indices_to_remove: indices.into_iter(),
        }
    }

    /// Shift removes a single element at index `i`.
    ///
    /// Parameters are forwarded from self because in Iterator::collect
    /// we cannot borrow self multiple times so we must
    /// forward all the necessary components.
    ///
    /// `i` must be self.indices_to_remove.next() or equivalent
    #[inline]
    unsafe fn shift_remove_index(
        map: &mut IndexMultimapCore<K, V>,
        prev_removed_idx: &mut Option<usize>,
        del: &mut usize,
        i: usize,
    ) -> (usize, K, V) {
        unsafe {
            let pairs_start = map.pairs.as_mut_ptr();
            // IDEA: We can get rid of this branch. It's not taken only on the first call to next().
            //       If we set prev_idx_to_remove = self.indices.first() - 1 = i - 1 at construction,
            //       then on first call diff = i - (i - 1) - 1 = i - i + 1 - 1 = 0
            //       Only issue is if self.indices.first() == 0, we can do saturating_sub(), at construction and here.
            //       Thus on first call diff == 0 always.
            //       However benchmarks with it seem inconclusive or even slower.
            //       If it really doesn't matter, this one is clearer about what we are doing.
            if let Some(prev_removed_idx) = prev_removed_idx {
                // Cover the empty slots with valid items that are between
                // previously removed index and the one that will be removed now.
                //
                // [head] [del - 1 empty slots] [prev_index] [   diff items    ] [i] ...
                //        ^-dst=src-del                      ^-src=prev_idx+1  ^-src+diff
                //                                           \----------/
                //                                              ^-shift by del to the start of empty slots
                // result:
                // [head] [diff items] [del empty slots] [i] ...
                // \-----------------/
                //   ^- contiguous valid items
                //
                // SAFETY:
                //  * src+diff are all valid items to be read,
                //    we haven't touched them yet so they must be valid
                //  * dst+diff are all valid to be written to,
                //    they may however overlap with src+diff
                //  * after copy the elements at new empty slots will never be read,
                //    only maybe overwritten
                let prev_removed_idx = *prev_removed_idx;
                let diff = i - prev_removed_idx - 1;
                if diff > 0 {
                    let src = pairs_start.add(prev_removed_idx + 1).cast_const();
                    let dst = pairs_start.add(prev_removed_idx + 1 - *del);
                    ptr::copy(src, dst, diff);
                }
            }

            *del += 1;
            // SAFETY:
            // * value at `i` must be valid since we haven't touched it yet
            // * we never read the value at `i` after
            let Bucket { key, value, .. } = ptr::read(pairs_start.add(i));
            *prev_removed_idx = Some(i);

            (i, key, value)
        }
    }
}

impl<K, V> Iterator for ShiftRemove<'_, K, V>
where
    K: Eq,
{
    type Item = (usize, K, V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices_to_remove.next() {
            Some(i) => Some(unsafe {
                Self::shift_remove_index(self.map, &mut self.prev_removed_idx, &mut self.del, i)
            }),
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices_to_remove.size_hint()
    }

    fn collect<B: FromIterator<Self::Item>>(mut self) -> B
    where
        Self: Sized,
    {
        (&mut self.indices_to_remove)
            .map(|i| unsafe {
                Self::shift_remove_index(self.map, &mut self.prev_removed_idx, &mut self.del, i)
            })
            .collect()
    }
}

impl<K, V> ExactSizeIterator for ShiftRemove<'_, K, V>
where
    K: Eq,
{
    fn len(&self) -> usize {
        self.indices_to_remove.len()
    }
}

impl<K, V> FusedIterator for ShiftRemove<'_, K, V> where K: Eq {}

impl<K, V> Drop for ShiftRemove<'_, K, V>
where
    K: Eq,
{
    fn drop(&mut self) {
        struct Guard<'a, 'b, K, V>(&'a mut ShiftRemove<'b, K, V>)
        where
            K: Eq;

        impl<'a, 'b, K, V> Drop for Guard<'a, 'b, K, V>
        where
            K: Eq,
        {
            fn drop(&mut self) {
                let inner = &mut *self.0;
                inner.for_each(drop);

                // Shift back the tail
                // Only way to get here is if we managed to remove and drop
                // all the items we needed items
                let del = inner.del;
                let orig_len = inner.orig_len;
                let map = &mut *inner.map;

                let prev_removed_idx = inner
                    .prev_removed_idx
                    .expect("expected to remove at least one pair");
                let tail_len = orig_len - prev_removed_idx - 1;
                if tail_len > 0 {
                    unsafe {
                        // [head] [del - 1 empty slots] [prev_idx] [    tail items    ] [orig_len]
                        //        ^-dst=src-del                    ^-src=prev_idx+1   ^-src+diff
                        //                                         \------------------/
                        //                                              ^-shift by del to the start of empty slots
                        // result:
                        // [head] [tail items] [del empty slots] [orig_len]
                        //
                        // SAFETY:
                        //  * src+tail_len are all valid items to be read,
                        //    we haven't touched them yet so they must be valid
                        //  * dst+tail_len are all valid to be written to,
                        //    they may however overlap with src+tail_len
                        //  * after copy the elements at new empty slots will never be read,
                        let pairs_start = map.pairs.as_mut_ptr();
                        let src = pairs_start.add(prev_removed_idx + 1).cast_const();
                        let dst = pairs_start.add(prev_removed_idx + 1 - del);
                        ptr::copy(src, dst, tail_len);
                    }
                }

                unsafe { map.pairs.set_len(orig_len - del) }
                mem::swap(&mut inner.indices_table, &mut map.indices);
                map.debug_assert_invariants();
            }
        }

        let guard = Guard(self);
        // Following may panic if K's or V's drop panics.
        // If that happens, keep trying to remove and drop items.
        // If any more panics occur, abort because of double panic.
        guard.0.for_each(drop);
    }
}

impl<'a, K, V> fmt::Debug for ShiftRemove<'a, K, V>
where
    K: fmt::Debug + Eq,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let items_iter = self.indices_to_remove.clone().map(|i| {
            assert!(i < self.orig_len);
            let bucket = unsafe { &*self.map.pairs.as_ptr().add(i) };
            (i, &bucket.key, &bucket.value)
        });
        if cfg!(feature = "test_debug") {
            f.debug_struct("ShiftRemove")
                .field(
                    "prev_removed_idx",
                    &format_args!("{:?}", self.prev_removed_idx),
                )
                .field("del", &self.del)
                .field("orig_len", &self.orig_len)
                .field(
                    "indices_to_remove",
                    &DebugIterAsNumberedCompactList::new(self.indices_to_remove.clone()),
                )
                .field(
                    "items_to_remove",
                    &DebugIterAsNumberedCompactList::new(items_iter),
                )
                .finish()
        } else {
            debug_iter_as_list(f, Some("ShiftRemove"), items_iter)
        }
    }
}

/// An iterator that swap removes pairs from [`IndexMultimap`].
///
/// This struct is created by [`IndexMultimap::swap_remove`] and
/// [`OccupiedEntry::swap_remove`], see their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::swap_remove`]: crate::IndexMultimap::swap_remove
pub struct SwapRemove<'a, K, V>
where
    // Needed by map.debug_assert_invariants if cfg!(more_debug_assertions) in Drop impl.
    // It's a bit annoying but since one cannot reasonably use a map without
    // `K: Eq`, then I don't mind too much about having this bound here.
    K: Eq,
{
    /* ---
    We take ownership of map's indices table.
    This ensures that an empty table is left if this struct is leaked
    without dropping. At construction map.pairs length is set to zero.
    These two things together ensure that in the case this struct is leaked
    map itself will be left in a valid and safe (but empty) state.

    Correct state will be restored in `Drop` implementation.
    `Drop` will also remove and drop any remaining items that should be removed
    but haven't yet.

    Implementation idea here is not to swap with the last element in `map.pairs`
    but with the last one that will be kept in the map.

    # Safety

    * indices must be already removed from map.indices
    * indices must be valid to index into map.pairs
    * indices must be unique and sorted
    * indices must be non-empty
    * on construction map.pairs.len must be set to 0
    * on construction we must take ownership of map.indices
      and leave an empty table in it's place
    --- */
    map: &'a mut IndexMultimapCore<K, V>,
    indices_table: IndicesTable,
    /// The original length of `map.pairs` prior to removing.
    orig_len: usize,
    /// Indices that need to be removed from `map.pairs`.
    indices_to_remove: Indices,
    /// `0..indices.len()`, used to index into `self.indices` to get indices to remove
    iter_forward: ops::Range<usize>,
    /// `0..indices.len() - 1`, used to index into `self.indices` backwards to determine which index to swap with
    iter_backward: ops::Range<usize>,
    /// `0..map.pairs.len()`, used to determine which index to swap with
    total_iter: ops::Range<usize>,
    /// `indices_to_remove[i]` where `i` is the previous index yielded by `self.iter_backward`,
    /// used to determine which index to swap with
    prev_backward: usize,
}

impl<'a, K, V> SwapRemove<'a, K, V>
where
    K: Eq,
{
    pub(super) fn new<Q>(
        map: &'a mut IndexMultimapCore<K, V>,
        hash: HashValue,
        key: &Q,
    ) -> Option<Self>
    where
        Q: ?Sized + Equivalent<K>,
        K: Eq,
    {
        let eq = equivalent(key, &map.pairs);
        match map.indices.remove_entry(hash.get(), eq) {
            Some(indices) => Some(unsafe { Self::new_unchecked(map, indices) }),
            None => None,
        }
    }

    /// # Safety
    ///
    /// * `bucket` must be alive and from the `map`
    pub(super) unsafe fn new_raw(
        map: &'a mut IndexMultimapCore<K, V>,
        bucket: IndicesBucket,
    ) -> Self {
        unsafe {
            let (indices, _) = map.indices.remove(bucket);
            Self::new_unchecked(map, indices)
        }
    }

    /// # Safety
    ///
    /// See the comment in the type definition for safety.
    unsafe fn new_unchecked(map: &'a mut IndexMultimapCore<K, V>, indices: Indices) -> Self {
        debug_assert!(!indices.is_empty());
        debug_assert!(*indices.last().unwrap() < map.pairs.len());

        let orig_len = map.pairs.len();
        unsafe { map.pairs.set_len(0) };
        let indices_table = mem::take(&mut map.indices);
        let indices_len = indices.len();
        let last = *indices.last().unwrap();
        Self {
            map,
            indices_table,
            orig_len,
            indices_to_remove: indices,
            iter_forward: 0..indices_len,
            iter_backward: (0..indices_len - 1),
            total_iter: (0..orig_len),
            prev_backward: last,
        }
    }

    /// Removes the element at map.pairs[index] and swaps it with last element
    /// that will be kept.
    ///
    /// # Safety
    ///
    /// * `index` must be in bounds for map.pairs buffer
    /// * map.pairs[index] must be valid for reads and writes
    #[inline]
    unsafe fn swap_remove_index(&mut self, index: usize) -> (usize, K, V) {
        unsafe {
            // SAFETY:
            // * `index` must be in bounds for map.pairs buffer
            // * map.pairs[index] must be valid for reads and writes
            // * remove will never be read again => we give ownership away
            let ptr = self.map.pairs.as_mut_ptr();
            let remove = ptr.add(index);
            let Bucket { key, value, .. } = ptr::read(remove);

            let idx_to_swap_with = self.index_to_swap_with(index);
            if let Some(idx_to_swap_with) = idx_to_swap_with {
                debug_assert!(index != idx_to_swap_with);
                // SAFETY:
                // * src and dst cannot be equal,
                //   `indices.index_to_swap_with` cannot ever return value equal to i
                // * src will never be read again
                let src = ptr.add(idx_to_swap_with);
                let dst = remove;
                ptr::copy_nonoverlapping(src, dst, 1);

                let hash = (*dst).hash;
                update_index_last(&mut self.indices_table, hash, idx_to_swap_with, index);
            }

            (index, key, value)
        }
    }

    /// Return the next index that will be removed.
    #[inline]
    fn next_idx_to_remove(&mut self) -> Option<usize> {
        self.iter_forward.next().map(|i| self.indices_to_remove[i])
    }

    #[inline]
    fn next_backward_idx(&mut self) -> Option<usize> {
        self.iter_backward
            .next_back()
            .map(|i| self.indices_to_remove[i])
    }

    /// Returns the next index from the back that is not to be removed and is
    /// larger than `current`.
    ///
    /// This is an index that `current` can be swapped with in swap_remove.
    /// Return None if all elements after current are to be removed.
    /// Thus there is no need to swap with anything.

    fn index_to_swap_with(&mut self, current: usize) -> Option<usize> {
        if current >= self.orig_len - self.indices_to_remove.len() {
            // current will never need to be swapped if it's outside of the new
            return None;
        }

        while let Some(i) = self.total_iter.next_back() {
            if i <= current {
                // I think this branch is never actually taken in real use cases.
                // It's because the current >= self.new_len would be triggered first
                // if we keep removing and swapping items in order.
                // But I'm not 100% sure
                // panic!("took this branch");
                return None;
            }

            #[allow(clippy::comparison_chain)]
            if i > self.prev_backward {
                return Some(i);
            } else if i == self.prev_backward {
                self.prev_backward = self.next_backward_idx().unwrap_or(0);
            }
        }
        None
    }
}

impl<K, V> Iterator for SwapRemove<'_, K, V>
where
    K: Eq,
{
    type Item = (usize, K, V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_idx_to_remove() {
            Some(i) => Some(unsafe { self.swap_remove_index(i) }),
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter_forward.size_hint()
    }

    fn collect<B: FromIterator<Self::Item>>(mut self) -> B
    where
        Self: Sized,
    {
        // Nothing else uses self.iter_forward, so we can take it
        let iter = mem::take(&mut self.iter_forward);
        iter.map(|i| unsafe {
            let i = self.indices_to_remove[i];
            self.swap_remove_index(i)
        })
        .collect()
    }
}

impl<K, V> ExactSizeIterator for SwapRemove<'_, K, V>
where
    K: Eq,
{
    fn len(&self) -> usize {
        self.iter_forward.len()
    }
}

impl<K, V> FusedIterator for SwapRemove<'_, K, V> where K: Eq {}

impl<K, V> Drop for SwapRemove<'_, K, V>
where
    K: Eq,
{
    fn drop(&mut self) {
        struct Guard<'a, 'b, K, V>(&'a mut SwapRemove<'b, K, V>)
        where
            K: Eq;

        impl<'a, 'b, K, V> Drop for Guard<'a, 'b, K, V>
        where
            K: Eq,
        {
            fn drop(&mut self) {
                let inner = &mut *self.0;
                inner.for_each(drop);
                // Only way to get here is if we managed to drop all the items we needed to remove
                let map = &mut *inner.map;
                unsafe {
                    map.pairs
                        .set_len(inner.orig_len - inner.indices_to_remove.len())
                }
                mem::swap(&mut inner.indices_table, &mut map.indices);
                map.debug_assert_invariants();
            }
        }

        let guard = Guard(self);
        // Following may panic if K's or V's drop panics.
        // Guard will try to keep removing and dropping items.
        // If any more panic, we abort because of double panic.
        guard.0.for_each(drop);
    }
}

impl<'a, K, V> fmt::Debug for SwapRemove<'a, K, V>
where
    K: fmt::Debug + Eq,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let items_iter = self.iter_forward.clone().map(|i| {
            let i = self.indices_to_remove[i];
            assert!(i < self.orig_len);
            let bucket = unsafe { &*self.map.pairs.as_ptr().add(i) };
            (i, &bucket.key, &bucket.value)
        });
        if cfg!(feature = "test_debug") {
            f.debug_struct("SwapRemove")
                .field("orig_len", &self.orig_len)
                .field("iter_forward", &self.iter_forward)
                .field("iter_backward", &self.iter_backward)
                .field("total_iter", &self.total_iter)
                .field("prev_backward", &self.prev_backward)
                .field(
                    "indices_to_remove",
                    &DebugIterAsNumberedCompactList::new(self.indices_to_remove.as_slice().iter()),
                )
                .field(
                    "items_to_remove",
                    &DebugIterAsNumberedCompactList::new(items_iter),
                )
                .finish()
        } else {
            debug_iter_as_list(f, Some("SwapRemove"), items_iter)
        }
    }
}

/// A draining iterator over the pairs of a [`IndexMultimap`].
///
/// This `struct` is created by the [`drain`] method on [`IndexMultimap`].
/// See its documentation for more.
///
/// [`drain`]: crate::IndexMultimap::drain
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct Drain<'a, K, V>
where
    K: Eq,
{
    /* ---
    Effectively a copy of `std::vec::Drain` implementation (as of Rust 1.68),
    with small modifications.

    We must take ownership of map.indices for the duration of this struct's life.
    This is to leave the map in consistent and valid (but empty) state in the
    case this struct is leaked. For that reason on construction map.pairs.len
    must be set to 0.

    Layout of map.pairs:
    [head]        [start] ... [end]         [tail_start] [tail_len - 1 items]
    ^-don't touch \-- to_remove --/         \-----------  tail  ------------/
                    ^-items to remove/drain   ^- shift left to cover removed items

    Result after drop:
    [head] [tail], new length of vec = start + tail_len
    --- */
    /// Pointer to map.
    map: NonNull<IndexMultimapCore<K, V>>,
    /// map.indices
    indices_table: IndicesTable,
    // Index of first item that's drained
    start: usize,
    /// Index of tail to preserve
    tail_start: usize,
    /// Length of tail
    tail_len: usize,
    /// Current remaining range to remove
    to_remove: slice::Iter<'a, Bucket<K, V>>,
}

// &self can only read, there is no interior mutability
unsafe impl<K, V> Sync for Drain<'_, K, V>
where
    K: Sync + Eq,
    V: Sync,
{
}
unsafe impl<K, V> Send for Drain<'_, K, V>
where
    K: Send + Eq,
    V: Send,
{
}

impl<'a, K, V> Drain<'a, K, V>
where
    K: Eq,
{
    pub(super) fn new<R>(map: &'a mut IndexMultimapCore<K, V>, range: R) -> Self
    where
        R: ops::RangeBounds<usize>,
        K: Eq,
    {
        let range = simplify_range(range, map.pairs.len());
        map.erase_indices(range.start, range.end);

        let indices_table = mem::take(&mut map.indices);
        let len = map.pairs.len();
        let ops::Range { start, end } = range;

        // SAFETY: simplify_range panics if given range is invalid for self.pairs
        unsafe {
            map.pairs.set_len(0);
            // Convert to pointer early
            let map = NonNull::from(map);
            // Go through raw pointer as long as possible
            let pairs = ptr::addr_of!((*map.as_ptr()).pairs);
            // SAFETY:
            //   range_slice will be invalidated if map.pairs' buffer is reallocated
            //   or we create a &mut to at least one element of that slice
            //   or we write to an element in that slice
            //     (like `*addr_of_mut!((*map.as_ptr()).pairs).add(start) = something`)
            // We never do anything from above while we need range_slice/to_remove
            let range_slice = slice::from_raw_parts((*pairs).as_ptr().add(start), end - start);
            Self {
                map,
                indices_table,
                start,
                tail_start: end,
                tail_len: len - end,
                to_remove: range_slice.iter(),
            }
        }
    }

    /// Returns the remaining items of this iterator as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &Slice<K, V> {
        Slice::from_slice(self.to_remove.as_slice())
    }
}

impl<K, V> fmt::Debug for Drain<'_, K, V>
where
    K: fmt::Debug + Eq,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if cfg!(feature = "test_debug") {
            f.debug_struct("Drain")
                .field("start", &self.start)
                .field("tail_start", &self.tail_start)
                .field("tail_len", &self.tail_len)
                .field(
                    "to_remove",
                    &DebugIterAsNumberedCompactList::new(self.as_slice().iter()),
                )
                .finish()
        } else {
            debug_iter_as_list(f, Some("Drain"), self.as_slice().iter())
        }
    }
}

impl<K, V> Iterator for Drain<'_, K, V>
where
    K: Eq,
{
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        self.to_remove
            .next()
            // SAFETY: elt is valid, aligned, initialized and we never use the read value again
            .map(|elt| unsafe { ptr::read(elt as *const Bucket<K, V>) }.key_value())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.to_remove.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for Drain<'_, K, V>
where
    K: Eq,
{
    #[inline]
    fn next_back(&mut self) -> Option<(K, V)> {
        self.to_remove
            .next_back()
            // SAFETY: elt is valid, initialized, aligned and we never use the read value again
            .map(|elt| unsafe { ptr::read(elt as *const Bucket<K, V>) }.key_value())
    }
}

impl<K, V> ExactSizeIterator for Drain<'_, K, V>
where
    K: Eq,
{
    fn len(&self) -> usize {
        self.to_remove.len()
    }
}

impl<K, V> FusedIterator for Drain<'_, K, V> where K: Eq {}

impl<K, V> Drop for Drain<'_, K, V>
where
    K: Eq,
{
    fn drop(&mut self) {
        /// Moves back the un-`Drain`ed elements to restore the original `Vec`.
        /// Swaps back the indices that we took from the map at construction.
        struct DropGuard<'r, 'a, K, V>
        where
            K: Eq,
        {
            drain: &'r mut Drain<'a, K, V>,
        }

        impl<'r, 'a, K, V> Drop for DropGuard<'r, 'a, K, V>
        where
            K: Eq,
        {
            fn drop(&mut self) {
                unsafe {
                    // Only way to get here is if we actually have drained (removed)
                    // and dropped the given range.
                    let drain = &mut *self.drain;
                    let map = drain.map.as_ptr();
                    let pairs = ptr::addr_of_mut!((*map).pairs);
                    if drain.tail_len > 0 {
                        // memmove back untouched tail, update to new length
                        let start = drain.start;
                        let tail = drain.tail_start;
                        if tail != start {
                            // [head] [drained items] [tail]
                            // ^- ptr ^- start        ^- tail
                            let ptr = (*pairs).as_mut_ptr();
                            let src = ptr.add(tail).cast_const();
                            let dst = ptr.add(start);
                            ptr::copy(src, dst, drain.tail_len);
                        }
                    }
                    (*pairs).set_len(drain.start + drain.tail_len);
                    ptr::swap(ptr::addr_of_mut!((*map).indices), &mut drain.indices_table);
                    (*map).debug_assert_invariants();
                }
            }
        }

        let to_drop = mem::replace(&mut self.to_remove, [].iter());
        let drop_len = to_drop.len();

        // ensure elements are moved back into their appropriate places, even when drop_in_place panics
        let guard = DropGuard { drain: self };

        if drop_len == 0 {
            return;
        }

        // as_slice() must only be called when iter.len() is > 0 because it also
        // gets touched by vec::Splice which may turn it into a dangling pointer
        // which would make it and the vec pointer point to different allocations
        // which would lead to invalid pointer arithmetic below.
        // (Not important in our case at the moment, but I'll leave the comment
        // here in case we decide to implement Splice for our map)
        let drop_ptr = to_drop.as_slice().as_ptr();

        unsafe {
            // slice::Iter can only gives us a &[T] but for drop_in_place
            // a pointer with mutable provenance is necessary. Therefore we must reconstruct
            // it from the original vec but also avoid creating a &mut to the front since that could
            // invalidate raw pointers to it which some unsafe code might rely on.
            //
            // [head]    [drained items] [undrained items] [drained items] [tail]
            // ^-vec_ptr ^- self.start   ^- drop_ptr                       ^- tail_start
            //                           \--- to_drop ---/

            // Go through a raw pointer as long as possible
            let pairs = ptr::addr_of_mut!((*guard.drain.map.as_ptr()).pairs);
            let pairs_start = (*pairs).as_mut_ptr();
            let drop_offset = drop_ptr.offset_from(pairs_start);
            // drop_ptr points into pairs, it must be 'greater than' or 'equal to' vec_ptr
            drop(to_drop); // Next line invalidates iter, make it explicit, that it cannot be used anymore
            let to_drop = ptr::slice_from_raw_parts_mut(pairs_start.offset(drop_offset), drop_len);
            ptr::drop_in_place(to_drop);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn swap_remove_index_to_swap_with() {
        // Removes all key=1, expected swap positions are all where key!=1
        fn test(insert: &[i32], current: usize, expected: &[usize]) {
            let mut map = IndexMultimapCore::<i32, i32>::new();
            for &k in insert {
                map.insert_append_full(HashValue(k as usize), k, 0);
            }

            let mut r = map.swap_remove(HashValue(1), &1).unwrap();

            // Clone internals so we can restore the original state and allow r to be dropped properly.
            let iter_forward = r.iter_forward.clone();
            let iter_backward = r.iter_backward.clone();
            let total_iter = r.total_iter.clone();
            let prev_backward = r.prev_backward;

            let mut swaps = Vec::new();
            while let Some(i) = r.index_to_swap_with(current) {
                swaps.push(i);
            }

            assert_eq!(&swaps, &expected);

            r.iter_forward = iter_forward;
            r.iter_backward = iter_backward;
            r.total_iter = total_iter;
            r.prev_backward = prev_backward;
        }

        let insert = [1, 1, 2, 3, 1, 1, 5, 4, 1];
        test(&insert, 0, &[7, 6, 3, 2]);
        test(&insert, 1, &[7, 6, 3, 2]);
        test(&insert, 4, &[]);

        let insert = [1, 1, 2, 3, 1, 1, 5, 4];
        test(&insert, 0, &[7, 6, 3, 2]);
        test(&insert, 1, &[7, 6, 3, 2]);
        test(&insert, 4, &[]);

        let insert = [1, 1, 2, 3, 5, 4];
        test(&insert, 0, &[5, 4, 3, 2]);

        let insert = [1, 1, 1, 1, 1, 1, 1, 1];
        test(&insert, 0, &[]);

        let insert = [1, 1, 2, 3, 1, 1, 5, 4, 1, 1, 1, 1, 1];
        test(&insert, 0, &[7, 6, 3, 2]);
        test(&insert, 1, &[7, 6, 3, 2]);
        test(&insert, 4, &[]);

        let insert = [2, 3, 5, 1, 1, 2, 3, 1, 1, 5, 4, 1];
        test(&insert, 3, &[10, 9, 6, 5]);
        test(&insert, 4, &[10, 9, 6, 5]);
        test(&insert, 7, &[]);

        let insert = [2, 2, 2, 3, 5, 4, 1];
        test(&insert, 6, &[]);
    }
}

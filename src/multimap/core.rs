//! This is the core implementation that doesn't depend on the hasher at all.
//!
//! The methods of `IndexMapCore` don't use any Hash properties of K.
//!
//! It's cleaner to separate them out, then the compiler checks that we are not
//! using Hash at all in these methods.
//!
//! However, we should probably not let this show in the public API or docs.

mod raw;

use alloc::format;
use hashbrown::raw::RawTable;

use crate::vec::{Drain, Vec};
use crate::TryReserveError;
use core::fmt;
use core::ops;

pub use self::raw::OccupiedEntry;
use crate::equivalent::Equivalent;
use crate::util::{is_sorted, is_unique, replace_sorted, simplify_range};
use crate::{Bucket, HashValue};

use super::{Subset, SubsetMut};

/// Core of the map that does not depend on S
///
pub(crate) struct IndexMultimapCore<K, V, Indices> {
    // ---
    // IMPL DETAILS:
    //
    // The implementation assumes that the indices in the `self.indices` are unique
    // and sorted after every complete operation.
    // ---
    /// indices mapping from the entry hash to its index.
    indices: RawTable<Indices>,
    /// pairs is a dense vec of key-value pairs in their order.
    pairs: Vec<Bucket<K, V>>,
}

#[inline(always)]
fn get_hash<K, V, Indices>(pairs: &[Bucket<K, V>]) -> impl Fn(&Indices) -> u64 + '_
where
    Indices: IndexStorage,
{
    // All pairs at `indices` must have equivalent keys,
    // just take the key from the first
    move |indices| pairs[indices[0]].hash.get()
}

#[inline]
fn equivalent<'a, K, V, Q, Indices>(
    key: &'a Q,
    pairs: &'a [Bucket<K, V>],
) -> impl Fn(&Indices) -> bool + 'a
where
    Q: ?Sized + Equivalent<K>,
    Indices: IndexStorage,
{
    // All pairs at `indices` must have equivalent keys,
    // just take the key from the first
    move |indices| Q::equivalent(key, &pairs[indices[0]].key)
}

#[inline]
fn eq_index<Indices>(index: usize) -> impl Fn(&Indices) -> bool
where
    Indices: IndexStorage,
{
    move |indices| {
        debug_assert!(is_sorted(indices), "expected indices to be sorted");
        indices.binary_search(&index).is_ok()
    }
}

#[inline]
fn update_index<Indices>(table: &mut RawTable<Indices>, hash: HashValue, old: usize, new: usize)
where
    Indices: IndexStorage,
{
    // Index to `old` in the indices
    let mut olds_index: usize = 0;
    let indices = table
        .get_mut(hash.get(), |indices| {
            debug_assert!(is_sorted(indices), "expected indices to be sorted");
            match indices.binary_search(&old) {
                Ok(i) => {
                    olds_index = i;
                    true
                }
                Err(_) => false,
            }
        })
        .expect("index not found")
        .as_mut_slice();

    if indices.len() == 1 {
        indices[olds_index] = new;
    } else {
        replace_sorted(indices, olds_index, new);
        debug_assert!(is_sorted(indices));
        debug_assert!(is_unique(indices));
    }
}

impl<K, V, Indices> IndexMultimapCore<K, V, Indices>
where
    Indices: IndexStorage,
{
    #[inline(always)]
    #[cfg(any(not(debug_assertions), not(feature = "more_debug_assertions")))]
    fn debug_assert_invariants(&self) {}

    #[allow(unsafe_code)]
    #[cfg(all(debug_assertions, feature = "std", feature = "more_debug_assertions"))]
    #[track_caller]
    fn debug_assert_invariants(&self) {
        let mut index_count = 0;
        let index_iter = unsafe { self.indices.iter().map(|indices| indices.as_ref()) };
        let mut used = crate::IndexSet::with_capacity(index_iter.len());
        for indices in index_iter {
            index_count += indices.len();
            assert!(is_sorted(indices), "found unsorted indices");
            assert!(
                indices.last().unwrap_or(&0) < &self.pairs.len(),
                "found out of bound index for entries in indices"
            );
            used.reserve(indices.len());

            for i in indices.as_slice() {
                assert!(used.insert(i), "found duplicate index in `indices`")
            }
        }

        assert_eq!(
            self.pairs.len(),
            index_count,
            "mismatch between entries and indices count"
        );
    }

    #[allow(unsafe_code)]
    #[cfg(all(
        debug_assertions,
        not(feature = "std"),
        feature = "more_debug_assertions"
    ))]
    #[track_caller]
    fn debug_assert_invariants(&self) {
        let mut index_count = 0;
        let index_iter = unsafe { self.indices.iter().map(|indices| indices.as_ref()) };
        let mut used = ::alloc::collections::BTreeSet::new();
        for indices in index_iter {
            index_count += indices.len();
            assert!(is_sorted(indices), "found unsorted indices");
            assert!(
                indices.last().unwrap_or(&0) < &self.entries.len(),
                "found out of bound index for entries in indices"
            );
            let needed_capacity = used.len() + indices.len();

            for i in indices.as_slice() {
                assert!(used.insert(i), "found duplicate index in `indices`")
            }
        }

        assert_eq!(
            self.entries.len(),
            index_count,
            "mismatch between entries and indices count"
        );
    }
}

impl<K, V, Indices> IndexMultimapCore<K, V, Indices> {
    /// The maximum capacity before the `entries` allocation would exceed `isize::MAX`.
    //const MAX_ENTRIES_CAPACITY: usize = (isize::MAX as usize) / mem::size_of::<Bucket<K, V>>();

    #[inline]
    pub(crate) const fn new() -> Self {
        IndexMultimapCore {
            indices: RawTable::new(),
            pairs: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn with_capacity(keys: usize, pairs: usize) -> Self {
        IndexMultimapCore {
            indices: RawTable::with_capacity(keys),
            pairs: Vec::with_capacity(pairs),
        }
    }

    #[inline]
    pub(crate) fn len_keys(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub(crate) fn len_pairs(&self) -> usize {
        self.pairs.len()
    }

    #[inline]
    pub(crate) fn capacity_keys(&self) -> usize {
        self.indices.capacity()
    }

    #[inline]
    pub(crate) fn capacity_pairs(&self) -> usize {
        self.pairs.capacity()
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.indices.clear();
        self.pairs.clear();
    }

    #[inline]
    pub(super) fn into_pairs(self) -> Vec<Bucket<K, V>> {
        self.pairs
    }

    #[inline]
    pub(super) fn as_pairs(&self) -> &[Bucket<K, V>] {
        &self.pairs
    }

    #[inline]
    pub(super) fn as_pairs_mut(&mut self) -> &mut [Bucket<K, V>] {
        &mut self.pairs
    }
}
impl<K, V, Indices> IndexMultimapCore<K, V, Indices>
where
    Indices: IndexStorage,
{
    pub(super) fn with_pairs<F>(&mut self, f: F)
    where
        F: FnOnce(&mut [Bucket<K, V>]),
        K: Eq,
    {
        f(&mut self.pairs);
        self.rebuild_hash_table();
        self.debug_assert_invariants();
    }

    pub(crate) fn truncate(&mut self, len: usize)
    where
        K: Eq,
    {
        if len < self.len_pairs() {
            self.erase_indices(len, self.pairs.len());
            self.pairs.truncate(len);
        }
    }

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    pub(crate) fn par_drain<R>(&mut self, range: R) -> rayon::vec::Drain<'_, Bucket<K, V>>
    where
        K: Send + Eq,
        V: Send,
        R: ops::RangeBounds<usize>,
    {
        use rayon::iter::ParallelDrainRange;
        let range = simplify_range(range, self.entries.len());
        self.erase_indices(range.start, range.end);
        self.entries.par_drain(range)
    }

    pub(crate) fn split_off(&mut self, at: usize) -> Self
    where
        K: Eq,
    {
        assert!(at <= self.pairs.len());
        let unique_keys = self.len_keys();

        self.erase_indices(at, self.pairs.len());
        let pairs = self.pairs.split_off(at);

        // TODO: We are most likely over allocating here.
        // Options:
        //    * keep as is and potentially over allocate
        //    * count unique keys in entries and potentially perform a rather expensive calculation
        //    * use RawTable::new and potentially allocate multiple times
        //
        // There cannot be more unique keys that was in the original map or there are new entries.
        let capacity = Ord::min(unique_keys, pairs.len());
        let mut indices = RawTable::with_capacity(capacity);
        raw::insert_bulk_no_grow(&mut indices, &[], &pairs);
        let new = Self { indices, pairs };

        if cfg!(debug_assertions) {
            self.debug_assert_invariants();
            new.debug_assert_invariants();
        }

        new
    }

    /// Reserve capacity for `additional` more key-value pairs.
    pub(crate) fn reserve(&mut self, additional_keys: usize, additional_pairs: usize) {
        self.indices.reserve(additional_keys, get_hash(&self.pairs));
        self.pairs.reserve(additional_pairs);
    }

    /// Reserve capacity for `additional` more key-value pairs, without over-allocating.
    pub(crate) fn reserve_exact(&mut self, additional_keys: usize, additional_pairs: usize) {
        self.indices.reserve(additional_keys, get_hash(&self.pairs));
        self.pairs.reserve_exact(additional_pairs);
    }

    /// Try to reserve capacity for `additional` more key-value pairs.
    pub(crate) fn try_reserve(
        &mut self,
        additional_keys: usize,
        additional_pairs: usize,
    ) -> Result<(), TryReserveError> {
        self.indices
            .try_reserve(additional_keys, get_hash(&self.pairs))
            .map_err(TryReserveError::from_hashbrown)?;
        self.pairs
            .try_reserve(additional_pairs)
            .map_err(TryReserveError::from_alloc)
    }

    /// Try to reserve capacity for `additional` more key-value pairs, without over-allocating.
    pub(crate) fn try_reserve_exact(
        &mut self,
        additional_keys: usize,
        additional_pairs: usize,
    ) -> Result<(), TryReserveError> {
        self.indices
            .try_reserve(additional_keys, get_hash(&self.pairs))
            .map_err(TryReserveError::from_hashbrown)?;
        self.pairs
            .try_reserve_exact(additional_pairs)
            .map_err(TryReserveError::from_alloc)
    }

    /// Shrink the capacity of the map with a lower bound
    pub(crate) fn shrink_to(&mut self, min_capacity: usize) {
        self.indices.shrink_to(min_capacity, get_hash(&self.pairs));
        self.pairs.shrink_to(min_capacity);
    }

    /// Remove the last key-value pair
    pub(crate) fn pop(&mut self) -> Option<(K, V)> {
        if let Some(entry) = self.pairs.pop() {
            let last_index = self.pairs.len();
            raw::erase_index(&mut self.indices, entry.hash, last_index);
            if cfg!(debug_assertions) {
                self.debug_assert_invariants();
            }
            Some((entry.key, entry.value))
        } else {
            None
        }
    }

    /// Append a key-value pair, *without* checking whether it already exists,
    /// and return the pair's new index.
    fn push(
        &mut self,
        hash: HashValue,
        key: K,
        value: V,
    ) -> (usize, hashbrown::raw::Bucket<Indices>) {
        let i = self.pairs.len();
        let bucket = self
            .indices
            .insert(hash.get(), Indices::one(i), get_hash(&self.pairs));
        self.pairs.push(Bucket { hash, key, value });
        self.debug_assert_invariants();
        (i, bucket)
    }

    /// Return the index in `entries` where an equivalent key can be found
    pub(crate) fn get_indices_of<Q>(&self, hash: HashValue, key: &Q) -> &[usize]
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        self.indices
            .get(hash.get(), eq)
            .map(IndexStorage::as_slice)
            .unwrap_or_default()
    }

    pub(crate) fn get<Q>(&self, hash: HashValue, key: &Q) -> Subset<'_, K, V, &'_ [usize]>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        if let Some(indices) = self.indices.get(hash.get(), eq) {
            Subset::new(&self.pairs, indices.as_slice())
        } else {
            Subset::new(&self.pairs, [].as_slice())
        }
    }

    /// Appends a new entry to the existing key, or insert the key.
    pub(crate) fn insert_append_full(&mut self, hash: HashValue, key: K, value: V) -> usize
    where
        K: Eq,
    {
        let i = self.pairs.len();
        let eq = equivalent(&key, &self.pairs);
        match self.indices.get_mut(hash.get(), eq) {
            Some(indices) => {
                indices.push(i);
            }
            None => {
                self.indices
                    .insert(hash.get(), Indices::one(i), get_hash(&self.pairs));
            }
        }
        self.pairs.push(Bucket { hash, key, value });
        self.debug_assert_invariants();
        i
    }

    /// Remove an entry by shifting all entries that follow it.
    ///
    /// Returns `true` is the entry was removed, `false` if not.
    pub(crate) fn shift_remove<Q>(&mut self, hash: HashValue, key: &Q) -> bool
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        match self.indices.remove_entry(hash.get(), eq) {
            Some(indices) => {
                self.shift_remove_finish_wo_collect(&indices);
                self.debug_assert_invariants();
                true
            }
            None => false,
        }
    }

    /// Remove an entry by shifting all entries that follow it
    pub(crate) fn shift_remove_full<Q>(
        &mut self,
        hash: HashValue,
        key: &Q,
    ) -> Option<(Indices, Vec<(K, V)>)>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        match self.indices.remove_entry(hash.get(), eq) {
            Some(indices) => {
                let pairs = self.shift_remove_finish_collect(indices.as_slice());
                self.debug_assert_invariants();
                Some((indices, pairs))
            }
            None => None,
        }
    }

    /// Remove an entry by shifting all entries that follow it
    pub(crate) fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        match self.pairs.get(index) {
            Some(pair) => {
                raw::erase_index(&mut self.indices, pair.hash, index);
                let entry = self.shift_remove_finish(index);
                self.debug_assert_invariants();
                Some(entry)
            }
            None => None,
        }
    }

    /// Remove an entry by shifting all entries that follow it
    ///
    /// The index should already be removed from `self.indices`.
    fn shift_remove_finish(&mut self, index: usize) -> (K, V) {
        // We need to go over the keys in decreasing order such that lower indices are still valid
        // There is probably a more efficient way of doing this:
        // * maybe use retain on entries and keep track how many items were removed before given item
        self.decrement_indices(index + 1, self.pairs.len(), 1);
        // Use Vec::remove to actually remove the entry.
        let Bucket { key, value, .. } = self.pairs.remove(index);
        (key, value)
    }

    /// Remove all the entries at given indices by shifting all entries that follow them
    ///
    /// The index should already be removed from `self.indices`.
    fn shift_remove_finish_collect(&mut self, indices: &[usize]) -> Vec<(K, V)> {
        debug_assert!(is_sorted(indices));
        debug_assert!(is_unique(indices));

        self.decrement_indices_batched(indices);

        let mut removed = Vec::with_capacity(indices.len());
        // TODO: Use drain_filter if it becomes stable
        for (shift, index) in indices.iter().enumerate() {
            // Since we iterate from front to back, every following element is shifted
            let Bucket { key, value, .. } = self.pairs.remove(index - shift);
            removed.push((key, value))
        }

        removed
    }

    /// Remove all the entries at given indices by shifting all entries that follow them
    ///
    /// The index should already be removed from `self.indices`
    /// but all the entries should be still in `self.entries`
    fn shift_remove_finish_wo_collect(&mut self, indices: &[usize]) {
        debug_assert!(is_sorted(indices));
        debug_assert!(is_unique(indices));

        self.decrement_indices_batched(indices);

        let mut indices = indices.iter();
        let mut next_remove_index = *indices.next().unwrap();
        let mut current_iter_index = 0;
        self.pairs.retain(|_| {
            let retain = if next_remove_index == current_iter_index {
                if let Some(&i) = indices.next() {
                    next_remove_index = i;
                };
                false
            } else {
                true
            };

            current_iter_index += 1;
            retain
        });
    }

    /// Decrements indexes by variable amount determined by how many ranges have been before it,
    /// (eg how many items were removed before it).
    ///
    /// * Indices need to be sorted in increasing order and be unique.
    ///
    /// Say we need to remove indices [2, 4, 7]
    /// then
    ///     indices [0, 1], don't get decremented
    ///     indices [3] get decremented by 1
    ///     indices [5, 6], get decremented by 2
    ///     indices [8, ...] get decremented by 3
    fn decrement_indices_batched(&mut self, indices: &[usize]) {
        for (offset, w) in (1..).zip(indices.windows(2)) {
            let (start, end) = (w[0], w[1]);
            debug_assert!(start < end);
            self.decrement_indices(start + 1, end, offset);
        }

        // Shift the tail after the last item
        let last = *indices.last().unwrap();
        if last < self.pairs.len() {
            self.decrement_indices(last + 1, self.pairs.len(), indices.len());
        }
    }

    /// Decrement all indices in the range `start..end` by `amount`.
    ///
    /// The index `start - amount` should not exist in `self.indices`.
    /// All entries should still be in their original positions.
    fn decrement_indices(&mut self, start: usize, end: usize, amount: usize) {
        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        let shifted_pairs = &self.pairs[start..end];
        if shifted_pairs.len() > self.indices.buckets() / 2 {
            // Shift all indices in range.
            for indices in self.indices_mut() {
                debug_assert!(is_sorted(indices));
                for i in indices.as_mut_slice() {
                    if *i >= end {
                        // early break as we go past end and our indices are sorted
                        break;
                    } else if start <= *i {
                        *i -= amount;
                    }
                }
            }
        } else {
            // Find each entry in range to shift its index.
            for (i, entry) in (start..end).zip(shifted_pairs) {
                update_index(&mut self.indices, entry.hash, i, i - amount);
            }
        }
    }

    /// Increment all indices in the range `start..end`.
    ///
    /// The index `end` should not exist in `self.indices`.
    /// All entries should still be in their original positions.
    fn increment_indices(&mut self, start: usize, end: usize) {
        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        let shifted_pairs = &self.pairs[start..end];
        if shifted_pairs.len() > self.indices.buckets() / 2 {
            // Shift all indices in range.
            for indices in self.indices_mut() {
                for i in indices.as_mut_slice() {
                    if start <= *i && *i < end {
                        *i += 1;
                    }
                }
            }
        } else {
            // Find each entry in range to shift its index, updated in reverse so
            // we never have duplicated indices that might have a hash collision.
            for (i, entry) in (start..end).zip(shifted_pairs).rev() {
                update_index(&mut self.indices, entry.hash, i, i + 1);
            }
        }
    }

    pub(super) fn move_index(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }

        let from_hash = self.pairs[from].hash;
        // Use a sentinel index so other indices don't collide.
        update_index(&mut self.indices, from_hash, from, usize::MAX);

        // Update all other indices and rotate the entry positions.
        #[allow(clippy::comparison_chain)]
        if from < to {
            self.decrement_indices(from + 1, to + 1, 1);
            self.pairs[from..=to].rotate_left(1);
        } else if to < from {
            self.increment_indices(to, from);
            self.pairs[to..=from].rotate_right(1);
        }

        // Change the sentinal index to its final position.
        update_index(&mut self.indices, from_hash, usize::MAX, to);
        self.debug_assert_invariants();
    }

    /// Remove an entry by swapping it with the last
    ///
    /// Returns `true` is the entry was removed, `false` if not.
    pub(crate) fn swap_remove<Q>(&mut self, hash: HashValue, key: &Q) -> bool
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        match self.indices.remove_entry(hash.get(), eq) {
            Some(index) => {
                self.swap_remove_finish_wo_collect(index.as_slice());
                self.debug_assert_invariants();
                true
            }
            None => false,
        }
    }

    /// Remove an entry by swapping it with the last
    pub(crate) fn swap_remove_full<Q>(
        &mut self,
        hash: HashValue,
        key: &Q,
    ) -> Option<(Indices, Vec<(K, V)>)>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        match self.indices.remove_entry(hash.get(), eq) {
            Some(indices) => {
                let removed = self.swap_remove_finish_collect_ordered(indices.as_slice());
                self.debug_assert_invariants();
                Some((indices, removed))
            }
            None => None,
        }
    }

    /// Remove an entry by swapping it with the last
    pub(crate) fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        match self.pairs.get(index) {
            Some(entry) => {
                raw::erase_index(&mut self.indices, entry.hash, index);
                let removed = self.swap_remove_finish(index);
                self.debug_assert_invariants();
                Some(removed)
            }
            None => None,
        }
    }

    /// Finish removing an entry by swapping it with the last
    ///
    /// The index should already be removed from `self.indices`.
    fn swap_remove_finish(&mut self, index: usize) -> (K, V) {
        // use swap_remove, but then we need to update the index that points
        // to the other entry that has to move
        let removed_pair = self.pairs.swap_remove(index);

        // correct index that points to the entry that had to swap places
        if let Some(moved_pair) = self.pairs.get(index) {
            // was not last element
            // examine new element in `index` and find it in indices
            let last = self.pairs.len();
            update_index(&mut self.indices, moved_pair.hash, last, index);
        }

        (removed_pair.key, removed_pair.value)
    }

    /// Finish removing an entries by swapping them with the last
    ///
    /// The indices should already be removed from `self.indices`.
    fn swap_remove_finish_collect_ordered(&mut self, indices: &[usize]) -> Vec<(K, V)> {
        // use swap_remove, but then we need to update the index that points
        // to the other entry that has to move

        // TODO: Can we do without reversing?
        let mut removed = Vec::with_capacity(indices.len());
        for &index in indices.iter().rev() {
            let Bucket { key, value, .. } = self.pairs.swap_remove(index);
            // correct index that points to the entry that had to swap places
            if let Some(moved_pair) = self.pairs.get(index) {
                // was not last element
                // examine new element in `index` and find it in indices
                let last = self.pairs.len();
                update_index(&mut self.indices, moved_pair.hash, last, index)
            }
            removed.push((key, value));
        }
        removed.reverse();
        removed
    }

    /// Finish removing an entry by swapping it with the last
    ///
    /// The index should already be removed from `self.indices`.
    fn swap_remove_finish_wo_collect(&mut self, indices: &[usize]) {
        // use swap_remove, but then we need to update the index that points
        // to the other entry that has to move
        for &index in indices.iter().rev() {
            let _removed_pair = self.pairs.swap_remove(index);

            // correct index that points to the entry that had to swap places
            if let Some(moved_pair) = self.pairs.get(index) {
                // was not last element
                // examine new element in `index` and find it in indices
                let last = self.pairs.len();
                update_index(&mut self.indices, moved_pair.hash, last, index);
            }
        }
    }

    /// Erase `start..end` from `indices`, and shift `end..` indices down to `start..`
    ///
    /// All of these items should still be at their original location in `entries`.
    /// This is used by `drain`, which will let `Vec::drain` do the work on `entries`.
    fn erase_indices(&mut self, start: usize, end: usize)
    where
        K: Eq,
    {
        let (init, shifted_pairs) = self.pairs.split_at(end);
        let (start_pairs, erased_pairs) = init.split_at(start);

        let erased = erased_pairs.len();
        let shifted = shifted_pairs.len();
        let half_capacity = self.indices.buckets() / 2;

        // Use a heuristic between different strategies
        if erased == 0 {
            // Degenerate case, nothing to do
        } else if start + shifted < half_capacity && start < erased {
            // Reinsert everything, as there are few kept indices
            self.indices.clear();

            // Reinsert stable indices, then shifted indices
            raw::insert_bulk_no_grow(&mut self.indices, &[], start_pairs);
            raw::insert_bulk_no_grow(&mut self.indices, start_pairs, shifted_pairs);
        } else if erased + shifted < half_capacity {
            // Find each affected index, as there are few to adjust

            // Find erased indices
            for (i, entry) in (start..).zip(erased_pairs) {
                raw::erase_index(&mut self.indices, entry.hash, i);
            }

            // Find shifted indices
            for ((new, old), entry) in (start..).zip(end..).zip(shifted_pairs) {
                update_index(&mut self.indices, entry.hash, old, new);
            }
        } else {
            // Sweep the whole table for adjustments
            self.erase_indices_sweep(start, end);
        }
    }

    pub(crate) fn retain_in_order<F>(&mut self, mut keep: F)
    where
        F: FnMut(&mut K, &mut V) -> bool,
        K: Eq,
    {
        let start_count = self.pairs.len();
        self.pairs
            .retain_mut(|pair| keep(&mut pair.key, &mut pair.value));
        if self.pairs.len() < start_count {
            self.rebuild_hash_table();
        }
        self.debug_assert_invariants();
    }

    fn rebuild_hash_table(&mut self)
    where
        K: Eq,
    {
        self.indices.clear();
        raw::insert_bulk_no_grow(&mut self.indices, &[], &self.pairs);
    }

    pub(crate) fn reverse(&mut self) {
        self.pairs.reverse();

        // No need to save hash indices, can easily calculate what they should
        // be, given that this is an in-place reversal.
        let len = self.pairs.len();
        for indices in self.indices_mut() {
            for i in indices.as_mut_slice() {
                *i = len - *i - 1;
            }
            indices.reverse();
        }
        self.debug_assert_invariants();
    }
}

impl<K, V, Indices> Clone for IndexMultimapCore<K, V, Indices>
where
    K: Clone,
    V: Clone,
    Indices: Clone + IndexStorage,
{
    fn clone(&self) -> Self {
        let mut new = Self::new();
        new.clone_from(self);
        new
    }

    fn clone_from(&mut self, other: &Self) {
        let hasher = get_hash(&other.pairs);
        self.indices.clone_from_with_hasher(&other.indices, hasher);
        if self.pairs.capacity() < other.pairs.len() {
            // If we must resize, match the indices capacity.
            let additional = other.pairs.len() - self.pairs.len();
            self.pairs.reserve(additional);
        }
        self.pairs.clone_from(&other.pairs);
    }
}

impl<K, V, Indices> fmt::Debug for IndexMultimapCore<K, V, Indices>
where
    K: fmt::Debug,
    V: fmt::Debug,
    Indices: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexMapCore")
            .field("indices", &raw::DebugIndices(&self.indices))
            .field(
                "entries",
                &self
                    .pairs
                    .iter()
                    .enumerate()
                    .map(|(i, a)| format!("{i}: {:?}", a))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Entry for an existing key-value pair or a vacant location to
/// insert one.
pub enum Entry<'a, K, V, Indices> {
    /// Existing slot with equivalent key.
    Occupied(OccupiedEntry<'a, K, V, Indices>),
    /// Vacant slot (no equivalent key in the map).
    Vacant(VacantEntry<'a, K, V, Indices>),
}

impl<'a, K, V, Indices> Entry<'a, K, V, Indices>
where
    Indices: IndexStorage,
{
    /// Gets a reference to the entry's key, either within the map if occupied,
    /// or else the new key that was used to find the entry.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// Returns the indices where the key-value pairs exists or will be inserted.
    /// The return value behaves like `&[`[`usize`]`]`
    pub fn indices(&self) -> EntryIndices<'_> {
        match self {
            Entry::Occupied(entry) => EntryIndices(SingleOrSlice::Slice(entry.indices())),
            Entry::Vacant(entry) => EntryIndices(SingleOrSlice::Single([entry.index()])),
        }
    }

    /// Modifies the entries if it is occupied.
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(SubsetMut<'_, K, V, &'_ [usize]>),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.as_subset_mut());
                Entry::Occupied(entry)
            }
            x => x,
        }
    }

    /// Inserts the given default value in the entry if it is vacant and returns
    /// a mutable iterator over it.
    /// That iterator will yield exactly one value.
    /// Otherwise a mutable iterator over an already existent values is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_insert(self, default: V) -> SubsetMut<'a, K, V, &'a [usize]> {
        match self {
            Entry::Occupied(entry) => entry.into_subset_mut(),
            Entry::Vacant(entry) => entry.insert_subset_mut(default),
        }
    }

    /// Inserts the result of the `call` function in the entry if it is vacant
    /// and returns a mutable iterator over it.
    /// That iterator will yield exactly one value.
    /// Otherwise a mutable iterator over an already existent values is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_insert_with<F>(self, call: F) -> SubsetMut<'a, K, V, &'a [usize]>
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_subset_mut(),
            Entry::Vacant(entry) => entry.insert_subset_mut(call()),
        }
    }

    /// Inserts the result of the `call` function with a reference to the entry's
    /// key if it is vacant, and returns a mutable iterator over it.
    /// That iterator will yield exactly one value.
    /// Otherwise a mutable iterator over an already existent values is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_insert_with_key<F>(self, call: F) -> SubsetMut<'a, K, V, &'a [usize]>
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_subset_mut(),

            Entry::Vacant(entry) => {
                let value = call(&entry.key);
                entry.insert_subset_mut(value)
            }
        }
    }

    /// Inserts a default-constructed value in the entry if it is vacant and
    /// returns a mutable iterator over it.
    /// That iterator will yield exactly one value.
    /// Otherwise a mutable iterator over an already existent values is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_default(self) -> SubsetMut<'a, K, V, &'a [usize]>
    where
        V: Default,
    {
        match self {
            Entry::Occupied(entry) => entry.into_subset_mut(),
            Entry::Vacant(entry) => entry.insert_subset_mut(V::default()),
        }
    }

    /// Insert provided `value` in the entry and return an occupied entry referring to it.
    pub fn insert_append(self, value: V) -> OccupiedEntry<'a, K, V, Indices> {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert_append_take_owned_key(value);
                entry
            }
            Entry::Vacant(entry) => entry.insert_entry(value),
        }
    }
}

impl<K, V, Indices> fmt::Debug for Entry<'_, K, V, Indices>
where
    K: fmt::Debug,
    V: fmt::Debug,
    Indices: fmt::Debug + IndexStorage,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Vacant(v) => f.debug_tuple(stringify!(Entry)).field(v).finish(),
            Entry::Occupied(o) => f.debug_tuple(stringify!(Entry)).field(o).finish(),
        }
    }
}

/// Wrapper of the indices of an [`Entry`]. Behaves like a `&[`[`usize`]`]`.
/// Always has at least one element.
///
/// Returned by the [`Entry::indices`] method. See it's documentation for more.
pub struct EntryIndices<'a>(SingleOrSlice<'a, usize>);

impl EntryIndices<'_> {
    pub fn as_slice(&self) -> &[usize] {
        match &self.0 {
            SingleOrSlice::Single(v) => v.as_slice(),
            SingleOrSlice::Slice(v) => v,
        }
    }
}

impl<'a> ops::Deref for EntryIndices<'a> {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            SingleOrSlice::Single(v) => v.as_slice(),
            SingleOrSlice::Slice(v) => v,
        }
    }
}

impl fmt::Debug for EntryIndices<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl PartialEq<&[usize]> for EntryIndices<'_> {
    fn eq(&self, other: &&[usize]) -> bool {
        &self.as_slice() == other
    }
}

enum SingleOrSlice<'a, T> {
    Single([T; 1]),
    Slice(&'a [T]),
}

impl<K, V, Indices> fmt::Debug for OccupiedEntry<'_, K, V, Indices>
where
    K: fmt::Debug,
    V: fmt::Debug,
    Indices: IndexStorage + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!(OccupiedEntry))
            .field("key", self.key())
            .field("pairs", &self.as_subset().iter())
            .finish()
    }
}

/// A view into a vacant entry in a [`IndexMultimap`].
/// It is part of the [`Entry`] enum.
///
/// [`IndexMultimap`]: super::IndexMultimap
pub struct VacantEntry<'a, K, V, Indices> {
    map: &'a mut IndexMultimapCore<K, V, Indices>,
    hash: HashValue,
    key: K,
}

impl<'a, K, V, Indices> VacantEntry<'a, K, V, Indices> {
    /// Gets a reference to the key that was used to find the entry.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Takes ownership of the key, leaving the entry vacant.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Returns the index where the key-value pair will be inserted.
    pub fn index(&self) -> usize {
        self.map.len_pairs()
    }

    /// Inserts the entry's key and the given value into the map,
    /// and returns a mutable reference to the value.
    pub fn insert(self, value: V) -> (usize, &'a K, &'a mut V)
    where
        Indices: IndexStorage,
    {
        let (i, _) = self.map.push(self.hash, self.key, value);
        let entry = &mut self.map.pairs[i];
        (i, &entry.key, &mut entry.value)
    }

    /// Inserts the entry's key and the given value into the map,
    /// and returns a mutable reference to the value.
    fn insert_subset_mut(self, value: V) -> SubsetMut<'a, K, V, &'a [usize]>
    where
        Indices: IndexStorage,
    {
        self.map.push_full(self.hash, self.key, value)
    }
}

impl<K, V, Indices> fmt::Debug for VacantEntry<'_, K, V, Indices>
where
    K: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(stringify!(VacantEntry))
            .field(self.key())
            .finish()
    }
}

/// Vector like types that can be used as a backing storage for [`IndexMultimap`]
/// to hold the indices of one key.
///
/// [`IndexMultimap`]: crate::IndexMultimap
pub trait IndexStorage
where
    Self: ops::Deref<Target = [usize]> + ops::DerefMut,
{
    /// Creates `self` with one element.
    fn one(v: usize) -> Self;
    /// Creates `self` with capacity.
    fn with_capacity(cap: usize) -> Self;
    /// Pushes one element to `self`.
    fn push(&mut self, v: usize);
    /// Removes one element from `self`.
    ///
    /// Should ***panic*** if `index` is out of bounds.
    fn remove(&mut self, index: usize) -> usize;
    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&mut usize) -> bool;
    /// Returns a view into `self` as a slice.
    fn as_slice(&self) -> &[usize];
    /// Returns a mutable view into `self` as a slice.
    fn as_mut_slice(&mut self) -> &mut [usize];
    fn shrink_to(&mut self, min_capacity: usize);
    fn shrink_to_fit(&mut self);
    fn reserve(&mut self, additional: usize);
    fn reserve_exact(&mut self, additional: usize);
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>;
    fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError>;
}

impl IndexStorage for Vec<usize> {
    #[inline]
    fn one(v: usize) -> Self {
        alloc::vec![v]
    }

    #[inline]
    fn with_capacity(cap: usize) -> Self {
        Vec::with_capacity(cap)
    }

    #[inline]
    fn push(&mut self, v: usize) {
        self.push(v)
    }
    #[inline]
    fn remove(&mut self, i: usize) -> usize {
        self.remove(i)
    }
    #[inline]
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&mut usize) -> bool,
    {
        self.retain_mut(f)
    }

    fn as_slice(&self) -> &[usize] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [usize] {
        self.as_mut_slice()
    }

    fn shrink_to(&mut self, min_capacity: usize) {
        self.shrink_to(min_capacity)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }

    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve(additional)
            .map_err(TryReserveError::from_alloc)
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.reserve_exact(additional)
    }

    fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve_exact(additional)
            .map_err(TryReserveError::from_alloc)
    }
}

// impl<const N: usize> IndexStorage for SmallVec<[usize; N]> {
//     #[inline]
//     fn one(v: usize) -> Self {
//         smallvec::smallvec![v]
//     }

//     #[inline]
//     fn with_capacity(cap: usize) -> Self {
//         SmallVec::with_capacity(cap)
//     }

//     #[inline]
//     fn push(&mut self, v: usize) {
//         self.push(v)
//     }
//     #[inline]
//     fn remove(&mut self, i: usize) -> usize {
//         self.remove(i)
//     }
//     #[inline]
//     fn retain<F>(&mut self, f: F)
//     where
//         F: FnMut(&mut usize) -> bool,
//     {
//         self.retain_mut(f)
//     }

//     fn as_slice(&self) -> &[usize] {
//         self.as_slice()
//     }

//     fn as_mut_slice(&mut self) -> &mut [usize] {
//         self.as_mut_slice()
//     }
//     fn shrink_to(&mut self, _min_capacity: usize) {}
//     fn shrink_to_fit(&mut self) {
//         self.shrink_to_fit()
//     }

//     fn reserve(&mut self, additional: usize) {
//         self.reserve(additional)
//     }

//     fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
//         self.try_reserve(additional).map_err(|e| match e {
//             smallvec::CollectionAllocErr::CapacityOverflow => TryReserveError::capacity_overflow(),
//             smallvec::CollectionAllocErr::AllocErr { layout } => TryReserveError::alloc(layout),
//         })
//     }

//     fn reserve_exact(&mut self, additional: usize) {
//         self.reserve_exact(additional)
//     }

//     fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
//         self.try_reserve_exact(additional).map_err(|e| match e {
//             smallvec::CollectionAllocErr::CapacityOverflow => TryReserveError::capacity_overflow(),
//             smallvec::CollectionAllocErr::AllocErr { layout } => TryReserveError::alloc(layout),
//         })
//     }
// }

// impl<const N: usize> IndexStorage for arrayvec::ArrayVec<usize, N> {
//     #[inline]
//     fn one(v: usize) -> Self {
//         let mut s = arrayvec::ArrayVec::new();
//         s.push(v);
//         s
//     }

//     #[inline]
//     fn with_capacity(cap: usize) -> Self {
//         arrayvec::ArrayVec::new()
//     }

//     /// ***Panics*** if the vector is already full.
//     #[inline]
//     fn push(&mut self, v: usize) {
//         self.push(v)
//     }

//     #[inline]
//     fn remove(&mut self, i: usize) -> usize {
//         self.remove(i)
//     }
//     #[inline]
//     fn retain<F>(&mut self, f: F)
//     where
//         F: FnMut(&mut usize) -> bool,
//     {
//         self.retain(f)
//     }

//     fn as_slice(&self) -> &[usize] {
//         self.as_slice()
//     }

//     fn as_mut_slice(&mut self) -> &mut [usize] {
//         self.as_mut_slice()
//     }
//     fn shrink_to(&mut self, _min_capacity: usize) {}
//     fn shrink_to_fit(&mut self) {}

//     fn reserve(&mut self, additional: usize) {
//         assert!(additional == 0, "cannot resize ArrayVec")
//     }

//     fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
//         assert!(additional == 0, "cannot resize ArrayVec");
//         Ok(())
//     }

//     fn reserve_exact(&mut self, additional: usize) {
//         assert!(additional == 0, "cannot resize ArrayVec")
//     }

//     fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
//         assert!(additional == 0, "cannot resize ArrayVec");
//         Ok(())
//     }
// }

#[test]
fn assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<IndexMultimapCore<i32, i32, Vec<usize>>>();
    assert_send_sync::<Entry<'_, i32, i32, Vec<usize>>>();
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    //use hashbrown::raw::RawTable;

    use crate::HashValue;

    use super::*;

    #[test]
    fn update_index_test() {
        let mut map = IndexMultimapCore::<i32, i32, Vec<usize>>::new();
        map.insert_append_full(HashValue(1), 1, 1);
        map.insert_append_full(HashValue(2), 2, 2);
        map.insert_append_full(HashValue(1), 1, 3);
        map.insert_append_full(HashValue(2), 2, 4);
        map.insert_append_full(HashValue(1), 1, 5);

        let table = &mut map.indices;
        //println!("{:#?}", &raw::DebugIndices(table));
        update_index(table, HashValue(1), 2, 7);
        //println!("{:#?}", &raw::DebugIndices(table));
    }
}

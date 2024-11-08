#![allow(unsafe_code)]

//! This is the core implementation that doesn't depend on the hasher at all.
//!
//! The methods of `IndexMapCore` don't use any Hash properties of K.
//!
//! It's cleaner to separate them out, then the compiler checks that we are not
//! using Hash at all in these methods.
//!
//! However, we should probably not let this show in the public API or docs.

use ::alloc::vec::Vec;
use ::core::{cmp, fmt, ops, panic};
use ::equivalent::Equivalent;
use ::hashbrown::hash_table;

pub use self::entry::{Entry, EntryIndices, IndexedEntry, OccupiedEntry, VacantEntry};
use self::indices::{Indices, UniqueSortedIter};
#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
pub use self::remove_iter::rayon::ParDrain;
pub use self::remove_iter::{Drain, ShiftRemove, SwapRemove};
pub use self::subsets::{
    Subset, SubsetIter, SubsetIterMut, SubsetKeys, SubsetMut, SubsetValues, SubsetValuesMut,
};
use crate::util::DebugIterAsNumberedCompactList;
use crate::{Bucket, HashValue, TryReserveError};

mod entry;
mod indices;
mod remove_iter;
mod subsets;

type IndicesTable = hashbrown::hash_table::HashTable<Indices>;

/// Core of the map that does not depend on S
pub(super) struct IndexMultimapCore<K, V> {
    // ---
    // IMPL DETAILS:
    //
    // # Invariants
    //
    // .. that must hold after every complete operation
    //   (eq after insertion, removal, completed drain etc).
    //
    // * Indices in self.indices are all unique
    // * Indies in self.indices are all valid to index into self.pairs
    // * Each entry in self.indices has it's indices in sorted order
    // * Each entry in self.indices has at least one index
    // * Indices of single entry in self.indices must point to all and only
    //   of the equivalent key pairs in self.pairs
    //
    // There is self.debug_assert_invariants() method which check's that these
    // invariants hold. It is behind a feature flag `more_debug_assertions`
    // and is compiled away in release builds in any case.
    //
    // ---
    /// indices mapping from the entry hash to its index.
    indices: IndicesTable,
    /// pairs is a dense vec of key-value pairs in their order.
    pairs: Vec<Bucket<K, V>>,
}

pub(super) struct IndexMultimapCoreRef<'a, K, V> {
    indices: &'a IndicesTable,
    pairs: &'a Vec<Bucket<K, V>>,
}

pub(super) struct IndexMultimapCoreRefMut<'a, K, V> {
    indices: &'a mut IndicesTable,
    pairs: &'a mut Vec<Bucket<K, V>>,
}

/// Inserts multiple pairs into a raw table without reallocating.
///
/// # Panics
///
/// 1. if there is not sufficient capacity already.
/// 2. potentially panics if `indices` and `current_pairs` don't match.
/// 3.
fn insert_bulk_no_grow<K, V>(
    indices_table: &mut IndicesTable,
    current_pairs: &[Bucket<K, V>],
    new_pairs: &[Bucket<K, V>],
) where
    K: Eq,
{
    let current_num_pairs = current_pairs.len();
    debug_assert_eq!(
        indices_table
            .iter()
            .map(|a| a.as_ref().len())
            .sum::<usize>(),
        current_num_pairs,
        "mismatch in the number of current pairs"
    );

    for (pair_index, pair) in (current_num_pairs..).zip(new_pairs) {
        match indices_table.find_mut(pair.hash.get(), |indices| {
            let i = indices[0];
            if i < current_num_pairs {
                current_pairs[i].key == pair.key
            } else {
                new_pairs[i - current_num_pairs].key == pair.key
            }
        }) {
            Some(indices) => indices.push(pair_index),
            None => {
                let indices = Indices::one(pair_index);
                indices_table.insert_unique(pair.hash.get(), indices, |_| {
                    panic!("we know that there is enough capacity, hasher should never be called")
                });
            }
        }
    }
}

/// # Panics
///
/// 1. Panics if `hash` and `index` do not match to any item in the `table`
#[inline]
fn erase_index(table: &mut IndicesTable, hash: HashValue, index: usize) {
    // Cache the index in indices as we find the bucket,
    // so that we don't need to find it again for indices.remove below
    let mut index_in_indices = None;
    let eq_index = |indices: &Indices| match indices.binary_search(&index) {
        Ok(i) => {
            index_in_indices = Some(i);
            true
        }
        Err(_) => false,
    };
    match table.find_entry(hash.get(), eq_index) {
        Ok(mut entry) => {
            let indices = entry.get_mut();
            if indices.len() == 1 {
                entry.remove();
            } else {
                indices.remove(
                    index_in_indices.expect("the index_in_indices was already found in indices"),
                );
            }
        }
        Err(_) => unreachable!("pair for index not found"),
    }
}

/// Erase the index but assumes that it's last in the key's indices.
/// Avoids the binary_searches of generic erase_index above.
///
/// Used by .pop() method, since we keep indices sorted and thus the index we
/// need to remove must be in the last position.
///
/// # Panics
///
/// 1. Panics if `hash` and `index` do not match an item in the `table`
#[inline]
fn erase_index_last(table: &mut IndicesTable, hash: HashValue, index: usize) {
    match table.find_entry(hash.get(), eq_index_last(index)) {
        Ok(mut entry) => {
            let indices = entry.get_mut();
            debug_assert_eq!(*indices.last().unwrap(), index);
            if indices.len() == 1 {
                entry.remove();
            } else {
                indices.pop();
            }
        }
        Err(_) => unreachable!("pair for index not found"),
    }
}

/// # Panics
///
/// 1. if `hash` and `old` do not match to any item in the `table`
/// 2. if `new` is already in the indices matching `hash` and `old` index
#[inline]
fn update_index(table: &mut IndicesTable, hash: HashValue, old: usize, new: usize) {
    // Index to `old` in the indices
    let mut olds_index: usize = 0;
    let indices = table
        .find_mut(hash.get(), |indices| match indices.binary_search(&old) {
            Ok(i) => {
                olds_index = i;
                true
            }
            Err(_) => false,
        })
        .expect("index not found");
    // `olds_index` in is bounds, cannot panic because of that
    // but can panic if `new` already exists in `indices`
    indices.replace(olds_index, new);
}

/// Update the index old to new in indices table.
/// Assumes that old is the last index in indices arrays.
///
/// This is used by swap_removes where we know that old is the very last index
/// in the whole map and thus in any indices array as well.
/// This avoids one binary search to find the position of old in indices array.
///
/// # Panics
///
/// 1. if `hash` and `old` do not match to any item in the `table`
/// 2. if `new` is already in the indices matching `hash` and `old` index
#[inline]
fn update_index_last(table: &mut IndicesTable, hash: HashValue, old: usize, new: usize) {
    // Index to `old` in the indices
    let indices = table
        .find_mut(hash.get(), eq_index_last(old))
        .expect("index not found");
    indices.replace(indices.len() - 1, new);
}

/// # Panics
///
/// The returned function will panic if:
///   1. `indices` passed to it are empty
///   2. `indices[0]` is out of bounds for `pairs`
#[inline(always)]
fn get_hash<K, V>(pairs: &[Bucket<K, V>]) -> impl Fn(&Indices) -> u64 + use<'_, K, V> {
    // All pairs at `indices` must have equivalent keys,
    // just take the key from the first
    move |indices| pairs[indices[0]].hash.get()
}

/// # Panics
///
/// The returned function will panic if:
///   1. `indices` passed to it are empty
///   2. `indices[0]` is out of bounds for `pairs`
#[inline]
fn equivalent<'a, K, V, Q>(key: &'a Q, pairs: &'a [Bucket<K, V>]) -> impl Fn(&Indices) -> bool + 'a
where
    Q: ?Sized + Equivalent<K>,
{
    // All pairs at `indices` must have equivalent keys,
    // just take the key from the first
    move |indices| Q::equivalent(key, &pairs[indices[0]].key)
}

#[allow(dead_code)]
#[inline]
fn eq_index(index: usize) -> impl Fn(&Indices) -> bool {
    move |indices| indices.binary_search(&index).is_ok()
}

/// # Panics
///
/// The returned function will panic if `indices` passed to it are empty.
#[inline]
fn eq_index_last(index: usize) -> impl Fn(&Indices) -> bool {
    move |indices| *indices.last().unwrap() == index
}

impl<K, V> IndexMultimapCore<K, V> {
    #[inline(always)]
    fn debug_assert_invariants(&self)
    where
        K: Eq,
    {
        self.as_ref().debug_assert_invariants();
    }

    #[inline(always)]
    fn debug_assert_indices(&self, indices: &[usize]) {
        self.as_ref().debug_assert_indices(indices);
    }
}

impl<K, V> IndexMultimapCore<K, V> {
    #[inline]
    pub(super) const fn new() -> Self {
        IndexMultimapCore {
            indices: IndicesTable::new(),
            pairs: Vec::new(),
        }
    }

    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    #[inline]
    pub(super) fn with_capacity(keys: usize, pairs: usize) -> Self {
        IndexMultimapCore {
            indices: IndicesTable::with_capacity(keys),
            pairs: Vec::with_capacity(pairs),
        }
    }

    #[inline]
    pub(super) fn len_keys(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub(super) fn len_pairs(&self) -> usize {
        self.pairs.len()
    }

    #[inline]
    pub(super) fn capacity_keys(&self) -> usize {
        self.indices.capacity()
    }

    #[inline]
    pub(super) fn capacity_pairs(&self) -> usize {
        self.pairs.capacity()
    }

    #[inline]
    pub(super) fn clear(&mut self) {
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
    pub(super) fn as_mut_pairs(&mut self) -> &mut [Bucket<K, V>] {
        &mut self.pairs
    }

    /// # Panics
    ///
    /// May panic if `f` modifies keys in such a way that there are more unique keys than before.  
    pub(super) fn with_pairs<F>(&mut self, f: F)
    where
        F: FnOnce(&mut [Bucket<K, V>]),
        K: Eq,
    {
        f(&mut self.pairs);
        self.as_ref_mut().rebuild_hash_table();
    }
}

impl<K, V> IndexMultimapCore<K, V> {
    /// Reserve capacity for `additional` more key-value pairs.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    pub(super) fn reserve(&mut self, additional_keys: usize, additional_pairs: usize) {
        self.indices.reserve(additional_keys, get_hash(&self.pairs));
        self.pairs.reserve(additional_pairs);
    }

    /// Reserve capacity for `additional` more key-value pairs, without over-allocating.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    pub(super) fn reserve_exact(&mut self, additional_keys: usize, additional_pairs: usize) {
        self.indices.reserve(additional_keys, get_hash(&self.pairs));
        self.pairs.reserve_exact(additional_pairs);
    }

    /// Try to reserve capacity for `additional` more key-value pairs.
    pub(super) fn try_reserve(
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
    pub(super) fn try_reserve_exact(
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
    ///
    /// # Panics
    ///
    /// Panics if we need to reallocate and that allocation fails.
    pub(super) fn shrink_to(&mut self, min_capacity: usize) {
        self.indices.shrink_to(min_capacity, get_hash(&self.pairs));
        self.pairs.shrink_to(min_capacity);
    }

    /// # Panics
    ///
    /// Panics if we need to reallocate and that allocation fails.
    pub(super) fn shrink_to_fit(&mut self) {
        self.shrink_to(0);
        self.indices.iter_mut().for_each(|a| a.shrink_to_fit());
    }

    /// Return the index in `entries` where an equivalent key can be found
    pub(super) fn get_indices_of<Q>(&self, hash: HashValue, key: &Q) -> &[usize]
    where
        Q: ?Sized + Equivalent<K>,
    {
        let indices = self
            .get_indices_of_core(hash, key)
            .map(|i| i.as_slice())
            .unwrap_or_default();
        if cfg!(debug_assertions) && !indices.is_empty() {
            self.debug_assert_indices(indices);
        }

        indices
    }

    pub(super) fn get_indices_of_core<Q>(&self, hash: HashValue, key: &Q) -> Option<&Indices>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        self.indices.find(hash.get(), eq)
    }

    pub(super) fn get<Q>(&self, hash: HashValue, key: &Q) -> Subset<'_, K, V>
    where
        Q: ?Sized + Equivalent<K>,
    {
        self.as_ref().get(hash, key)
    }

    pub(super) fn get_mut<Q>(&mut self, hash: HashValue, key: &Q) -> SubsetMut<'_, K, V>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        if let Some(indices) = self.indices.find(hash.get(), eq) {
            self.debug_assert_indices(indices);
            SubsetMut::new(&mut self.pairs, indices.as_unique_slice())
        } else {
            SubsetMut::empty()
        }
    }

    pub(super) fn get_all_by_index(&self, index: usize) -> Subset<'_, K, V>
    where
        K: Eq,
    {
        let bucket = &self.pairs[index];
        self.get(bucket.hash, &bucket.key)
    }

    pub(super) fn get_all_mut_by_index(&mut self, index: usize) -> SubsetMut<'_, K, V>
    where
        K: Eq,
    {
        let bucket = &self.pairs[index];
        let hash = bucket.hash;
        let key = &bucket.key;
        let eq = equivalent(key, &self.pairs);
        if let Some(indices) = self.indices.find(hash.get(), eq) {
            self.debug_assert_indices(indices);
            SubsetMut::new(&mut self.pairs, indices.as_unique_slice())
        } else {
            SubsetMut::empty()
        }
    }

    pub(super) fn entry(&mut self, hash: HashValue, key: K) -> Entry<'_, K, V>
    where
        K: Eq,
    {
        Entry::new(self, hash, key)
    }

    fn indices_mut(&mut self) -> impl Iterator<Item = &mut Indices> {
        self.indices.iter_mut()
    }

    /// Appends a new entry to the existing key, or insert the key.
    pub(super) fn insert_append_full(&mut self, hash: HashValue, key: K, value: V) -> usize
    where
        K: Eq,
    {
        let i = self.pairs.len();
        let eq = equivalent(&key, &self.pairs);
        match self.indices.find_mut(hash.get(), eq) {
            Some(indices) => {
                // Cannot panic as none of the entries in `self.indices` can contain `i`
                indices.push(i);
            }
            None => {
                self.indices
                    .insert_unique(hash.get(), Indices::one(i), get_hash(&self.pairs));
            }
        }
        self.pairs.push(Bucket { hash, key, value });
        i
    }

    pub(super) fn insert_at(
        &mut self,
        index: usize,
        hash: HashValue,
        key: K,
        value: V,
    ) -> Option<(usize, OccupiedEntry<'_, K, V>)>
    where
        K: Eq,
    {
        self.as_ref_mut()
            .insert_at_into(index, hash, key, value)
            .ok()
    }

    /// Append a key-value pair, *without* checking whether it already exists,
    /// and return the pair's new index.
    fn push(
        &mut self,
        hash: HashValue,
        key: K,
        value: V,
    ) -> (usize, hash_table::OccupiedEntry<'_, Indices>) {
        let i = self.pairs.len();
        let entry = self
            .indices
            .insert_unique(hash.get(), Indices::one(i), get_hash(&self.pairs));
        self.pairs.push(Bucket { hash, key, value });
        //#[allow(unsafe_code)]
        //self.debug_assert_indices(bucket.get());
        (i, entry)
    }

    pub(super) fn reverse(&mut self)
    where
        K: Eq,
    {
        self.pairs.reverse();

        // No need to save hash indices, can easily calculate what they should
        // be, given that this is an in-place reversal.
        let len = self.pairs.len();
        for indices in self.indices_mut() {
            // SAFETY: Following keeps indices unique and sorted
            unsafe {
                let indices = indices.as_mut_slice();
                for i in indices.iter_mut() {
                    *i = len - *i - 1;
                }
                indices.reverse();
            }
        }
        self.debug_assert_invariants();
    }

    pub(super) fn sort_keys(&mut self)
    where
        K: Ord,
    {
        self.pairs.sort_by(move |a, b| K::cmp(&a.key, &b.key));
        self.rebuild_hash_table();
        self.debug_assert_invariants();
    }

    pub(super) fn sort_by<F>(&mut self, mut cmp: F)
    where
        F: FnMut(&K, &V, &K, &V) -> cmp::Ordering,
        K: Eq,
    {
        self.pairs
            .sort_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        self.rebuild_hash_table();
        self.debug_assert_invariants();
    }

    pub(super) fn sort_unstable_by<F>(&mut self, mut cmp: F)
    where
        F: FnMut(&K, &V, &K, &V) -> cmp::Ordering,
        K: Eq,
    {
        self.pairs
            .sort_unstable_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        self.rebuild_hash_table();
        self.debug_assert_invariants();
    }

    pub(super) fn sort_unstable_keys(&mut self)
    where
        K: Ord,
    {
        self.pairs
            .sort_unstable_by(move |a, b| K::cmp(&a.key, &b.key));
        self.rebuild_hash_table();
        self.debug_assert_invariants();
    }

    pub(super) fn sort_by_cached_key<T, F>(&mut self, mut sort_key: F)
    where
        T: Ord,
        F: FnMut(&K, &V) -> T,
        K: Eq,
    {
        self.pairs
            .sort_by_cached_key(move |a| sort_key(&a.key, &a.value));
        self.rebuild_hash_table();
        self.debug_assert_invariants();
    }

    /// Remove the last key-value pair
    pub(super) fn pop(&mut self) -> Option<(K, V)>
    where
        K: Eq,
    {
        if let Some(entry) = self.pairs.pop() {
            let last_index = self.pairs.len();
            // last_index must also be last in the key's indices,
            // use erase_index_last which assumes that that's the case
            // and avoids unnecessary binary searches.
            erase_index_last(&mut self.indices, entry.hash, last_index);
            self.debug_assert_invariants();
            Some((entry.key, entry.value))
        } else {
            None
        }
    }

    /// Remove an entry by shifting all entries that follow it
    #[inline]
    pub(super) fn shift_remove<Q>(
        &mut self,
        hash: HashValue,
        key: &Q,
    ) -> Option<ShiftRemove<'_, K, V>>
    where
        Q: ?Sized + Equivalent<K>,
        K: Eq,
    {
        self.debug_assert_invariants();
        ShiftRemove::new(self, hash, key)
    }

    /// Remove an entry by shifting all entries that follow it
    pub(super) fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)>
    where
        K: Eq,
    {
        self.as_ref_mut().shift_remove_index(index)
    }

    /// Remove an entry by swapping it with the last
    #[inline]
    pub(super) fn swap_remove<Q>(
        &mut self,
        hash: HashValue,
        key: &Q,
    ) -> Option<SwapRemove<'_, K, V>>
    where
        Q: ?Sized + Equivalent<K>,
        K: Eq,
    {
        self.debug_assert_invariants();
        SwapRemove::new(self, hash, key)
    }

    /// Remove an entry by swapping it with the last
    pub(super) fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)>
    where
        K: Eq,
    {
        self.as_ref_mut().swap_remove_index(index)
    }

    pub(super) fn retain_in_order<F>(&mut self, mut keep: F)
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

    #[inline]
    pub(super) fn drain<R>(&mut self, range: R) -> Drain<'_, K, V>
    where
        R: ops::RangeBounds<usize>,
        K: Eq,
    {
        self.debug_assert_invariants();
        Drain::new(self, range)
    }

    pub(super) fn truncate(&mut self, len: usize)
    where
        K: Eq,
    {
        if len < self.len_pairs() {
            unsafe { self.erase_indices(len, self.pairs.len()) };
            self.pairs.truncate(len);
            self.debug_assert_invariants();
        }
    }

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    pub(super) fn par_drain<R>(&mut self, range: R) -> ParDrain<'_, K, V>
    where
        K: Send + Eq,
        V: Send,
        R: ops::RangeBounds<usize>,
    {
        ParDrain::new(self, range)
    }

    /// # Panics
    ///
    /// Panics if `at` is out of bounds, eg `at > self.pairs.len()`.
    /// Note that `at == self.pairs.len()` is ok, but the returned map is empty.
    pub(super) fn split_off(&mut self, at: usize) -> Self
    where
        K: Eq,
    {
        assert!(at <= self.pairs.len());
        let unique_keys = self.len_keys();

        unsafe { self.erase_indices(at, self.pairs.len()) };
        let pairs = self.pairs.split_off(at);

        // TODO: We are most likely over allocating here.
        // Options:
        //    * keep as is and potentially over allocate
        //    * count unique keys in entries and potentially perform a rather expensive calculation
        //    * use RawTable::new and potentially allocate multiple times
        //
        // There cannot be more unique keys that was in the original map or there are new entries.
        let capacity = Ord::min(unique_keys, pairs.len());
        let mut indices = IndicesTable::with_capacity(capacity);
        // Cannot panic as we just created empty table with enough capacity.
        insert_bulk_no_grow(&mut indices, &[], &pairs);
        let new = Self { indices, pairs };

        self.debug_assert_invariants();
        new.debug_assert_invariants();

        new
    }

    /// # Panics
    ///
    /// * if `from` or `to` are out of bounds.
    pub(super) fn move_index(&mut self, from: usize, to: usize)
    where
        K: Eq,
    {
        if from == to {
            return;
        }

        assert!(
            from < self.pairs.len(),
            "index from=`{from}` is out of bounds for slice of length `{}`",
            self.len_pairs()
        );
        assert!(
            to < self.pairs.len(),
            "index to=`{to}` is out of bounds for slice of length `{}`",
            self.len_pairs()
        );

        let from_hash = self.pairs[from].hash;
        // Use a sentinel index so other indices don't collide.
        // * cannot panic as self.pairs and self.indices must be in sync and usize::MAX cannot exist
        //   in self.indices
        update_index(&mut self.indices, from_hash, from, usize::MAX);

        // Update all other indices and rotate the entry positions.
        #[allow(clippy::comparison_chain)]
        if from < to {
            // [head] [from]     [from + 1] ... [to - 1] [to]   [tail]
            // [head] [from + 1] [from + 2] ... [to]     [from] [tail]

            // SAFETY:
            //  * we changed `from` to `usize::MAX`
            //  * self.pairs and self.indices are in sync except `from` index
            //  * self.paris and self.indices are synced up below

            // * following two calls cannot panic as to and from are checked to be in bounds
            unsafe { self.as_ref_mut().decrement_indices(from + 1, to + 1, 1) };
            self.pairs[from..=to].rotate_left(1);
        } else {
            // [head] [to]   [to + 1] ... [from - 1] [from]     [tail]
            // [head] [from] [to]     ... [from - 2] [from - 1] [tail]

            // SAFETY: same as above branch
            unsafe { self.as_ref_mut().increment_indices(to, from) };
            self.pairs[to..=from].rotate_right(1);
        }

        // Change the sentinel index to its final position.
        // * cannot panic we just updated this index and `to` was removed from `self.indices` by
        //   either decrement or increment calls
        update_index_last(&mut self.indices, from_hash, usize::MAX, to);
        self.debug_assert_invariants();
    }

    pub(super) fn swap_indices(&mut self, a: usize, b: usize)
    where
        K: Eq,
    {
        self.as_ref_mut().swap_indices(a, b);
    }

    /// Erase `start..end` from `indices`, and shift `end..` indices down to `start..`
    ///
    /// All of these items should still be at their original location in `entries`.
    /// This is used by `drain`, which will let `Vec::drain` do the work on `entries`.
    ///
    /// # Safety
    ///
    /// * The caller must ensure that after calling this function they fix up the map again, that is
    ///   sync `self.indices` and `self.pairs` again.
    ///
    /// # Panics
    ///
    /// 1. if `start > end` or `end > self.pairs.len()`, that is they are invalid for slicing `self.pairs`
    ///
    /// 2. may panic if `self.indices` and `self.pairs` are out of sync for indices in range `start..end`
    ///
    ///    If `self.indices` and `self.pairs` are out of sync, then this method could update random
    ///    indices, decrement same index more than once, create duplicate indices and/or panic.
    ///
    /// If this method panics, any number of indices in range `start..end` may have been updated,
    /// and rest remaining as was.
    unsafe fn erase_indices(&mut self, start: usize, end: usize)
    where
        K: Eq,
    {
        let (init, shifted_pairs) = self.pairs.split_at(end);
        let (start_pairs, erased_pairs) = init.split_at(start);

        let erased = erased_pairs.len();
        let shifted = shifted_pairs.len();
        let half_capacity = self.indices.capacity() / 2;

        // Use a heuristic between different strategies
        if erased == 0 {
            // Degenerate case, nothing to do
        } else if start + shifted < half_capacity && start < erased {
            // Reinsert everything, as there are few kept indices
            self.indices.clear();

            // Reinsert stable indices, then shifted indices
            // These cannot panic because:
            //   * self.indices had more entries than we insert
            //   * `current_pairs` match with what's in `self.indices` at each step
            insert_bulk_no_grow(&mut self.indices, &[], start_pairs);
            insert_bulk_no_grow(&mut self.indices, start_pairs, shifted_pairs);
        } else if erased + shifted < half_capacity {
            // Find each affected index, as there are few to adjust

            // Find erased indices
            for (i, entry) in (start..).zip(erased_pairs) {
                erase_index(&mut self.indices, entry.hash, i); // may panic for 1.
            }

            // Find shifted indices
            for ((new, old), entry) in (start..).zip(end..).zip(shifted_pairs) {
                // may panic for 1.
                // cannot panic:
                //   2. every index new was removed by erase_index above, or will be removed in previous iterations of this loop
                update_index(&mut self.indices, entry.hash, old, new);
            }
        } else {
            // Sweep the whole table for adjustments, as there are a lot to adjust
            // SAFETY: requirements are same as for this call
            unsafe { self.erase_indices_sweep(start, end) };
        }
    }

    /// Sweep the whole table to erase indices `start..end` and shift `end..` indices down to `start..`
    ///
    /// # Safety
    ///
    /// * The caller must ensure that after calling this function they fix up the map again, that is
    ///   sync `self.indices` and `self.pairs` again.
    unsafe fn erase_indices_sweep(&mut self, start: usize, end: usize) {
        let offset = end - start;
        self.indices.retain(|indices| {
            // SAFETY:
            //  a) we remove all the indices in range start..end,
            //     thus any starting index >= end turned into
            //     index in range start..end will be unique.
            //     Now as we use constant offset
            //     and self.indices as whole also contains only unique indices,
            //     means that the resulting indices will be unique
            //  b) retain preserves the order and using constant offset
            //     cannot change the order
            //  d) above points together ensure that individual set
            //     of indices remain unique and sorted and the whole
            //     indices table will also contain unique indices
            unsafe {
                indices.retain_mut(|i| {
                    if *i >= end {
                        *i -= offset;
                        true
                    } else {
                        *i < start
                    }
                })
            };
            !indices.is_empty()
        });
    }

    /// # Panics
    ///
    /// * if there is not sufficient capacity in `self.indices` to insert
    ///   indices of pairs in `self.pairs`
    fn rebuild_hash_table(&mut self)
    where
        K: Eq,
    {
        self.as_ref_mut().rebuild_hash_table();
    }
    pub(super) fn as_ref_mut(&mut self) -> IndexMultimapCoreRefMut<'_, K, V> {
        IndexMultimapCoreRefMut::new(self)
    }

    pub(super) fn as_ref(&self) -> IndexMultimapCoreRef<'_, K, V> {
        IndexMultimapCoreRef::new(self)
    }
}

impl<K, V> Clone for IndexMultimapCore<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut new = Self::new();
        new.clone_from(self);
        new
    }

    fn clone_from(&mut self, other: &Self) {
        self.indices.clone_from(&other.indices);
        if self.pairs.capacity() < other.pairs.len() {
            let additional = other.pairs.len() - self.pairs.len();
            self.pairs.reserve(additional);
        }
        self.pairs.clone_from(&other.pairs);
    }
}

impl<K, V> fmt::Debug for IndexMultimapCore<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexMapCore")
            .field("indices", &DebugIndices(self))
            .field(
                "entries",
                &DebugIterAsNumberedCompactList::new(self.pairs.iter().map(Bucket::refs)),
            )
            .finish()
    }
}

struct DebugIndices<'a, K, V>(&'a IndexMultimapCore<K, V>);

impl<K, V> fmt::Debug for DebugIndices<'_, K, V>
where
    K: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use more readable output - each bucket is always one line
        // SAFETY: we're not letting any of the buckets escape this function
        let indices = self.0.indices.iter();
        let mut list = f.debug_map();
        for i in indices {
            let key = self.0.pairs[i[0]].key_ref();
            list.entry(&format_args!("{key:?}"), &format_args!("{:?}", i));
        }
        list.finish()
    }
}

impl<'a, K, V> IndexMultimapCoreRef<'a, K, V> {
    fn new(map: &'a IndexMultimapCore<K, V>) -> Self {
        let indices = &map.indices;
        let pairs = &map.pairs;
        unsafe { Self::new_unchecked(indices, pairs) }
    }

    /// # Safety
    ///
    /// * `indices` and `pairs` must be from the same `IndexMultimapCore`
    unsafe fn new_unchecked(indices: &'a IndicesTable, pairs: &'a Vec<Bucket<K, V>>) -> Self {
        Self { indices, pairs }
    }

    #[inline]
    pub(super) fn len_keys(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub(super) fn len_pairs(&self) -> usize {
        self.pairs.len()
    }

    #[inline]
    pub(super) fn capacity_keys(&self) -> usize {
        self.indices.capacity()
    }

    #[inline]
    pub(super) fn capacity_pairs(&self) -> usize {
        self.pairs.capacity()
    }

    pub(super) fn get<Q>(&self, hash: HashValue, key: &Q) -> Subset<'a, K, V>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        if let Some(indices) = self.indices.find(hash.get(), eq) {
            //self.debug_assert_indices(indices);
            Subset::new(&self.pairs, indices)
        } else {
            Subset::empty()
        }
    }

    pub(super) fn get_all_by_index(&self, index: usize) -> Subset<'a, K, V>
    where
        K: Eq,
    {
        let bucket = &self.pairs[index];
        self.get(bucket.hash, &bucket.key)
    }
}

impl<'a, K, V> IndexMultimapCoreRef<'a, K, V> {
    #[inline(always)]
    #[cfg(any(not(debug_assertions), not(feature = "more_debug_assertions")))]
    fn debug_assert_invariants(&self) {}

    #[cfg(all(debug_assertions, feature = "std", feature = "more_debug_assertions"))]
    #[track_caller]
    fn debug_assert_invariants(&self)
    where
        K: Eq,
    {
        let mut index_count = 0; // Count the total number of indices in self.indices
        let index_iter = self.indices.iter();
        let mut seen_indices = crate::IndexSet::with_capacity(index_iter.len());
        for indices in index_iter {
            index_count += indices.len();
            assert!(!indices.is_empty(), "found empty indices");
            assert!(crate::util::is_sorted(indices), "found unsorted indices");
            assert!(
                indices.last().unwrap() < &self.pairs.len(),
                "found out of bound index for entries in indices"
            );
            seen_indices.reserve(indices.len());

            let Bucket { hash, key, .. } = &self.pairs[*indices.first().unwrap()];
            // Probably redundant check, if for every entry in self.indices
            // all of those indices point to pairs with equivalent keys and
            // the indices are all unique and the number of indices matches the
            // number of pairs, then there must be exactly indices.len() pairs
            // with given key in self.pairs.
            let count = self.pairs.iter().filter(|b| b.key_ref() == key).count();
            assert_eq!(
                count,
                indices.len(),
                "indices do not contain indices to all of the pairs with this key"
            );
            for i in indices.as_slice() {
                assert!(seen_indices.insert(i), "found duplicate index in `indices`");
                let Bucket {
                    hash: hash2,
                    key: key2,
                    ..
                } = &self.pairs[*i];
                assert_eq!(
                    hash, hash2,
                    "indices of single entry point to different hashes"
                );
                assert!(
                    key == key2,
                    "indices of single entry point to different keys"
                );
            }
        }

        assert_eq!(
            self.pairs.len(),
            index_count,
            "mismatch between pairs and indices count"
        );

        // This is probably unnecessary, if the number of indices in self.indices
        // equals the number of pairs in self.paris and all of the indices in
        // self.indices are unique then there must be an index for each pair
        for (i, Bucket { hash, .. }) in self.pairs.iter().enumerate() {
            let indices = self.indices.find(hash.get(), eq_index(i));
            assert!(
                indices.is_some(),
                "expected a pair to have a matching entry in indices table"
            );
        }
    }

    #[cfg(all(
        debug_assertions,
        not(feature = "std"),
        feature = "more_debug_assertions"
    ))]
    #[track_caller]
    fn debug_assert_invariants(&self)
    where
        K: Eq,
    {
        let mut index_count = 0;
        let index_iter = unsafe { self.indices.iter().map(|indices| indices.as_ref()) };
        let mut seen_indices = ::alloc::collections::BTreeSet::new();
        for indices in index_iter {
            index_count += indices.len();
            assert!(!indices.is_empty(), "found empty indices");
            assert!(crate::util::is_sorted(indices), "found unsorted indices");
            assert!(
                indices.last().unwrap() < &self.pairs.len(),
                "found out of bound index for entries in indices"
            );

            let Bucket { hash, key, .. } = &self.pairs[*indices.first().unwrap()];
            let count = self.pairs.iter().filter(|b| b.key_ref() == key).count();
            assert_eq!(
                count,
                indices.len(),
                "indices do not contain indices to all of the pairs with this key"
            );
            for i in indices.as_slice() {
                assert!(seen_indices.insert(i), "found duplicate index in `indices`");
                let Bucket {
                    hash: hash2,
                    key: key2,
                    ..
                } = &self.pairs[*i];
                assert_eq!(
                    hash, hash2,
                    "indices of single entry point to different hashes"
                );
                assert!(
                    key == key2,
                    "indices of single entry point to different keys"
                );
            }
        }

        assert_eq!(
            self.pairs.len(),
            index_count,
            "mismatch between pairs and indices count"
        );

        // This is probably unnecessary, if the number of indices in self.indices
        // equals the number of pairs in self.paris and all of the indices in
        // self.indices are unique then there must be an index for each pair in self.pairs.
        for (i, Bucket { hash, .. }) in self.pairs.iter().enumerate() {
            let indices = self.indices.get(hash.get(), eq_index(i));
            assert!(
                indices.is_some(),
                "expected a pair to have a matching entry in indices table"
            );
        }
    }

    #[inline(always)]
    #[cfg(any(not(debug_assertions), not(feature = "more_debug_assertions")))]
    fn debug_assert_indices(&self, _indices: &[usize]) {}

    #[cfg(all(debug_assertions, feature = "more_debug_assertions"))]
    #[track_caller]
    fn debug_assert_indices(&self, indices: &[usize]) {
        assert!(crate::util::is_sorted_and_unique(indices));
        assert!(!indices.is_empty());
        assert!(indices.last().unwrap_or(&0) < &self.pairs.len());
    }
}

impl<'a, K, V> IndexMultimapCoreRefMut<'a, K, V> {
    fn new(map: &'a mut IndexMultimapCore<K, V>) -> Self {
        let indices = &mut map.indices;
        let pairs = &mut map.pairs;
        unsafe { Self::new_unchecked(indices, pairs) }
    }

    /// # Safety
    ///
    /// * `indices` and `pairs` must be from the same `IndexMultimapCore`
    unsafe fn new_unchecked(
        indices: &'a mut IndicesTable,
        pairs: &'a mut Vec<Bucket<K, V>>,
    ) -> Self {
        Self { indices, pairs }
    }

    #[inline]
    pub(super) fn len_keys(&self) -> usize {
        self.as_ref().len_keys()
    }

    #[inline]
    pub(super) fn len_pairs(&self) -> usize {
        self.as_ref().len_pairs()
    }

    #[inline]
    pub(super) fn capacity_keys(&self) -> usize {
        self.as_ref().capacity_keys()
    }

    #[inline]
    pub(super) fn capacity_pairs(&self) -> usize {
        self.as_ref().capacity_pairs()
    }

    #[inline(always)]
    fn reborrow(&mut self) -> IndexMultimapCoreRefMut<'_, K, V> {
        unsafe { IndexMultimapCoreRefMut::new_unchecked(self.indices, self.pairs) }
    }

    fn as_ref(&self) -> IndexMultimapCoreRef<'_, K, V> {
        unsafe { IndexMultimapCoreRef::new_unchecked(self.indices, self.pairs) }
    }

    pub(super) fn get_all_mut_by_index(&mut self, index: usize) -> SubsetMut<'_, K, V>
    where
        K: Eq,
    {
        self.reborrow().into_all_mut_by_index(index)
    }

    pub(super) fn into_all_mut_by_index(self, index: usize) -> SubsetMut<'a, K, V>
    where
        K: Eq,
    {
        let bucket = &self.pairs[index];
        let hash = bucket.hash;
        let key = &bucket.key;
        let eq = equivalent(key, &self.pairs);
        if let Some(indices) = self.indices.find(hash.get(), eq) {
            self.as_ref().debug_assert_indices(indices);
            SubsetMut::new(self.pairs, indices.as_unique_slice())
        } else {
            SubsetMut::empty()
        }
    }

    pub(super) fn insert_at_into(
        mut self,
        index: usize,
        hash: HashValue,
        key: K,
        value: V,
    ) -> Result<(usize, OccupiedEntry<'a, K, V>), Self>
    where
        K: Eq,
    {
        if index > self.pairs.len() {
            return Err(self);
        }

        // this needs to be the first thing we do
        unsafe { self.increment_indices(index, self.len_pairs()) };

        let bucket = Bucket { hash, key, value };
        self.pairs.insert(index, bucket);

        // need to insert the pair first, so that the indices match up
        let key = &self.pairs[index].key;
        let indices = self.indices.entry(
            hash.get(),
            equivalent(key, &self.pairs),
            get_hash(&self.pairs),
        );

        let (index, entry) = match indices {
            hash_table::Entry::Occupied(mut entry) => {
                let index = entry.get_mut().insert_sorted(index);
                (index, entry)
            }
            hash_table::Entry::Vacant(entry) => (0, entry.insert(Indices::one(index))),
        };

        Ok((index, OccupiedEntry::new(self.pairs, entry, hash, None)))
    }

    /// Remove an entry by shifting all entries that follow it
    pub(super) fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)>
    where
        K: Eq,
    {
        match self.pairs.get(index) {
            Some(pair) => {
                erase_index(&mut self.indices, pair.hash, index);
                unsafe { self.decrement_indices(index + 1, self.pairs.len(), 1) };
                // Use Vec::remove to actually remove the entry.
                let Bucket { key, value, .. } = self.pairs.remove(index);
                self.as_ref().debug_assert_invariants();
                Some((key, value))
            }
            None => None,
        }
    }

    /// Remove an entry by swapping it with the last
    pub(super) fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)>
    where
        K: Eq,
    {
        match self.pairs.get(index) {
            Some(entry) => {
                // cannot panic as is pair was in self.pairs, then index must be in self.indices
                erase_index(&mut self.indices, entry.hash, index);
                // use swap_remove, but then we need to update the index that points
                // to the other entry that has to move
                let removed_pair = self.pairs.swap_remove(index); // cannot panic

                // correct index that points to the entry that had to swap places
                if let Some(moved_pair) = self.pairs.get(index) {
                    // was not last element
                    // examine new element in `index` and find it in indices
                    let last = self.pairs.len();
                    // Cannot panic:
                    //  1. moved_pair was last, so moved_pair.hash and last must match in self.indices
                    //  2. `index` was erased from the self.indices before
                    update_index_last(&mut self.indices, moved_pair.hash, last, index);
                }

                self.as_ref().debug_assert_invariants();
                Some((removed_pair.key, removed_pair.value))
            }
            None => None,
        }
    }

    /// # Panics
    ///
    /// * Panics if `from` or `to` are out of bounds, `>= self.pairs.len()`.
    pub(super) fn move_index(&mut self, from: usize, to: usize)
    where
        K: Eq,
    {
        if from == to {
            return;
        }

        let len = self.pairs.len();
        assert!(
            from < len,
            "index `from={from}` is out of bounds for slice of length `{len}`"
        );
        assert!(
            to < len,
            "index `to={to}` is out of bounds for slice of length `{len}`"
        );

        let from_hash = self.pairs[from].hash;
        // Use a sentinel index so other indices don't collide.
        //
        // Cannot panic:
        //  1. from_hash and from must be in indices, as self is in sync
        //  2. usize::MAX cannot be in self.indices as index cannot be so large
        update_index(&mut self.indices, from_hash, from, usize::MAX);

        // Update all other indices and rotate the entry positions.
        #[allow(clippy::comparison_chain)]
        if from < to {
            // [head] [from]     [from + 1] ... [to - 1] [to]   [tail]
            // [head] [from + 1] [from + 2] ... [to]     [from] [tail]
            // so every index in range from+1..=to must be decremented by 1 and `from` changed to `to`
            //
            // SAFETY: map in synced up later
            //
            // Cannot panic:
            //  1. from < to is checked above
            //  1. from and to are checked to be < self.pairs.len(), thus from + 1 and to + 1 <= self.pairs.len(),
            //    eg from + 1 and to + 1 are in bounds for slicing
            //  2. start-amount=from +1-1 = from, is not in self.indices as it's changed to usize::MAX
            //  3. self.indices and self.pairs are synced up except the from index as we haven't touched them otherwise
            unsafe { self.decrement_indices(from + 1, to + 1, 1) };
            self.pairs[from..=to].rotate_left(1);
        } else if to < from {
            // [head] [to]   [to + 1] ... [from - 1] [from]     [tail]
            // [head] [from] [to]     ... [from - 2] [from - 1] [tail]
            // so every index in range to..=from-1 must be incremented by 1 and `from` changed to `to`
            //
            // SAFETY: same as above branch
            //
            // Cannot panic:
            //  1. to < from is checked above
            //  ... other points from above branch apply
            unsafe { self.increment_indices(to, from) };
            self.pairs[to..=from].rotate_right(1);
        }

        // Change the sentinel index to its final position,
        // eg sync up self.pairs and self.indices
        //
        // Cannot panic
        //  1. `from_hash` and `usize::MAX` must be in `self.indices` as we just
        //     updated it above and haven't changed it between
        //  2. `to` cannot be in `self.indices` as `decrement_indices` or
        //     `increment_indices` removed that index
        update_index_last(&mut self.indices, from_hash, usize::MAX, to);

        self.as_ref().debug_assert_invariants();
    }

    /// # Panics
    ///
    /// * Panics if `a` or `b` are out of bounds, `>= self.pairs.len()`.
    pub(super) fn swap_indices(&mut self, a: usize, b: usize)
    where
        K: Eq,
    {
        if a == b {
            return;
        }

        let a_item = &self.pairs[a];
        let b_item = &self.pairs[b];

        // step 1: swap indices
        if a_item.key == b_item.key {
            // both indices belong to the same entry,
            // if we swap entries, indices are still correct
            // nothing to do
        } else {
            /// get_many_mut cannot panic as we checked for a and b key equivalency above
            const ITEMS: usize = 2;
            match self.indices.get_many_mut::<ITEMS>(
                [a_item.hash.get(), b_item.hash.get()],
                |i, k| match i {
                    0 => self.pairs[k[0]].key == a_item.key,
                    1 => self.pairs[k[0]].key == b_item.key,
                    ITEMS.. => unreachable!(
                        "i is the index of item in given hashes array, must be <= {ITEMS}"
                    ),
                },
            ) {
                [Some(a_indices), Some(b_indices)] => {
                    let index_a = a_indices.binary_search(&a).expect("index a not found");
                    let index_b = b_indices.binary_search(&b).expect("index b not found");

                    a_indices.replace(index_a, b);
                    b_indices.replace(index_b, a);
                }
                _ => unreachable!(
                    "indices a and b must be in self.indices as they are in self.pairs"
                ),
            }
        }

        // step 2: swap pairs

        // Cannot panic: a and b are already used to index into self.pairs,
        // and self.pairs has not changed in between
        self.pairs.swap(a, b);

        self.as_ref().debug_assert_invariants();
    }

    /// Rebuild the hash table
    ///
    /// # Panics
    ///
    /// Panic if there is not enough capacity in `self.indices` to hold all `self.pairs`'s unique keys.
    ///
    /// This can only happen, if a new key was added to `self.pairs` but not to `self.indices` prior to calling this method.
    fn rebuild_hash_table(&mut self)
    where
        K: Eq,
    {
        self.indices.clear();
        // * cannot panic because self.indices and passed current_pairs are empty
        // * can panic for not enough capacity, if new pair was added to self.pairs but not to self.indices
        insert_bulk_no_grow(self.indices, &[], self.pairs);
    }

    /// Decrements indexes by variable amount determined by how many ranges have been before it,
    /// (eg how many items were removed before it).
    ///
    /// Say we need to remove indices [2, 4, 7], then
    /// * indices [0, 1], don't need decrementing
    /// * indices [3] need to be decremented by 1
    /// * indices [5, 6], need to be decremented by 2
    /// * indices [8, ...] need to be decremented by 3.
    ///
    /// The result should be a contiguous range of indices `0..self.pairs.len()-indices.len()` in `self.indices`.
    ///
    /// This is achieved by calling this method with indices `[2, 4, 7]`.
    ///
    /// # Safety
    ///
    /// * The caller must ensure that after calling this function they fix up the map again, that is
    ///   sync `self.indices` and `self.pairs` again.
    ///
    /// # Panics
    ///
    /// 1. may panic if any of the `indices` is in `self.indices`
    ///
    ///    None of the `indices` should exist in `self.indices`.
    ///    Otherwise `self.indices` will include duplicate indices which violates our invariants.
    ///    It will be very difficult to fix after or worse this method can panic.
    ///
    /// 2. may panic if `self.indices` and `self.pairs` are out of sync for indices not in `indices`
    ///
    ///    If `self.indices` and `self.pairs` are out of sync, then this method could update random
    ///    indices, decrement same index more than once, create duplicate indices and/or panic.
    ///
    /// If this method panics, any number of indices may have been updated, and rest remaining as was.
    unsafe fn decrement_indices_batched(&mut self, indices: &Indices) {
        match self.indices.len() {
            0 => {}
            1 => {
                // if there is only 1 key left in the map,
                // then the indices must be sequential 0..len
                for indices in self.indices.iter_mut() {
                    // don't use self.pairs.len() because values
                    // to be removed may not be removed yet
                    let len = indices.len();
                    indices.clear();
                    indices.extend(UniqueSortedIter::from_range(0..len));
                }
            }
            _ if indices.len() == 1 => {
                // fastest if removing only one index
                // Shift the tail after the last item
                let i = *indices.last().unwrap();
                if i < self.pairs.len() {
                    // SAFETY: caller promises to sync map up
                    // Cannot panic:
                    //   1. for out of bound indexing, `i+1 <= self.pairs.len()`
                    // can panic for other listed reasons
                    unsafe { self.decrement_indices(i + 1, self.pairs.len(), 1) };
                    // may panic
                }
            }
            _ => {
                // if removing more than 1 index it's faster to iterate over indices in the map once
                // and loop over removed indices multiple times, rather then other way around.
                let first = *indices.first().unwrap();
                let last = *indices.last().unwrap();

                for indices_in_map in self.indices.iter_mut() {
                    for i in unsafe { indices_in_map.as_mut_slice() } {
                        if *i < first {
                            continue;
                        } else if *i > last {
                            *i -= indices.len();
                        } else {
                            let offset = indices.partition_point(|a| *a < *i);
                            *i -= offset;
                        }
                    }
                }
            }
        }
    }

    /// Decrement all indices in the range `start..end` by `amount`.
    ///
    /// This will leave a hole in the indices table (indices `end-amount..end` will be missing)
    /// and `self.pairs` and `self.indices` will be out of sync.
    /// It's up to the caller to sync them up again after.
    ///
    /// # Safety
    ///
    /// * The caller must ensure that after calling this function they fix up the map again, that is
    ///   sync `self.indices` and `self.pairs` again.
    ///
    /// # Panics
    ///
    /// 1. if `start > end` or `end > self.pairs.len()`, that is they are invalid for slicing `self.pairs`
    ///
    /// 2. may panic if any of the indices `start-amount..start` is in `self.indices`
    ///
    ///    None of the indices `start-amount..start` should exist in `self.indices`.
    ///    Otherwise `self.indices` will include duplicate indices which violates our invariants.
    ///    It will be very difficult to fix after or worse this method can panic.
    ///
    /// 3. may panic if `self.indices` and `self.pairs` are out of sync for indices in range `start..end`
    ///
    ///    If `self.indices` and `self.pairs` are out of sync, then this method could update random
    ///    indices, decrement same index more than once, create duplicate indices and/or panic.
    ///
    /// If this method panics, any number of indices in range `start..end` may have been updated,
    /// and rest remaining as was.
    unsafe fn decrement_indices(&mut self, start: usize, end: usize, amount: usize) {
        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        let shifted_pairs = &self.pairs[start..end]; // may panic
        if shifted_pairs.len() > self.indices.len() / 2 {
            // Shift all indices in range.
            for indices in &mut *self.indices {
                for i in unsafe { indices.as_mut_slice() } {
                    if *i >= end {
                        // early break as we go past end and our indices are sorted
                        break;
                    } else if start <= *i {
                        *i -= amount;
                    }
                }
            }
        } else {
            // IDEA: if there are runs of equivalent keys in the range,
            // then we could update them in one go and don't need to find the entry for every item

            // Find each entry in range to shift its index.
            for (i, entry) in (start..end).zip(shifted_pairs) {
                update_index(self.indices, entry.hash, i, i - amount); // may panic
            }
        }
    }

    /// Increment all indices in the range `start..end` by `1`.
    ///
    /// This will leave a hole in the indices table (index `start` will be missing)
    /// and `self.pairs` and `self.indices` will be out of sync.
    /// It's up to the caller to sync them up again after.
    ///
    /// # Safety
    ///
    /// * The caller must ensure that after calling this function they fix up the map again, that is
    ///   sync `self.indices` and `self.pairs` again.
    ///
    /// # Panics
    ///
    /// 1. if `start > end` or `end > self.pairs.len()`, that is they are invalid for slicing `self.pairs`
    ///
    /// 2. may panic if index `end` is in `self.indices`
    ///
    ///    Index `end` should not exist in `self.indices`.
    ///    Otherwise `self.indices` will include duplicate indices which violates our invariants.
    ///    It will be very difficult to fix after or worse this method can panic.
    ///
    /// 3. may panic if `self.indices` and `self.pairs` are out of sync for indices in range `start..end`
    ///
    ///    If `self.indices` and `self.pairs` are out of sync, then this method could update random
    ///    indices, increment same index more than once, create duplicate indices and/or panic.
    ///
    /// If this method panics, any number of indices in range `start..end` may have been updated,
    /// and rest remaining as was.
    unsafe fn increment_indices(&mut self, start: usize, end: usize) {
        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        let shifted_pairs = &self.pairs[start..end]; // may panic
        if shifted_pairs.len() > self.indices.len() / 2 {
            // Shift all indices in range.
            for indices in self.indices.iter_mut() {
                for i in unsafe { indices.as_mut_slice() } {
                    if start <= *i && *i < end {
                        // cannot overflow, if end == usize::MAX and i < end,
                        // then i+1 == usize::MAX at most
                        *i += 1;
                    }
                }
            }
        } else {
            // Find each entry in range to shift its index, updated in reverse so
            // we never have duplicated indices that might have a hash collision.
            for (i, entry) in (start..end).zip(shifted_pairs).rev() {
                update_index(&mut self.indices, entry.hash, i, i + 1); // may panic
            }
        }
    }
}

#[test]
fn assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<IndexMultimapCore<i32, i32>>();
    assert_send_sync::<Entry<'_, i32, i32>>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HashValue;

    // #[test]
    // fn insert_bulk_no_grow_test() {
    //     fn bucket(k: usize) -> Bucket<usize, usize> {
    //         Bucket {
    //             hash: HashValue(k),
    //             key: k,
    //             value: k,
    //         }
    //     }

    //     let mut table = IndicesTable::with_capacity(10);
    //     let mut pairs = Vec::new();

    //     let new_pairs = vec![bucket(1)];
    //     insert_bulk_no_grow(&mut table, &pairs, &new_pairs);
    //     pairs.extend(new_pairs);
    //     assert_eq!(DebugIndices(&table), &[[0].as_slice()]);

    //     let new_pairs = vec![bucket(1), bucket(1)];
    //     insert_bulk_no_grow(&mut table, &pairs, &new_pairs);
    //     pairs.extend(new_pairs);
    //     assert_eq!(DebugIndices(&table), &[[0, 1, 2].as_slice()]);

    //     let new_pairs = vec![bucket(2), bucket(1)];
    //     insert_bulk_no_grow(&mut table, &pairs, &new_pairs);
    //     assert_eq!(
    //         DebugIndices(&table),
    //         &[[0, 1, 2, 4].as_slice(), [3].as_slice()]
    //     );
    //     pairs.extend(new_pairs);
    // }

    #[test]
    fn update_index_test() {
        let mut map = IndexMultimapCore::<i32, i32>::new();
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

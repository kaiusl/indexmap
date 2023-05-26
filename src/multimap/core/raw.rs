#![allow(unsafe_code)]
//! This module encapsulates the `unsafe` access to `hashbrown::raw::RawTable`,
//! mostly in dealing with its bucket "pointers".

use alloc::format;
use alloc::vec::Vec;
use core::fmt;
use core::iter::{Copied, FusedIterator};
use core::mem;
use core::ops::{self, Range};
use core::ptr::{self, addr_of, addr_of_mut, NonNull};
use core::slice;

use crate::multimap::core::update_index_last;
use crate::TryReserveError;
use crate::{
    map::Slice,
    multimap::{SubsetIter, SubsetIterMut, SubsetKeys, SubsetValues, SubsetValuesMut},
    util::{is_sorted_and_unique, is_unique_sorted, simplify_range},
    Equivalent,
};

use super::super::{Subset, SubsetMut};
use super::{
    equivalent, update_index, Bucket, Entry, HashValue, IndexMultimapCore, Unique, UniqueSorted,
    VacantEntry,
};
use hashbrown::raw::RawTable;

type RawBucket<Indices> = hashbrown::raw::Bucket<Indices>;

/// Inserts multiple pairs into a raw table without reallocating.
///
/// ***Panics*** if there is not sufficient capacity already.
///
/// `current_pairs` must not contain `new_pairs`
pub(super) fn insert_bulk_no_grow<K, V, Indices>(
    indices: &mut RawTable<UniqueSorted<Indices>>,
    current_pairs: &[Bucket<K, V>],
    new_pairs: &[Bucket<K, V>],
) where
    K: Eq,
    Indices: IndexStorage,
{
    let current_num_pairs = current_pairs.len();
    debug_assert_eq!(
        // SAFETY: we have mutable reference to the table and buckets don't escape
        unsafe { indices.iter().map(|a| a.as_ref().as_slice().len()) }.sum::<usize>(),
        current_num_pairs,
        "mismatch in the number of current pairs"
    );

    for (pair_index, pair) in (current_num_pairs..).zip(new_pairs) {
        match indices.get_mut(pair.hash.get(), |indices| {
            let i = indices[0];
            if i < current_num_pairs {
                current_pairs[i].key == pair.key
            } else {
                new_pairs[i - current_num_pairs].key == pair.key
            }
        }) {
            Some(indices) => indices.push(pair_index),
            None => {
                // Need to check every time as we don't know how many unique keys are in the `new_pairs`
                // Cannot use `new_pairs.len()` as regular IndexMap does.
                assert!(indices.capacity() > indices.len());
                // SAFETY: we asserted that sufficient capacity exists for this pair and that this pair is not in the table
                unsafe {
                    indices.insert_no_grow(pair.hash.get(), UniqueSorted::one(pair_index));
                }
            }
        }
    }
}

#[inline]
/// Binds the mutable borrow of T to the borrow of bucket.
unsafe fn bucket_as_mut<T>(bucket: &mut RawBucket<T>) -> &mut T {
    unsafe { bucket.as_mut() }
}

#[inline]
pub(super) fn erase_index<Indices>(
    table: &mut RawTable<UniqueSorted<Indices>>,
    hash: HashValue,
    index: usize,
) where
    Indices: IndexStorage,
{
    // Cache the index in indices as we find the bucket,
    // so that we don't need to find it again for indices.remove below
    let mut index_in_indices = None;
    let eq_index = |indices: &UniqueSorted<Indices>| {
        debug_assert!(
            is_sorted_and_unique(indices.as_slice()),
            "expected indices to be sorted and unique"
        );
        match indices.binary_search(&index) {
            Ok(i) => {
                index_in_indices = Some(i);
                true
            }
            Err(_) => false,
        }
    };
    match table.find(hash.get(), eq_index) {
        Some(mut bucket) => {
            // SAFETY: we have &mut to table and thus to the bucket
            let indices = unsafe { bucket_as_mut(&mut bucket) };
            if indices.len() == 1 {
                // SAFETY: the bucket cannot escape as &mut to indices is dropped
                unsafe { table.erase(bucket) };
            } else {
                indices.remove(index_in_indices.expect("expected to find index"));
            }
        }
        None => unreachable!("pair for index not found"),
    }
}

/// Erase the index but assumes that it's last in the key's indices.
/// Avoids the binary_searches of generic erase_index above.
///
/// Used by .pop() method, since we keep indices sorted and thus the index we
/// need to remove must be in the last position.
#[inline]
pub(super) fn erase_index_last<Indices>(
    table: &mut RawTable<UniqueSorted<Indices>>,
    hash: HashValue,
    index: usize,
) where
    Indices: IndexStorage,
{
    match table.find(hash.get(), super::eq_index_last(index)) {
        Some(mut bucket) => {
            // SAFETY: we have &mut to table and thus to the bucket
            let indices = unsafe { bucket_as_mut(&mut bucket) };
            debug_assert_eq!(*indices.last().unwrap(), index);
            if indices.len() == 1 {
                // SAFETY: the bucket cannot escape as &mut to indices is dropped
                unsafe { table.erase(bucket) };
            } else {
                indices.pop();
            }
        }
        None => unreachable!("pair for index not found"),
    }
}

pub(crate) struct DebugIndices<'a, Indices>(pub &'a RawTable<UniqueSorted<Indices>>);
impl<Indices> fmt::Debug for DebugIndices<'_, Indices>
where
    Indices: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use more readable output - each bucket is always one line
        // SAFETY: we're not letting any of the buckets escape this function
        let indices = unsafe { self.0.iter().map(|bucket| bucket.as_ref()) }
            .map(|a| format!("{:?}", a.as_inner()));
        f.debug_list().entries(indices).finish()
    }
}

impl<Indices> PartialEq<&[&[usize]]> for DebugIndices<'_, Indices>
where
    Indices: IndexStorage,
{
    fn eq(&self, other: &&[&[usize]]) -> bool {
        let mut indices = unsafe {
            self.0
                .iter()
                .map(|bucket| bucket.as_ref())
                .map(|a| a.as_slice())
        }
        .collect::<Vec<_>>();
        indices.sort_by(|a, b| a[0].cmp(&b[0]));

        for (&this, &other) in indices.iter().zip(*other) {
            if this != other {
                return false;
            }
        }
        true
    }
}

impl<K, V, Indices> IndexMultimapCore<K, V, Indices>
where
    Indices: IndexStorage,
{
    /// Sweep the whole table to erase indices start..end
    pub(super) fn erase_indices_sweep(&mut self, start: usize, end: usize) {
        // SAFETY: we're not letting any of the buckets escape this function
        unsafe {
            let offset = end - start;
            for mut bucket in self.indices.iter() {
                let indices = bucket_as_mut(&mut bucket);
                indices.retain(|i| {
                    if *i >= end {
                        *i -= offset;
                        true
                    } else {
                        *i < start
                    }
                });
                if indices.len() == 0 {
                    self.indices.erase(bucket);
                }
            }
        }
    }

    pub(crate) fn entry(&mut self, hash: HashValue, key: K) -> Entry<'_, K, V, Indices>
    where
        K: Eq,
    {
        let eq = equivalent(&key, &self.pairs);
        match self.indices.find(hash.get(), eq) {
            // SAFETY: The entry is created with a live raw bucket, at the same time
            // we have a &mut reference to the map, so it can not be modified further.
            Some(raw_bucket) => Entry::Occupied(unsafe {
                OccupiedEntry::new_unchecked(self, raw_bucket, hash, Some(key))
            }),
            None => Entry::Vacant(VacantEntry {
                map: self,
                hash,
                key,
            }),
        }
    }

    pub(super) fn indices_mut(&mut self) -> impl Iterator<Item = &mut UniqueSorted<Indices>> {
        // SAFETY: we're not letting any of the buckets escape this function,
        // only the item references that are appropriately bound to `&mut self`.
        unsafe { self.indices.iter().map(|bucket| bucket.as_mut()) }
    }

    /// Return the raw bucket for the given index
    fn find_index(&self, index: usize) -> RawBucket<UniqueSorted<Indices>> {
        // We'll get a "nice" bounds-check from indexing `self.pairs`,
        // and then we expect to find it in the table as well.
        let hash = self.pairs[index].hash.get();
        self.indices
            .find(hash, |i| i.contains(&index))
            .expect("index not found")
    }

    pub(crate) fn swap_indices(&mut self, a: usize, b: usize)
    where
        K: Eq,
    {
        if a == b {
            return;
        }
        // SAFETY: Can't take two `get_mut` references from one table, so we
        // must use raw buckets to do the swap. This is still safe because we
        // are locally sure they won't dangle, and we write them individually.
        unsafe {
            let raw_bucket_a = self.find_index(a);
            let raw_bucket_b = self.find_index(b);
            if core::ptr::eq(raw_bucket_a.as_ptr(), raw_bucket_b.as_ptr()) {
                // both indices belong to the same entry,
                // if we swap entries indices are still correct
                // nothing to do
            } else {
                let raw_bucket_a = raw_bucket_a.as_mut();
                let index_a = raw_bucket_a
                    .iter()
                    .position(|&i| i == a)
                    .expect("index not found");

                let raw_bucket_b = raw_bucket_b.as_mut();
                let index_b = raw_bucket_b
                    .iter()
                    .position(|&i| i == b)
                    .expect("index not found");
                raw_bucket_a.replace(index_a, b);
                raw_bucket_b.replace(index_b, a);

                //replace_sorted(raw_bucket_a, index_a, b);
                //replace_sorted(raw_bucket_b, index_b, a);
            }
        }
        self.pairs.swap(a, b);

        self.debug_assert_invariants();
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.shrink_to(0);
        // SAFETY: we own the RawTable and don't let the bucket's escape
        unsafe { self.indices.iter().for_each(|a| a.as_mut().shrink_to_fit()) };
    }

    /// Decrement all indices in the range `start..end` by `amount`.
    ///
    /// The index `start - amount` should not exist in `self.indices`.
    /// All entries should still be in their original positions.
    pub(crate) fn decrement_indices(&mut self, start: usize, end: usize, amount: usize) {
        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        let shifted_pairs = &self.pairs[start..end];
        if shifted_pairs.len() > self.indices.buckets() / 2 {
            // Shift all indices in range.
            for indices in self.indices_mut() {
                debug_assert!(is_sorted_and_unique(indices.as_slice()));
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
    pub(crate) fn increment_indices(&mut self, start: usize, end: usize) {
        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        let shifted_pairs = &self.pairs[start..end];
        if shifted_pairs.len() > self.indices.buckets() / 2 {
            // Shift all indices in range.
            for indices in self.indices_mut() {
                for i in unsafe { indices.as_mut_slice() } {
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

    pub(crate) fn reverse(&mut self)
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
}

/// A view into an occupied entry in a [`IndexMultimap`].
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: super::Entry
/// [`IndexMultimap`]: super::super::IndexMultimap
pub struct OccupiedEntry<'a, K, V, Indices> {
    // SAFETY:
    //   * The lifetime of the map reference also constrains the raw bucket,
    //     which is essentially a raw pointer into the map indices.
    //   * `bucket` must point into map.indices
    //   * `bucket` must be live at the creation moment
    //   * we must be the only ones with the pointer to `bucket`
    // Thus by also taking &mut map, no one else can modify the map and we are
    // sole modifiers of the bucket.
    // None of our methods directly modify map.indices (except when we consume self),
    // which means that the internal raw table won't reallocate and the bucket
    // must be alive for the lifetime of self.
    map: &'a mut IndexMultimapCore<K, V, Indices>,
    raw_bucket: RawBucket<UniqueSorted<Indices>>,
    hash: HashValue,
    key: Option<K>,
}

// `hashbrown::raw::Bucket` is only `Send`, not `Sync`.
// SAFETY: `&self` only accesses the bucket to read it.
unsafe impl<K: Sync, V: Sync, Indices: Sync> Sync for OccupiedEntry<'_, K, V, Indices> {}

impl<'a, K, V, Indices> OccupiedEntry<'a, K, V, Indices>
where
    Indices: IndexStorage,
{
    /// SAFETY:
    ///   * `bucket` must point into map.indices
    ///   * `bucket` must be live at the creation moment
    ///   * we must be the only ones with the `bucket` pointer
    #[inline]
    pub(super) unsafe fn new_unchecked(
        map: &'a mut IndexMultimapCore<K, V, Indices>,
        bucket: RawBucket<UniqueSorted<Indices>>,
        hash: HashValue,
        key: Option<K>,
    ) -> Self {
        Self {
            map,
            raw_bucket: bucket,
            hash,
            key,
        }
    }

    /// Returns the number of key-value pairs in this entry.
    #[allow(clippy::len_without_is_empty)] // There is always at least one pair
    pub fn len(&self) -> usize {
        self.indices().len()
    }

    /// Appends a new key-value pair to this entry.
    ///
    /// This method will clone the key.
    pub fn insert_append(&mut self, value: V)
    where
        K: Clone,
    {
        let index = self.map.len_pairs();
        let key = self.clone_key();
        self.map.pairs.push(Bucket {
            hash: self.hash,
            key,
            value,
        });

        unsafe { self.raw_bucket.as_mut() }.push(index);

        self.map.debug_assert_indices(self.indices());
    }

    /// Appends a new key-value pair to this entry by taking the owned key.
    ///
    /// This method should only be called once after creation of pair enum.
    /// Panics otherwise.
    pub(super) fn insert_append_take_owned_key(&mut self, value: V) {
        let index = self.map.pairs.len();

        let key = self.key.take().unwrap();
        self.map.pairs.push(Bucket {
            hash: self.hash,
            key,
            value,
        });

        unsafe { self.raw_bucket.as_mut() }.push(index);

        self.map.debug_assert_indices(self.indices());
    }

    /// Gets a reference to the entry's first key in the map.
    ///
    /// Other keys can be accessed through [`get`](Self::get) method.
    ///
    /// Note that this may not be the key that was used to find the pair.
    /// There may be an observable difference if the key type has any
    /// distinguishing features outside of [`Hash`] and [`Eq`], like
    /// extra fields or the memory address of an allocation.
    ///
    /// [`Hash`]: core::hash::Hash
    pub fn key(&self) -> &K {
        &self.map.pairs[self.indices()[0]].key
    }

    fn clone_key(&self) -> K
    where
        K: Clone,
    {
        match self.key.as_ref() {
            Some(k) => k.clone(),
            None => {
                debug_assert_eq!(self.indices().last(), Some(&(self.map.pairs.len() - 1)));
                // The only way we don't have owned key is if we already inserted it.
                // (either by Entry::insert_append or VacantEntry::insert).
                // The key that was used to get this entry thus must be in the last pair
                self.map
                    .pairs
                    .last()
                    .expect("expected occupied pair to have at least one key-value pair")
                    .key
                    .clone()
            }
        }
    }

    /// Returns the indices of all the pairs associated with this entry in the map.
    #[inline]
    pub fn indices(&self) -> &[usize] {
        // SAFETY: we have &mut map keep keeping the bucket stable
        unsafe { self.raw_bucket.as_ref() }.as_slice()
    }

    /// Returns a reference to the `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth(&self, n: usize) -> Option<(usize, &K, &V)> {
        match self.indices().get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.map.as_pairs()[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a reference to the `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        match self.indices().get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.map.as_mut_pairs()[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Return a reference to the first pair in this entry.
    pub fn first(&self) -> (usize, &K, &V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &self.map.as_pairs()[index];
        (index, key, value)
    }

    /// Return a reference to the first pair in this entry.
    pub fn first_mut(&mut self) -> (usize, &K, &mut V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &mut self.map.as_mut_pairs()[index];
        (index, key, value)
    }

    /// Returns a reference to the last pair in this entry.
    pub fn last(&self) -> (usize, &K, &V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &self.map.as_pairs()[index];
        (index, key, value)
    }

    /// Returns a reference to the last pair in this entry.
    pub fn last_mut(&mut self) -> (usize, &K, &mut V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &mut self.map.as_mut_pairs()[index];
        (index, key, value)
    }

    /// Remove all the key-value pairs for this entry and return an iterator
    /// over all the removed pairs.
    ///
    /// Like [`Vec::swap_remove`], the pairs are removed by swapping them with the
    /// last element of the map and popping them off.
    /// **This perturbs the position of what used to be the last element!**
    ///
    /// # Laziness
    ///
    /// To avoid any unnecessary allocations the pairs are actually removed when
    /// the returned iterator is consumed.
    /// For convenience, dropping the iterator will remove and drop
    /// all the remaining items that are meant to be removed.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without
    /// being dropped (is *leaked*),
    /// the map may have lost and leaked elements arbitrarily,
    /// including pairs not associated with this entry.
    pub fn swap_remove(self) -> SwapRemove<'a, K, V, Indices>
    where
        K: Eq,
    {
        // SAFETY: This is safe because it can only happen once (self is consumed)
        // and bucket has not been removed from the map.indices
        unsafe { SwapRemove::new_raw(self.map, self.raw_bucket) }
    }

    /// Remove all the key-value pairs for this entry and
    /// return an iterator over all the removed items,
    /// or [`None`] if the `key` was not in the map.
    ///
    /// Like [`Vec::remove`], the pairs are removed by shifting all of the
    /// elements that follow them, preserving their relative order.
    /// **This perturbs the index of all of those elements!**
    ///
    /// # Laziness
    ///
    /// To avoid any unnecessary allocations the pairs are removed when iterator is
    /// consumed. For convenience, dropping the iterator will remove and drop
    /// all the remaining items that are meant to be removed.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without
    /// being dropped (is *leaked*),
    /// the map may have lost and leaked elements arbitrarily,
    /// including pairs not associated with this entry.
    pub fn shift_remove(self) -> ShiftRemove<'a, K, V, Indices>
    where
        K: Eq,
    {
        // SAFETY: This is safe because it can only happen once (self is consumed)
        // and bucket has not been removed from the map.indices
        unsafe { ShiftRemove::new_raw(self.map, self.raw_bucket) }
    }

    /// Returns an iterator over all the pairs in this entry.
    pub fn iter(&self) -> SubsetIter<'_, K, V, Copied<slice::Iter<'_, usize>>> {
        SubsetIter::new(
            &self.map.pairs,
            // SAFETY: we have &mut map keep keeping the bucket stable
            unsafe { self.raw_bucket.as_ref() }.iter().copied(),
        )
    }

    /// Returns a mutable iterator over all the pairs in this entry.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V, Copied<slice::Iter<'_, usize>>> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        let indices_iter = Unique::from(indices.slice_iter());
        SubsetIterMut::new(&mut self.map.pairs, indices_iter.copied())
    }

    /// Returns an iterator over all the keys in this entry.
    ///
    /// Note that the iterator yields one key for each pair which are all equivalent.
    /// But there may be an observable differences if the key type has any
    /// distinguishing features outside of [`Hash`] and [`Eq`], like
    /// extra fields or the memory address of an allocation.
    pub fn keys(&self) -> SubsetKeys<'_, K, V, Copied<slice::Iter<'_, usize>>> {
        SubsetKeys::new(
            &self.map.pairs,
            // SAFETY: we have &mut map keep keeping the bucket stable
            unsafe { self.raw_bucket.as_ref() }.iter().copied(),
        )
    }

    /// Returns an iterator over all the values in this entry.
    pub fn values(&self) -> SubsetValues<'_, K, V, Copied<slice::Iter<'_, usize>>> {
        SubsetValues::new(
            &self.map.pairs,
            // SAFETY: we have &mut map keep keeping the bucket stable
            unsafe { self.raw_bucket.as_ref() }.iter().copied(),
        )
    }

    /// Returns a mutable iterator over all the values in this entry.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V, Copied<slice::Iter<'_, usize>>> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        debug_assert!(is_unique_sorted(indices.as_slice()));
        debug_assert!(indices.last().unwrap_or(&0) < &self.map.pairs.len());
        let indices_iter = Unique::from(indices.slice_iter());
        SubsetValuesMut::new(&mut self.map.pairs, indices_iter.copied())
    }

    /// Converts into an iterator over all the values in this entry.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V, Copied<slice::Iter<'a, usize>>> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        let indices_iter = Unique::from(indices.slice_iter());
        SubsetValuesMut::new(&mut self.map.pairs, indices_iter.copied())
    }

    /// Returns a slice like construct with all the values associated with this entry in the map.
    pub fn as_subset(&self) -> Subset<'_, K, V, &'_ [usize]> {
        Subset::new(&self.map.pairs, self.indices())
    }

    /// Returns a slice like construct with all values associated with this entry in the map.
    ///
    /// If you need a reference which may outlive the destruction of the
    /// pair, see [`into_mut`](Self::into_mut).
    pub fn as_subset_mut(&mut self) -> SubsetMut<'_, K, V, &'_ [usize]> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetMut::new(&mut self.map.pairs, indices.into())
    }

    /// Converts into  a slice like construct with all the values associated with
    /// this pair in the map, with a lifetime bound to the map itself.
    pub fn into_subset_mut(self) -> SubsetMut<'a, K, V, &'a [usize]> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetMut::new(&mut self.map.pairs, indices.into())
    }
}

impl<'a, K, V, Indices> IntoIterator for &'a OccupiedEntry<'_, K, V, Indices>
where
    Indices: IndexStorage,
{
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V, Copied<slice::Iter<'a, usize>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, Indices> IntoIterator for &'a mut OccupiedEntry<'_, K, V, Indices>
where
    Indices: IndexStorage,
{
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V, Copied<slice::Iter<'a, usize>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, K, V, Indices> IntoIterator for OccupiedEntry<'a, K, V, Indices>
where
    Indices: IndexStorage,
{
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V, Copied<slice::Iter<'a, usize>>>;

    fn into_iter(self) -> Self::IntoIter {
        let indices = unsafe { self.raw_bucket.as_ref() };
        debug_assert!(is_unique_sorted(indices.as_slice()));
        debug_assert!(indices.last().unwrap_or(&0) < &self.map.pairs.len());
        let indices_iter = Unique::from(indices.slice_iter());
        SubsetIterMut::new(&mut self.map.pairs, indices_iter.copied())
    }
}

impl<'a, K, V, Indices> VacantEntry<'a, K, V, Indices> {
    pub(super) fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V, Indices>
    where
        Indices: IndexStorage,
    {
        let (_, bucket) = self.map.push(self.hash, self.key, value);
        if cfg!(debug_assertions) {
            let indices = unsafe { bucket.as_ref() };
            self.map.debug_assert_indices(indices.as_slice());
        }

        // SAFETY: push returns a live, valid bucket from the self.map
        unsafe { OccupiedEntry::new_unchecked(self.map, bucket, self.hash, None) }
    }
}

/// An iterator that shift removes pairs from [`IndexMultimap`].
///
/// This struct is created by [`IndexMultimap::shift_remove`] and
/// [`OccupiedEntry::shift_remove`], see their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::shift_remove`]: crate::IndexMultimap::shift_remove
pub struct ShiftRemove<'a, K, V, Indices>
where
    // Needed by Drop impl
    Indices: IndexStorage,
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
    map: &'a mut IndexMultimapCore<K, V, Indices>,
    /// map.indices we took ownership of
    indices_table: RawTable<UniqueSorted<Indices>>,
    /// The index of pair that was removed previously.
    prev_removed_idx: Option<usize>,
    /// The number of items that have been removed thus far.
    del: usize,
    /// The original length of `vec` prior to removing.
    orig_len: usize,
    /// The indices that will be removed.
    indices_to_remove: UniqueSorted<Indices::IntoIter>,
}

impl<'a, K: 'a, V: 'a, Indices: 'a> ShiftRemove<'a, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
    pub(super) fn new<Q>(
        map: &'a mut IndexMultimapCore<K, V, Indices>,
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
        map: &'a mut IndexMultimapCore<K, V, Indices>,
        bucket: RawBucket<UniqueSorted<Indices>>,
    ) -> Self {
        unsafe {
            let indices = map.indices.remove(bucket);
            Self::new_unchecked(map, indices)
        }
    }

    /// # Safety
    ///
    /// See the comment in the type definition for safety.
    unsafe fn new_unchecked(
        map: &'a mut IndexMultimapCore<K, V, Indices>,
        indices: UniqueSorted<Indices>,
    ) -> Self {
        debug_assert!(indices.is_empty() || *indices.last().unwrap() < map.pairs.len());

        map.decrement_indices_batched(indices.as_slice());
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
        map: &mut IndexMultimapCore<K, V, Indices>,
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

impl<K, V, Indices> Iterator for ShiftRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
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

impl<K, V, Indices> ExactSizeIterator for ShiftRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
    fn len(&self) -> usize {
        self.indices_to_remove.len()
    }
}

impl<K, V, Indices> FusedIterator for ShiftRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
}

impl<K, V, Indices> Drop for ShiftRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
    fn drop(&mut self) {
        struct Guard<'a, 'b, K, V, Indices>(&'a mut ShiftRemove<'b, K, V, Indices>)
        where
            K: Eq,
            Indices: IndexStorage;

        impl<'a, 'b, K, V, Indices> Drop for Guard<'a, 'b, K, V, Indices>
        where
            K: Eq,
            Indices: IndexStorage,
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

impl<'a, K, V, Indices> fmt::Debug for ShiftRemove<'a, K, V, Indices>
where
    K: fmt::Debug + Eq,
    V: fmt::Debug,
    Indices: fmt::Debug + IndexStorage,
    Indices::IntoIter: fmt::Debug + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if cfg!(feature = "test_debug") {
            f.debug_struct("ShiftRemove")
                .field("map", &self.map)
                .field("indices_table", &DebugIndices(&self.indices_table))
                .field("prev_removed_idx", &self.prev_removed_idx)
                .field("del", &self.del)
                .field("orig_len", &self.orig_len)
                .field("indices_to_remove", &self.indices_to_remove)
                .finish()
        } else {
            let iter = self.indices_to_remove.clone().map(|i| {
                let bucket = &self.map.pairs[i];
                (i, &bucket.key, &bucket.value)
            });
            f.debug_struct("ShiftRemove").field("left", &iter).finish()
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
pub struct SwapRemove<'a, K, V, Indices>
where
    // Needed by Drop impl
    Indices: IndexStorage,
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
    map: &'a mut IndexMultimapCore<K, V, Indices>,
    indices_table: RawTable<UniqueSorted<Indices>>,
    /// The original length of `map.pairs` prior to removing.
    orig_len: usize,
    /// Indices that need to be removed from `map.pairs`.
    indices_to_remove: UniqueSorted<Indices>,
    /// `0..indices.len()`, used to index into `self.indices` to get indices to remove
    iter_forward: Range<usize>,
    /// `0..indices.len() - 1`, used to index into `self.indices` backwards to determine which index to swap with
    iter_backward: Range<usize>,
    /// `0..map.pairs.len()`, used to determine which index to swap with
    total_iter: Range<usize>,
    /// `indices_to_remove[i]` where `i` is the previous index yielded by `self.iter_backward`,
    /// used to determine which index to swap with
    prev_backward: usize,
}

impl<'a, K, V, Indices> SwapRemove<'a, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
    pub(super) fn new<Q>(
        map: &'a mut IndexMultimapCore<K, V, Indices>,
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
        map: &'a mut IndexMultimapCore<K, V, Indices>,
        bucket: RawBucket<UniqueSorted<Indices>>,
    ) -> Self {
        unsafe {
            let indices = map.indices.remove(bucket);
            Self::new_unchecked(map, indices)
        }
    }

    /// # Safety
    ///
    /// See the comment in the type definition for safety.
    unsafe fn new_unchecked(
        map: &'a mut IndexMultimapCore<K, V, Indices>,
        indices: UniqueSorted<Indices>,
    ) -> Self {
        debug_assert!(!indices.is_empty());
        debug_assert!(is_sorted_and_unique(indices.as_slice()));
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

impl<K, V, Indices> Iterator for SwapRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
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

impl<K, V, Indices> ExactSizeIterator for SwapRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
    fn len(&self) -> usize {
        self.iter_forward.len()
    }
}

impl<K, V, Indices> FusedIterator for SwapRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
}

impl<K, V, Indices> Drop for SwapRemove<'_, K, V, Indices>
where
    Indices: IndexStorage,
    K: Eq,
{
    fn drop(&mut self) {
        struct Guard<'a, 'b, K, V, Indices>(&'a mut SwapRemove<'b, K, V, Indices>)
        where
            Indices: IndexStorage,
            K: Eq;

        impl<'a, 'b, K, V, Indices> Drop for Guard<'a, 'b, K, V, Indices>
        where
            Indices: IndexStorage,
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

impl<'a, K, V, Indices> fmt::Debug for SwapRemove<'a, K, V, Indices>
where
    K: fmt::Debug + Eq,
    V: fmt::Debug,
    Indices: fmt::Debug + IndexStorage,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if cfg!(feature = "test_debug") {
            f.debug_struct("SwapRemove")
                .field("map", &self.map)
                .field("indices_table", &DebugIndices(&self.indices_table))
                .field("orig_len", &self.orig_len)
                .field("indices", &self.indices_to_remove.as_inner())
                .field("iter_forward", &self.iter_forward)
                .field("iter_backward", &self.iter_backward)
                .field("total_iter", &self.total_iter)
                .field("prev_backward", &self.prev_backward)
                .finish()
        } else {
            let iter = self.iter_forward.clone().map(|i| {
                let i = self.indices_to_remove[i];
                let bucket = &self.map.pairs[i];
                (i, &bucket.key, &bucket.value)
            });
            f.debug_struct("SwapRemove").field("left", &iter).finish()
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
pub struct Drain<'a, K, V, Indices>
where
    K: Eq,
    Indices: IndexStorage,
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
    map: NonNull<IndexMultimapCore<K, V, Indices>>,
    /// map.indices
    indices_table: RawTable<UniqueSorted<Indices>>,
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
unsafe impl<K, V, Indices> Sync for Drain<'_, K, V, Indices>
where
    K: Sync + Eq,
    V: Sync,
    Indices: Sync + IndexStorage,
{
}
unsafe impl<K, V, Indices> Send for Drain<'_, K, V, Indices>
where
    K: Send + Eq,
    V: Send,
    Indices: Send + IndexStorage,
{
}

impl<'a, K, V, Indices> Drain<'a, K, V, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    pub(super) fn new<R>(map: &'a mut IndexMultimapCore<K, V, Indices>, range: R) -> Self
    where
        R: ops::RangeBounds<usize>,
        K: Eq,
    {
        let range = simplify_range(range, map.pairs.len());
        map.erase_indices(range.start, range.end);

        let indices_table = mem::take(&mut map.indices);
        let len = map.pairs.len();
        let Range { start, end } = range;

        // SAFETY: simplify_range panics if given range is invalid for self.pairs
        unsafe {
            map.pairs.set_len(0);
            // Convert to pointer early
            let map = NonNull::from(map);
            // Go through raw pointer as long as possible
            let pairs = addr_of!((*map.as_ptr()).pairs);
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

impl<K, V, Indices> fmt::Debug for Drain<'_, K, V, Indices>
where
    K: fmt::Debug + Eq,
    V: fmt::Debug,
    Indices: IndexStorage,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain").field(&self.as_slice()).finish()
    }
}

impl<K, V, Indices> Iterator for Drain<'_, K, V, Indices>
where
    K: Eq,
    Indices: IndexStorage,
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

impl<K, V, Indices> DoubleEndedIterator for Drain<'_, K, V, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    #[inline]
    fn next_back(&mut self) -> Option<(K, V)> {
        self.to_remove
            .next_back()
            // SAFETY: elt is valid, initialized, aligned and we never use the read value again
            .map(|elt| unsafe { ptr::read(elt as *const Bucket<K, V>) }.key_value())
    }
}

impl<K, V, Indices> ExactSizeIterator for Drain<'_, K, V, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    fn len(&self) -> usize {
        self.to_remove.len()
    }
}

impl<K, V, Indices> FusedIterator for Drain<'_, K, V, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
}

impl<K, V, Indices> Drop for Drain<'_, K, V, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    fn drop(&mut self) {
        /// Moves back the un-`Drain`ed elements to restore the original `Vec`.
        /// Swaps back the indices that we took from the map at construction.
        struct DropGuard<'r, 'a, K, V, Indices>
        where
            K: Eq,
            Indices: IndexStorage,
        {
            drain: &'r mut Drain<'a, K, V, Indices>,
        }

        impl<'r, 'a, K, V, Indices> Drop for DropGuard<'r, 'a, K, V, Indices>
        where
            K: Eq,
            Indices: IndexStorage,
        {
            fn drop(&mut self) {
                unsafe {
                    // Only way to get here is if we actually have drained (removed)
                    // and dropped the given range.
                    let drain = &mut *self.drain;
                    let map = drain.map.as_ptr();
                    let pairs = addr_of_mut!((*map).pairs);
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
                    ptr::swap(addr_of_mut!((*map).indices), &mut drain.indices_table);
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
            let pairs = addr_of_mut!((*guard.drain.map.as_ptr()).pairs);
            let pairs_start = (*pairs).as_mut_ptr();
            let drop_offset = drop_ptr.offset_from(pairs_start);
            // drop_ptr points into pairs, it must be 'greater than' or 'equal to' vec_ptr
            drop(to_drop); // Next line invalidates iter, make it explicit, that it cannot be used anymore
            let to_drop = ptr::slice_from_raw_parts_mut(pairs_start.offset(drop_offset), drop_len);
            ptr::drop_in_place(to_drop);
        }
    }
}

/// Vector like types that can be used as a backing storage for [`IndexMultimap`]
/// to hold the indices of one key.
///
/// # Safety
///
/// All methods must behave like the ones on [`alloc::vec::Vec`] (this includes panics).
///
/// This trait is `unsafe` because the safety of [`IndexMultimap`] depends on the
/// fact that methods on this trait behave like the ones on [`alloc::vec::Vec`].
/// However there is no feasibly way for us to check that they do behave as
/// expected or guard against any funny business.
///
/// [`IndexMultimap`]: crate::IndexMultimap
pub unsafe trait IndexStorage
where
    Self: ops::Deref<Target = [usize]> + ops::DerefMut,
{
    /// Creates `self` with one element.
    ///
    /// Must
    fn one(v: usize) -> Self;
    /// Creates `self` with capacity.
    fn with_capacity(cap: usize) -> Self;
    /// Pushes one element to `self`.
    fn push(&mut self, v: usize);
    /// Removes one element from `self`.
    ///
    /// Should ***panic*** if `index` is out of bounds.
    fn remove(&mut self, index: usize) -> usize;
    fn pop(&mut self) -> Option<usize>;
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

    type IntoIter: Iterator<Item = usize> + DoubleEndedIterator + ExactSizeIterator + FusedIterator;
    fn into_iter(self) -> Self::IntoIter;
}

unsafe impl IndexStorage for Vec<usize> {
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
    fn pop(&mut self) -> Option<usize> {
        self.pop()
    }

    #[inline]
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&mut usize) -> bool,
    {
        self.retain_mut(f)
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        self.as_slice()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [usize] {
        self.as_mut_slice()
    }

    #[inline]
    fn shrink_to(&mut self, min_capacity: usize) {
        self.shrink_to(min_capacity)
    }

    #[inline]
    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }

    #[inline]
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve(additional)
            .map_err(TryReserveError::from_alloc)
    }

    #[inline]
    fn reserve_exact(&mut self, additional: usize) {
        self.reserve_exact(additional)
    }

    #[inline]
    fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve_exact(additional)
            .map_err(TryReserveError::from_alloc)
    }

    type IntoIter = alloc::vec::IntoIter<usize>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIterator::into_iter(self)
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

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn insert_bulk_no_grow_test() {
    //     fn bucket(k: usize) -> Bucket<usize, usize> {
    //         Bucket {
    //             hash: HashValue(k),
    //             key: k,
    //             value: k,
    //         }
    //     }

    //     let mut table = RawTable::<Vec<usize>>::with_capacity(10);
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
    fn swap_remove_index_to_swap_with() {
        // Removes all key=1, expected swap positions are all where key!=1
        fn test(insert: &[i32], current: usize, expected: &[usize]) {
            let mut map = IndexMultimapCore::<i32, i32, Vec<usize>>::new();
            for &k in insert {
                map.insert_append_full(HashValue(k as usize), k, 0);
            }

            let mut r = map.swap_remove(HashValue(1), &1).unwrap();

            let mut swaps = Vec::new();
            while let Some(i) = r.index_to_swap_with(current) {
                swaps.push(i);
            }

            assert_eq!(&swaps, &expected);
            mem::forget(r);
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

#![allow(unsafe_code)]
//! This module encapsulates the `unsafe` access to `hashbrown::raw::RawTable`,
//! mostly in dealing with its bucket "pointers".

use core::{fmt, iter::Copied, ops, slice};

use crate::{
    multimap::{SubsetIter, SubsetIterMut, SubsetKeys, SubsetValues, SubsetValuesMut},
    util::{is_sorted, is_unique_sorted, simplify_range},
    Equivalent,
};

use super::super::{Subset, SubsetMut};
use super::{equivalent, Bucket, Entry, HashValue, IndexMultimapCore, IndexStorage, VacantEntry};
use alloc::{format, vec::Drain, vec::Vec};
use hashbrown::raw::RawTable;

type RawBucket<Indices> = hashbrown::raw::Bucket<Indices>;

/// Inserts multiple pairs into a raw table without reallocating.
///
/// ***Panics*** if there is not sufficient capacity already.
///
/// `current_pairs` must not contain `new_pairs`
pub(super) fn insert_bulk_no_grow<K, V, Indices>(
    indices: &mut RawTable<Indices>,
    current_pairs: &[Bucket<K, V>],
    new_pairs: &[Bucket<K, V>],
) where
    K: Eq,
    Indices: IndexStorage,
{
    let current_num_pairs = current_pairs.len();
    debug_assert_eq!(
        // SAFETY: we have mutable reference to the table and buckets don't escape
        unsafe { indices.iter().map(|a| a.as_ref().len()) }.sum::<usize>(),
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
                assert!(indices.capacity() - indices.len() > 0);
                // SAFETY: we asserted that sufficient capacity exists for this pair and that this pair is not in the table
                unsafe {
                    indices.insert_no_grow(pair.hash.get(), Indices::one(pair_index));
                }
            }
        }
    }
}

#[inline]
pub(super) fn erase_index<Indices>(table: &mut RawTable<Indices>, hash: HashValue, index: usize)
where
    Indices: IndexStorage,
{
    match table.find(hash.get(), super::eq_index(index)) {
        Some(bucket) => {
            // SAFETY: we have &mut to table and thus to the bucket
            let indices = unsafe { bucket.as_mut() };
            if indices.len() == 1 {
                // Drop indices so they won't be used accidentally later
                #[allow(clippy::drop_ref)]
                drop(indices);
                // SAFETY: the bucket cannot escape as &mut to indices is dropped
                unsafe { table.erase(bucket) };
            } else {
                debug_assert!(is_sorted(indices), "expected indices to be sorted");
                let idx = indices.binary_search(&index).unwrap();
                indices.remove(idx);
            }
        }
        None => unreachable!("pair for index not found"),
    }
}

pub(crate) struct DebugIndices<'a, Indices>(pub &'a RawTable<Indices>);
impl<Indices> fmt::Debug for DebugIndices<'_, Indices>
where
    Indices: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use more readable output - each bucket is always one line
        // SAFETY: we're not letting any of the buckets escape this function
        let indices =
            unsafe { self.0.iter().map(|bucket| bucket.as_ref()) }.map(|a| format!("{:?}", a));
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
            for bucket in self.indices.iter() {
                let indices = bucket.as_mut();
                indices.retain(|i| {
                    if *i >= end {
                        *i -= offset;
                        true
                    } else {
                        *i < start
                    }
                });
                if indices.len() == 0 {
                    // so we don't use it accidentally later
                    #[allow(clippy::drop_ref)]
                    drop(indices);
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
                OccupiedEntry::from_parts(self, raw_bucket, hash, Some(key))
            }),
            None => Entry::Vacant(VacantEntry {
                map: self,
                hash,
                key,
            }),
        }
    }

    pub(super) fn indices_mut(&mut self) -> impl Iterator<Item = &mut Indices> {
        // SAFETY: we're not letting any of the buckets escape this function,
        // only the item references that are appropriately bound to `&mut self`.
        unsafe { self.indices.iter().map(|bucket| bucket.as_mut()) }
    }

    /// Return the raw bucket for the given index
    fn find_index(&self, index: usize) -> RawBucket<Indices> {
        // We'll get a "nice" bounds-check from indexing `self.pairs`,
        // and then we expect to find it in the table as well.
        let hash = self.pairs[index].hash.get();
        self.indices
            .find(hash, |i| i.contains(&index))
            .expect("index not found")
    }

    pub(crate) fn swap_indices(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        // SAFETY: Can't take two `get_mut` references from one table, so we
        // must use raw buckets to do the swap. This is still safe because we
        // are locally sure they won't dangle, and we write them individually.
        unsafe {
            let raw_bucket_a = self
                .find_index(a)
                .as_mut()
                .iter_mut()
                .find(|&&mut i| i == a)
                .expect("index not found");
            let raw_bucket_b = self
                .find_index(b)
                .as_mut()
                .iter_mut()
                .find(|&&mut i| i == b)
                .expect("index not found");
            *raw_bucket_a = b;
            *raw_bucket_b = a;
        }
        self.pairs.swap(a, b);
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.shrink_to(0);
        // SAFETY: we own the RawTable and don't let the bucket's escape
        unsafe { self.indices.iter().for_each(|a| a.as_mut().shrink_to_fit()) };
    }

    pub(crate) fn get_mut<Q>(
        &mut self,
        hash: HashValue,
        key: &Q,
    ) -> SubsetMut<'_, K, V, &'_ [usize]>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.pairs);
        if let Some(indices) = self.indices.get(hash.get(), eq) {
            debug_assert!(is_sorted(indices));
            debug_assert!(is_unique_sorted(indices));
            debug_assert!(*indices.last().unwrap_or(&0) < self.pairs.len());
            // SAFETY: indices come from the map, thus must be valid for pairs and unique
            unsafe { SubsetMut::new_unchecked(&mut self.pairs, indices.as_slice()) }
        } else {
            SubsetMut::empty()
        }
    }

    /// Push a key-value pair, *without* checking whether it already exists,
    /// and return the Subset over all the pairs for given key.
    pub(crate) fn push_full(
        &mut self,
        hash: HashValue,
        key: K,
        value: V,
    ) -> SubsetMut<'_, K, V, &'_ [usize]> {
        let i = self.pairs.len();
        let bucket = self
            .indices
            .insert(hash.get(), Indices::one(i), super::get_hash(&self.pairs));
        self.pairs.push(Bucket { hash, key, value });
        // SAFETY: we have &mut to self, thus no one else can have any references to the bucket
        //         bucket must be valid for at least '_
        let indices = unsafe { bucket.as_ref() }.as_slice();
        debug_assert_eq!(indices.len(), 1);
        debug_assert!(indices[0] < self.pairs.len());
        if cfg!(debug_assertions) {
            self.debug_assert_invariants();
        }
        // SAFETY: we just inserted the bucket, it can only have one index in it.
        //         It also must be valid as we just inserted the pair into self.pairs
        unsafe { SubsetMut::new_unchecked(&mut self.pairs, indices) }
    }

    /// This is unsafe because it can violate our maps assumptions about self.indices.
    /// Which can lead to unsoundness down the line in mutable subsets and their iterators.
    ///
    /// This happens if the returned Drain is leaked. This leaks all the pairs in
    /// the range `range.start..`. Thus first self.indices can contain out of bounds indices.
    /// This is ok by itself as it will only lead to panics.
    /// However if after the leaking there is an insertion, we insert a duplicate index into indices.
    /// If that insertion happens to be behind an existing key, get_mut will return a SubsetMut which
    /// inturn return aliased mutable references.
    pub(crate) unsafe fn _drain<R>(&mut self, range: R) -> Drain<'_, Bucket<K, V>>
    where
        R: ops::RangeBounds<usize>,
        K: Eq,
    {
        let range = simplify_range(range, self.pairs.len());
        self.erase_indices(range.start, range.end);
        self.pairs.drain(range)
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
    raw_bucket: RawBucket<Indices>,
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
    pub(super) unsafe fn from_parts(
        map: &'a mut IndexMultimapCore<K, V, Indices>,
        bucket: RawBucket<Indices>,
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

    /// Appends a new key-value pair to this entry by cloning the key
    /// used to find this pair. This allows to keep using this entry if it's
    /// needed.
    pub fn insert_append(&mut self, value: V)
    where
        K: Clone,
    {
        let index = self.map.pairs.len();
        self.map.pairs.push(Bucket {
            hash: self.hash,
            key: self.clone_key(),
            value,
        });

        unsafe { self.raw_bucket.as_mut() }.push(index);

        self.map.debug_assert_invariants();
    }

    /// Appends a new key-value pair to this entry by taking the owned key.
    ///
    /// This method should only be called once after creation of pair enum.
    /// Panics otherwise.
    pub(super) fn insert_append_take_owned_key(&mut self, value: V) {
        let index = self.map.pairs.len();

        let key = self.key.take().unwrap();
        debug_assert_eq!(self.indices().last(), Some(&(self.map.pairs.len() - 1)));
        self.map.pairs.push(Bucket {
            hash: self.hash,
            key,
            value,
        });

        unsafe { self.raw_bucket.as_mut() }.push(index);

        self.map.debug_assert_invariants();
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
                let Bucket { key, value, .. } = &self.map.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a reference to the `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        match self.indices().get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.map.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Return a reference to the first pair in this entry.
    pub fn first(&self) -> (usize, &K, &V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &self.map.pairs[index];
        (index, key, value)
    }

    /// Return a reference to the first pair in this entry.
    pub fn first_mut(&mut self) -> (usize, &K, &mut V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &mut self.map.pairs[index];
        (index, key, value)
    }

    /// Returns a reference to the last pair in this entry.
    pub fn last(&self) -> (usize, &K, &V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &self.map.pairs[index];
        (index, key, value)
    }

    /// Returns a reference to the last pair in this entry.
    pub fn last_mut(&mut self) -> (usize, &K, &mut V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &mut self.map.pairs[index];
        (index, key, value)
    }

    /// Remove all the values stored in the map for this entry.
    ///
    /// Like [`Vec::swap_remove`], the values are removed by swapping it with the
    /// last element of the map and popping it off. **This perturbs
    /// the position of what used to be the last element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove(self) {
        // SAFETY: This is safe because it can only happen once (self is consumed)
        // and bucket has not been removed from the map.indices
        let index = unsafe { self.map.indices.remove(self.raw_bucket) };
        self.map.swap_remove_finish_wo_collect(index.as_slice());
        self.map.debug_assert_invariants();
    }

    /// Remove and return all the values stored in the map for this entry.
    ///
    /// Like [`Vec::swap_remove`], the values are removed by swapping it with the
    /// last element of the map and popping it off. **This perturbs
    /// the position of what used to be the last element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_full(self) -> (Indices, Vec<(K, V)>) {
        // SAFETY: This is safe because it can only happen once (self is consumed)
        // and bucket has not been removed from the map.indices
        let indices = unsafe { self.map.indices.remove(self.raw_bucket) };
        let removed = self
            .map
            .swap_remove_finish_collect_ordered(indices.as_slice());
        self.map.debug_assert_invariants();
        (indices, removed)
    }

    /// Remove all the values stored in the map for this entry.
    ///
    /// Like [`Vec::remove`], the values are removed by shifting all of the
    /// elements that follow it, preserving their relative order.
    /// **This perturbs the index of all of those elements!**
    ///
    /// Computes in **O(n)** time (average).
    pub fn shift_remove(self) {
        // SAFETY: This is safe because it can only happen once (self is consumed)
        // and bucket has not been removed from the map.indices
        let index = unsafe { self.map.indices.remove(self.raw_bucket) };
        self.map.shift_remove_finish_wo_collect(index.as_slice());
        self.map.debug_assert_invariants();
    }

    /// Remove and return all the values stored in the map for this entry.
    ///
    /// Like [`Vec::remove`], the values are removed by shifting all of the
    /// elements that follow it, preserving their relative order.
    /// **This perturbs the index of all of those elements!**
    ///
    /// Computes in **O(n)** time (average).
    pub fn shift_remove_full(self) -> (Indices, Vec<(K, V)>) {
        // SAFETY: This is safe because it can only happen once (self is consumed)
        // and bucket has not been removed from the map.indices
        let indices = unsafe { self.map.indices.remove(self.raw_bucket) };
        let removed = self.map.shift_remove_finish_collect(indices.as_slice());
        self.map.debug_assert_invariants();
        (indices, removed)
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
        // SAFETY: we have &mut map keep keeping the bucket stable
        //  indices come from the map and must be unique, valid
        unsafe {
            SubsetIterMut::new_unchecked(
                &mut self.map.pairs,
                self.raw_bucket.as_ref().iter().copied(),
            )
        }
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
        // SAFETY: we have &mut map keep keeping the bucket stable
        //  indices come from the map and must be unique, valid
        unsafe {
            SubsetValuesMut::new_unchecked(
                &mut self.map.pairs,
                self.raw_bucket.as_ref().iter().copied(),
            )
        }
    }

    /// Converts into an iterator over all the values in this entry.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V, Copied<slice::Iter<'a, usize>>> {
        // SAFETY:
        //  * we have &mut map keep keeping the bucket stable
        //    SubsetValuesMut will take the &'a mut map.pairs and keep it,
        //    keeping the map mutably borrowed for 'a
        //  * indices come from the map, thus must be valid for pairs and unique
        unsafe {
            SubsetValuesMut::new_unchecked(
                &mut self.map.pairs,
                self.raw_bucket.as_ref().iter().copied(),
            )
        }
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
        // SAFETY: indices come from the map, thus must be valid for pairs and unique
        unsafe { SubsetMut::new_unchecked(&mut self.map.pairs, indices) }
    }

    /// Converts into  a slice like construct with all the values associated with
    /// this pair in the map, with a lifetime bound to the map itself.
    pub fn into_subset_mut(self) -> SubsetMut<'a, K, V, &'a [usize]> {
        // SAFETY:
        //  * we have &mut map keep keeping the bucket stable
        //    SubsetMut will take the &'a mut map.pairs and keep it,
        //    keeping the map mutably borrowed for 'a
        //  * indices come from the map, thus must be valid for pairs and unique
        unsafe { SubsetMut::new_unchecked(&mut self.map.pairs, self.raw_bucket.as_ref()) }
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
        unsafe {
            SubsetIterMut::new_unchecked(
                &mut self.map.pairs,
                self.raw_bucket.as_ref().iter().copied(),
            )
        }
    }
}

impl<'a, K, V, Indices> VacantEntry<'a, K, V, Indices> {
    pub(super) fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V, Indices>
    where
        Indices: IndexStorage,
    {
        let (_, bucket) = self.map.push(self.hash, self.key, value);
        // SAFETY: push returns a live, valid bucket from the self.map
        unsafe { OccupiedEntry::from_parts(self.map, bucket, self.hash, None) }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn insert_bulk_no_grow_test() {
        fn bucket(k: usize) -> Bucket<usize, usize> {
            Bucket {
                hash: HashValue(k),
                key: k,
                value: k,
            }
        }

        let mut table = RawTable::<Vec<usize>>::with_capacity(10);
        let mut pairs = Vec::new();

        let new_pairs = vec![bucket(1)];
        insert_bulk_no_grow(&mut table, &pairs, &new_pairs);
        pairs.extend(new_pairs);
        assert_eq!(DebugIndices(&table), &[[0].as_slice()]);

        let new_pairs = vec![bucket(1), bucket(1)];
        insert_bulk_no_grow(&mut table, &pairs, &new_pairs);
        pairs.extend(new_pairs);
        assert_eq!(DebugIndices(&table), &[[0, 1, 2].as_slice()]);

        let new_pairs = vec![bucket(2), bucket(1)];
        insert_bulk_no_grow(&mut table, &pairs, &new_pairs);
        assert_eq!(
            DebugIndices(&table),
            &[[0, 1, 2, 4].as_slice(), [3].as_slice()]
        );
        pairs.extend(new_pairs);
    }
}

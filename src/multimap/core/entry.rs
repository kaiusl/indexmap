#![allow(unsafe_code)]

use core::{fmt, ops};

use super::{
    equivalent, IndexMultimapCore, IndicesBucket, ShiftRemove, Subset, SubsetIter, SubsetIterMut,
    SubsetKeys, SubsetMut, SubsetValues, SubsetValuesMut, SwapRemove,
};
use crate::util::{DebugIterAsList, DebugIterAsNumberedCompactList};
use crate::{Bucket, HashValue};

/// Entry for an existing key-value pair or a vacant location to
/// insert one.
pub enum Entry<'a, K, V> {
    /// Existing slot with equivalent key.
    Occupied(OccupiedEntry<'a, K, V>),
    /// Vacant slot (no equivalent key in the map).
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V> {
    #[inline]
    pub(super) fn new(map: &'a mut IndexMultimapCore<K, V>, hash: HashValue, key: K) -> Self
    where
        K: Eq,
    {
        let eq = equivalent(&key, &map.pairs);
        match map.indices.find(hash.get(), eq) {
            // SAFETY: The entry is created with a live raw bucket, at the same time
            // we have a &mut reference to the map, so it can not be modified further.
            Some(raw_bucket) => Entry::Occupied(unsafe {
                OccupiedEntry::new_unchecked(map, raw_bucket, hash, Some(key))
            }),
            // SAFETY: There is no entry for given key
            None => Entry::Vacant(unsafe { VacantEntry::new_unchecked(map, hash, key) }),
        }
    }

    /// Gets a reference to the entry's key, either within the map if occupied,
    /// or else the new key that was used to find the entry.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    pub fn into_key(self) -> Option<K> {
        match self {
            Entry::Occupied(e) => e.into_key(),
            Entry::Vacant(e) => Some(e.into_key()),
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
        F: FnOnce(SubsetMut<'_, K, V>),
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
    pub fn or_insert(self, default: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => entry.insert_entry(default),
        }
    }

    /// Inserts the result of the `call` function in the entry if it is vacant
    /// and returns a mutable iterator over it.
    /// That iterator will yield exactly one value.
    /// Otherwise a mutable iterator over an already existent values is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_insert_with<F>(self, call: F) -> OccupiedEntry<'a, K, V>
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => entry.insert_entry(call()),
        }
    }

    /// Inserts the result of the `call` function with a reference to the entry's
    /// key if it is vacant, and returns a mutable iterator over it.
    /// That iterator will yield exactly one value.
    /// Otherwise a mutable iterator over an already existent values is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_insert_with_key<F>(self, call: F) -> OccupiedEntry<'a, K, V>
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => {
                let value = call(&entry.key);
                entry.insert_entry(value)
            }
        }
    }

    /// Inserts a default-constructed value in the entry if it is vacant and
    /// returns a mutable iterator over it.
    /// That iterator will yield exactly one value.
    /// Otherwise a mutable iterator over an already existent values is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_default(self) -> OccupiedEntry<'a, K, V>
    where
        V: Default,
    {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => entry.insert_entry(V::default()),
        }
    }

    /// Insert provided `value` in the entry and return an occupied entry referring to it.
    pub fn insert_append(self, value: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert_append_take_owned_key(value);
                entry
            }
            Entry::Vacant(entry) => entry.insert_entry(value),
        }
    }
}

impl<K, V> fmt::Debug for Entry<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
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

/// A view into a vacant entry in a [`IndexMultimap`].
/// It is part of the [`Entry`] enum.
///
/// [`IndexMultimap`]: super::IndexMultimap
pub struct VacantEntry<'a, K, V> {
    map: &'a mut IndexMultimapCore<K, V>,
    hash: HashValue,
    key: K,
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// SAFETY:
    ///   * key must not exist in the map
    #[inline]
    unsafe fn new_unchecked(map: &'a mut IndexMultimapCore<K, V>, hash: HashValue, key: K) -> Self {
        Self { map, hash, key }
    }

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
    pub fn insert(self, value: V) -> (usize, &'a K, &'a mut V) {
        let (i, _) = self.map.push(self.hash, self.key, value);
        let entry = &mut self.map.pairs[i];
        (i, &entry.key, &mut entry.value)
    }

    fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        let (_, bucket) = self.map.push(self.hash, self.key, value);
        if cfg!(debug_assertions) {
            let indices = unsafe { bucket.as_ref() };
            self.map.debug_assert_indices(indices.as_slice());
        }

        // SAFETY: push returns a live, valid bucket from the self.map
        unsafe { OccupiedEntry::new_unchecked(self.map, bucket, self.hash, None) }
    }
}

impl<K, V> fmt::Debug for VacantEntry<'_, K, V>
where
    K: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct(stringify!(VacantEntry));
        if cfg!(feature = "test_debug") {
            s.field("key", &format_args!("{:?}", self.key()));
        } else {
            s.field("key", self.key());
        }
        s.finish()
    }
}

/// A view into an occupied entry in a [`IndexMultimap`].
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: super::Entry
/// [`IndexMultimap`]: super::super::IndexMultimap
pub struct OccupiedEntry<'a, K, V> {
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
    map: &'a mut IndexMultimapCore<K, V>,
    raw_bucket: IndicesBucket,
    hash: HashValue,
    key: Option<K>,
}

// `hashbrown::raw::Bucket` is only `Send`, not `Sync`.
// SAFETY: `&self` only accesses the bucket to read it.
unsafe impl<K: Sync, V: Sync> Sync for OccupiedEntry<'_, K, V> {}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// SAFETY:
    ///   * `bucket` must point into map.indices
    ///   * `bucket` must be live at the creation moment
    ///   * we must be the only ones with the `bucket` pointer
    #[inline]
    unsafe fn new_unchecked(
        map: &'a mut IndexMultimapCore<K, V>,
        bucket: IndicesBucket,
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
    fn insert_append_take_owned_key(&mut self, value: V) {
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

    pub fn into_key(self) -> Option<K> {
        self.key
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

    /// Returns a mutable reference to the `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        match self.indices().get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.map.as_mut_pairs()[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Converts `self` into a long lived mutable reference to the `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn into_nth(self, n: usize) -> Option<(usize, &'a K, &'a mut V)> {
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

    /// Return a mutable reference to the first pair in this entry.
    pub fn first_mut(&mut self) -> (usize, &K, &mut V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &mut self.map.as_mut_pairs()[index];
        (index, key, value)
    }

    /// Converts `self` into a long lived mutable reference to the first pair in this entry.
    pub fn into_first(self) -> (usize, &'a K, &'a mut V) {
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

    /// Returns a mutable reference to the last pair in this entry.
    pub fn last_mut(&mut self) -> (usize, &K, &mut V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &mut self.map.as_mut_pairs()[index];
        (index, key, value)
    }

    /// Converts `self` into a long lived mutable reference to the last pair in this entry.
    pub fn into_last(self) -> (usize, &'a K, &'a mut V) {
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
    pub fn swap_remove(self) -> SwapRemove<'a, K, V>
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
    pub fn shift_remove(self) -> ShiftRemove<'a, K, V>
    where
        K: Eq,
    {
        // SAFETY: This is safe because it can only happen once (self is consumed)
        // and bucket has not been removed from the map.indices
        unsafe { ShiftRemove::new_raw(self.map, self.raw_bucket) }
    }

    /// Returns an iterator over all the pairs in this entry.
    pub fn iter(&self) -> SubsetIter<'_, K, V> {
        SubsetIter::new(
            &self.map.pairs,
            // SAFETY: we have &mut map keep keeping the bucket stable
            unsafe { self.raw_bucket.as_ref() }.iter(),
        )
    }

    /// Returns a mutable iterator over all the pairs in this entry.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetIterMut::new(&mut self.map.pairs, indices.as_unique_slice().iter())
    }

    /// Returns an iterator over all the keys in this entry.
    ///
    /// Note that the iterator yields one key for each pair which are all equivalent.
    /// But there may be an observable differences if the key type has any
    /// distinguishing features outside of [`Hash`] and [`Eq`], like
    /// extra fields or the memory address of an allocation.
    pub fn keys(&self) -> SubsetKeys<'_, K, V> {
        SubsetKeys::new(
            &self.map.pairs,
            // SAFETY: we have &mut map keep keeping the bucket stable
            unsafe { self.raw_bucket.as_ref() }.iter(),
        )
    }

    /// Converts into a mutable iterator over all the keys in this subset.
    pub fn into_keys(self) -> SubsetKeys<'a, K, V> {
        SubsetKeys::new(
            &self.map.pairs,
            // SAFETY: we have &mut map keep keeping the bucket stable
            unsafe { self.raw_bucket.as_ref() }.iter(),
        )
    }

    /// Returns an iterator over all the values in this entry.
    pub fn values(&self) -> SubsetValues<'_, K, V> {
        SubsetValues::new(
            &self.map.pairs,
            // SAFETY: we have &mut map keep keeping the bucket stable
            unsafe { self.raw_bucket.as_ref() }.iter(),
        )
    }

    /// Returns a mutable iterator over all the values in this entry.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetValuesMut::new(&mut self.map.pairs, indices.as_unique_slice().iter())
    }

    /// Converts into an iterator over all the values in this entry.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetValuesMut::new(&mut self.map.pairs, indices.as_unique_slice().iter())
    }

    /// Returns a slice like construct with all the values associated with this entry in the map.
    pub fn as_subset(&self) -> Subset<'_, K, V> {
        Subset::new(&self.map.pairs, self.indices())
    }

    pub fn into_subset(self) -> Subset<'a, K, V> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        Subset::new(&self.map.pairs, indices.as_slice())
    }

    /// Returns a slice like construct with all values associated with this entry in the map.
    ///
    /// If you need a reference which may outlive the destruction of the
    /// pair, see [`into_mut`](Self::into_mut).
    pub fn as_subset_mut(&mut self) -> SubsetMut<'_, K, V> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetMut::new(&mut self.map.pairs, indices.as_unique_slice())
    }

    /// Converts into a slice like construct with all the values associated with
    /// this pair in the map, with a lifetime bound to the map itself.
    pub fn into_subset_mut(self) -> SubsetMut<'a, K, V> {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetMut::new(&mut self.map.pairs, indices.as_unique_slice())
    }
}

impl<'a, K, V> IntoIterator for &'a OccupiedEntry<'_, K, V> {
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut OccupiedEntry<'_, K, V> {
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, K, V> IntoIterator for OccupiedEntry<'a, K, V> {
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        let indices = unsafe { self.raw_bucket.as_ref() };
        SubsetIterMut::new(&mut self.map.pairs, indices.as_unique_slice().iter())
    }
}

impl<K, V> fmt::Debug for OccupiedEntry<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct(stringify!(OccupiedEntry));
        if cfg!(feature = "test_debug") {
            s.field("key", &format_args!("{:?}", self.key()));
            s.field("pairs", &DebugIterAsNumberedCompactList::new(self.iter()));
        } else {
            s.field("key", self.key());
            s.field("pairs", &DebugIterAsList::new(self.iter()));
        }
        s.finish()
    }
}

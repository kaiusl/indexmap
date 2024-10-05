#![allow(unsafe_code)]

use ::alloc::vec::Vec;
use ::core::{fmt, ops, ptr};

use hashbrown::hash_table;

use super::indices::Indices;
use super::{
    equivalent, get_hash, IndexMultimapCore, ShiftRemove, Subset, SubsetIter, SubsetIterMut,
    SubsetKeys, SubsetMut, SubsetValues, SubsetValuesMut, SwapRemove,
};
use crate::multimap::core::IndicesTable;
use crate::util::{
    check_unique_and_in_bounds, try_simplify_range, DebugIterAsList, DebugIterAsNumberedCompactList,
};
use crate::{Bucket, HashValue, TryReserveError};

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
        let pairs = &mut map.pairs;
        let eq = equivalent(&key, pairs);
        let entry = map.indices.entry(hash.get(), eq, get_hash(pairs));
        match entry {
            hash_table::Entry::Occupied(entry) => {
                Entry::Occupied(unsafe { OccupiedEntry::new(pairs, entry, hash, Some(key)) })
            }
            hash_table::Entry::Vacant(entry) => {
                Entry::Vacant(unsafe { VacantEntry::new(pairs, entry, hash, key) })
            }
        }
    }

    /// Gets a reference to the entry's key, either within the map if occupied,4
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
    /// The return value behaves like <code>&[[usize]]</code>
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

    /// Inserts the values from `iter` using the entry's key into the map.
    ///
    /// Return [`Entry::Vacant`] if entry is still vacant,
    /// which happens if entry was vacant and `iter` was empty.
    /// Otherwise returns [`Entry::Occupied`].
    pub fn insert_append_many<I>(self, iter: I) -> Self
    where
        I: IntoIterator<Item = V>,
        K: Clone + Eq,
    {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert_append_many(iter);
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => entry.insert_many(iter),
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

/// Wrapper of the indices of an [`Entry`]. Behaves like a <code>&[[usize]]</code>.
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
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct VacantEntry<'a, K, V> {
    pairs: &'a mut Vec<Bucket<K, V>>,
    indices_entry: hash_table::VacantEntry<'a, Indices>,
    hash: HashValue,
    key: K,
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    #[inline]
    unsafe fn new(
        pairs: &'a mut Vec<Bucket<K, V>>,
        indices_entry: hash_table::VacantEntry<'a, Indices>,
        hash: HashValue,
        key: K,
    ) -> Self {
        Self {
            pairs,
            indices_entry,
            hash,
            key,
        }
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
        self.pairs.len()
    }

    /// Inserts the entry's key and the given value into the map,
    /// and returns a mutable reference to the value.
    pub fn insert(self, value: V) -> (usize, &'a K, &'a mut V) {
        let i = self.pairs.len();
        let _ = self.indices_entry.insert(Indices::one(i));

        self.pairs.push(Bucket {
            hash: self.hash,
            key: self.key,
            value,
        });
        let entry = self.pairs.last_mut().unwrap();
        (i, &entry.key, &mut entry.value)
    }

    fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        let i = self.pairs.len();
        let entry = self.indices_entry.insert(Indices::one(i));
        self.pairs.push(Bucket {
            hash: self.hash,
            key: self.key,
            value,
        });

        // if cfg!(debug_assertions) {
        //     let indices = bucket.get();
        //     self.map.debug_assert_indices(indices.as_slice());
        // }

        unsafe { OccupiedEntry::new(self.pairs, entry, self.hash, None) }
    }

    /// Inserts the values from `iter` using the entry's key into the map.
    ///
    /// Return [`Entry::Vacant`] if iterator was empty, [`Entry::Occupied`] otherwise.
    pub fn insert_many<T>(self, iter: T) -> Entry<'a, K, V>
    where
        T: IntoIterator<Item = V>,
        K: Clone + Eq,
    {
        let iter = iter.into_iter();
        let start_len_pairs = self.pairs.len();
        self.pairs.extend(iter.map(|v| Bucket {
            hash: self.hash,
            key: self.key.clone(),
            value: v,
        }));

        if start_len_pairs != self.pairs.len() {
            let indices = Indices::from_range(start_len_pairs..self.pairs.len());
            let bucket = self.indices_entry.insert(indices);

            Entry::Occupied(unsafe {
                OccupiedEntry::new(self.pairs, bucket, self.hash, Some(self.key))
            })
        } else {
            Entry::Vacant(self)
        }
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
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct OccupiedEntry<'a, K, V> {
    pairs: &'a mut Vec<Bucket<K, V>>,
    indices_entry: hash_table::OccupiedEntry<'a, Indices>,
    hash: HashValue,
    key: Option<K>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    #[inline]
    unsafe fn new(
        pairs: &'a mut Vec<Bucket<K, V>>,
        entry: hash_table::OccupiedEntry<'a, Indices>,
        hash: HashValue,
        key: Option<K>,
    ) -> Self {
        Self {
            pairs,
            indices_entry: entry,
            hash,
            key,
        }
    }

    /// Returns the number of key-value pairs in this entry.
    #[allow(clippy::len_without_is_empty)] // There is always at least one pair
    #[inline]
    pub fn len(&self) -> usize {
        self.indices().len()
    }

    /// Returns the total number of key-value pairs this entry can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        // We'll need to reallocate either if the indices vec is full or pairs vec is full.
        let free_space_in_map = self.pairs.capacity() - self.pairs.len();
        usize::min(
            self.indices_entry.get().capacity(),
            self.len() + free_space_in_map,
        )
    }

    /// Reserve capacity for `additional` more key-value pairs for this entry.
    pub fn reserve(&mut self, additional: usize) {
        self.pairs.reserve(additional);
        self.indices_entry.get_mut().reserve(additional)
    }

    /// Reserve capacity for `additional` more key-value pairs for this entry, without over-allocating.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.pairs.reserve_exact(additional);
        self.indices_entry.get_mut().reserve_exact(additional);
    }

    /// Try to reserve capacity for `additional` more key-value pairs for this entry.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.pairs
            .try_reserve(additional)
            .map_err(TryReserveError::from_alloc)?;
        self.indices_entry.get_mut().try_reserve(additional)
    }

    /// Try to reserve capacity for `additional` more key-value pairs for this entry, without over-allocating.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.pairs
            .try_reserve_exact(additional)
            .map_err(TryReserveError::from_alloc)?;
        self.indices_entry.get_mut().try_reserve_exact(additional)
    }

    /// Shrink the capacity of this entry with a lower bound.
    ///     
    /// The capacity will remain at least as large as both the length and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.indices_entry.get_mut().shrink_to(min_capacity);
    }

    /// Shrinks the capacity of this entry as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator
    /// may still inform the vector that there is space for a few more elements.
    pub fn shrink_to_fit(&mut self) {
        self.indices_entry.get_mut().shrink_to_fit()
    }

    /// Appends a new key-value pair to this entry.
    ///
    /// This method will clone the key.
    pub fn insert_append(&mut self, value: V)
    where
        K: Clone,
    {
        let index = self.pairs.len();
        let key = self.clone_key();
        self.pairs.push(Bucket {
            hash: self.hash,
            key,
            value,
        });

        self.indices_entry.get_mut().push(index);

        // self.map.debug_assert_indices(self.indices());
    }

    /// Inserts the values from `iter` using the entry's key into the map.
    ///
    /// This method will clone the key.
    pub fn insert_append_many<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = V>,
        K: Clone + Eq,
    {
        // Impl:
        // a) use vec.extend twice in order to make use of std's specializations
        //    of extend (for example if iter is ExactSizeIterator)
        //    downside is that will iterate twice over the items,
        //    in general case we still may resize multiple times
        // b) reserve ourselves and iterate once, but we cannot make use of the specializations,
        //    so we may need to resize multiple times
        let iter = iter.into_iter();
        let start_len_pairs = self.pairs.len();
        let key = self.clone_key();
        self.pairs.extend(iter.map(|v| Bucket {
            hash: self.hash,
            key: key.clone(),
            value: v,
        }));

        let indices = self.indices_entry.get_mut();
        debug_assert!(*indices.last().unwrap() < start_len_pairs);
        // SAFETY: indices is from map, if map has `start_len_pairs` then the
        // maximum index that could be in `indices` is `start_len_pairs - 1`,
        // thus the range below is guaranteed to yield larger values than currently
        // in the indices
        // A range will also yield unique items in sorted order.
        unsafe { indices.extend(start_len_pairs..self.pairs.len()) };

        //self.map.debug_assert_invariants();
    }

    /// Appends a new key-value pair to this entry by taking the owned key.
    ///
    /// This method should only be called once after creation of pair enum.
    /// Panics otherwise.
    fn insert_append_take_owned_key(&mut self, value: V) {
        let index = self.pairs.len();

        let key = self.key.take().unwrap();
        self.pairs.push(Bucket {
            hash: self.hash,
            key,
            value,
        });

        self.indices_entry.get_mut().push(index);

        //self.map.debug_assert_indices(self.indices());
    }

    /// Gets a reference to the entry's first key in the map.
    ///
    /// Other keys can be accessed through [`nth`](Self::nth) (and similar) method.
    ///
    /// Note that this may not be the key that was used to find the pair.
    /// There may be an observable difference if the key type has any
    /// distinguishing features outside of [`Hash`] and [`Eq`], like
    /// extra fields or the memory address of an allocation.
    ///
    /// [`Hash`]: ::core::hash::Hash
    pub fn key(&self) -> &K {
        &self.pairs[self.indices()[0]].key
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
                debug_assert_eq!(self.indices().last(), Some(&(self.pairs.len() - 1)));
                // The only way we don't have owned key is if we already inserted it.
                // (either by Entry::insert_append or VacantEntry::insert).
                // The key that was used to get this entry thus must be in the last pair
                self.pairs
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
        self.indices_entry.get().as_slice()
    }

    /// Returns a reference to the `n`th pair in this subset or [`None`] if <code>n >= self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn nth(&self, n: usize) -> Option<(usize, &K, &V)> {
        match self.indices().get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a mutable reference to the `n`th pair in this subset or [`None`]
    ///  if <code>n >= self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        match self.indices().get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Converts `self` into a long lived mutable reference to the `n`th pair in
    ///  this subset or [`None`] if <code>n >= self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn into_nth(self, n: usize) -> Option<(usize, &'a K, &'a mut V)> {
        match self.indices().get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Return a reference to the first pair in this entry.
    pub fn first(&self) -> (usize, &K, &V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &self.pairs[index];
        (index, key, value)
    }

    /// Return a mutable reference to the first pair in this entry.
    pub fn first_mut(&mut self) -> (usize, &K, &mut V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &mut self.pairs[index];
        (index, key, value)
    }

    /// Converts `self` into a long lived mutable reference to the first pair in this entry.
    pub fn into_first(self) -> (usize, &'a K, &'a mut V) {
        let index = self.indices()[0];
        let Bucket { key, value, .. } = &mut self.pairs[index];
        (index, key, value)
    }

    /// Returns a reference to the last pair in this entry.
    pub fn last(&self) -> (usize, &K, &V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &self.pairs[index];
        (index, key, value)
    }

    /// Returns a mutable reference to the last pair in this entry.
    pub fn last_mut(&mut self) -> (usize, &K, &mut V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &mut self.pairs[index];
        (index, key, value)
    }

    /// Converts `self` into a long lived mutable reference to the last pair in this entry.
    pub fn into_last(self) -> (usize, &'a K, &'a mut V) {
        let index = *self.indices().last().unwrap();
        let Bucket { key, value, .. } = &mut self.pairs[index];
        (index, key, value)
    }

    /// Returns a immutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are <code>0 <= index < self.[len]\()</code>
    ///
    /// [len]: Self::len
    pub fn get_range<R>(&self, range: R) -> Option<Subset<'_, K, V>>
    where
        R: ops::RangeBounds<usize>,
    {
        let indices = self.indices();
        let range = try_simplify_range(range, indices.len())?;
        match indices.get(range) {
            Some(indices) => Some(unsafe { Subset::from_slice_unchecked(self.pairs, indices) }),
            None => None,
        }
    }

    /// Returns a mutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are <code>0 <= index < self.[len]\()</code>
    ///
    /// [len]: Self::len
    pub fn get_range_mut<R>(&mut self, range: R) -> Option<SubsetMut<'_, K, V>>
    where
        R: ops::RangeBounds<usize>,
    {
        let indices = self.indices_entry.get().as_unique_slice();
        match indices.get_range(range) {
            Some(indices) => Some(unsafe { SubsetMut::from_slice_unchecked(self.pairs, indices) }),
            None => None,
        }
    }

    /// Converts `self` into a mutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are <code>0 <= index < self.[len]\()</code>
    ///
    /// [len]: Self::len
    pub fn into_range<R>(self, range: R) -> Option<SubsetMut<'a, K, V>>
    where
        R: ops::RangeBounds<usize>,
    {
        let indices = self.indices_entry.into_mut().as_unique_slice();
        match indices.get_range(range) {
            Some(indices) => Some(unsafe { SubsetMut::from_slice_unchecked(self.pairs, indices) }),
            None => None,
        }
    }

    /// Returns mutable references to many items at once or [`None`] if any index
    /// is out-of-bounds, or if the same index was passed more than once.
    pub fn get_many_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> Option<[(usize, &K, &mut V); N]> {
        unsafe { Self::get_many_mut_core(self.pairs, self.indices_entry.get(), indices) }
    }

    /// Returns mutable references to many items at once or [`None`] if any index
    /// is out-of-bounds, or if the same index was passed more than once.
    pub fn into_many_mut<const N: usize>(
        self,
        indices: [usize; N],
    ) -> Option<[(usize, &'a K, &'a mut V); N]> {
        unsafe { Self::get_many_mut_core(self.pairs, self.indices_entry.into_mut(), indices) }
    }

    #[inline]
    unsafe fn get_many_mut_core<'b, const N: usize>(
        pairs: &'b mut [Bucket<K, V>],
        subset_indices: &[usize],
        get_indices: [usize; N],
    ) -> Option<[(usize, &'b K, &'b mut V); N]> {
        let pairs_len = pairs.len();
        if !check_unique_and_in_bounds(&get_indices, subset_indices.len()) {
            return None;
        }

        let pairs = pairs.as_mut_ptr();
        Some(get_indices.map(|i| {
            let i = unsafe { *subset_indices.get_unchecked(i) };
            debug_assert!(i < pairs_len, "index out of bounds");
            let Bucket { ref key, value, .. } = unsafe { &mut *pairs.add(i) };
            (i, key, value)
        }))
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
    ///
    /// [`Vec::swap_remove`]: ::alloc::vec::Vec::swap_remove
    pub fn swap_remove(self) -> SwapRemove<'a, K, V>
    where
        K: Eq,
    {
        let (indices, entry) = self.indices_entry.remove();
        let indices_table = entry.into_table();
        unsafe { SwapRemove::new_unchecked(indices_table, self.pairs, indices) }
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
    ///
    /// [`Vec::remove`]: ::alloc::vec::Vec::remove
    pub fn shift_remove(self) -> ShiftRemove<'a, K, V>
    where
        K: Eq,
    {
        let (indices, entry) = self.indices_entry.remove();
        let indices_table = entry.into_table();
        unsafe { ShiftRemove::new_unchecked(indices_table, self.pairs, indices) }
    }

    /// Retains only the elements specified by the predicate in this entry.
    ///
    /// In other words, remove all elements e such that `f(&mut e)` returns [`false`].
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    /// This can be more efficient than [`IndexMultimap::retain`] since we don't
    /// need to traverse all the pairs in the map.
    ///
    /// [`IndexMultimap::retain`]: crate::IndexMultimap::retain
    pub fn retain<F>(self, mut keep: F) -> Option<Self>
    where
        F: FnMut(usize, &K, &mut V) -> bool,
        K: Eq,
    {
        /// Drop guard in case dropping any K/V panics or the user provided predicate panics
        struct BackShiftOnDrop<'map, 'b, K, V>
        where
            K: Eq,
        {
            pairs: &'b mut Vec<Bucket<K, V>>,
            indices_entry: Option<hash_table::OccupiedEntry<'map, Indices>>,
            start_pairs_count: usize,
            prev_removed_index: usize,
            delete_count: usize,
            is_dropped: bool,
        }

        impl<'map, 'b, K, V> BackShiftOnDrop<'map, 'b, K, V>
        where
            K: Eq,
        {
            fn finish(&mut self) -> Option<&'map mut IndicesTable> {
                if self.is_dropped {
                    return None;
                }

                let indices_table = self.indices_entry.take().unwrap().into_table();

                // Restore map to valid state
                if self.delete_count != 0 {
                    // Step 1: Shift back the tail
                    let move_count = self.start_pairs_count - self.prev_removed_index - 1;
                    if move_count > 0 {
                        // Before: [c c hole hole hole prev_removed/hole tail tail ]
                        // Before: [c c tail tail hole     hole          hole hole ]
                        let pairs_ptr = self.pairs.as_mut_ptr();
                        let hole_start = unsafe {
                            pairs_ptr.add(self.prev_removed_index - (self.delete_count - 1))
                        };
                        let src = unsafe { pairs_ptr.add(self.prev_removed_index + 1) };
                        unsafe { ptr::copy(src, hole_start, move_count) }
                    }

                    // Step 2: Set correct pairs length
                    unsafe {
                        self.pairs
                            .set_len(self.start_pairs_count - self.delete_count)
                    };

                    // Step 3: Rebuild the hash table
                    IndexMultimapCore::rebuild_hash_table_core(self.pairs, indices_table);
                }

                self.is_dropped = true;

                Some(indices_table)
            }
        }

        impl<'map, 'b, K, V> Drop for BackShiftOnDrop<'map, 'b, K, V>
        where
            K: Eq,
        {
            fn drop(&mut self) {
                self.finish();
            }
        }

        let start_set_count = self.len();
        let mut guard = {
            let start_pairs_count = self.pairs.len();
            BackShiftOnDrop {
                pairs: self.pairs,
                indices_entry: Some(self.indices_entry),
                start_pairs_count,
                prev_removed_index: 0,
                delete_count: 0,
                is_dropped: false,
            }
        };

        let mut indices = guard.indices_entry.as_ref().unwrap().get().iter();
        let pairs_ptr = guard.pairs.as_mut_ptr();

        // Step 1: Check until first removed value, there is nothing to backshift until then
        while let Some(&index) = indices.next() {
            let bucket = unsafe { &mut *pairs_ptr.add(index) };
            let should_remove = !keep(index, &mut bucket.key, &mut bucket.value);

            if should_remove {
                // Advance early to avoid double drop if `drop_in_place` panicked.
                guard.delete_count += 1;
                guard.prev_removed_index = index;
                // SAFETY: we never touch bucket again
                unsafe { ptr::drop_in_place(bucket) };
                break;
            }
        }

        // Step 2: Check remaining values and backshift all the values between previously removed and the most recently removed value
        for &index in indices {
            let bucket = unsafe { &mut *pairs_ptr.add(index) };
            let should_remove = !keep(index, &mut bucket.key, &mut bucket.value);

            if should_remove {
                // Advance early to avoid double drop if `drop_in_place` panicked.
                guard.delete_count += 1;
                let prev_removed_index = guard.prev_removed_index;
                guard.prev_removed_index = index;

                // Before: [kept kept  hole hole  prev_removed/hole shift shift shift  index/new_hole unchecked unchecked ]
                // After : [kept kept shift shift       shift        hole  hole  hole      hole       unchecked unchecked ]
                let move_count = index - prev_removed_index - 1;
                if move_count > 0 {
                    let hole_start =
                        unsafe { pairs_ptr.add(prev_removed_index - (guard.delete_count - 2)) };
                    let src = unsafe { pairs_ptr.add(prev_removed_index + 1) };
                    unsafe { ptr::copy(src, hole_start, move_count) }
                }

                // SAFETY: we never touch this bucket again
                unsafe { ptr::drop_in_place(bucket) };
            }
        }

        let delete_count = guard.delete_count;
        // Step 3: Backshift the tail, set correct pairs length and rebuild the hash table (indices)
        let indices_table = guard.finish().unwrap();
        drop(guard);

        //self.map.debug_assert_invariants();

        // Step 4: Return a new OccupiedEntry if we didn't remove all the items
        //   and we have a key to look up the new indices bucket
        //   (since we need to rebuild the indices table,
        //   current indices bucket is not valid anymore)
        if start_set_count == delete_count {
            None
        } else {
            match &self.key {
                Some(key) => {
                    let eq = equivalent(key, self.pairs);
                    let indices = indices_table.find_entry(self.hash.get(), eq).unwrap();
                    Some(unsafe { OccupiedEntry::new(self.pairs, indices, self.hash, self.key) })
                }
                None => None,
            }
        }
    }

    /// Returns an iterator over all the pairs in this entry.
    pub fn iter(&self) -> SubsetIter<'_, K, V> {
        unsafe { SubsetIter::from_slice_unchecked(self.pairs, self.indices_entry.get().iter()) }
    }

    /// Returns a mutable iterator over all the pairs in this entry.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V> {
        let indices = self.indices_entry.get();
        unsafe { SubsetIterMut::from_slice_unchecked(self.pairs, indices.as_unique_slice().iter()) }
    }

    /// Returns an iterator over all the keys in this entry.
    ///
    /// Note that the iterator yields one key for each pair which are all equivalent.
    /// But there may be an observable differences if the key type has any
    /// distinguishing features outside of [`Hash`] and [`Eq`], like
    /// extra fields or the memory address of an allocation.
    ///
    /// [`Hash`]: ::core::hash::Hash
    /// [`Eq`]: ::core::cmp::Eq
    pub fn keys(&self) -> SubsetKeys<'_, K, V> {
        unsafe { SubsetKeys::from_slice_unchecked(self.pairs, self.indices_entry.get().iter()) }
    }

    /// Converts into a mutable iterator over all the keys in this subset.
    pub fn into_keys(self) -> SubsetKeys<'a, K, V> {
        unsafe {
            SubsetKeys::from_slice_unchecked(self.pairs, self.indices_entry.into_mut().iter())
        }
    }

    /// Returns an iterator over all the values in this entry.
    pub fn values(&self) -> SubsetValues<'_, K, V> {
        unsafe { SubsetValues::from_slice_unchecked(self.pairs, self.indices_entry.get().iter()) }
    }

    /// Returns a mutable iterator over all the values in this entry.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V> {
        let indices = self.indices_entry.get();
        unsafe {
            SubsetValuesMut::from_slice_unchecked(self.pairs, indices.as_unique_slice().iter())
        }
    }

    /// Converts into an iterator over all the values in this entry.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V> {
        let indices = self.indices_entry.into_mut();
        unsafe {
            SubsetValuesMut::from_slice_unchecked(self.pairs, indices.as_unique_slice().iter())
        }
    }

    /// Returns a slice like construct with all the values associated with this entry in the map.
    pub fn as_subset(&self) -> Subset<'_, K, V> {
        unsafe { Subset::from_slice_unchecked(self.pairs, self.indices()) }
    }

    pub fn into_subset(self) -> Subset<'a, K, V> {
        let indices = self.indices_entry.into_mut();
        unsafe { Subset::from_slice_unchecked(self.pairs, indices.as_slice()) }
    }

    /// Returns a slice like construct with all values associated with this entry in the map.
    ///
    /// If you need a references which may outlive the destruction of this entry,
    /// see [`into_subset_mut`](Self::into_subset_mut).
    pub fn as_subset_mut(&mut self) -> SubsetMut<'_, K, V> {
        let indices = self.indices_entry.get();
        unsafe { SubsetMut::from_slice_unchecked(self.pairs, indices.as_unique_slice()) }
    }

    /// Converts into a slice like construct with all the values associated with
    /// this entry in the map, with a lifetime bound to the map itself.
    pub fn into_subset_mut(self) -> SubsetMut<'a, K, V> {
        let indices = self.indices_entry.into_mut();
        unsafe { SubsetMut::from_slice_unchecked(self.pairs, indices.as_unique_slice()) }
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
        let indices = self.indices_entry.into_mut();
        unsafe { SubsetIterMut::from_slice_unchecked(self.pairs, indices.as_unique_slice().iter()) }
    }
}

impl<K, V> Extend<V> for OccupiedEntry<'_, K, V>
where
    K: Clone + Eq,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = V>,
    {
        self.insert_append_many(iter)
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

#![allow(clippy::question_mark)]
#![allow(clippy::manual_map)]
#![deny(unsafe_op_in_unsafe_fn)]

//! `IndexMultimap` is a hash table where the iteration order of the key-value
//! pairs is independent of the hash values of the keys and each key supports
//! multiple associated values.
//!
//! # Note about wording
//!
//! To make reading docs easier we make some clarifications about wording.
//! * A *key-value pair* or simply *pair* refers to a single inserted key-value pair.
//! * An *entry* refers to all the *key-value pair*s which have an equivalent key
//!   according to the [`Equivalent`] trait. The keys in an *entry* may have
//!   observable differences if the key type has any distinguishing features
//!   outside of [`Hash`] and [`Eq`], like extra fields or the memory address
//!   of an allocation.
//!
//! These definitions are used throughout this module and it's submodules.

use ::alloc::boxed::Box;
use ::alloc::vec::Vec;
use ::core::cmp::Ordering;
use ::core::fmt;
use ::core::hash::{BuildHasher, Hash, Hasher};
use ::core::ops::{self, Index, IndexMut, RangeBounds};
#[cfg(feature = "std")]
use ::std::collections::hash_map::RandomState;

pub use self::core::{
    Drain, Entry, EntryIndices, IndexStorage, OccupiedEntry, ShiftRemove, Subset,
    SubsetIndexStorage, SubsetIter, SubsetIterMut, SubsetKeys, SubsetMut, SubsetValues,
    SubsetValuesMut, SwapRemove, ToIndexIter, VacantEntry,
};

use self::core::IndexMultimapCore;
use crate::equivalent::Equivalent;
use crate::map::{IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys, Slice, Values, ValuesMut};
use crate::util::{try_simplify_range, DebugIterAsList, debug_iter_as_list};
use crate::{Bucket, HashValue, TryReserveError};

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
pub mod serde_seq;
// TODO: add rayon impls
//#[cfg(feature = "rayon")]
//#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
//pub use crate::rayon::multimap as rayon;

mod core;
mod slice;
#[cfg(test)]
mod tests;

/// A hash table where the iteration order of the key-value pairs is independent
/// of the hash values of the keys and each key supports multiple associated values.
///
/// The interface is closely compatible with the standard `HashMap`, but also
/// has additional features.
///
/// # Note about docs
///
/// To make reading docs easier we make some clarifications about wording.
/// * A *key-value pair* refers to a single inserted key-value pair.
/// * An *entry* refers to all the *key-value pair*s which have an equivalent key
///   according to the [`Equivalent`] trait. The keys in an *entry* may have
///   observable differences if the key type has any distinguishing features
///   outside of [`Hash`] and [`Eq`], like extra fields or the memory address
///   of an allocation.
///
/// # Order
///
/// The key-value pairs have a consistent order that is determined by
/// the sequence of insertion and removal calls on the map. The order does
/// not depend on the keys or the hash function at all.
///
/// All iterators traverse the map in *the order*.
///
/// The insertion order is preserved, with **notable exceptions** like the
/// [`swap_remove()`] method. Methods such as [`sort_by()`] of
/// course result in a new order, depending on the sorting order.
///
/// # Indices
///
/// The key-value pairs are indexed in a compact range without holes in the
/// range `0..self.`[`len_pairs()`]. For example, the method [`get()`] looks up
/// an entry with equivalent key, and the method [`get_index()`]
/// looks up the key-value pair by index.
///
/// # Examples
///
/// ```
/// use indexmap::IndexMap;
///
/// // count the frequency of each letter in a sentence.
/// let mut letters = IndexMap::new();
/// for ch in "a short treatise on fungi".chars() {
///     *letters.entry(ch).or_insert(0) += 1;
/// }
///
/// assert_eq!(letters[&'s'], 2);
/// assert_eq!(letters[&'t'], 3);
/// assert_eq!(letters[&'u'], 1);
/// assert_eq!(letters.get(&'y'), None);
/// ```
///
/// [`swap_remove()`]: Self::swap_remove
/// [`sort_by()`]: Self::sort_by
/// [`len_pairs()`]: Self::len_pairs
/// [`get()`]: Self::get
/// [`get_index()`]: Self::get_index
#[cfg(feature = "std")]
pub struct IndexMultimap<K, V, S = RandomState, Indices = Vec<usize>> {
    core: IndexMultimapCore<K, V, Indices>,
    hash_builder: S,
}
#[cfg(not(feature = "std"))]
pub struct IndexMultimap<K, V, S, Indices = Vec<usize>> {
    pub(crate) core: IndexMultimapCore<K, V, Indices>,
    hash_builder: S,
}

impl<K, V, S, Indices> Clone for IndexMultimap<K, V, S, Indices>
where
    K: Clone,
    V: Clone,
    S: Clone,
    Indices: Clone + IndexStorage,
{
    fn clone(&self) -> Self {
        IndexMultimap {
            core: self.core.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.core.clone_from(&other.core);
        self.hash_builder.clone_from(&other.hash_builder);
    }
}

#[cfg(not(feature = "test_debug"))]
impl<K, V, S, Indices> fmt::Debug for IndexMultimap<K, V, S, Indices>
where
    K: fmt::Debug,
    V: fmt::Debug,
    Indices: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.iter().zip(0usize..).map(|((k, v), i)| (i, k, v));
        debug_iter_as_list(f, Some("IndexMultimap"), iter)
    }
}

#[cfg(feature = "test_debug")]
impl<K, V, S, Indices> fmt::Debug for IndexMultimap<K, V, S, Indices>
where
    K: fmt::Debug,
    V: fmt::Debug,
    Indices: ops::Deref<Target = [usize]>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Let the inner `IndexMultimapCore` print all of its details
        f.debug_struct("IndexMultimap")
            .field("core", &self.core)
            .finish()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<K, V> IndexMultimap<K, V> {
    /// Create a new map. (Does not allocate.)
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    /// Create a new map with capacity for `keys` unique keys and `pairs` total
    /// key-value pairs. (Does not allocate if both `keys` and `pairs` are zero.)
    ///
    /// Computes in **O(n)** time.
    #[inline]
    pub fn with_capacity(keys: usize, pairs: usize) -> Self {
        Self::with_capacity_and_hasher(keys, pairs, Default::default())
    }
}

impl<K, V, S, Indices> IndexMultimap<K, V, S, Indices> {
    /// Create a new map with capacity for `keys` unique keys and `pairs` total
    /// key-value pairs. (Does not allocate if both `keys` and `entries` are zero.)
    ///
    /// Computes in **O(n)** time.
    #[inline]
    pub fn with_capacity_and_hasher(keys: usize, pairs: usize, hash_builder: S) -> Self {
        if keys == 0 && pairs == 0 {
            Self::with_hasher(hash_builder)
        } else {
            IndexMultimap {
                core: IndexMultimapCore::with_capacity(keys, pairs),
                hash_builder,
            }
        }
    }

    /// Create a new map with `hash_builder`.
    ///
    /// This function is `const`, so it can be called in `static` contexts.
    #[inline]
    pub const fn with_hasher(hash_builder: S) -> Self {
        IndexMultimap {
            core: IndexMultimapCore::new(),
            hash_builder,
        }
    }

    /// Computes in **O(1)** time.
    pub fn capacity_keys(&self) -> usize {
        self.core.capacity_keys()
    }

    /// Computes in **O(1)** time.
    pub fn capacity_pairs(&self) -> usize {
        self.core.capacity_pairs()
    }

    /// Return a reference to the map's [`BuildHasher`].
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Return the number of unique keys in the map.
    ///
    /// Computes in **O(1)** time.
    #[inline]
    pub fn len_keys(&self) -> usize {
        self.core.len_keys()
    }

    /// Return the total number of key-value pairs in the map.
    ///
    /// Computes in **O(1)** time.
    #[inline]
    pub fn len_pairs(&self) -> usize {
        self.core.len_pairs()
    }

    /// Returns `true` if the map contains no key-value pairs.
    ///
    /// Computes in **O(1)** time.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_keys() == 0
    }

    /// Remove all key-value pairs in the map, while preserving its capacity.
    ///
    /// Computes in **O(n)** time.
    pub fn clear(&mut self) {
        self.core.clear();
    }

    /// Return an iterator over the key-value pairs of the map, in their order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self.core.as_pairs())
    }

    /// Return a mutable iterator over the key-value pairs of the map, in their order.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut::new(self.core.as_mut_pairs())
    }

    /// Return an iterator over the keys of the map, in their order.
    ///
    /// Note that the iterator contains a key for every key-value pair, that is
    /// there may be duplicates.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys::new(self.core.as_pairs())
    }

    /// Return an owning iterator over the keys of the map, in their order.
    ///
    /// Note that the iterator contains a key for every key-value pair, that is
    /// there may be duplicates.
    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys::new(self.core.into_pairs())
    }

    /// Return an iterator over the values of the map, in their order.
    pub fn values(&self) -> Values<'_, K, V> {
        Values::new(self.core.as_pairs())
    }

    /// Return an iterator over mutable references to the values of the map,
    /// in their order
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut::new(self.core.as_mut_pairs())
    }

    /// Return an owning iterator over the values of the map, in their order.
    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues::new(self.core.into_pairs())
    }
}
impl<K, V, S, Indices> IndexMultimap<K, V, S, Indices>
where
    K: Hash + Eq,
    S: BuildHasher,
    Indices: IndexStorage,
{
    /// Shortens the map, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than the map's current length, this has no effect.
    pub fn truncate(&mut self, len: usize) {
        self.core.truncate(len);
    }

    // /// Clears the [`IndexMultimap`] in the given index range, returning those
    // /// key-value pairs as a drain iterator.
    // ///
    // /// The range may be any type that implements [`RangeBounds`]`<usize>`,
    // /// including all of the [`std::ops`]`::Range*` types, or even a tuple pair of
    // /// [`Bound`] start and end values. To drain the map entirely, use [`RangeFull`]
    // /// like `self.`[`drain`](Self::drain)`(..)`.
    // ///
    // /// This shifts down all entries following the drained range to fill the
    // /// gap, and keeps the allocated memory for reuse.
    // ///
    // /// # Panics
    // ///
    // /// Panics if the starting point is greater than the end point or if
    // /// the end point is greater than the length of the map.
    // ///
    // /// # Leaking
    // ///
    // /// If the returned iterator goes out of scope without being dropped (due
    // /// to [`mem::forget`], for example), the map may have lost and leaked
    // /// elements arbitrarily, including elements outside the range. This may
    // /// lead to unexpected panics.
    // ///
    // /// [`RangeFull`]: ::core::ops::RangeFull
    // /// [`Bound`]: ::core::ops::Bound
    // /// [`mem::forget`]: ::core::mem::forget
    // pub fn drain<R>(&mut self, range: R) -> Drain<'_, K, V>
    // where
    //     R: RangeBounds<usize>,
    // {
    //     Drain::new(self.core.drain(range))
    // }

    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated map containing the elements in the range
    /// `[at, len)`. After the call, the original map will be left containing
    /// the elements `[0, at)` with its previous capacity unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    pub fn split_off(&mut self, at: usize) -> Self
    where
        S: Clone,
    {
        Self {
            core: self.core.split_off(at),
            hash_builder: self.hash_builder.clone(),
        }
    }

    /// Reserve capacity for `additional_keys` more unique keys and
    /// `additional_pairs` more key-value pairs.
    ///
    /// Computes in **O(n)** time.
    pub fn reserve(&mut self, additional_keys: usize, additional_pairs: usize) {
        self.core.reserve(additional_keys, additional_pairs);
    }

    /// Reserve capacity for `additional_keys` more unique keys and
    /// `additional_pairs` more key-value pairs, without over-allocating.
    ///
    /// Unlike [`reserve`], this does not deliberately over-allocate the entry capacity to avoid
    /// frequent re-allocations. However, the underlying data structures may still have internal
    /// capacity requirements, and the allocator itself may give more space than requested, so this
    /// cannot be relied upon to be precisely minimal.
    ///
    /// Computes in **O(n)** time.
    ///
    /// [`reserve`]: Self::reserve
    pub fn reserve_exact(&mut self, additional_keys: usize, additional_pairs: usize) {
        self.core.reserve_exact(additional_keys, additional_pairs);
    }

    /// Try to reserve capacity for `additional_keys` more unique keys and
    /// `additional_pairs` more key-value pairs.
    ///
    /// Computes in **O(n)** time.
    pub fn try_reserve(
        &mut self,
        additional_keys: usize,
        additional_pairs: usize,
    ) -> Result<(), TryReserveError> {
        self.core.try_reserve(additional_keys, additional_pairs)
    }

    /// Try to reserve capacity for `additional_keys` more unique keys and
    /// `additional_pairs` more key-value pairs, without over-allocating.
    ///
    /// Unlike [`try_reserve`], this does not deliberately over-allocate the entry capacity to avoid
    /// frequent re-allocations. However, the underlying data structures may still have internal
    /// capacity requirements, and the allocator itself may give more space than requested, so this
    /// cannot be relied upon to be precisely minimal.
    ///
    /// Computes in **O(n)** time.
    ///
    /// [`try_reserve`]: Self::try_reserve
    pub fn try_reserve_exact(
        &mut self,
        additional_keys: usize,
        additional_pairs: usize,
    ) -> Result<(), TryReserveError> {
        self.core
            .try_reserve_exact(additional_keys, additional_pairs)
    }

    /// Shrink the capacity of the map as much as possible.
    ///
    /// Computes in **O(n)** time.
    pub fn shrink_to_fit(&mut self) {
        self.core.shrink_to_fit();
    }

    /// Shrink the capacity of the map with a lower limit.
    ///
    /// Note that this method does not shrink the capacity of the indices
    /// related to a single key.
    ///
    /// Computes in **O(n)** time.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.core.shrink_to(min_capacity);
    }

    // Convenience method over hash_key, however in some cases we cannot borrow whole self.
    #[inline]
    fn hash<Q>(&self, key: &Q) -> HashValue
    where
        Q: ?Sized + Hash,
    {
        Self::hash_key(&self.hash_builder, key)
    }

    #[inline]
    fn hash_key<Q>(hash_builder: &S, key: &Q) -> HashValue
    where
        Q: ?Sized + Hash,
    {
        let mut h = hash_builder.build_hasher();
        key.hash(&mut h);
        HashValue(h.finish() as usize)
    }

    /// Insert a key-value pair in the map, last in order.
    ///
    /// If an equivalent key already exists in the map, the new key-value pair
    /// is added behind the equivalent entry.
    ///
    /// If no equivalent key existed in the map, the new key-value pair is
    /// inserted as a new entry.
    ///
    /// Computes in **O(1)** time (amortized average).
    ///
    /// See also [`entry`](#method.entry) if you you want to insert *or* modify
    /// the corresponding key-value pair.
    pub fn insert_append(&mut self, key: K, value: V) -> usize {
        let hash = self.hash(&key);
        self.core.insert_append_full(hash, key, value)
    }

    /// Get the given key’s corresponding entry in the map for insertion and/or
    /// in-place manipulation.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, Indices> {
        let hash = self.hash(&key);
        self.core.entry(hash, key)
    }

    /// Return `true` if an equivalent to `key` exists in the map.
    ///
    /// Computes in **O(1)** time (average).
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K>,
    {
        !self.get_indices_of(key).is_empty()
    }

    /// Return a subset of key-value pairs corresponding to given `key`.
    ///
    /// If the `key` has no equivalent in the map, then the returned subset is empty.
    ///
    /// Computes in **O(1)** time (average).
    pub fn get<Q>(&self, key: &Q) -> Subset<'_, K, V, &'_ [usize]>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        if self.is_empty() {
            Subset::empty()
        } else {
            let hash = self.hash(key);
            self.core.get(hash, key)
        }
    }

    /// Return a mutable subset of key-value pairs corresponding to given `key`.
    ///
    /// If the `key` has no equivalent in the map, then the returned subset is empty.
    ///
    /// Computes in **O(1)** time (average).
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> SubsetMut<'_, K, V, &'_ [usize]>
    where
        Q: Hash + Equivalent<K>,
    {
        if self.is_empty() {
            SubsetMut::empty()
        } else {
            let hash = self.hash(key);
            self.core.get_mut(hash, key)
        }
    }

    /// Return all the indices for `key`.
    ///
    /// If the `key` has no equivalent in the map, then the returned slice is empty.
    ///
    /// Computes in **O(1)** time (average).
    pub fn get_indices_of<Q>(&self, key: &Q) -> &[usize]
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        if self.is_empty() {
            &[]
        } else {
            let hash = self.hash(key);
            self.core.get_indices_of(hash, key)
        }
    }

    /// Remove all the key-value pairs with key equivalent to given `key` and
    /// return an iterator over all the removed items,
    /// or [`None`] if the `key` was not in the map.
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
    /// including pairs not associated with given key.
    pub fn swap_remove<Q>(&mut self, key: &Q) -> Option<SwapRemove<'_, K, V, Indices>>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        if self.is_empty() {
            return None;
        }
        let hash = self.hash(key);
        self.core.swap_remove(hash, key)
    }

    /// Remove all the key-value pairs with key equivalent to given `key` and
    /// return an iterator over all the removed items,
    /// or [`None`] if the `key` was not in the map.
    ///
    /// Like [`Vec::remove`], the pairs are removed by shifting all of the
    /// elements that follow them, preserving their relative order.
    /// **This perturbs the index of all of those elements!**
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
    /// including pairs not associated with given key.
    pub fn shift_remove<Q>(&mut self, key: &Q) -> Option<ShiftRemove<'_, K, V, Indices>>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        if self.is_empty() {
            return None;
        }
        let hash = self.hash(key);
        self.core.shift_remove(hash, key)
    }

    /// Remove the last key-value pair.
    ///
    /// This preserves the order of the remaining elements.
    ///
    /// Computes in **O(1)** time (average).
    pub fn pop(&mut self) -> Option<(K, V)> {
        self.core.pop()
    }

    /// Scan through each key-value pair in the map and keep those where the
    /// closure `keep` returns `true`.
    ///
    /// The elements are visited in order, and remaining elements keep their
    /// order.
    ///
    /// Computes in **O(n)** time (average).
    pub fn retain<F>(&mut self, mut keep: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.core.retain_in_order(move |k, v| keep(k, v));
    }

    /// Sort the map's key-value pairs in place using the comparison
    /// function `cmp`.
    ///
    /// The comparison function receives two key-value pairs to compare (you
    /// can sort by keys or values or their combination as needed).
    ///
    /// Computes in **O(n log n + c)** time and **O(n)** space where *n* is
    /// the length of the map and *c* the capacity. The sort is stable.
    pub fn sort_by<F>(&mut self, cmp: F)
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        self.core.sort_by(cmp)
    }

    /// Sort the key-value pairs of the map and return a by-value iterator of
    /// the key-value pairs with the result.
    ///
    /// The sort is stable.
    pub fn sorted_by<F>(self, mut cmp: F) -> IntoIter<K, V>
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        let mut entries = self.core.into_pairs();
        entries.sort_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        IntoIter::new(entries)
    }

    /// Sort the map's key-value pairs by the default ordering of the keys, but
    /// may not preserve the order of equal elements.
    ///
    /// See [`sort_unstable_by`](Self::sort_unstable_by) for details.
    pub fn sort_unstable_keys(&mut self)
    where
        K: Ord,
    {
        self.core.sort_unstable_keys()
    }

    /// Sort the map's key-value pairs in place using the comparison function `cmp`, but
    /// may not preserve the order of equal elements.
    ///
    /// The comparison function receives two key-value pairs to compare (you
    /// can sort by keys or values or their combination as needed).
    ///
    /// Computes in **O(n log n + c)** time where *n* is
    /// the length of the map and *c* is the capacity. The sort is unstable.
    pub fn sort_unstable_by<F>(&mut self, cmp: F)
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        self.core.sort_unstable_by(cmp);
    }

    /// Sort the key-value pairs of the map and return a by-value iterator of
    /// the key-value pairs with the result.
    ///
    /// The sort is unstable.
    #[inline]
    pub fn sorted_unstable_by<F>(self, mut cmp: F) -> IntoIter<K, V>
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        let mut entries = self.core.into_pairs();
        entries.sort_unstable_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        IntoIter::new(entries)
    }

    /// Sort the map’s key-value pairs in place using a sort-key extraction function.
    ///
    /// During sorting, the function is called at most once per entry, by using temporary storage
    /// to remember the results of its evaluation. The order of calls to the function is
    /// unspecified and may change between versions of `indexmap` or the standard library.
    ///
    /// Computes in **O(m n + n log n + c)** time () and **O(n)** space, where the function is
    /// **O(m)**, *n* is the length of the map, and *c* the capacity. The sort is stable.
    pub fn sort_by_cached_key<T, F>(&mut self, sort_key: F)
    where
        T: Ord,
        F: FnMut(&K, &V) -> T,
    {
        self.core.sort_by_cached_key(sort_key)
    }

    /// Reverses the order of the map’s key-value pairs in place.
    ///
    /// Computes in **O(n)** time and **O(1)** space.
    pub fn reverse(&mut self) {
        self.core.reverse()
    }

    /// Clears the [`IndexMultimap`] in the given index range, returning those
    /// key-value pairs as a drain iterator.
    ///
    /// The range may be any type that implements `RangeBounds<usize>`,
    /// including all of the `std::ops::Range*` types, or even a tuple pair of
    /// `Bound` start and end values. To drain the map entirely, use `RangeFull`
    /// like `map.drain(..)`.
    ///
    /// This shifts down all entries following the drained range to fill the
    /// gap, and keeps the allocated memory for reuse.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without being dropped
    /// (due to [`mem::forget`], for example), the map may have lost and
    /// leaked elements arbitrarily, including elements outside the range.
    ///
    /// # Panics
    ///
    /// This method panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the map.
    ///
    /// [`mem::forget`]: ::core::mem::forget
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, K, V, Indices>
    where
        R: ops::RangeBounds<usize>,
    {
        self.core.drain(range)
    }

    /// Remove the key-value pair by index
    ///
    /// Valid indices are `0 <= index < self.`[`len_pairs()`].
    ///
    /// Like [`Vec::swap_remove`], the pair is removed by swapping it with the
    /// last element of the map and popping it off.
    /// **This perturbs the position of what used to be the last element!**
    ///
    /// Computes in **O(1)** time (average).
    ///
    /// [`len_pairs()`]: Self::len_pairs
    pub fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        self.core.swap_remove_index(index)
    }

    /// Remove the key-value pair by index
    ///
    /// Valid indices are `0 <= index < self.`[`len_pairs()`].
    ///
    /// Like [`Vec::remove`], the pair is removed by shifting all of the
    /// elements that follow it, preserving their relative order.
    /// **This perturbs the index of all of those elements!**
    ///
    /// Computes in **O(n)** time (average).
    ///
    /// [`len_pairs()`]: Self::len_pairs
    pub fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        self.core.shift_remove_index(index)
    }

    /// Moves the position of a key-value pair from one index to another
    /// by shifting all other pairs in-between.
    ///
    /// * If `from < to`, the other pairs will shift down while the targeted pair moves up.
    /// * If `from > to`, the other pairs will shift up while the targeted pair moves down.
    ///
    /// ***Panics*** if `from` or `to` are out of bounds.
    ///
    /// Computes in **O(n)** time (average).
    pub fn move_index(&mut self, from: usize, to: usize) {
        self.core.move_index(from, to)
    }

    /// Swaps the position of two key-value pairs in the map.
    ///
    /// ***Panics*** if `a` or `b` are out of bounds.
    pub fn swap_indices(&mut self, a: usize, b: usize) {
        self.core.swap_indices(a, b)
    }
}

impl<K, V, S, Indices> IndexMultimap<K, V, S, Indices>
where
    Indices: IndexStorage,
{
    /// Returns a slice of all the key-value pairs in the map.
    ///
    /// Computes in **O(1)** time.
    pub fn as_slice(&self) -> &Slice<K, V> {
        Slice::from_slice(self.core.as_pairs())
    }

    /// Returns a mutable slice of all the key-value pairs in the map.
    ///
    /// Computes in **O(1)** time.
    pub fn as_mut_slice(&mut self) -> &mut Slice<K, V> {
        Slice::from_mut_slice(self.core.as_mut_pairs())
    }

    /// Converts into a boxed slice of all the key-value pairs in the map.
    ///
    /// Note that this will drop the inner hash table and any excess capacity.
    pub fn into_boxed_slice(self) -> Box<Slice<K, V>> {
        Slice::from_boxed(self.core.into_pairs().into_boxed_slice())
    }

    /// Get a reference to key-value pair by index.
    ///
    /// Valid indices are `0 <= index < self.`[`len_pairs()`]
    ///
    /// Computes in **O(1)** time.
    ///
    /// [`len_pairs()`]: Self::len_pairs
    pub fn get_index(&self, index: usize) -> Option<(&K, &V)> {
        self.core.as_pairs().get(index).map(Bucket::refs)
    }

    /// Get a mutable reference to key-value pair by index.
    ///
    /// Valid indices are `0 <= index < self.`[`len_pairs()`]
    ///
    /// Computes in **O(1)** time.
    ///
    /// [`len_pairs()`]: Self::len_pairs
    pub fn get_index_mut(&mut self, index: usize) -> Option<(&K, &mut V)> {
        self.core.as_mut_pairs().get_mut(index).map(Bucket::ref_mut)
    }

    /// Returns a slice of key-value pairs in the given range of indices.
    ///
    /// Valid indices are `0 <= index < self.`[`len_pairs()`]
    ///
    /// Computes in **O(1)** time.
    ///
    /// [`len_pairs()`]: Self::len_pairs
    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<&Slice<K, V>> {
        let entries = self.core.as_pairs();
        let range = try_simplify_range(range, entries.len())?;
        entries.get(range).map(Slice::from_slice)
    }

    /// Returns a mutable slice of key-value pairs in the given range of indices.
    ///
    /// Valid indices are `0 <= index < self.`[`len_pairs()`]
    ///
    /// Computes in **O(1)** time.
    ///
    /// [`len_pairs()`]: Self::len_pairs
    pub fn get_range_mut<R: RangeBounds<usize>>(&mut self, range: R) -> Option<&mut Slice<K, V>> {
        let entries = self.core.as_mut_pairs();
        let range = try_simplify_range(range, entries.len())?;
        entries.get_mut(range).map(Slice::from_mut_slice)
    }

    /// Get the first key-value pair.
    ///
    /// Computes in **O(1)** time.
    pub fn first(&self) -> Option<(&K, &V)> {
        self.core.as_pairs().first().map(Bucket::refs)
    }

    /// Get the first key-value pair, with mutable access to the value.
    ///
    /// Computes in **O(1)** time.
    pub fn first_mut(&mut self) -> Option<(&K, &mut V)> {
        self.core.as_mut_pairs().first_mut().map(Bucket::ref_mut)
    }

    /// Get the last key-value pair.
    ///
    /// Computes in **O(1)** time.
    pub fn last(&self) -> Option<(&K, &V)> {
        self.core.as_pairs().last().map(Bucket::refs)
    }

    /// Get the last key-value pair, with mutable access to the value.
    ///
    /// Computes in **O(1)** time.
    pub fn last_mut(&mut self) -> Option<(&K, &mut V)> {
        self.core.as_mut_pairs().last_mut().map(Bucket::ref_mut)
    }
}

/// Access [`IndexMultimap`] values at indexed positions.
///
/// # Examples
///
/// ```
/// use indexmap::IndexMap;
///
/// let mut map = IndexMap::new();
/// for word in "Lorem ipsum dolor sit amet".split_whitespace() {
///     map.insert(word.to_lowercase(), word.to_uppercase());
/// }
/// assert_eq!(map[0], "LOREM");
/// assert_eq!(map[1], "IPSUM");
/// map.reverse();
/// assert_eq!(map[0], "AMET");
/// assert_eq!(map[1], "SIT");
/// map.sort_keys();
/// assert_eq!(map[0], "AMET");
/// assert_eq!(map[1], "DOLOR");
/// ```
///
/// ```should_panic
/// use indexmap::IndexMap;
///
/// let mut map = IndexMap::new();
/// map.insert("foo", 1);
/// println!("{:?}", map[10]); // panics!
/// ```
impl<K, V, S, Indices> Index<usize> for IndexMultimap<K, V, S, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    type Output = V;

    /// Returns a reference to the value at the supplied `index`.
    ///
    /// ***Panics*** if `index` is out of bounds.
    fn index(&self, index: usize) -> &V {
        self.get_index(index)
            .expect("IndexMultimap: index out of bounds")
            .1
    }
}

/// Access [`IndexMultimap`] values at indexed positions.
///
/// Mutable indexing allows changing / updating indexed values
/// that are already present.
///
/// You can **not** insert new values with index syntax, use `.insert()`.
///
/// # Examples
///
/// ```
/// use indexmap::IndexMap;
///
/// let mut map = IndexMap::new();
/// for word in "Lorem ipsum dolor sit amet".split_whitespace() {
///     map.insert(word.to_lowercase(), word.to_string());
/// }
/// let lorem = &mut map[0];
/// assert_eq!(lorem, "Lorem");
/// lorem.retain(char::is_lowercase);
/// assert_eq!(map["lorem"], "orem");
/// ```
///
/// ```should_panic
/// use indexmap::IndexMap;
///
/// let mut map = IndexMap::new();
/// map.insert("foo", 1);
/// map[10] = 1; // panics!
/// ```
impl<K, V, S, Indices> IndexMut<usize> for IndexMultimap<K, V, S, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    /// Returns a mutable reference to the value at the supplied `index`.
    ///
    /// ***Panics*** if `index` is out of bounds.
    fn index_mut(&mut self, index: usize) -> &mut V {
        self.get_index_mut(index)
            .expect("IndexMultimap: index out of bounds")
            .1
    }
}

impl<K, V, Indices, S> IntoIterator for IndexMultimap<K, V, S, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.core.into_pairs())
    }
}

impl<'a, K, V, Indices, S> IntoIterator for &'a IndexMultimap<K, V, S, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, Indices, S> IntoIterator for &'a mut IndexMultimap<K, V, S, Indices>
where
    K: Eq,
    Indices: IndexStorage,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, S, Indices> FromIterator<(K, V)> for IndexMultimap<K, V, S, Indices>
where
    K: Hash + Eq,
    S: BuildHasher + Default,
    Indices: IndexStorage,
{
    /// Create an [`IndexMultimap`] from the sequence of key-value pairs in the
    /// iterable.
    ///
    /// `from_iter` uses the same logic as `extend`. See
    /// [`extend`](#method.extend) for more details.
    fn from_iter<I>(pairs_iterable: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let iter = pairs_iterable.into_iter();
        let (low, _) = iter.size_hint();
        let mut map = Self::with_capacity_and_hasher(low, low, Default::default());
        map.extend(iter);
        map
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<K, V, Indices, const N: usize> From<[(K, V); N]> for IndexMultimap<K, V, RandomState, Indices>
where
    K: Hash + Eq,
    Indices: IndexStorage,
{
    /// # Examples
    ///
    /// ```
    /// use indexmap::IndexMap;
    ///
    /// let map1 = IndexMap::from([(1, 2), (3, 4)]);
    /// let map2: IndexMap<_, _> = [(1, 2), (3, 4)].into();
    /// assert_eq!(map1, map2);
    /// ```
    fn from(pairs: [(K, V); N]) -> Self {
        Self::from_iter(pairs)
    }
}

impl<K, V, S, Indices> Extend<(K, V)> for IndexMultimap<K, V, S, Indices>
where
    K: Hash + Eq,
    S: BuildHasher,
    Indices: IndexStorage,
{
    /// Extend the map with all key-value pairs in the iterable.
    ///
    /// This is equivalent to calling [`insert_append`](#method.insert_append) for each of
    /// them in order, which means that for keys that already existed
    /// in the map, the new value is added to the back behind the equivalent key.
    ///
    /// New keys are inserted in the order they appear in the sequence. If
    /// equivalents of a key occur more than once, the last corresponding value
    /// prevails.
    fn extend<I>(&mut self, pairs_iterable: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        // (Note: this is a copy of `std`/`hashbrown`'s reservation logic.)
        // Keys may be already present or show multiple times in the iterator.
        // Reserve the entire hint lower bound if the map is empty.
        // Otherwise reserve half the hint (rounded up), so the map
        // will only resize twice in the worst case.
        let iter = pairs_iterable.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };
        self.reserve(reserve, reserve);
        iter.for_each(move |(k, v)| {
            self.insert_append(k, v);
        });
    }
}

impl<'a, K, V, Indices, S> Extend<(&'a K, &'a V)> for IndexMultimap<K, V, S, Indices>
where
    K: Hash + Eq + Copy,
    V: Copy,
    Indices: IndexStorage,
    S: BuildHasher,
{
    /// Extend the map with all key-value pairs in the iterable.
    ///
    /// See the first extend method for more details.
    fn extend<I>(&mut self, pairs_iterable: I)
    where
        I: IntoIterator<Item = (&'a K, &'a V)>,
    {
        self.extend(
            pairs_iterable
                .into_iter()
                .map(|(&key, &value)| (key, value)),
        );
    }
}

impl<K, V, S, Indices> Default for IndexMultimap<K, V, S, Indices>
where
    S: Default,
{
    /// Return an empty [`IndexMultimap`]
    fn default() -> Self {
        Self::with_capacity_and_hasher(0, 0, S::default())
    }
}

/// This implementation does not consider the order as a part of equality.
/// Use [`IndexMultimap::as_slice`] and compare the results if the order is important.
impl<K, V1, S1, Indices1, V2, S2, Indices2> PartialEq<IndexMultimap<K, V2, S2, Indices2>>
    for IndexMultimap<K, V1, S1, Indices1>
where
    K: Hash + Eq,
    V1: PartialEq<V2>,
    S1: BuildHasher,
    S2: BuildHasher,
    Indices1: IndexStorage,
    Indices2: IndexStorage,
{
    fn eq(&self, other: &IndexMultimap<K, V2, S2, Indices2>) -> bool {
        if self.len_pairs() != other.len_pairs() {
            return false;
        }

        self.iter().all(|(key, this_v)| {
            other
                .get(key)
                .into_iter()
                .any(|(_, _, other_v)| this_v == other_v)
        })
    }
}

impl<K, V, S, Indices> Eq for IndexMultimap<K, V, S, Indices>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
    Indices: IndexStorage,
{
}

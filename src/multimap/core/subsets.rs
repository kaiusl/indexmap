#![allow(unsafe_code)]

// This module contains subset structs and iterators that can give references to
// some subset of key-value pairs in the map.
//
// # Impl details
//
// Subsets (and their iterators) are effectively a pointer to slice of pairs
// and a set of indices used to index into that slice.
// A subset or it's iterator will only touch item's to which it's indices point to.
// That means it's okay for multiple mutable subsets to be alive at the same time
// if their indices are disjoint.
// In that case there is no way to create multiple aliasing mutable references.
//
// This detail can be used by `get_many_mut` methods or `SubsetMut::split_at_mut`
// or similar methods.
//
// It's worth pointing out that while the non-mut subset and iterators can be
// implemented in safe Rust, they are unsound in the context if multiple mutable
// subsets can be alive at the same time. The reason being mutable subsets can
// return non-mutable iterators or be turned into non-mutable subset.
// A safe implementation however needs to create a
// reference to whole pairs slice. This in turn can potentially create
// aliasing references if there are multiple mutable subsets alive at
// that time.

use core::ops::RangeBounds;
use core::ptr::NonNull;
use core::slice;

use ::core::fmt;
use ::core::iter::FusedIterator;
use ::core::marker::PhantomData;

use super::indices::{UniqueIter, UniqueSlice};
use crate::util::{debug_iter_as_list, debug_iter_as_numbered_compact_list, try_simplify_range};
use crate::Bucket;

/// Slice like construct over a subset of the key-value pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`IndexMultimap::get`] and [`OccupiedEntry::get`] methods.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::get`]: crate::IndexMultimap::get
/// [`OccupiedEntry::get`]: crate::multimap::OccupiedEntry::get
pub struct Subset<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // # Safety
    //
    // * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    // * every index in indices must be in bounds to index into pairs
    //
    // * Also see the top of the module for more details. *
    pairs: NonNull<Bucket<K, V>>,
    pairs_len: usize,
    indices: &'a [usize],
    marker: PhantomData<&'a [Bucket<K, V>]>,
}

impl<'a, K, V> Subset<'a, K, V> {
    /// # Safety
    ///
    /// * `indices` must be in bounds to index into `pairs`.
    pub(super) unsafe fn from_slice_unchecked(
        pairs: &'a [Bucket<K, V>],
        indices: &'a [usize],
    ) -> Self {
        let pairs_len = pairs.len();
        let pairs = unsafe { NonNull::new_unchecked(pairs.as_ptr().cast_mut()) };
        unsafe { Self::from_raw_unchecked(pairs, pairs_len, indices) }
    }

    /// # Safety
    ///
    /// * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    /// * `indices` must be in bounds to index into `pairs`
    pub(super) unsafe fn from_raw_unchecked(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        indices: &'a [usize],
    ) -> Self {
        Self {
            pairs,
            pairs_len,
            indices,
            marker: PhantomData,
        }
    }

    pub(crate) fn empty() -> Self {
        // SAFETY: no actual access of pairs will ever be made
        unsafe { Self::from_slice_unchecked(&[], &[]) }
    }

    /// Returns the number of pairs in this subset
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this subset is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn indices(&self) -> &'a [usize] {
        &self.indices
    }

    /// Returns a reference to the `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth(&self, n: usize) -> Option<(usize, &'a K, &'a V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.pairs_len, self.indices.get(n)) }
    }

    /// Return a reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first(&self) -> Option<(usize, &'a K, &'a V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.pairs_len, self.indices.first()) }
    }

    /// Returns a reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last(&self) -> Option<(usize, &'a K, &'a V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.pairs_len, self.indices.last()) }
    }

    /// Returns a immutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are *0 <= index < self.len()*
    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<Subset<'a, K, V>> {
        let range = try_simplify_range(range, self.indices.len())?;
        match self.indices.get(range) {
            Some(indices) => {
                Some(unsafe { Subset::from_raw_unchecked(self.pairs, self.pairs_len, indices) })
            }
            None => None,
        }
    }

    /// # Safety
    ///
    /// * If `index` is `Some`, it must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn get_item_unchecked<'b>(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        index: Option<&usize>,
    ) -> Option<(usize, &'b K, &'b V)> {
        match index {
            Some(&index) => {
                debug_assert!(index < pairs_len, "index out of bounds");
                // SAFETY: caller must guarantee that `index` is in bounds
                let Bucket { key, value, .. } = unsafe { &*pairs.as_ptr().add(index) };
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetIter::from_raw_unchecked(self.pairs, self.pairs_len, self.indices.iter()) }
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yields one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetKeys::from_raw_unchecked(self.pairs, self.pairs_len, self.indices.iter()) }
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetValues::from_raw_unchecked(self.pairs, self.pairs_len, self.indices.iter()) }
    }

    /// Divides one subset into two non-mutable subsets at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)`` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub fn split_at(&self, mid: usize) -> (Subset<'a, K, V>, Subset<'a, K, V>) {
        let (left, right) = self.indices.split_at(mid);
        unsafe {
            (
                Subset::from_raw_unchecked(self.pairs, self.pairs_len, left),
                Subset::from_raw_unchecked(self.pairs, self.pairs_len, right),
            )
        }
    }

    /// Returns the first and all the rest of the elements of the subset, or `None` if it is empty.
    pub fn split_first(&self) -> Option<((usize, &'a K, &'a V), Subset<'a, K, V>)> {
        unsafe { Self::split_one(self.pairs, self.pairs_len, self.indices.split_first()) }
    }

    /// Returns the last and all the rest of the elements of the subset, or `None` if it is empty.
    pub fn split_last(&self) -> Option<((usize, &'a K, &'a V), Subset<'a, K, V>)> {
        unsafe { Self::split_one(self.pairs, self.pairs_len, self.indices.split_last()) }
    }

    /// Internal impl of split_first/last.
    ///
    /// # Safety
    ///
    /// * If `indices` is `Some`, then all the indices must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn split_one<'b>(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        indices: Option<(&usize, &'b [usize])>,
    ) -> Option<((usize, &'b K, &'b V), Subset<'b, K, V>)> {
        match indices {
            Some((&split_index, rest_indices)) => {
                debug_assert!(split_index < pairs_len, "index out of bounds");
                let Bucket { key, value, .. } = unsafe { &*pairs.as_ptr().add(split_index) };
                let split = (split_index, key, value);
                let rest = unsafe { Subset::from_raw_unchecked(pairs, pairs_len, rest_indices) };
                Some((split, rest))
            }
            None => None,
        }
    }

    /// Takes the first element out of the subset and returns a long lived
    /// reference to it, or `None` if subset is empty.
    ///
    /// The returned element will remain in the map/pairs slice but not in this subset.
    pub fn take_first(&mut self) -> Option<(usize, &'a K, &'a V)> {
        unsafe { self.take_one(self.indices.split_first()) }
    }

    /// Takes the last element out of the subset and returns a long lived
    /// reference to it, or `None` if subset is empty.
    ///
    /// The returned element will remain in the map/pairs slice but not in this subset.
    pub fn take_last(&mut self) -> Option<(usize, &'a K, &'a V)> {
        unsafe { self.take_one(self.indices.split_last()) }
    }

    /// Internal impl of take_first/last.
    ///
    /// # Safety
    ///
    /// * If `indices` is `Some`, then all the indices must be in bounds to index into `pairs`.
    unsafe fn take_one(
        &mut self,
        indices: Option<(&usize, &'a [usize])>,
    ) -> Option<(usize, &'a K, &'a V)> {
        match indices {
            Some((&take_index, rest)) => {
                self.indices = rest;
                debug_assert!(take_index < self.pairs_len, "index out of bounds");
                let Bucket { ref key, value, .. } =
                    unsafe { &*self.pairs.as_ptr().add(take_index) };
                Some((take_index, key, value))
            }
            None => None,
        }
    }
}

impl<'a, K, V> IntoIterator for Subset<'a, K, V> {
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a Subset<'_, K, V> {
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V> core::ops::Index<usize> for Subset<'a, K, V> {
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        let index = self.indices[index];
        debug_assert!(index < self.pairs_len, "index out of bounds");
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { &(*self.pairs.as_ptr().add(index)).value }
    }
}

impl<'a, K, V> Clone for Subset<'a, K, V> {
    fn clone(&self) -> Self {
        Self {
            pairs: self.pairs,
            pairs_len: self.pairs_len,
            indices: self.indices,
            marker: PhantomData,
        }
    }
}

impl<'a, K, V> fmt::Debug for Subset<'a, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_subset(f, "Subset", self.iter())
    }
}

/// Slice like construct over a subset of the pairs in the [`IndexMultimap`]
/// with mutable access to the values.
///
/// This `struct` is created by the [`IndexMultimap::get_mut`],
/// [`OccupiedEntry::get_mut`], [`Entry::or_insert`] and other similar methods.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::get_mut`]: crate::IndexMultimap::get_mut
/// [`OccupiedEntry::get_mut`]: crate::multimap::OccupiedEntry::get_mut
/// [`Entry::or_insert`]: crate::multimap::Entry::or_insert
pub struct SubsetMut<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // # Safety
    //
    // * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    // * every index in indices must be in bounds to index into pairs
    //
    // * Also see the top of the module for more details. *
    pairs: NonNull<Bucket<K, V>>,
    pairs_len: usize,
    indices: &'a UniqueSlice<usize>,
    marker: PhantomData<&'a mut [Bucket<K, V>]>,
}

impl<'a, K, V> SubsetMut<'a, K, V> {
    /// # Safety
    ///
    /// * `ìndices` must be in bounds to index into `pairs`.
    pub(crate) unsafe fn from_slice_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: &'a UniqueSlice<usize>,
    ) -> Self {
        let pairs_len = pairs.len();
        let pairs = unsafe { NonNull::new_unchecked(pairs.as_mut_ptr()) };
        unsafe { Self::from_raw_unchecked(pairs, pairs_len, indices) }
    }

    /// # Safety
    ///
    /// * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    /// * `indices` must be in bounds to index into `pairs`
    pub(super) unsafe fn from_raw_unchecked(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        indices: &'a UniqueSlice<usize>,
    ) -> Self {
        Self {
            pairs,
            pairs_len,
            indices,
            marker: PhantomData,
        }
    }

    pub(crate) fn empty() -> Self {
        // SAFETY: no access will ever be actually performed
        unsafe { Self::from_slice_unchecked(&mut [], &UniqueSlice::from_slice_unchecked(&[])) }
    }

    /// Returns the number of pairs in this subset.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this subset is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a slice of indices where the key-value pairs of this subset are
    /// located in the map.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns a reference to an `n`th key-value pair in this subset or `None` if `n >= self.len()`.
    pub fn nth(&self, n: usize) -> Option<(usize, &K, &V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.pairs_len, self.indices.get(n)) }
    }

    /// Returns a mutable reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.pairs_len, self.indices.get(n)) }
    }

    /// Converts `self` into a long lived mutable reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn into_nth(self, n: usize) -> Option<(usize, &'a K, &'a mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.pairs_len, self.indices.get(n)) }
    }

    /// Return a reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first(&self) -> Option<(usize, &K, &V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.pairs_len, self.indices.first()) }
    }

    /// Return a mutable reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.pairs_len, self.indices.first()) }
    }

    /// Converts `self` into long lived mutable reference to the first pair in this subset or `None` if this subset is empty.
    pub fn into_first_mut(self) -> Option<(usize, &'a K, &'a mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.pairs_len, self.indices.first()) }
    }

    /// Returns a reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last(&self) -> Option<(usize, &K, &V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.pairs_len, self.indices.last()) }
    }

    /// Returns a mutable reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.pairs_len, self.indices.last()) }
    }

    /// Converts `self` into long lived mutable reference to the last pair in this subset or `None` if this subset is empty.
    pub fn into_last_mut(self) -> Option<(usize, &'a K, &'a mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.pairs_len, self.indices.last()) }
    }

    /// # Safety
    ///
    /// * If `index` is `Some`, it must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn get_item_unchecked<'b>(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        index: Option<&usize>,
    ) -> Option<(usize, &'b K, &'b V)> {
        match index {
            Some(&index) => {
                debug_assert!(index < pairs_len, "index out of bounds");
                // SAFETY: caller must guarantee that `index` is in bounds
                let Bucket { key, value, .. } = unsafe { &*pairs.as_ptr().add(index) };
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// # Safety
    ///
    /// * If `index` is `Some`, it must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn get_item_unchecked_mut<'b>(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        index: Option<&usize>,
    ) -> Option<(usize, &'b K, &'b mut V)> {
        match index {
            Some(&index) => {
                debug_assert!(index < pairs_len, "index out of bounds");
                // SAFETY: caller must guarantee that `index` is in bounds
                let Bucket { key, value, .. } = unsafe { &mut *pairs.as_ptr().add(index) };
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a immutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are *0 <= index < self.len()*
    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<Subset<'_, K, V>> {
        match self.indices.get_range(range) {
            Some(indices) => {
                Some(unsafe { Subset::from_raw_unchecked(self.pairs, self.pairs_len, indices) })
            }
            None => None,
        }
    }

    /// Returns a mutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are *0 <= index < self.len()*
    pub fn get_range_mut<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Option<SubsetMut<'_, K, V>> {
        match self.indices.get_range(range) {
            Some(indices) => {
                Some(unsafe { SubsetMut::from_raw_unchecked(self.pairs, self.pairs_len, indices) })
            }
            None => None,
        }
    }

    /// Converts `self` into a mutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are *0 <= index < self.len()*
    pub fn into_range<R: RangeBounds<usize>>(self, range: R) -> Option<SubsetMut<'a, K, V>> {
        match self.indices.get_range(range) {
            Some(indices) => {
                Some(unsafe { SubsetMut::from_raw_unchecked(self.pairs, self.pairs_len, indices) })
            }
            None => None,
        }
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetIter::from_raw_unchecked(
                self.pairs,
                self.pairs_len,
                self.indices.as_slice().iter(),
            )
        }
    }

    /// Returns a mutable iterator over all the pairs in this subset.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetIterMut::from_raw_unchecked(self.pairs, self.pairs_len, self.indices.iter())
        }
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yield one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetKeys::from_raw_unchecked(
                self.pairs,
                self.pairs_len,
                self.indices.as_slice().iter(),
            )
        }
    }

    /// Converts into a iterator over all the keys in this subset.
    pub fn into_keys(self) -> SubsetKeys<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetKeys::from_raw_unchecked(
                self.pairs,
                self.pairs_len,
                self.indices.as_slice().into_iter(),
            )
        }
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetValues::from_raw_unchecked(
                self.pairs,
                self.pairs_len,
                self.indices.as_slice().iter(),
            )
        }
    }

    /// Returns a mutable iterator over all the values in this subset.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetValuesMut::from_raw_unchecked(self.pairs, self.pairs_len, self.indices.iter())
        }
    }

    /// Converts `self` into a mutable iterator over all the values in this subset.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetValuesMut::from_raw_unchecked(self.pairs, self.pairs_len, self.indices.iter())
        }
    }

    /// Borrows `self` as an immutable subset of same pairs.
    pub fn as_subset(&self) -> Subset<'_, K, V> {
        unsafe { Subset::from_raw_unchecked(self.pairs, self.pairs_len, self.indices) }
    }

    /// Converts `self` into an immutable subset of same pairs.
    pub fn into_subset(self) -> Subset<'a, K, V> {
        unsafe { Subset::from_raw_unchecked(self.pairs, self.pairs_len, self.indices) }
    }

    /// Divides one mutable subset into two non-mutable subsets at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)`` (excluding the index `len` itself).
    ///
    /// If you need a longer lived subsets, see [`Self::split_at_into].
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub fn split_at(&self, mid: usize) -> (Subset<'_, K, V>, Subset<'_, K, V>) {
        let (left, right) = self.indices.split_at(mid);
        unsafe {
            (
                Subset::from_raw_unchecked(self.pairs, self.pairs_len, left),
                Subset::from_raw_unchecked(self.pairs, self.pairs_len, right),
            )
        }
    }

    /// Divides one mutable subset into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)`` (excluding the index `len` itself).
    ///
    /// If you need a longer lived subsets, see [`Self::split_at_into`].
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub fn split_at_mut(&mut self, mid: usize) -> (SubsetMut<'_, K, V>, SubsetMut<'_, K, V>) {
        let (left, right) = self.indices.split_at(mid);
        unsafe {
            (
                SubsetMut::from_raw_unchecked(self.pairs, self.pairs_len, left),
                SubsetMut::from_raw_unchecked(self.pairs, self.pairs_len, right),
            )
        }
    }

    /// Divides one mutable subset into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)`` (excluding the index `len` itself).
    ///
    /// This method consumes `self` in order to return a longer lived subsets.
    /// If don't need it or need to keep original complete subset around,
    /// see [`Self::split_at_mut`] or [`Self::split_at`].
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub fn split_at_into(self, mid: usize) -> (SubsetMut<'a, K, V>, SubsetMut<'a, K, V>) {
        let (left, right) = self.indices.split_at(mid);
        unsafe {
            (
                SubsetMut::from_raw_unchecked(self.pairs, self.pairs_len, left),
                SubsetMut::from_raw_unchecked(self.pairs, self.pairs_len, right),
            )
        }
    }

    /// Returns the first and all the rest of the elements of the subset, or `None` if it is empty.
    pub fn split_first(&self) -> Option<((usize, &'_ K, &'_ V), Subset<'_, K, V>)> {
        unsafe { Self::split_one(self.pairs, self.pairs_len, self.indices.split_first()) }
    }

    /// Returns the first and all the rest of the elements of the subset, or `None` if it is empty.
    pub fn split_first_mut(&mut self) -> Option<((usize, &'_ K, &'_ mut V), SubsetMut<'_, K, V>)> {
        unsafe { Self::split_one_mut(self.pairs, self.pairs_len, self.indices.split_first()) }
    }

    /// Returns the last and all the rest of the elements of the subset, or `None` if it is empty.
    pub fn split_last(&self) -> Option<((usize, &'_ K, &'_ V), Subset<'_, K, V>)> {
        unsafe { Self::split_one(self.pairs, self.pairs_len, self.indices.split_last()) }
    }

    /// Returns the last and all the rest of the elements of the subset, or `None` if it is empty.
    pub fn split_last_mut(&mut self) -> Option<((usize, &'_ K, &'_ mut V), SubsetMut<'_, K, V>)> {
        unsafe { Self::split_one_mut(self.pairs, self.pairs_len, self.indices.split_last()) }
    }

    /// Internal impl of split_first/last.
    ///
    /// # Safety
    ///
    /// * If `indices` is `Some`, then all the indices must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn split_one<'b>(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        indices: Option<(&usize, &'b UniqueSlice<usize>)>,
    ) -> Option<((usize, &'b K, &'b V), Subset<'b, K, V>)> {
        match indices {
            Some((&split_index, rest_indices)) => {
                debug_assert!(split_index < pairs_len, "index out of bounds");
                let Bucket { key, value, .. } = unsafe { &*pairs.as_ptr().add(split_index) };
                let split = (split_index, key, value);
                let rest = unsafe { Subset::from_raw_unchecked(pairs, pairs_len, rest_indices) };
                Some((split, rest))
            }
            None => None,
        }
    }

    /// Internal impl of split_first/last_mut.
    ///
    /// # Safety
    ///
    /// * If `indices` is `Some`, then all the indices must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn split_one_mut<'b>(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        indices: Option<(&usize, &'b UniqueSlice<usize>)>,
    ) -> Option<((usize, &'b K, &'b mut V), SubsetMut<'b, K, V>)> {
        match indices {
            Some((&index, rest)) => {
                debug_assert!(index < pairs_len, "index out of bounds");
                let Bucket { ref key, value, .. } = unsafe { &mut *pairs.as_ptr().add(index) };
                let split = (index, key, value);
                let rest = unsafe { SubsetMut::from_raw_unchecked(pairs, pairs_len, rest) };
                Some((split, rest))
            }
            None => None,
        }
    }

    /// Takes the first element out of the subset and returns a long lived
    /// reference to it, or `None` if subset is empty.
    ///
    /// The returned element will remain in the map/pairs slice but not in this subset.
    pub fn take_first(&mut self) -> Option<(usize, &'a K, &'a mut V)> {
        unsafe { self.take_one(self.indices.split_first()) }
    }

    /// Takes the last element out of the subset and returns a long lived
    /// reference to it, or `None` if subset is empty.
    ///
    /// The returned element will remain in the map/pairs slice but not in this subset.
    pub fn take_last(&mut self) -> Option<(usize, &'a K, &'a mut V)> {
        unsafe { self.take_one(self.indices.split_last()) }
    }

    /// Internal impl of take_first/last.
    ///
    /// # Safety
    ///
    /// * If `indices` is `Some`, then all the indices must be in bounds to index into `pairs`.
    unsafe fn take_one(
        &mut self,
        indices: Option<(&usize, &'a UniqueSlice<usize>)>,
    ) -> Option<(usize, &'a K, &'a mut V)> {
        match indices {
            Some((&take_index, rest)) => {
                self.indices = rest;
                debug_assert!(take_index < self.pairs_len, "index out of bounds");
                let Bucket { ref key, value, .. } =
                    unsafe { &mut *self.pairs.as_ptr().add(take_index) };
                Some((take_index, key, value))
            }
            None => None,
        }
    }
}

impl<'a, K, V> fmt::Debug for SubsetMut<'a, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_subset(f, "SubsetMut", self.iter())
    }
}

impl<'a, K, V> IntoIterator for SubsetMut<'a, K, V> {
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe {
            SubsetIterMut::from_raw_unchecked(self.pairs, self.pairs_len, self.indices.iter())
        }
    }
}

impl<'a, K, V> IntoIterator for &'a SubsetMut<'_, K, V> {
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut SubsetMut<'_, K, V> {
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V> core::ops::Index<usize> for SubsetMut<'_, K, V> {
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        let index = self.indices[index];
        debug_assert!(index < self.pairs_len, "index out of bounds");
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { &(*self.pairs.as_ptr().add(index)).value }
    }
}

impl<K, V> core::ops::IndexMut<usize> for SubsetMut<'_, K, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let index = self.indices[index];
        debug_assert!(index < self.pairs_len, "index out of bounds");
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { &mut (*self.pairs.as_ptr().add(index)).value }
    }
}

macro_rules! iter_methods {
    (@get_item $self:ident, $index_value:expr, $index:ident, $pair:ident, $($return:tt)*) => {
        match $index_value {
            Some(&$index) => {
                debug_assert!($index < $self.pairs_len, "expected indices to be in bounds");
                // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
                let $pair = unsafe { &*$self.pairs.as_ptr().add($index) };
                Some($($return)*)
            }
            None => None,
        }
    };
    ($name:ident, $ty:ty, $item:ty, $index:ident, $pair:ident, $($return:tt)*) => {

        impl<'a, K, V> $ty
        {
            /// # Safety
            ///
            /// * `indices` must be in bounds to index into `pairs`.
            pub(super) unsafe fn from_slice_unchecked(pairs: &'a [Bucket<K, V>], indices: slice::Iter<'a, usize>) -> Self {
                let pairs_len = pairs.len();
                let pairs = unsafe { NonNull::new_unchecked(pairs.as_ptr().cast_mut()) };
                unsafe { Self::from_raw_unchecked(pairs, pairs_len, indices) }
            }

            /// # Safety
            ///
            /// * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
            /// * `indices` must be in bounds to index into `pairs`
            pub(super) unsafe fn from_raw_unchecked(
                pairs: NonNull<Bucket<K, V>>,
                pairs_len: usize,
                indices: slice::Iter<'a, usize>,
            ) -> Self {
                Self {
                    pairs,
                    pairs_len,
                    indices,
                    marker: PhantomData,
                }
            }
        }

        impl<'a, K, V> Iterator for $ty {
            type Item = $item;

            fn next(&mut self) -> Option<Self::Item> {
                iter_methods!(@get_item self, self.indices.next(), $index, $pair, $($return)*)
            }

            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                iter_methods!(@get_item self, self.indices.nth(n), $index, $pair, $($return)*)
            }

            fn last(self) -> Option<Self::Item>
            where
                Self: Sized,
            {
                iter_methods!(@get_item self, self.indices.last(), $index, $pair, $($return)*)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.indices.size_hint()
            }

            fn count(self) -> usize
            where
                Self: Sized,
            {
                self.indices.count()
            }

            fn collect<B: FromIterator<Self::Item>>(self) -> B
            where
                Self: Sized,
            {
                self.indices
                    .filter_map(|&$index| {
                        debug_assert!($index < self.pairs_len);
                        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
                        let $pair = unsafe { &*self.pairs.as_ptr().add($index) };
                        Some($($return)*)
                    })
                    .collect()
            }
        }

        impl<'a, K, V> DoubleEndedIterator for $ty {
            fn next_back(&mut self) -> Option<Self::Item> {
                iter_methods!(@get_item self, self.indices.next_back(), $index, $pair, $($return)*)
            }

            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                iter_methods!(@get_item self, self.indices.nth_back(n), $index, $pair, $($return)*)
            }
        }

        impl<'a, K, V> ExactSizeIterator for $ty {
            fn len(&self) -> usize {
                self.indices.len()
            }
        }

        impl<'a, K, V> FusedIterator for $ty {}

        impl<'a, K, V> Clone for $ty {
            fn clone(&self) -> Self {
                Self {
                    pairs: self.pairs,
                    pairs_len: self.pairs_len,
                    indices: self.indices.clone(),
                    marker: PhantomData
                }
            }
        }


        impl<'a, K, V> fmt::Debug for $ty
        where
            K: fmt::Debug,
            V: fmt::Debug,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let pairs = self.indices.clone().map(|$index| {
                    debug_assert!(*$index < self.pairs_len);
                    // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
                    let $pair = unsafe { &*self.pairs.as_ptr().add(*$index) };
                    $($return)*
                });
                debug_subset(f, stringify!($name), pairs)
            }
        }
    };
}

/// An iterator over a subset of the pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::iter`] and [`SubsetMut::iter`].
/// See their documentation for more.
///
/// [`Subset::iter`]: super::Subset::iter
/// [`SubsetMut::iter`]: super::SubsetMut::iter
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetIter<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // # Safety
    //
    // * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    // * every index in indices must be in bounds to index into pairs
    //
    // * Also see the top of the module for more details. *
    pairs: NonNull<Bucket<K, V>>,
    pairs_len: usize,
    indices: slice::Iter<'a, usize>,
    marker: PhantomData<&'a [Bucket<K, V>]>,
}

iter_methods!(
    SubsetIter,
    SubsetIter<'a, K, V>,
    (usize, &'a K, &'a V),
    index,
    pair,
    (index, &pair.key, &pair.value)
);

/// An iterator over a subset of the keys in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::keys`] and [`SubsetMut::keys`].
/// See their documentation for more.
///
/// [`Subset::keys`]: super::Subset::keys
/// [`SubsetMut::keys`]: super::SubsetMut::keys
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetKeys<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // # Safety
    //
    // * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    // * every index in indices must be in bounds to index into pairs
    //
    // * Also see the top of the module for more details. *
    pairs: NonNull<Bucket<K, V>>,
    pairs_len: usize,
    indices: slice::Iter<'a, usize>,
    marker: PhantomData<&'a [Bucket<K, V>]>,
}

iter_methods!(
    SubsetKeys,
    SubsetKeys<'a, K, V>,
    &'a K,
    index,
    pair,
    &pair.key
);

/// An iterator over a subset of the values in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::values`] and [`SubsetMut::values`].
/// See their documentation for more.
///
/// [`Subset::values`]: super::Subset::values
/// [`SubsetMut::values`]: super::SubsetMut::values
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetValues<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // # Safety
    //
    // * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    // * every index in indices must be in bounds to index into pairs
    //
    // * Also see the top of the module for more details. *
    pairs: NonNull<Bucket<K, V>>,
    pairs_len: usize,
    indices: slice::Iter<'a, usize>,
    marker: PhantomData<&'a [Bucket<K, V>]>,
}

iter_methods!(
    SubsetValues,
    SubsetValues<'a, K, V>,
    &'a V,
    index,
    pair,
    &pair.value
);

/// A mutable iterator over a subset of the pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`SubsetMut::iter_mut`] method.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`SubsetMut::iter_mut`]: crate::multimap::SubsetMut::iter_mut
pub struct SubsetIterMut<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // # Safety
    //
    // * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    // * every index in indices must be in bounds to index into pairs
    //
    // * Also see the top of the module for more details. *
    pairs: NonNull<Bucket<K, V>>,
    pairs_len: usize,
    indices: UniqueIter<slice::Iter<'a, usize>>,
    // What self.pairs really is, constructors should take this to bind the lifetime properly
    marker: PhantomData<&'a mut [Bucket<K, V>]>,
}

impl<'a, K, V> SubsetIterMut<'a, K, V> {
    /// # Safety
    ///
    /// * `indices` must be in bounds to index into `pairs`.
    pub(super) unsafe fn from_slice_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        let pairs_len = pairs.len();
        let pairs = unsafe { NonNull::new_unchecked(pairs.as_mut_ptr()) };
        unsafe { Self::from_raw_unchecked(pairs, pairs_len, indices) }
    }

    /// # Safety
    ///
    /// * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    /// * `indices` must be in bounds to index into `pairs`
    pub(super) unsafe fn from_raw_unchecked(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            pairs,
            pairs_len,
            indices,
            marker: PhantomData,
        }
    }

    fn get_items(&self) -> impl Iterator<Item = (usize, &K, &V)> {
        // SAFETY:
        //  * `self.indices` only contains valid indices to index into `self.pairs`.
        //  * self.pairs is never modified (items are not removed and the pointer is not changed)
        //  * self.indices is an iterator that return unique indices,
        //    thus we cannot have returned item at this index,
        //    hence it's ok to create a reference and it cannot invalidate any
        //    previously returned mutable references
        //  * the lifetime of returned values is bound to the borrow of self by
        //    this function call. Thus any further call to Iterator methods will
        //    invalidate these references as it should.
        self.indices.clone().map(|&i| {
            debug_assert!(i < self.pairs_len);
            let pair = unsafe { &*self.pairs.as_ptr().add(i) };
            (i, &pair.key, &pair.value)
        })
    }

    /// # Safety
    ///
    /// * pairs_start = self.pairs.start
    /// * pairs_len = self.pairs_len
    /// * index must be from the iterator self.indices
    ///
    /// * pairs_start must be a pointer to the start of pairs_len contiguous initialized buckets
    /// * index must be valid to index into pairs (that is index < pairs_len)
    /// * this method cannot be called twice with the same index while the previously
    ///   returned references are alive
    #[inline]
    unsafe fn get_item_mut(
        pairs_start: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        index: usize,
    ) -> (usize, &'a K, &'a mut V) {
        debug_assert!(index < pairs_len);
        // SAFETY:
        //   * self.pairs is never changed after the construction,
        //     points to the start of pairs slice
        //   * self.indices are unique => we cannot return aliasing
        //     mutable references
        //   * self.indices is also an iterator => cannot use same index twice
        //   * `self.indices` only contains valid indices to index into `self.pairs`.
        //   * no one else can have access to the pairs slice as we
        //     borrowed it mutably for 'a => it's valid to return
        //     &'a mut into the slice
        let pair = unsafe { &mut *pairs_start.as_ptr().add(index) };
        (index, &pair.key, &mut pair.value)
    }
}

impl<'a, K, V> Iterator for SubsetIterMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    type Item = (usize, &'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices.next() {
            Some(&index) => Some(unsafe { Self::get_item_mut(self.pairs, self.pairs_len, index) }),
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.indices.count()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        match self.indices.last() {
            Some(&index) => Some(unsafe { Self::get_item_mut(self.pairs, self.pairs_len, index) }),
            None => None,
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.indices.nth(n) {
            Some(&index) => Some(unsafe { Self::get_item_mut(self.pairs, self.pairs_len, index) }),
            None => None,
        }
    }

    fn collect<B: FromIterator<Self::Item>>(self) -> B
    where
        Self: Sized,
    {
        self.indices
            .map(|&index| unsafe { Self::get_item_mut(self.pairs, self.pairs_len, index) })
            .collect()
    }
}

impl<'a, K, V> DoubleEndedIterator for SubsetIterMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.indices.next_back() {
            Some(&index) => Some(unsafe { Self::get_item_mut(self.pairs, self.pairs_len, index) }),
            None => None,
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match self.indices.nth_back(n) {
            Some(&index) => Some(unsafe { Self::get_item_mut(self.pairs, self.pairs_len, index) }),
            None => None,
        }
    }
}

impl<'a, K, V> ExactSizeIterator for SubsetIterMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<'a, K, V> FusedIterator for SubsetIterMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
}

impl<K, V: fmt::Debug> fmt::Debug for SubsetIterMut<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_subset(f, "SubsetIterMut", self.get_items())
    }
}

/// A mutable iterator over a subset of all the pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`SubsetMut::values_mut`] method.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`SubsetMut::values_mut`]: crate::multimap::SubsetMut::values_mut
pub struct SubsetValuesMut<'a, K, V> {
    // SAFETY: see `SubsetIterMut`'s definition
    inner: SubsetIterMut<'a, K, V>,
}

impl<'a, K, V> SubsetValuesMut<'a, K, V> {
    /// # Safety
    ///
    /// * `indices` must be in bounds to index into `pairs`.
    #[inline]
    pub(super) unsafe fn from_slice_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            inner: unsafe { SubsetIterMut::from_slice_unchecked(pairs, indices) },
        }
    }

    /// # Safety
    ///
    /// * `pairs` must be a valid pointer to `pairs_len` contiguous initialized buckets
    /// * `indices` must be in bounds to index into `pairs`
    pub(super) unsafe fn from_raw_unchecked(
        pairs: NonNull<Bucket<K, V>>,
        pairs_len: usize,
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            inner: unsafe { SubsetIterMut::from_raw_unchecked(pairs, pairs_len, indices) },
        }
    }
}

impl<'a, K, V> Iterator for SubsetValuesMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some((_, _, v)) => Some(v),
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.inner.count()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        match self.inner.last() {
            Some((_, _, v)) => Some(v),
            None => None,
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.inner.nth(n) {
            Some((_, _, v)) => Some(v),
            None => None,
        }
    }

    fn collect<B: FromIterator<Self::Item>>(self) -> B
    where
        Self: Sized,
    {
        self.inner.map(|(_, _, v)| v).collect()
    }
}

impl<'a, K, V> DoubleEndedIterator for SubsetValuesMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.inner.next_back() {
            Some((_, _, v)) => Some(v),
            None => None,
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match self.inner.nth_back(n) {
            Some((_, _, v)) => Some(v),
            None => None,
        }
    }
}

impl<'a, K, V> ExactSizeIterator for SubsetValuesMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for SubsetValuesMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
}

impl<K, V: fmt::Debug> fmt::Debug for SubsetValuesMut<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let items = self.inner.get_items().map(|(_, _, v)| v);
        debug_subset(f, "SubsetValuesMut", items)
    }
}

fn debug_subset<I>(f: &mut fmt::Formatter<'_>, name: &'static str, iter: I) -> fmt::Result
where
    I: Iterator,
    I::Item: fmt::Debug,
{
    if cfg!(feature = "test_debug") {
        debug_iter_as_numbered_compact_list(f, Some(name), iter)
    } else {
        debug_iter_as_list(f, Some(name), iter)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::super::indices::UniqueSlice;
    use super::*;
    use crate::util::is_unique;
    use crate::HashValue;

    #[test]
    fn unique() {
        fn bucket(k: usize, v: usize) -> Bucket<usize, usize> {
            Bucket {
                hash: HashValue(k),
                key: k,
                value: v,
            }
        }

        let mut pairs = vec![
            bucket(1, 11),
            bucket(2, 21),
            bucket(1, 12),
            bucket(1, 13),
            bucket(2, 22),
            bucket(1, 14),
        ];
        // SAFETY: inner is unique slice of usize, no interior  mutability and it's not mutable iterator
        let indices = unsafe { UniqueSlice::from_slice_unchecked([0usize, 2, 3, 5].as_slice()) };

        let iter1 = unsafe { SubsetIterMut::from_slice_unchecked(&mut pairs, indices.iter()) };
        let items = iter1.collect::<Vec<_>>();
        assert!(is_unique(&items));

        let mut iter2 = unsafe { SubsetIterMut::from_slice_unchecked(&mut pairs, indices.iter()) };
        let items = [
            iter2.next(),
            iter2.nth(1),
            iter2.next_back(),
            iter2.next(),
            iter2.nth_back(1),
            iter2.last(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        assert!(is_unique(&items));
    }
}

#![allow(unsafe_code)]

//! This module contains subset structs and iterators that can give references to
//! some subset of key-value pairs in the map.

/*

# Impl details

Subsets (and their iterators) are effectively a pointer to slice of pairs
and a set of indices used to index into that slice.
A subset or it's iterator will only touch item's to which it's indices point to.
That means it's okay for multiple mutable subsets to be alive at the same time
if their indices are disjoint.
In that case there is no way to create multiple aliasing mutable references.

This detail can be used by `get_many_mut` methods or `SubsetMut::split_at_mut`
or similar methods.

It's worth pointing out that while the non-mut subset and iterators can be
implemented using real slices &[Bucket<K, V>], it would be unsound to create them
from mutable subsets if we allow multiple disjoint mutable subsets to be alive
at the same time.

A safe implementation however needs to create a
reference to whole pairs slice, which would invalidate all other mutable subsets
from accessing their elements.
A case of don't mix pointers and references.

Thus under the hood all structs here use the RawIndexSlice and RawIndexSliceMut
which internally use pointers but expose mostly safe API as one would actually
have a &[T] or &mut [T].

*/

use ::core::iter::FusedIterator;
use ::core::marker::PhantomData;
use ::core::ops::RangeBounds;
use ::core::ptr::NonNull;
use ::core::{fmt, ops, slice};

use super::indices::{UniqueIter, UniqueSlice};
use crate::util::{
    check_unique_and_in_bounds, debug_iter_as_list, debug_iter_as_numbered_compact_list,
    try_simplify_range,
};
use crate::Bucket;

/// Immutable raw slice which behaves like `&[T]` without ever creating a reference to whole slice.
pub(crate) struct RawIndexSlice<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    marker: PhantomData<&'a [T]>,
}

impl<'a, T> RawIndexSlice<'a, T> {
    fn new(slice: &'a [T]) -> Self {
        let len = slice.len();
        // SAFETY: we don't ever return mutable reference into the slice
        let ptr = NonNull::new(slice.as_ptr() as *mut T).unwrap();
        Self {
            ptr,
            len,
            marker: PhantomData,
        }
    }
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            marker: self.marker,
        }
    }

    // We can return long lived reference because we never expose any mutability.
    fn get(&self, index: usize) -> Option<&'a T> {
        if index < self.len {
            // SAFETY:
            //  * we own &'a [T]
            //  * index is checked to be in bounds
            //  * lifetime is bound to 'a
            Some(unsafe { &*self.ptr.as_ptr().add(index) })
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Mutable raw slice which behaves like `&mut [T]` without ever creating a reference to whole slice.
/// Elements can only be accessed through indexing.s
pub(crate) struct RawIndexSliceMut<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> RawIndexSliceMut<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        let ptr = NonNull::new(slice.as_mut_ptr()).unwrap();
        Self {
            ptr,
            len,
            marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// After this call the caller must use the two raw slices without creating aliasing references.
    unsafe fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            marker: self.marker,
        }
    }

    fn reborrow(&mut self) -> RawIndexSliceMut<'_, T> {
        RawIndexSliceMut {
            ptr: self.ptr,
            len: self.len,
            marker: PhantomData,
        }
    }

    fn as_raw_slice(&self) -> RawIndexSlice<'_, T> {
        RawIndexSlice {
            ptr: self.ptr,
            len: self.len,
            marker: PhantomData,
        }
    }

    fn into_raw_slice(self) -> RawIndexSlice<'a, T> {
        RawIndexSlice {
            ptr: self.ptr,
            len: self.len,
            marker: PhantomData,
        }
    }

    fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY:
            //  * we own &'a mut [T]
            //  * index is checked to be in bounds
            //  * lifetime is bound to a borrow of self
            Some(unsafe { &*self.ptr.as_ptr().add(index) })
        } else {
            None
        }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            // SAFETY:
            //  * we own &'a mut [T]
            //  * index is checked to be in bounds
            //  * lifetime is bound to a borrow of self
            Some(unsafe { &mut *self.ptr.as_ptr().add(index) })
        } else {
            None
        }
    }

    fn into_mut(self, index: usize) -> Option<&'a mut T> {
        if index < self.len {
            // SAFETY:
            //  * we own &'a mut [T]
            //  * index is checked to be in bounds
            //  * lifetime is bound to a 'a and self is consumed
            Some(unsafe { &mut *self.ptr.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// * any index used to call this method can only be used to index into the slice once
    ///   That is the index used in this call cannot be used in any of the other `get` calls after.
    ///
    /// Note that this method still does the bounds checking. Unsafe part is returning a long lifetime.
    unsafe fn get_mut_long_lifetime(&mut self, index: usize) -> Option<&'a mut T> {
        if index < self.len {
            // SAFETY:
            //  * we own &'a mut [T]
            //  * index is checked to be in bounds
            //  * caller guarantees that the index is not used in any other `get` calls, eg they won't create a
            //    mutable reference to this element for at least 'a lifetime
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Slice like construct over a subset of the key-value pairs in the [`IndexMultimap`].
///
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct Subset<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // * Also see the top of the module for more details. *
    pairs: RawIndexSlice<'a, Bucket<K, V>>,
    indices: &'a [usize],
}

impl<'a, K, V> Subset<'a, K, V> {
    pub(super) fn new(pairs: &'a [Bucket<K, V>], indices: &'a [usize]) -> Self {
        let raw_slice = RawIndexSlice::new(pairs);
        Self::from_raw_slice(raw_slice, indices)
    }

    pub(super) fn from_raw_slice(
        pairs: RawIndexSlice<'a, Bucket<K, V>>,
        indices: &'a [usize],
    ) -> Self {
        Self { pairs, indices }
    }

    pub(crate) fn empty() -> Self {
        Self::new(&[], &[])
    }

    /// Returns the number of pairs in this subset
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns [`true`] if this subset is empty, [`false`] otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn indices(&self) -> &'a [usize] {
        self.indices
    }

    /// Returns a reference to the `n`th pair in this subset or [`None`] if <code>n >= self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn nth(&self, n: usize) -> Option<(usize, &'a K, &'a V)> {
        let index = *self.indices.get(n)?;
        let b = self.pairs.get(index)?;
        Some((index, &b.key, &b.value))
    }

    /// Return a reference to the first pair in this subset or [`None`] if this subset is empty.
    pub fn first(&self) -> Option<(usize, &'a K, &'a V)> {
        let index = *self.indices.first()?;
        let b = self.pairs.get(index)?;
        Some((index, &b.key, &b.value))
    }

    /// Returns a reference to the last pair in this subset or [`None`] if this subset is empty.
    pub fn last(&self) -> Option<(usize, &'a K, &'a V)> {
        let index = *self.indices.last()?;
        let b = self.pairs.get(index)?;
        Some((index, &b.key, &b.value))
    }

    /// Returns a immutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are <code>0 <= index < self.[len]\()</code>
    ///
    /// [len]: Self::len
    pub fn get_range<R>(&self, range: R) -> Option<Subset<'a, K, V>>
    where
        R: RangeBounds<usize>,
    {
        let range = try_simplify_range(range, self.indices.len())?;
        let indices = self.indices.get(range)?;
        Some(Subset::from_raw_slice(self.pairs.clone(), indices))
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'a, K, V> {
        SubsetIter::from_raw_slice(self.pairs.clone(), self.indices.iter())
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yields one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'a, K, V> {
        SubsetKeys::from_raw_slice(self.pairs.clone(), self.indices.iter())
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'a, K, V> {
        SubsetValues::from_raw_slice(self.pairs.clone(), self.indices.iter())
    }

    /// Divides one subset into two non-mutable subsets at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if <code>mid > self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn split_at(&self, mid: usize) -> (Subset<'a, K, V>, Subset<'a, K, V>) {
        let (left, right) = self.indices.split_at(mid);

        (
            Subset::from_raw_slice(self.pairs.clone(), left),
            Subset::from_raw_slice(self.pairs.clone(), right),
        )
    }

    /// Returns the first and all the rest of the elements of the subset, or [`None`] if it is empty.
    pub fn split_first(&self) -> Option<((usize, &'a K, &'a V), Subset<'a, K, V>)> {
        let (&first_index, rest) = self.indices.split_first()?;
        let first = self.pairs.get(first_index)?;
        let first = (first_index, &first.key, &first.value);
        Some((first, Subset::from_raw_slice(self.pairs.clone(), rest)))
    }

    /// Returns the last and all the rest of the elements of the subset, or [`None`] if it is empty.
    pub fn split_last(&self) -> Option<((usize, &'a K, &'a V), Subset<'a, K, V>)> {
        let (&last_index, rest) = self.indices.split_last()?;
        let last = self.pairs.get(last_index)?;
        let first = (last_index, &last.key, &last.value);
        Some((first, Subset::from_raw_slice(self.pairs.clone(), rest)))
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

impl<'a, K, V> ops::Index<usize> for Subset<'a, K, V> {
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        let index = self.indices[index];
        &self.pairs.get(index).unwrap().value
    }
}

impl<'a, K, V> Clone for Subset<'a, K, V> {
    fn clone(&self) -> Self {
        Self {
            pairs: self.pairs.clone(),
            indices: self.indices,
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
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetMut<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // * Also see the top of the module for more details. *
    pairs: RawIndexSliceMut<'a, Bucket<K, V>>,
    indices: &'a UniqueSlice<usize>,
}

impl<'a, K, V> SubsetMut<'a, K, V> {
    pub(crate) fn new(pairs: &'a mut [Bucket<K, V>], indices: &'a UniqueSlice<usize>) -> Self {
        let pairs = RawIndexSliceMut::new(pairs);
        Self::from_raw_slice(pairs, indices)
    }

    pub(crate) fn from_raw_slice(
        pairs: RawIndexSliceMut<'a, Bucket<K, V>>,
        indices: &'a UniqueSlice<usize>,
    ) -> Self {
        Self { pairs, indices }
    }

    pub(crate) fn empty() -> Self {
        Self::new(&mut [], UniqueSlice::empty())
    }

    /// Returns the number of pairs in this subset.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns [`true`] if this subset is empty, [`false`] otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a slice of indices where the key-value pairs of this subset are
    /// located in the map.
    pub fn indices(&self) -> &[usize] {
        self.indices
    }

    /// Returns a reference to an `n`th key-value pair in this subset or [`None`] if <code>n >= self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn nth(&self, n: usize) -> Option<(usize, &K, &V)> {
        self.as_subset().nth(n)
    }

    /// Returns a mutable reference to an `n`th pair in this subset or [`None`] if <code>n >= self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        let index = *self.indices.get(n)?;
        let b = self.pairs.get_mut(index)?;
        Some((index, &b.key, &mut b.value))
    }

    /// Converts `self` into a long lived mutable reference to an `n`th pair in this subset or [`None`] if <code>n >= self.[len]\()</code>.
    ///
    /// [len]: Self::len
    pub fn into_nth_mut(self, n: usize) -> Option<(usize, &'a K, &'a mut V)> {
        let index = *self.indices.get(n)?;
        let b = self.pairs.into_mut(index)?;
        Some((index, &b.key, &mut b.value))
    }

    /// Return a reference to the first pair in this subset or [`None`] if this subset is empty.
    pub fn first(&self) -> Option<(usize, &K, &V)> {
        self.as_subset().first()
    }

    /// Return a mutable reference to the first pair in this subset or [`None`] if this subset is empty.
    pub fn first_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        let index = *self.indices.first()?;
        let b = self.pairs.get_mut(index)?;
        Some((index, &b.key, &mut b.value))
    }

    /// Converts `self` into long lived mutable reference to the first pair in this subset or [`None`] if this subset is empty.
    pub fn into_first_mut(self) -> Option<(usize, &'a K, &'a mut V)> {
        let index = *self.indices.first()?;
        let b = self.pairs.into_mut(index)?;
        Some((index, &b.key, &mut b.value))
    }

    /// Returns a reference to the last pair in this subset or [`None`] if this subset is empty.
    pub fn last(&self) -> Option<(usize, &K, &V)> {
        self.as_subset().last()
    }

    /// Returns a mutable reference to the last pair in this subset or [`None`] if this subset is empty.
    pub fn last_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        let index = *self.indices.last()?;
        let b = self.pairs.get_mut(index)?;
        Some((index, &b.key, &mut b.value))
    }

    /// Converts `self` into long lived mutable reference to the last pair in this subset or [`None`] if this subset is empty.
    pub fn into_last_mut(self) -> Option<(usize, &'a K, &'a mut V)> {
        let index = *self.indices.last()?;
        let b = self.pairs.into_mut(index)?;
        Some((index, &b.key, &mut b.value))
    }

    /// Returns a immutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are <code>0 <= index < self.[len]\()</code>
    ///
    /// [len]: Self::len
    pub fn get_range<R>(&self, range: R) -> Option<Subset<'_, K, V>>
    where
        R: RangeBounds<usize>,
    {
        self.as_subset().get_range(range)
    }

    /// Returns a mutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are <code>0 <= index < self.[len]\()</code>
    ///
    /// [len]: Self::len
    pub fn get_range_mut<R>(&mut self, range: R) -> Option<SubsetMut<'_, K, V>>
    where
        R: RangeBounds<usize>,
    {
        let indices = self.indices.get_range(range)?;
        Some(SubsetMut::from_raw_slice(self.pairs.reborrow(), indices))
    }

    /// Converts `self` into a mutable subset of key-value pairs in the given range of indices.
    ///
    /// Valid indices are <code>0 <= index < self.[len]\()</code>
    ///
    /// [len]: Self::len
    pub fn into_range<R>(self, range: R) -> Option<SubsetMut<'a, K, V>>
    where
        R: RangeBounds<usize>,
    {
        let indices = self.indices.get_range(range)?;
        Some(SubsetMut::from_raw_slice(self.pairs, indices))
    }

    /// Returns mutable references to many items at once or [`None`] if any index
    /// is out-of-bounds, or if the same index was passed more than once.
    pub fn get_many_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> Option<[(usize, &K, &mut V); N]> {
        if !check_unique_and_in_bounds(&indices, self.indices.len()) {
            return None;
        }

        Some(indices.map(move |i| {
            let index = *self.indices.get(i).unwrap();
            let b = unsafe { self.pairs.get_mut_long_lifetime(index).unwrap() };
            (index, &b.key, &mut b.value)
        }))
    }

    /// Returns mutable references to many items at once or [`None`] if any index
    /// is out-of-bounds, or if the same index was passed more than once.
    pub fn into_many_mut<const N: usize>(
        mut self,
        indices: [usize; N],
    ) -> Option<[(usize, &'a K, &'a mut V); N]> {
        if !check_unique_and_in_bounds(&indices, self.indices.len()) {
            return None;
        }

        Some(indices.map(move |i| {
            let index = *self.indices.get(i).unwrap();
            let b = unsafe { self.pairs.get_mut_long_lifetime(index).unwrap() };
            (index, &b.key, &mut b.value)
        }))
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'_, K, V> {
        self.as_subset().iter()
    }

    /// Returns a mutable iterator over all the pairs in this subset.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V> {
        SubsetIterMut::from_raw_slice(self.pairs.reborrow(), self.indices.iter())
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yield one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'_, K, V> {
        self.as_subset().keys()
    }

    /// Converts into a iterator over all the keys in this subset.
    pub fn into_keys(self) -> SubsetKeys<'a, K, V> {
        self.into_subset().keys()
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'_, K, V> {
        self.as_subset().values()
    }

    /// Returns a mutable iterator over all the values in this subset.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V> {
        SubsetValuesMut::from_raw_slice(self.pairs.reborrow(), self.indices.iter())
    }

    /// Converts `self` into a mutable iterator over all the values in this subset.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V> {
        SubsetValuesMut::from_raw_slice(self.pairs, self.indices.iter())
    }

    /// Borrows `self` as an immutable subset of same pairs.
    pub fn as_subset(&self) -> Subset<'_, K, V> {
        Subset::from_raw_slice(self.pairs.as_raw_slice(), self.indices)
    }

    /// Converts `self` into an immutable subset of same pairs.
    pub fn into_subset(self) -> Subset<'a, K, V> {
        Subset::from_raw_slice(self.pairs.into_raw_slice(), self.indices)
    }

    /// Divides one mutable subset into two non-mutable subsets at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// If you need a longer lived subsets, see [`split_at_into`].
    ///
    /// # Panics
    ///
    /// Panics if <code>mid > self.[len]\()</code>.
    ///
    /// [`split_at_into`]: Self::split_at_into
    /// [len]: Self::len
    pub fn split_at(&self, mid: usize) -> (Subset<'_, K, V>, Subset<'_, K, V>) {
        self.as_subset().split_at(mid)
    }

    /// Divides one mutable subset into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// If you need a longer lived subsets, see [`split_at_into`].
    ///
    /// # Panics
    ///
    /// Panics if <code>mid > self.[len]\()</code>.
    ///
    /// [`split_at_into`]: Self::split_at_into
    /// [len]: Self::len
    pub fn split_at_mut(&mut self, mid: usize) -> (SubsetMut<'_, K, V>, SubsetMut<'_, K, V>) {
        let (left, right) = self.indices.split_at(mid);

        let pairs1 = self.pairs.reborrow();
        // SAFETY: since self.indices are all unique, then left and right are completely disjoint,
        //   Thus created SubsetMut's cannot access same elements in `pairs`.
        let pairs2 = unsafe { pairs1.clone() };

        (
            SubsetMut::from_raw_slice(pairs1, left),
            SubsetMut::from_raw_slice(pairs2, right),
        )
    }

    /// Divides one mutable subset into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// This method consumes `self` in order to return a longer lived subsets.
    /// If don't need it or need to keep original complete subset around,
    /// see [`split_at_mut`] or [`split_at`].
    ///
    /// # Panics
    ///
    /// Panics if <code>mid > self.[len]\()</code>.
    ///
    /// [`split_at`]: Self::split_at
    /// [`split_at_mut`]: Self::split_at_mut
    /// [len]: Self::len
    pub fn split_into(self, mid: usize) -> (SubsetMut<'a, K, V>, SubsetMut<'a, K, V>) {
        let (left, right) = self.indices.split_at(mid);

        (
            // SAFETY: since self.indices are all unique, then left and right are completely disjoint,
            //   Thus created SubsetMut's cannot access same elements in `pairs``.
            SubsetMut::from_raw_slice(unsafe { self.pairs.clone() }, left),
            SubsetMut::from_raw_slice(self.pairs, right),
        )
    }

    /// Returns the first and all the rest of the elements of the subset, or [`None`] if it is empty.
    pub fn split_first(&self) -> Option<((usize, &'_ K, &'_ V), Subset<'_, K, V>)> {
        self.as_subset().split_first()
    }

    /// Returns the first and all the rest of the elements of the subset, or [`None`] if it is empty.
    pub fn split_first_mut(&mut self) -> Option<((usize, &'_ K, &'_ mut V), SubsetMut<'_, K, V>)> {
        let (&first_index, rest) = self.indices.split_first()?;
        // SAFETY: since self.indices are all unique, then first and rest are completely disjoint,
        //   Thus created SubsetMut cannot access the element corresponding to `first_index`.
        let mut pairs = self.pairs.reborrow();
        let first = unsafe { pairs.get_mut_long_lifetime(first_index).unwrap() };
        let first = (first_index, &first.key, &mut first.value);
        Some((first, SubsetMut::from_raw_slice(pairs, rest)))
    }

    /// Returns the last and all the rest of the elements of the subset, or [`None`] if it is empty.
    pub fn split_last(&self) -> Option<((usize, &'_ K, &'_ V), Subset<'_, K, V>)> {
        self.as_subset().split_last()
    }

    /// Returns the last and all the rest of the elements of the subset, or [`None`] if it is empty.
    pub fn split_last_mut(&mut self) -> Option<((usize, &'_ K, &'_ mut V), SubsetMut<'_, K, V>)> {
        let (&last_index, rest) = self.indices.split_last()?;
        // SAFETY: since self.indices are all unique, then last and rest are completely disjoint,
        //   Thus created SubsetMut cannot access the element corresponding to `last_index`.
        let mut pairs = self.pairs.reborrow();
        let last = unsafe { pairs.get_mut_long_lifetime(last_index).unwrap() };
        let last = (last_index, &last.key, &mut last.value);
        Some((last, SubsetMut::from_raw_slice(pairs, rest)))
    }

    /// Takes the first element out of the subset and returns a long lived
    /// reference to it, or [`None`] if subset is empty.
    ///
    /// The returned element will remain in the map/pairs slice but not in this subset.
    pub fn take_first(&mut self) -> Option<(usize, &'a K, &'a mut V)> {
        let (&first_index, rest) = self.indices.split_first()?;
        self.indices = rest;
        // SAFETY: since self.indices are all unique, then first and rest are completely disjoint,
        //   Since we replace self.indices with rest then, self cannot access the element corresponding
        //   to `first_index` anymore.
        let first = unsafe { self.pairs.get_mut_long_lifetime(first_index).unwrap() };
        Some((first_index, &first.key, &mut first.value))
    }

    /// Takes the last element out of the subset and returns a long lived
    /// reference to it, or [`None`] if subset is empty.
    ///
    /// The returned element will remain in the map/pairs slice but not in this subset.
    pub fn take_last(&mut self) -> Option<(usize, &'a K, &'a mut V)> {
        let (&last_index, rest) = self.indices.split_last()?;
        self.indices = rest;
        // SAFETY: since self.indices are all unique, then last and rest are completely disjoint,
        //   Since we replace self.indices with rest then, self cannot access the element corresponding
        //   to `last_index` anymore.
        let last = unsafe { self.pairs.get_mut_long_lifetime(last_index).unwrap() };
        Some((last_index, &last.key, &mut last.value))
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
        SubsetIterMut::from_raw_slice(self.pairs, self.indices.iter())
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

impl<K, V> ops::Index<usize> for SubsetMut<'_, K, V> {
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        let index = self.indices[index];
        let b = self.pairs.get(index).unwrap();
        &b.value
    }
}

impl<K, V> ops::IndexMut<usize> for SubsetMut<'_, K, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let index = self.indices[index];
        let b = self.pairs.get_mut(index).unwrap();
        &mut b.value
    }
}

macro_rules! iter_methods {
    (@get_item $self:ident, $index_value:expr, $index:ident, $pair:ident, $($return:tt)*) => {
        match $index_value {
            Some(&$index) => {
                debug_assert!($index < $self.pairs.len(), "expected indices to be in bounds");
                let $pair = $self.pairs.get($index)?;
                Some($($return)*)
            }
            None => None,
        }
    };
    ($name:ident, $ty:ty, $item:ty, $index:ident, $pair:ident, $($return:tt)*) => {

        impl<'a, K, V> $ty
        {
            pub(super) fn new(pairs: &'a [Bucket<K, V>], indices: slice::Iter<'a, usize>) -> Self {
                Self::from_raw_slice(RawIndexSlice::new(pairs), indices)
            }

            pub(super) fn from_raw_slice(pairs: RawIndexSlice<'a, Bucket<K, V>>, indices: slice::Iter<'a, usize>) -> Self {
                Self {
                    pairs,
                    indices
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
                        debug_assert!($index < self.pairs.len());

                        let $pair = self.pairs.get($index).unwrap();
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
                    pairs: self.pairs.clone(),
                    indices: self.indices.clone(),
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
                    debug_assert!(*$index < self.pairs.len());
                    let $pair = self.pairs.get(*$index).unwrap();
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
/// [`Subset::iter`]: crate::multimap::Subset::iter
/// [`SubsetMut::iter`]: crate::multimap::SubsetMut::iter
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetIter<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // * Also see the top of the module for more details. *
    pairs: RawIndexSlice<'a, Bucket<K, V>>,
    indices: slice::Iter<'a, usize>,
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
/// [`Subset::keys`]: crate::multimap::Subset::keys
/// [`SubsetMut::keys`]: crate::multimap::SubsetMut::keys
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetKeys<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // * Also see the top of the module for more details. *
    pairs: RawIndexSlice<'a, Bucket<K, V>>,
    indices: slice::Iter<'a, usize>,
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
/// [`Subset::values`]: crate::multimap::Subset::values
/// [`SubsetMut::values`]: crate::multimap::SubsetMut::values
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetValues<'a, K, V> {
    // # Guarantees
    //
    // * implementation will only create references to pairs that `indices` point to
    //   that is, it's okay to create mutable references to pairs that are
    //   disjoint from this subset's pairs while this subset is alive
    //
    // * Also see the top of the module for more details. *
    pairs: RawIndexSlice<'a, Bucket<K, V>>,
    indices: slice::Iter<'a, usize>,
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
    // * Also see the top of the module for more details. *
    pairs: RawIndexSliceMut<'a, Bucket<K, V>>,
    indices: UniqueIter<slice::Iter<'a, usize>>,
}

impl<'a, K, V> SubsetIterMut<'a, K, V> {
    pub(super) fn new(
        pairs: &'a mut [Bucket<K, V>],
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        let pairs = RawIndexSliceMut::new(pairs);
        Self::from_raw_slice(pairs, indices)
    }

    pub(super) fn from_raw_slice(
        pairs: RawIndexSliceMut<'a, Bucket<K, V>>,
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self { pairs, indices }
    }

    fn get_items(&self) -> impl Iterator<Item = (usize, &K, &V)> {
        self.indices.clone().map(|&i| {
            let pair = self.pairs.get(i).unwrap();
            (i, &pair.key, &pair.value)
        })
    }

    fn get_item(
        &mut self,
        index: impl FnOnce(&mut UniqueIter<slice::Iter<'a, usize>>) -> Option<&'a usize>,
    ) -> Option<(usize, &'a K, &'a mut V)> {
        match index(&mut self.indices) {
            Some(&index) => {
                // SAFETY:
                //   * self.indices are unique => we cannot return aliasing
                //     mutable references
                //   * self.indices is also an iterator => cannot use same index twice
                // Hence it's ok to return a &'a mut V
                let b = unsafe { self.pairs.get_mut_long_lifetime(index) };
                b.map(|b| (index, &b.key, &mut b.value))
            }
            None => None,
        }
    }
}

impl<'a, K, V> Iterator for SubsetIterMut<'a, K, V> {
    type Item = (usize, &'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.get_item(|i| i.next())
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

    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.get_item(|i| i.last())
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.get_item(move |i| i.nth(n))
    }

    fn collect<B: FromIterator<Self::Item>>(mut self) -> B
    where
        Self: Sized,
    {
        self.indices
            .map(|&index| {
                // SAFETY:
                //   * self.indices are unique => we cannot return aliasing
                //     mutable references
                //   * self.indices is also an iterator => cannot use same index twice
                // Hence it's ok to return a &'a mut V
                let b = unsafe { self.pairs.get_mut_long_lifetime(index) }.unwrap();
                (index, &b.key, &mut b.value)
            })
            .collect()
    }
}

impl<'a, K, V> DoubleEndedIterator for SubsetIterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.get_item(|i| i.next_back())
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.get_item(|i| i.nth_back(n))
    }
}

impl<'a, K, V> ExactSizeIterator for SubsetIterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<'a, K, V> FusedIterator for SubsetIterMut<'a, K, V> {}

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
    #[inline]
    pub(super) fn new(
        pairs: &'a mut [Bucket<K, V>],
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            inner: SubsetIterMut::new(pairs, indices),
        }
    }

    pub(super) fn from_raw_slice(
        pairs: RawIndexSliceMut<'a, Bucket<K, V>>,
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            inner: SubsetIterMut::from_raw_slice(pairs, indices),
        }
    }
}

impl<'a, K, V> Iterator for SubsetValuesMut<'a, K, V> {
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

impl<'a, K, V> DoubleEndedIterator for SubsetValuesMut<'a, K, V> {
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

impl<'a, K, V> ExactSizeIterator for SubsetValuesMut<'a, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for SubsetValuesMut<'a, K, V> {}

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
    use ::alloc::vec;
    use ::alloc::vec::Vec;

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

        let iter1 = SubsetIterMut::new(&mut pairs, indices.iter());
        let items = iter1.collect::<Vec<_>>();
        assert!(is_unique(&items));

        let mut iter2 = SubsetIterMut::new(&mut pairs, indices.iter());
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

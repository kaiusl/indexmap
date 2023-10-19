#![allow(unsafe_code)]

// This module contains subset structs and iterators that can give references to
// some subset of key-value pairs in the map.

use core::slice;

use ::core::fmt;
use ::core::iter::FusedIterator;
use ::core::marker::PhantomData;

use super::indices::{UniqueIter, UniqueSlice};
use crate::util::{debug_iter_as_list, debug_iter_as_numbered_compact_list};
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
    // # Safety
    //
    // * every index in indices must be in bounds to index into pairs
    pairs: &'a [Bucket<K, V>],
    indices: &'a [usize],
}

impl<'a, K, V> Subset<'a, K, V> {
    /// # Safety
    ///
    /// * `indices` must be in bounds to index into `pairs`.
    pub(super) unsafe fn new_unchecked(pairs: &'a [Bucket<K, V>], indices: &'a [usize]) -> Self {
        Self { pairs, indices }
    }

    pub(crate) fn empty() -> Self {
        Self {
            pairs: Default::default(),
            indices: Default::default(),
        }
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
        unsafe { Self::get_item_unchecked(self.pairs, self.indices.get(n)) }
    }

    /// Return a reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first(&self) -> Option<(usize, &'a K, &'a V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.indices.first()) }
    }

    /// Returns a reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last(&self) -> Option<(usize, &'a K, &'a V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.indices.last()) }
    }

    /// # Safety
    ///
    /// * If `index` is `Some`, it must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn get_item_unchecked<'b>(
        pairs: &'b [Bucket<K, V>],
        index: Option<&usize>,
    ) -> Option<(usize, &'b K, &'b V)> {
        match index {
            Some(&index) => {
                debug_assert!(index < pairs.len(), "index out of bounds");
                // SAFETY: caller must guarantee that `index` is in bounds
                let Bucket { key, value, .. } = unsafe { pairs.get_unchecked(index) };
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetIter::new_unchecked(self.pairs, self.indices.iter()) }
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yields one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetKeys::new_unchecked(self.pairs, self.indices.iter()) }
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetValues::new_unchecked(self.pairs, self.indices.iter()) }
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
        debug_assert!(index < self.pairs.len(), "index out of bounds");
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        &unsafe { self.pairs.get_unchecked(index) }.value
    }
}

impl<'a, K, V> Clone for Subset<'a, K, V> {
    fn clone(&self) -> Self {
        Self {
            pairs: self.pairs,
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
/// This `struct` is created by the [`IndexMultimap::get_mut`],
/// [`OccupiedEntry::get_mut`], [`Entry::or_insert`] and other similar methods.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::get_mut`]: crate::IndexMultimap::get_mut
/// [`OccupiedEntry::get_mut`]: crate::multimap::OccupiedEntry::get_mut
/// [`Entry::or_insert`]: crate::multimap::Entry::or_insert
pub struct SubsetMut<'a, K, V> {
    // # Safety
    //
    // * every index in indices must be in bounds to index into pairs
    pairs: &'a mut [Bucket<K, V>],
    indices: &'a UniqueSlice<usize>,
}

impl<'a, K, V> SubsetMut<'a, K, V> {
    /// # Safety
    ///
    /// * `ìndices` must be in bounds to index into `pairs`.
    pub(crate) unsafe fn new_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: &'a UniqueSlice<usize>,
    ) -> Self {
        Self { pairs, indices }
    }

    pub(crate) fn empty() -> Self {
        // SAFETY: no access will ever be actually performed
        Self {
            pairs: Default::default(),
            indices: Default::default(),
        }
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
        unsafe { Self::get_item_unchecked(self.pairs, self.indices.get(n)) }
    }

    /// Returns a mutable reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.indices.get(n)) }
    }

    /// Converts `self` into a long lived mutable reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn into_nth(self, n: usize) -> Option<(usize, &'a K, &'a mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.indices.get(n)) }
    }

    /// Return a reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first(&self) -> Option<(usize, &K, &V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.indices.first()) }
    }

    /// Return a mutable reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.indices.first()) }
    }

    /// Converts `self` into long lived mutable reference to the first pair in this subset or `None` if this subset is empty.
    pub fn into_first_mut(self) -> Option<(usize, &'a K, &'a mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.indices.first()) }
    }

    /// Returns a reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last(&self) -> Option<(usize, &K, &V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked(self.pairs, self.indices.last()) }
    }

    /// Returns a mutable reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.indices.last()) }
    }

    /// Converts `self` into long lived mutable reference to the last pair in this subset or `None` if this subset is empty.
    pub fn into_last_mut(self) -> Option<(usize, &'a K, &'a mut V)> {
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        unsafe { Self::get_item_unchecked_mut(self.pairs, self.indices.last()) }
    }

    /// # Safety
    ///
    /// * If `index` is `Some`, it must be in bounds to index into `pairs`.
    #[inline]
    unsafe fn get_item_unchecked<'b>(
        pairs: &'b [Bucket<K, V>],
        index: Option<&usize>,
    ) -> Option<(usize, &'b K, &'b V)> {
        match index {
            Some(&index) => {
                debug_assert!(index < pairs.len(), "index out of bounds");
                // SAFETY: caller must guarantee that `index` is in bounds
                let Bucket { key, value, .. } = unsafe { pairs.get_unchecked(index) };
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
        pairs: &'b mut [Bucket<K, V>],
        index: Option<&usize>,
    ) -> Option<(usize, &'b K, &'b mut V)> {
        match index {
            Some(&index) => {
                debug_assert!(index < pairs.len(), "index out of bounds");
                // SAFETY: caller must guarantee that `index` is in bounds
                let Bucket { key, value, .. } = unsafe { pairs.get_unchecked_mut(index) };
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetIter::new_unchecked(self.pairs, self.indices.as_slice().iter()) }
    }

    /// Returns a mutable iterator over all the pairs in this subset.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetIterMut::new_unchecked(self.pairs, self.indices.iter()) }
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yield one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetKeys::new_unchecked(self.pairs, self.indices.as_slice().iter()) }
    }

    /// Converts into a iterator over all the keys in this subset.
    pub fn into_keys(self) -> SubsetKeys<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetKeys::new_unchecked(self.pairs, self.indices.as_slice().into_iter()) }
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetValues::new_unchecked(self.pairs, self.indices.as_slice().iter()) }
    }

    /// Returns a mutable iterator over all the values in this subset.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetValuesMut::new_unchecked(self.pairs, self.indices.iter()) }
    }

    /// Converts into a mutable iterator over all the values in this subset.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V> {
        // SAFETY: `self.indices` (and consequently it's iterator) only contains
        // valid indices to index into `self.pairs`.
        unsafe { SubsetValuesMut::new_unchecked(self.pairs, self.indices.iter()) }
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
        unsafe { SubsetIterMut::new_unchecked(self.pairs, self.indices.iter()) }
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
        debug_assert!(index < self.pairs.len(), "index out of bounds");
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        &unsafe { self.pairs.get_unchecked(index) }.value
    }
}

impl<K, V> core::ops::IndexMut<usize> for SubsetMut<'_, K, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let index = self.indices[index];
        debug_assert!(index < self.pairs.len(), "index out of bounds");
        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
        &mut unsafe { self.pairs.get_unchecked_mut(index) }.value
    }
}

macro_rules! iter_methods {
    (@get_item $self:ident, $index_value:expr, $index:ident, $pair:ident, $($return:tt)*) => {
        match $index_value {
            Some(&$index) => {
                debug_assert!($index < $self.pairs.len());
                // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
                let $pair = unsafe { $self.pairs.get_unchecked($index) };
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
            /// * `ìndices` must be in bounds to index into `pairs`.
            pub(super) unsafe fn new_unchecked(pairs: &'a [Bucket<K, V>], indices: slice::Iter<'a, usize>) -> Self {
                Self { pairs, indices }
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
                        // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
                        let $pair = unsafe { self.pairs.get_unchecked($index) };
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
                    // SAFETY: `self.indices` only contains valid indices to index into `self.pairs`.
                    let $pair = unsafe { self.pairs.get_unchecked(*$index) };
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
    // # Safety
    //
    // * every index in indices must be in bounds to index into pairs
    pairs: &'a [Bucket<K, V>],
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
/// [`Subset::keys`]: super::Subset::keys
/// [`SubsetMut::keys`]: super::SubsetMut::keys
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetKeys<'a, K, V> {
    // # Safety
    //
    // * every index in indices must be in bounds to index into pairs
    pairs: &'a [Bucket<K, V>],
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
/// [`Subset::values`]: super::Subset::values
/// [`SubsetMut::values`]: super::SubsetMut::values
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetValues<'a, K, V> {
    // # Safety
    //
    // * every index in indices must be in bounds to index into pairs
    pairs: &'a [Bucket<K, V>],
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
    // # Safety
    //
    // * every index in indices must be in bounds to index into pairs
    // * pairs_start must be a pointer to the start of pairs_len contiguous initialized buckets
    pairs_start: *mut Bucket<K, V>,
    pairs_len: usize,
    indices: UniqueIter<slice::Iter<'a, usize>>,
    // What self.pairs really is, constructors should take this to bind the lifetime properly
    _marker: PhantomData<&'a mut [Bucket<K, V>]>,
}

impl<'a, K, V> SubsetIterMut<'a, K, V> {
    /// # Safety
    ///
    /// * `indices` must be in bounds to index into `pairs`.
    pub(super) unsafe fn new_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            pairs_start: pairs.as_mut_ptr(),
            pairs_len: pairs.len(),
            indices,
            _marker: PhantomData,
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
            let pair = unsafe { &*self.pairs_start.add(i) };
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
        pairs_start: *mut Bucket<K, V>,
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
        let pair = unsafe { &mut *pairs_start.add(index) };
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
            Some(&index) => {
                Some(unsafe { Self::get_item_mut(self.pairs_start, self.pairs_len, index) })
            }
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
            Some(&index) => {
                Some(unsafe { Self::get_item_mut(self.pairs_start, self.pairs_len, index) })
            }
            None => None,
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.indices.nth(n) {
            Some(&index) => {
                Some(unsafe { Self::get_item_mut(self.pairs_start, self.pairs_len, index) })
            }
            None => None,
        }
    }

    fn collect<B: FromIterator<Self::Item>>(self) -> B
    where
        Self: Sized,
    {
        self.indices
            .map(|&index| unsafe { Self::get_item_mut(self.pairs_start, self.pairs_len, index) })
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
            Some(&index) => {
                Some(unsafe { Self::get_item_mut(self.pairs_start, self.pairs_len, index) })
            }
            None => None,
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match self.indices.nth_back(n) {
            Some(&index) => {
                Some(unsafe { Self::get_item_mut(self.pairs_start, self.pairs_len, index) })
            }
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
    // SAFETY: see SubsetIterMut's definition
    inner: SubsetIterMut<'a, K, V>,
}

impl<'a, K, V> SubsetValuesMut<'a, K, V> {
    /// # Safety
    ///
    /// * `ìndices` must be in bounds to index into `pairs`.
    #[inline]
    pub(super) unsafe fn new_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            inner: unsafe { SubsetIterMut::new_unchecked(pairs, indices) },
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

        let iter1 = unsafe { SubsetIterMut::new_unchecked(&mut pairs, indices.iter()) };
        let items = iter1.collect::<Vec<_>>();
        assert!(is_unique(&items));

        let mut iter2 = unsafe { SubsetIterMut::new_unchecked(&mut pairs, indices.iter()) };
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

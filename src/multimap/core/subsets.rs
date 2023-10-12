#![allow(unsafe_code)]

// This module contains subset structs and iterators that can give references to
// some subset of key-value pairs in the map.
//
// The overall structure of each object is:
//   pairs: reference/pointer to all the pairs in the map,
//   indices: slice or an iterator with indices
//
// At the moment all implementations assume that all of the indices are in
// bound for indexing into pairs. Each access still does a bound check but
// the behavior for out of bounds access is a panic.
// This is ok for now as we control the indices and the panic would indicate a
// bug in our implementation.
// However this requires more thought if we end up providing methods that allow
// the user to provide the indices. In that case out of bound indices are a real
// possibility that needs to be dealt with.
// ---

use ::core::fmt;
use ::core::iter::FusedIterator;
use ::core::marker::PhantomData;
use core::{iter, slice};

use crate::util::{debug_iter_as_list, debug_iter_as_numbered_compact_list};
use crate::Bucket;

use super::indices::{UniqueIter, UniqueSlice};

/// Slice like construct over a subset of all the key-value pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`IndexMultimap::get`] and [`OccupiedEntry::get`] methods.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::get`]: crate::IndexMultimap::get
/// [`OccupiedEntry::get`]: crate::multimap::OccupiedEntry::get
pub struct Subset<'a, K, V> {
    // See the comment on top of this module for impl details
    pairs: &'a [Bucket<K, V>],
    indices: &'a [usize],
}

impl<'a, K, V> Subset<'a, K, V> {
    /// Indices should be in bounds for pairs, **panics** can occur otherwise.
    pub(super) fn new(pairs: &'a [Bucket<K, V>], indices: &'a [usize]) -> Self {
        debug_assert!(indices.iter().max().unwrap_or(&0) < &pairs.len());
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
        match self.indices.get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Return a reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first(&self) -> Option<(usize, &'a K, &'a V)> {
        match self.indices.first() {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last(&self) -> Option<(usize, &'a K, &'a V)> {
        match self.indices.last() {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'a, K, V> {
        SubsetIter::new(self.pairs, self.indices.iter())
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yields one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'a, K, V> {
        SubsetKeys::new(self.pairs, self.indices.iter())
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'a, K, V> {
        SubsetValues::new(self.pairs, self.indices.iter())
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
        &self.pairs[index].value
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

/// Slice like construct over a subset of all the pairs in the [`IndexMultimap`]
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
    pairs: &'a mut [Bucket<K, V>],
    indices: &'a UniqueSlice<usize>,
}

impl<'a, K, V> SubsetMut<'a, K, V> {
    #[cfg(not(debug_assertions))]
    #[inline(always)]
    fn debug_assert_invariants(&self) {}

    #[cfg(debug_assertions)]
    #[track_caller]
    fn debug_assert_invariants(&self) {
        use crate::util::is_unique;

        debug_assert!(is_unique(&self.indices));
        debug_assert!(self
            .indices
            .iter()
            .max()
            .map(|&i| i < self.pairs.len())
            .unwrap_or(true));
    }

    pub(super) fn new(pairs: &'a mut [Bucket<K, V>], indices: &'a UniqueSlice<usize>) -> Self {
        //let inner = indices.as_inner();
        assert!(indices.is_empty() || *indices.last().unwrap() < pairs.len());
        Self { pairs, indices }
    }

    // /// Indices should be in bounds for pairs, **panics** can occur otherwise.
    // ///
    // /// SAFETY: `indices` must be unique
    // pub(crate) unsafe fn new_unchecked(pairs: &'a mut [Bucket<K, V>], indices: Indices) -> Self {
    //     Self {
    //         pairs,
    //         indices,
    //         __non_exhaustive: (),
    //     }
    // }

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
        match self.indices.get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a mutable reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn nth_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
        match self.indices.get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Converts `self` into a long lived mutable reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn into_nth(self, n: usize) -> Option<(usize, &'a K, &'a mut V)> {
        match self.indices.get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Return a reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first(&self) -> Option<(usize, &K, &V)> {
        match self.indices.first() {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Return a mutable reference to the first pair in this subset or `None` if this subset is empty.
    pub fn first_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        match self.indices.first() {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Converts `self` into long lived mutable reference to the first pair in this subset or `None` if this subset is empty.
    pub fn into_first_mut(self) -> Option<(usize, &'a K, &'a mut V)> {
        match self.indices.first() {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last(&self) -> Option<(usize, &K, &V)> {
        match self.indices.last() {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a mutable reference to the last pair in this subset or `None` if this subset is empty.
    pub fn last_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        match self.indices.last() {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Converts `self` into long lived mutable reference to the last pair in this subset or `None` if this subset is empty.
    pub fn into_last_mut(&mut self) -> Option<(usize, &K, &mut V)> {
        match self.indices.last() {
            Some(&index) => {
                let Bucket { key, value, .. } = &mut self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'_, K, V> {
        self.debug_assert_invariants();
        SubsetIter::new(self.pairs, self.indices.as_slice().iter())
    }

    /// Returns a mutable iterator over all the pairs in this subset.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V> {
        self.debug_assert_invariants();
        SubsetIterMut::new(self.pairs, self.indices.iter())
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yield one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'_, K, V> {
        self.debug_assert_invariants();
        SubsetKeys::new(self.pairs, self.indices.as_slice().iter())
    }

    /// Converts into a iterator over all the keys in this subset.
    pub fn into_keys(self) -> SubsetKeys<'a, K, V> {
        SubsetKeys::new(self.pairs, self.indices.as_slice().into_iter())
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'_, K, V> {
        self.debug_assert_invariants();
        SubsetValues::new(self.pairs, self.indices.as_slice().iter())
    }

    /// Returns a mutable iterator over all the values in this subset.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V> {
        self.debug_assert_invariants();
        SubsetValuesMut::new(self.pairs, self.indices.iter())
    }

    /// Converts into a mutable iterator over all the values in this subset.
    pub fn into_values(self) -> SubsetValuesMut<'a, K, V> {
        self.debug_assert_invariants();
        SubsetValuesMut::new(self.pairs, self.indices.iter())
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
        self.debug_assert_invariants();
        SubsetIterMut::new(self.pairs, self.indices.iter())
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
        self.nth(index).expect("index out of bounds").2
    }
}

impl<K, V> core::ops::IndexMut<usize> for SubsetMut<'_, K, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.nth_mut(index).expect("index out of bounds").2
    }
}

macro_rules! iter_methods {
    (@get_item $self:ident, $index_value:expr, $index:ident, $pair:ident, $($return:tt)*) => {
        match $index_value {
            Some($index) => {
                let $pair = $self
                    .pairs
                    .get(*$index)
                    .expect("expected indices to be in bounds");
                Some($($return)*)
            }
            None => None,
        }
    };
    ($name:ident, $ty:ty, $item:ty, $index:ident, $pair:ident, $($return:tt)*) => {

        impl<'a, K, V> $ty
        {
            /// * `indices` should be in bounds for `pairs`,
            ///   otherwise the iterator will panic
            pub(super) fn new(pairs: &'a [Bucket<K, V>], indices: slice::Iter<'a, usize>) -> Self {
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
                    .filter_map(|$index| {
                        let $pair = self.pairs
                            .get(*$index)
                            .expect("expected indices to be in bounds");
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
                    let $pair = &self.pairs[*$index];
                    $($return)*
                });
                debug_subset(f, stringify!($name), pairs)
            }
        }
    };
}

/// An iterator over a subset of all the pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::iter`] and [`SubsetMut::iter`].
/// See their documentation for more.
///
/// [`Subset::iter`]: super::Subset::iter
/// [`SubsetMut::iter`]: super::SubsetMut::iter
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetIter<'a, K, V> {
    // See the comment on top of the super module (subsets.rs) for impl details
    pairs: &'a [Bucket<K, V>],
    indices: slice::Iter<'a, usize>,
}

iter_methods!(
    SubsetIter,
    SubsetIter<'a, K, V>,
    (usize, &'a K, &'a V),
    index,
    pair,
    (*index, &pair.key, &pair.value)
);

/// An iterator over a subset of all the keys in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::keys`] and [`SubsetMut::keys`].
/// See their documentation for more.
///
/// [`Subset::keys`]: super::Subset::keys
/// [`SubsetMut::keys`]: super::SubsetMut::keys
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetKeys<'a, K, V> {
    // See the comment on top of the super module (subsets.rs) for impl details
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

/// An iterator over a subset of all the values in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::values`] and [`SubsetMut::values`].
/// See their documentation for more.
///
/// [`Subset::values`]: super::Subset::values
/// [`SubsetMut::values`]: super::SubsetMut::values
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetValues<'a, K, V> {
    // See the comment on top of the super module (subsets.rs) for impl details
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

/// A mutable iterator over a subset of all the pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`SubsetMut::iter_mut`] method.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`SubsetMut::iter_mut`]: crate::multimap::SubsetMut::iter_mut
pub struct SubsetIterMut<'a, K, V> {
    // ---
    // See the comment on top of the super module (subsets.rs) for impl details
    //
    // # SAFETY
    //
    // * `indices` must be unique,
    //   otherwise iterator will return multiple unique references to the same
    //   value
    // ---
    pairs_start: *mut Bucket<K, V>,
    pairs_len: usize,
    indices: UniqueIter<slice::Iter<'a, usize>>,
    // What self.pairs really is, constructors should take this to bind the lifetime properly
    _marker: PhantomData<&'a mut [Bucket<K, V>]>,
}

impl<'a, K, V> SubsetIterMut<'a, K, V> {
    pub(super) fn new(
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
        //  * we assert i is in bounds
        //  * self.pairs is not modified never
        //  * self.indices is an iterator that return unique indices,
        //    thus we cannot have returned item at this index,
        //    hence it's ok to create a reference and it cannot invalidate any
        //    previously returned mutable references
        //  * the lifetime of returned values is bound to the borrow of self by
        //    this function call. Thus any further call to Iterator methods will
        //    invalidate these references as it should.
        self.indices.clone().map(|&i| unsafe {
            assert!(i < self.pairs_len);
            let pair = &*self.pairs_start.add(i);
            (i, &pair.key, &pair.value)
        })
    }

    /// # Safety
    ///
    /// * pairs_start = self.pairs.start
    /// * pairs_len = self.pairs_len
    /// * index must be from the iterator self.indices
    #[inline]
    unsafe fn get_item_mut(
        pairs_start: *mut Bucket<K, V>,
        pairs_len: usize,
        index: usize,
    ) -> (usize, &'a K, &'a mut V) {
        assert!(index < pairs_len);
        // SAFETY:
        //   * self.pairs is never changed after the construction,
        //     points to the start of pairs slice
        //   * self.indices are unique => we cannot return aliasing
        //     mutable references
        //   * self.indices is also an iterator => cannot use same index twice
        //   * index is asserted to be in bounds
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

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.indices.nth(n) {
            Some(&index) => {
                Some(unsafe { Self::get_item_mut(self.pairs_start, self.pairs_len, index) })
            }
            None => None,
        }
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

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.indices.count()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
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
    #[inline]
    pub(super) fn new(
        pairs: &'a mut [Bucket<K, V>],
        indices: UniqueIter<slice::Iter<'a, usize>>,
    ) -> Self {
        Self {
            inner: SubsetIterMut::new(pairs, indices),
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

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.inner.nth(n) {
            Some((_, _, v)) => Some(v),
            None => None,
        }
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

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.inner.count()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
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

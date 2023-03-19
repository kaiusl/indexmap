#![allow(unsafe_code)]

use core::fmt;
use core::iter::FusedIterator;
use core::marker::PhantomData;

use crate::{
    util::{is_unique, is_unique_sorted},
    Bucket,
};

use super::{internal, SubsetIndexStorage, SubsetIter, SubsetKeys, SubsetValues, ToIndexIter};

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
pub struct SubsetMut<'a, K, V, Indices> {
    // ---
    // See the comment on top of the super module (subsets.rs) for impl details
    //
    // # SAFETY
    //
    // * `indices` must be unique
    //   This is required if we want to make .iter_mut() and other iterators safe
    //   for end user. See the methods and [`SubsetIterMut`] for details.
    // ---
    pub(super) pairs: &'a mut [Bucket<K, V>],
    pub(super) indices: Indices,
    // Marker so we cannot construct it from other modules without
    // going through the constructor methods
    __non_exhaustive: (),
}

impl<'a, K, V, Indices> SubsetMut<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    /// Indices should be in bounds for pairs, **panics** can occur otherwise.
    ///
    /// SAFETY: `indices` must be unique
    pub(crate) unsafe fn new_unchecked(pairs: &'a mut [Bucket<K, V>], indices: Indices) -> Self {
        Self {
            pairs,
            indices,
            __non_exhaustive: (),
        }
    }

    pub(crate) fn empty() -> Self {
        // SAFETY: no access will ever be actually performed
        Self {
            pairs: Default::default(),
            indices: Default::default(),
            __non_exhaustive: (),
        }
    }

    #[cfg(not(debug_assertions))]
    #[inline(always)]
    fn debug_assert_invariants(&self) {}

    #[cfg(debug_assertions)]
    #[track_caller]
    fn debug_assert_invariants(&self) {
        debug_assert!(is_unique(&self.indices));
        debug_assert!(self
            .indices
            .iter()
            .max()
            .map(|&i| i < self.pairs.len())
            .unwrap_or(true));
    }

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetIter::new(self.pairs, self.indices.index_iter(internal::Guard))
    }

    /// Returns a mutable iterator over all the pairs in this subset.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        // SAFETY: Self's invariants are same as the iterator's
        unsafe {
            SubsetIterMut::new_unchecked(self.pairs, self.indices.index_iter(internal::Guard))
        }
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yield one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetKeys::new(self.pairs, self.indices.index_iter(internal::Guard))
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetValues::new(self.pairs, self.indices.index_iter(internal::Guard))
    }

    /// Returns a mutable iterator over all the values in this subset.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        // SAFETY: Self's invariants are same as the iterator's
        unsafe {
            SubsetValuesMut::new_unchecked(self.pairs, self.indices.index_iter(internal::Guard))
        }
    }

    /// Converts into a mutable iterator over all the values in this subset.
    pub fn into_values(
        self,
    ) -> SubsetValuesMut<'a, K, V, <Indices as SubsetIndexStorage>::IntoIter> {
        self.debug_assert_invariants();
        // SAFETY: Self's invariants are same as the iterator's
        unsafe {
            SubsetValuesMut::new_unchecked(
                self.pairs,
                self.indices.into_index_iter(internal::Guard),
            )
        }
    }
}

impl<'a, K, V, Indices> fmt::Debug for SubsetMut<'a, K, V, Indices>
where
    K: fmt::Debug,
    V: fmt::Debug,
    Indices: fmt::Debug + SubsetIndexStorage,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("pairsMut")
            // Print only pairs that are actually referred to by indices
            .field("pairs", &self.iter())
            .field("indices", &self.indices)
            .finish()
    }
}

impl<'a, K, V, Indices> IntoIterator for SubsetMut<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V, <Indices as SubsetIndexStorage>::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.debug_assert_invariants();
        // SAFETY: Self can only be created from the map, which only stores unique and valid indices
        unsafe {
            SubsetIterMut::new_unchecked(self.pairs, self.indices.into_index_iter(internal::Guard))
        }
    }
}

/// A mutable iterator over a subset of all the pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`SubsetMut::iter_mut`] method.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`SubsetMut::iter_mut`]: crate::multimap::SubsetMut::iter_mut
pub struct SubsetIterMut<'a, K, V, I> {
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
    indices: I,
    // What self.pairs really is, constructors should take this to bind the lifetime properly
    _marker: PhantomData<&'a mut [Bucket<K, V>]>,
}

impl<'a, K, V, I> SubsetIterMut<'a, K, V, I>
where
    I: Iterator<Item = usize>,
{
    /// * `indices` should be in bounds for `pairs`,
    ///   otherwise the iterator will panic
    ///
    /// # SAFETY
    ///
    /// * `indices` must be unique,
    ///   otherwise iterator will return multiple unique references to the same value
    pub(in crate::multimap) unsafe fn new_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: I,
    ) -> Self {
        Self {
            pairs_start: pairs.as_mut_ptr(),
            pairs_len: pairs.len(),
            indices,
            _marker: PhantomData,
        }
    }

    unsafe fn get_item(
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
        //   * self.indices are in bounds for indexing into self.pairs,
        //     that is index < self.pairs.len() => .add is safe
        //     These requirements are either asserted in safe constructors
        //     or documented on unsafe ones.
        //   * no one else can have access to the pairs slice as we
        //     borrowed it mutably for 'a => it's valid to return
        //     &'a mut into the slice
        let pair = unsafe { &mut *pairs_start.add(index) };
        (index, &pair.key, &mut pair.value)
    }
}

impl<'a, K, V, I> Iterator for SubsetIterMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize>,
{
    type Item = (usize, &'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.indices.next() {
            Some(index) => Some(unsafe { Self::get_item(self.pairs_start, self.pairs_len, index) }),
            None => None,
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.indices.nth(n) {
            Some(index) => Some(unsafe { Self::get_item(self.pairs_start, self.pairs_len, index) }),
            None => None,
        }
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        match self.indices.last() {
            Some(index) => Some(unsafe { Self::get_item(self.pairs_start, self.pairs_len, index) }),
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
            .map(|index| unsafe { Self::get_item(self.pairs_start, self.pairs_len, index) })
            .collect()
    }
}

impl<'a, K, V, I> DoubleEndedIterator for SubsetIterMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize> + DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: see the safety doc in `next` method
        match self.indices.next_back() {
            Some(index) => Some(unsafe { Self::get_item(self.pairs_start, self.pairs_len, index) }),
            None => None,
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // SAFETY: see the safety doc in `next` method
        match self.indices.nth_back(n) {
            Some(index) => Some(unsafe { Self::get_item(self.pairs_start, self.pairs_len, index) }),
            None => None,
        }
    }
}

impl<'a, K, V, I> ExactSizeIterator for SubsetIterMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize> + ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<'a, K, V, I> FusedIterator for SubsetIterMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize> + FusedIterator,
{
}

impl<K, V: fmt::Debug, I> fmt::Debug for SubsetIterMut<'_, K, V, I>
where
    K: fmt::Debug,
    V: fmt::Debug,
    I: Iterator<Item = usize> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pairs = self
            .indices
            .clone()
            .map(|index| unsafe { Self::get_item(self.pairs_start, self.pairs_len, index) });
        f.debug_list().entries(pairs).finish()
    }
}

/// A mutable iterator over a subset of all the pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`SubsetMut::values_mut`] method.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`SubsetMut::values_mut`]: crate::multimap::SubsetMut::values_mut
pub struct SubsetValuesMut<'a, K, V, I> {
    // SAFETY: see SubsetIterMut's definition
    inner: SubsetIterMut<'a, K, V, I>,
}

impl<'a, K, V, I> SubsetValuesMut<'a, K, V, I>
where
    I: Iterator<Item = usize>,
{
    /// * `indices` should be in bounds for `pairs`,
    ///   otherwise the iterator will panic
    ///
    /// # SAFETY
    ///
    /// * `indices` must be unique,
    ///   otherwise iterator will return multiple unique references to the same value
    pub(in crate::multimap) unsafe fn new_unchecked(
        pairs: &'a mut [Bucket<K, V>],
        indices: I,
    ) -> Self {
        Self {
            // SAFETY: we forward the requirements
            inner: unsafe { SubsetIterMut::new_unchecked(pairs, indices) },
        }
    }
}

impl<'a, K, V, I> Iterator for SubsetValuesMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize>,
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

impl<'a, K, V, I> DoubleEndedIterator for SubsetValuesMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize> + DoubleEndedIterator,
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

impl<'a, K, V, I> ExactSizeIterator for SubsetValuesMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize> + ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V, I> FusedIterator for SubsetValuesMut<'a, K, V, I>
where
    K: 'a,
    V: 'a,
    I: Iterator<Item = usize> + FusedIterator,
{
}

impl<K, V: fmt::Debug, I> fmt::Debug for SubsetValuesMut<'_, K, V, I>
where
    K: fmt::Debug,
    V: fmt::Debug,
    I: Iterator<Item = usize> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::{util::is_unique, HashValue};
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn debug() {
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
        let indices = [0, 2, 3, 5];

        let _iter1 = unsafe { SubsetIterMut::new_unchecked(&mut pairs, indices.iter().copied()) };
        //println!("{:#?}", iter1);
        //println!("{:#?}", iter1);
    }

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
        let indices = [0, 2, 3, 5];

        let iter1 = unsafe { SubsetIterMut::new_unchecked(&mut pairs, indices.iter().copied()) };
        //println!("{iter1:#?}");
        //iter.clone();
        let items = iter1.collect::<Vec<_>>();
        let values = items.iter().map(|a| *a.2).collect::<Vec<_>>();
        assert!(is_unique(&items));
        assert!(is_unique(&values));

        let mut iter2 =
            unsafe { SubsetIterMut::new_unchecked(&mut pairs, indices.iter().copied()) };

        //let b = items[0];
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
        let values = items.iter().map(|a| *a.2).collect::<Vec<_>>();
        assert!(is_unique(&items));
        assert!(is_unique(&values));

        // let mut iter3 = pairs.iter_mut();// unsafe { pairsMut::new(&mut pairs, indices.iter()) };
        // let i1 = iter3.next().unwrap();
        // let i2 = iter3.next().unwrap();
        // //let v1 = i1.2;

        // //let b = &items[0];
        // //let b = i1.2;
        // move_item(iter3);
        // let v = i1;

        // //let a = &pairs;
        // let v = i2;
    }
}

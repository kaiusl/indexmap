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

mod raw;

use core::{fmt, iter::FusedIterator};

use alloc::vec::Vec;

use crate::Bucket;

pub use raw::{SubsetIterMut, SubsetMut, SubsetValuesMut};

/// Slice like construct over a subset of all the key-value pairs in the [`IndexMultimap`].
///
/// This `struct` is created by the [`IndexMultimap::get`] and [`OccupiedEntry::get`] methods.
/// See their documentation for more.
///
/// [`IndexMultimap`]: crate::IndexMultimap
/// [`IndexMultimap::get`]: crate::IndexMultimap::get
/// [`OccupiedEntry::get`]: crate::multimap::OccupiedEntry::get
pub struct Subset<'a, K, V, Indices> {
    // See the comment on top of this module for impl details
    pairs: &'a [Bucket<K, V>],
    indices: Indices,
}

impl<'a, K, V, Indices> Subset<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    /// Indices should be in bounds for pairs, **panics** can occur otherwise.
    pub(crate) fn new(pairs: &'a [Bucket<K, V>], indices: Indices) -> Self {
        debug_assert!(indices.iter().max().unwrap_or(&0) < &pairs.len());
        Self { pairs, indices }
    }

    pub(crate) fn empty() -> Self
    where
        Indices: Default,
    {
        Self {
            pairs: Default::default(),
            indices: Default::default(),
        }
    }

    /// Returns number the number of pairs in this subset
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this subset is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns a reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn get(&self, n: usize) -> Option<(usize, &K, &V)> {
        match self.indices.get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
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

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        SubsetIter::new(self.pairs, self.indices.index_iter(internal::Guard))
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yields one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        SubsetKeys::new(self.pairs, self.indices.index_iter(internal::Guard))
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        SubsetValues::new(self.pairs, self.indices.index_iter(internal::Guard))
    }
}

impl<'a, K, V, Indices> IntoIterator for Subset<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V, <Indices as SubsetIndexStorage>::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        SubsetIter::new(self.pairs, self.indices.into_index_iter(internal::Guard))
    }
}

impl<'a, K, V, Indices> IntoIterator for &'a Subset<'_, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V, <Indices as ToIndexIter<'a>>::Iter>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, Indices> core::ops::Index<usize> for Subset<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        let index = self.indices[index];
        &self.pairs[index].value
    }
}

impl<'a, K, V, Indices> Clone for Subset<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage + Clone,
{
    fn clone(&self) -> Self {
        Self {
            pairs: self.pairs,
            indices: self.indices.clone(),
        }
    }
}

impl<'a, K, V, Indices> fmt::Debug for Subset<'a, K, V, Indices>
where
    K: fmt::Debug,
    V: fmt::Debug,
    Indices: SubsetIndexStorage + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubSet")
            // Print only pairs that are actually referred to by indices
            .field("pairs", &self.iter())
            .field("indices", &self.indices)
            .finish()
    }
}

impl<'a, K, V, Indices> SubsetMut<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    /// Returns number the number of pairs in this subset
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
    pub fn get(&self, n: usize) -> Option<(usize, &K, &V)> {
        match self.indices.get(n) {
            Some(&index) => {
                let Bucket { key, value, .. } = &self.pairs[index];
                Some((index, key, value))
            }
            None => None,
        }
    }

    /// Returns a mutable reference to an `n`th pair in this subset or `None` if `n >= self.len()`.
    pub fn get_mut(&mut self, n: usize) -> Option<(usize, &K, &mut V)> {
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
}

impl<'a, K, V, Indices> IntoIterator for &'a SubsetMut<'_, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    type Item = (usize, &'a K, &'a V);
    type IntoIter = SubsetIter<'a, K, V, <Indices as ToIndexIter<'a>>::Iter>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, Indices> IntoIterator for &'a mut SubsetMut<'_, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = SubsetIterMut<'a, K, V, <Indices as ToIndexIter<'a>>::Iter>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, Indices> core::ops::Index<usize> for SubsetMut<'_, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds").2
    }
}

impl<K, V, Indices> core::ops::IndexMut<usize> for SubsetMut<'_, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds").2
    }
}

macro_rules! iter_methods {
    (@get_item $self:ident, $index_value:expr, $index:ident, $pair:ident, $($return:tt)*) => {
        match $index_value {
            Some($index) => {
                let $pair = $self
                    .pairs
                    .get($index)
                    .expect("expected indices to be in bounds");
                Some($($return)*)
            }
            None => None,
        }
    };
    ($name:ty, $item:ty, $index:ident, $pair:ident, $($return:tt)*) => {

        impl<'a, K, V, I> $name
        where
            I: Iterator<Item = usize>,
        {
            /// * `indices` should be in bounds for `pairs`,
            ///   otherwise the iterator will panic
            pub(super) fn new(pairs: &'a [Bucket<K, V>], indices: I) -> Self {
                Self { pairs, indices }
            }
        }

        impl<'a, K, V, I> Iterator for $name
        where
            I: Iterator<Item = usize>,
        {
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
                            .get($index)
                            .expect("expected indices to be in bounds");
                        Some($($return)*)
                    })
                    .collect()
            }
        }

        impl<'a, K, V, I> DoubleEndedIterator for $name
        where
            I: Iterator<Item = usize> + DoubleEndedIterator,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                iter_methods!(@get_item self, self.indices.next_back(), $index, $pair, $($return)*)
            }

            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                iter_methods!(@get_item self, self.indices.nth_back(n), $index, $pair, $($return)*)
            }
        }

        impl<'a, K, V, I> ExactSizeIterator for $name
        where
            I: Iterator<Item = usize> + ExactSizeIterator,
        {
            fn len(&self) -> usize {
                self.indices.len()
            }
        }

        impl<'a, K, V, I> FusedIterator for $name where
            I: Iterator<Item = usize> + FusedIterator
        {
        }

        impl<'a, K, V, I> Clone for $name
        where
            I: Clone,
        {
            fn clone(&self) -> Self {
                Self {
                    pairs: self.pairs,
                    indices: self.indices.clone(),
                }
            }
        }

        impl<'a, K, V, I> fmt::Debug for $name
        where
            K: fmt::Debug,
            V: fmt::Debug,
            I: Clone + Iterator<Item = usize>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let pairs = self.indices.clone().map(|$index| {
                    let $pair = &self.pairs[$index];
                    $($return)*
                });
                f.debug_list().entries(pairs).finish()
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
pub struct SubsetIter<'a, K, V, I> {
    // See the comment on top of the super module (subsets.rs) for impl details
    pairs: &'a [Bucket<K, V>],
    indices: I,
}

iter_methods!(
    SubsetIter<'a, K, V, I>,
    (usize, &'a K, &'a V),
    index,
    pair,
    (index, &pair.key, &pair.value)
);

/// An iterator over a subset of all the keys in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::keys`] and [`SubsetMut::keys`].
/// See their documentation for more.
///
/// [`Subset::keys`]: super::Subset::keys
/// [`SubsetMut::keys`]: super::SubsetMut::keys
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetKeys<'a, K, V, I> {
    // See the comment on top of the super module (subsets.rs) for impl details
    pairs: &'a [Bucket<K, V>],
    indices: I,
}

//subpair_iter_impls!(SubsetKeys<'a, K, V, I>, &'a K, Bucket::key_ref);

iter_methods!(SubsetKeys<'a, K, V, I>, &'a K, index, pair, &pair.key);

/// An iterator over a subset of all the values in the [`IndexMultimap`].
///
/// This `struct` is created by the [`Subset::values`] and [`SubsetMut::values`].
/// See their documentation for more.
///
/// [`Subset::values`]: super::Subset::values
/// [`SubsetMut::values`]: super::SubsetMut::values
/// [`IndexMultimap`]: crate::IndexMultimap
pub struct SubsetValues<'a, K, V, I> {
    // See the comment on top of the super module (subsets.rs) for impl details
    pairs: &'a [Bucket<K, V>],
    indices: I,
}

iter_methods!(SubsetValues<'a, K, V, I>, &'a V, index, pair, &pair.value);

mod internal {
    pub struct Guard;
    pub trait Sealed {}
    pub struct Bounds<T>(T);
    impl<T> Sealed for Bounds<T> {}
    impl<'a> Sealed for &'a [usize] {}
    impl Sealed for alloc::vec::Vec<usize> {}
}

/// GAT workaround for [`SubsetIndexStorage`] providing an associated `Iter`
/// type that could borrow from self.
///
/// Users don't really need to worry about the details of this trait or that it even exists.
pub trait ToIndexIter<'a>
where
    Self: internal::Sealed,
{
    type Iter: Iterator<Item = usize>
        + DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
        + Clone;
}

/// Used by `Subset` and other similar structs to
/// be generic over the indices container.
///
/// Users don't really need to worry about the details of this trait or that it even exists.
pub trait SubsetIndexStorage
where
    Self: internal::Sealed + for<'a> ToIndexIter<'a> + core::ops::Deref<Target = [usize]> + Default,
{
    type IntoIter: Iterator<Item = usize> + DoubleEndedIterator + ExactSizeIterator + FusedIterator;

    #[doc(hidden)]
    fn into_index_iter(self, _: internal::Guard) -> Self::IntoIter;

    #[doc(hidden)]
    fn index_iter(&self, _: internal::Guard) -> <Self as ToIndexIter<'_>>::Iter;
}

impl<'a> ToIndexIter<'a> for &[usize] {
    type Iter = ::core::iter::Copied<::core::slice::Iter<'a, usize>>;
}

impl<'a> SubsetIndexStorage for &'a [usize] {
    type IntoIter = ::core::iter::Copied<::core::slice::Iter<'a, usize>>;

    fn into_index_iter(self, _: internal::Guard) -> Self::IntoIter {
        core::iter::IntoIterator::into_iter(self).copied()
    }

    fn index_iter(&self, _: internal::Guard) -> <Self as ToIndexIter<'_>>::Iter {
        <[usize]>::iter(self).copied()
    }
}

impl<'a> ToIndexIter<'a> for Vec<usize> {
    type Iter = ::core::iter::Copied<::core::slice::Iter<'a, usize>>;
}
impl SubsetIndexStorage for alloc::vec::Vec<usize> {
    type IntoIter = alloc::vec::IntoIter<usize>;

    fn into_index_iter(self, _: internal::Guard) -> Self::IntoIter {
        core::iter::IntoIterator::into_iter(self)
    }

    fn index_iter(&self, _: internal::Guard) -> <Self as ToIndexIter<'_>>::Iter {
        <[usize]>::iter(self).copied()
    }
}

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

use crate::Bucket;

pub use raw::{SubsetIndexStorage, SubsetIterMut, ToIndexIter};

pub(crate) use self::raw::internal;

use super::core::Unique;

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

    /// Returns the number of pairs in this subset
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

    /// Returns a reference to the `n`th pair in this subset or `None` if `n >= self.len()`.
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
    pairs: &'a mut [Bucket<K, V>],
    indices: Unique<Indices>,
}

impl<'a, K, V, Indices> SubsetMut<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
    pub(crate) fn new(pairs: &'a mut [Bucket<K, V>], indices: Unique<Indices>) -> Self {
        let inner = indices.as_inner();
        assert!(inner.is_empty() || *indices.last().unwrap() < pairs.len());
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

    /// Returns an iterator over all the pairs in this subset.
    pub fn iter(&self) -> SubsetIter<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetIter::new(
            self.pairs,
            self.indices.as_inner().index_iter(internal::Guard),
        )
    }

    /// Returns a mutable iterator over all the pairs in this subset.
    pub fn iter_mut(&mut self) -> SubsetIterMut<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetIterMut::new(
            self.pairs,
            self.indices.index_iter(internal::Guard),
        )
    }

    /// Returns an iterator over all the keys in this subset.
    ///
    /// Note that the iterator yield one key for each pair.
    /// That is there may be duplicate keys.
    pub fn keys(&self) -> SubsetKeys<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetKeys::new(
            self.pairs,
            self.indices.as_inner().index_iter(internal::Guard),
        )
    }

    /// Returns an iterator over all the values in this subset.
    pub fn values(&self) -> SubsetValues<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetValues::new(
            self.pairs,
            self.indices.as_inner().index_iter(internal::Guard),
        )
    }

    /// Returns a mutable iterator over all the values in this subset.
    pub fn values_mut(&mut self) -> SubsetValuesMut<'_, K, V, <Indices as ToIndexIter<'_>>::Iter> {
        self.debug_assert_invariants();
        SubsetValuesMut::new(self.pairs, self.indices.index_iter(internal::Guard))
    }

    /// Converts into a mutable iterator over all the values in this subset.
    pub fn into_values(
        self,
    ) -> SubsetValuesMut<'a, K, V, <Indices as SubsetIndexStorage>::IntoIter> {
        self.debug_assert_invariants();
        SubsetValuesMut::new(self.pairs, self.indices.into_index_iter(internal::Guard))
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
        SubsetIterMut::new(self.pairs, self.indices.into_index_iter(internal::Guard))
    }
}

impl<'a, K, V, Indices> SubsetMut<'a, K, V, Indices>
where
    Indices: SubsetIndexStorage,
{
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
    #[inline]
    pub(crate) fn new(pairs: &'a mut [Bucket<K, V>], indices: Unique<I>) -> Self {
        Self {
            inner: SubsetIterMut::new(pairs, indices),
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

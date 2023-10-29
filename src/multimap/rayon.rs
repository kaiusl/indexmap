//! Parallel iterator types for [`IndexMultimap`] with [rayon].
//!
//! You will rarely need to interact with this module directly unless you need to name one of the
//! iterator types.
//!
//! [rayon]: https://docs.rs/rayon/1.0/rayon

use crate::rayon::collect;
use crate::util::debug_iter_as_list;
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;

use crate::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::ops::RangeBounds;

use crate::Bucket;
use crate::IndexMultimap;

pub use super::core::ParDrain;

impl<K, V, S> IntoParallelIterator for IndexMultimap<K, V, S>
where
    K: Send,
    V: Send,
{
    type Item = (K, V);
    type Iter = IntoParIter<K, V>;

    fn into_par_iter(self) -> Self::Iter {
        IntoParIter {
            entries: self.into_pairs(),
        }
    }
}

/// A parallel owning iterator over the entries of a [`IndexMultimap`].
///
/// This `struct` is created by the [`into_par_iter`] method on [`IndexMultimap`]
/// (provided by [rayon]'s [`IntoParallelIterator`] trait).
/// See its documentation for more.
///
/// [`into_par_iter`]: IndexMultimap::into_par_iter
/// [rayon]: https://docs.rs/rayon/1.0/rayon
pub struct IntoParIter<K, V> {
    entries: Vec<Bucket<K, V>>,
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IntoParIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::refs);
        debug_iter_as_list(f, Some("IntoParIter"), iter)
    }
}

impl<K: Send, V: Send> ParallelIterator for IntoParIter<K, V> {
    type Item = (K, V);

    parallel_iterator_methods!(Bucket::key_value);
}

impl<K: Send, V: Send> IndexedParallelIterator for IntoParIter<K, V> {
    indexed_parallel_iterator_methods!(Bucket::key_value);
}

impl<'a, K, V, S> IntoParallelIterator for &'a IndexMultimap<K, V, S>
where
    K: Sync,
    V: Sync,
{
    type Item = (&'a K, &'a V);
    type Iter = ParIter<'a, K, V>;

    fn into_par_iter(self) -> Self::Iter {
        ParIter {
            entries: self.as_pairs(),
        }
    }
}

/// A parallel iterator over the entries of a [`IndexMultimap`].
///
/// This `struct` is created by the [`par_iter`] method on [`IndexMultimap`]
/// (provided by [rayon]'s [`IntoParallelRefIterator`] trait).
/// See its documentation for more.
///
/// [`par_iter`]: ../struct.IndexMultimap.html#method.par_iter
/// [rayon]: https://docs.rs/rayon/1.0/rayon
pub struct ParIter<'a, K, V> {
    entries: &'a [Bucket<K, V>],
}

impl<K, V> Clone for ParIter<'_, K, V> {
    fn clone(&self) -> Self {
        ParIter { ..*self }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::refs);
        debug_iter_as_list(f, Some("ParIter"), iter)
    }
}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    parallel_iterator_methods!(Bucket::refs);
}

impl<K: Sync, V: Sync> IndexedParallelIterator for ParIter<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::refs);
}

impl<'a, K, V, S> IntoParallelIterator for &'a mut IndexMultimap<K, V, S>
where
    K: Sync + Send,
    V: Send,
{
    type Item = (&'a K, &'a mut V);
    type Iter = ParIterMut<'a, K, V>;

    fn into_par_iter(self) -> Self::Iter {
        ParIterMut {
            entries: self.as_mut_pairs(),
        }
    }
}

/// A parallel mutable iterator over the entries of a [`IndexMultimap`].
///
/// This `struct` is created by the [`par_iter_mut`] method on [`IndexMultimap`]
/// (provided by [rayon]'s [`IntoParallelRefMutIterator`] trait).
/// See its documentation for more.
///
/// [`par_iter_mut`]: ../struct.IndexMultimap.html#method.par_iter_mut
/// [rayon]: https://docs.rs/rayon/1.0/rayon
pub struct ParIterMut<'a, K, V> {
    entries: &'a mut [Bucket<K, V>],
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIterMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::refs);
        debug_iter_as_list(f, Some("ParIterMut"), iter)
    }
}

impl<'a, K: Sync + Send, V: Send> ParallelIterator for ParIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    parallel_iterator_methods!(Bucket::ref_mut);
}

impl<K: Sync + Send, V: Send> IndexedParallelIterator for ParIterMut<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::ref_mut);
}

impl<'a, K, V, S> ParallelDrainRange<usize> for &'a mut IndexMultimap<K, V, S>
where
    K: Send + Eq + Hash,
    V: Send,
    S: BuildHasher,
{
    type Item = (K, V);
    type Iter = ParDrain<'a, K, V>;

    fn par_drain<R: RangeBounds<usize>>(self, range: R) -> Self::Iter {
        self.par_drain_inner(range)
    }
}

/// Parallel iterator methods and other parallel methods.
///
/// The following methods **require crate feature `"rayon"`**.
///
/// See also the [`IntoParallelIterator`] implementations.
impl<K, V, S> IndexMultimap<K, V, S>
where
    K: Sync,
    V: Sync,
{
    /// Return a parallel iterator over the keys of the map.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the map is still preserved for operations like `reduce` and `collect`.
    pub fn par_keys(&self) -> ParKeys<'_, K, V> {
        ParKeys {
            entries: self.as_pairs(),
        }
    }

    /// Return a parallel iterator over the values of the map.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the map is still preserved for operations like `reduce` and `collect`.
    pub fn par_values(&self) -> ParValues<'_, K, V> {
        ParValues {
            entries: self.as_pairs(),
        }
    }
}

impl<K, V, S> IndexMultimap<K, V, S>
where
    K: Hash + Eq + Sync,
    V: Sync,
    S: BuildHasher,
{
    /// Returns [`true`] if `self` contains all of the same key-value pairs as `other`,
    /// regardless of each map's indexed order, determined in parallel.
    pub fn par_eq<V2, S2>(&self, other: &IndexMultimap<K, V2, S2>) -> bool
    where
        V: PartialEq<V2>,
        V2: Sync,
        S2: BuildHasher + Sync,
    {
        self.len_pairs() == other.len_pairs()
            && self.par_iter().all(move |(key, this_v)| {
                other
                    .get(key)
                    .into_iter()
                    .any(|(_, _, other_v)| this_v == other_v)
            })
    }
}

/// A parallel iterator over the keys of a [`IndexMultimap`].
///
/// This `struct` is created by the [`par_keys`] method on [`IndexMultimap`].
/// See its documentation for more.
///
/// [`par_keys`]: IndexMultimap::par_keys
pub struct ParKeys<'a, K, V> {
    entries: &'a [Bucket<K, V>],
}

impl<K, V> Clone for ParKeys<'_, K, V> {
    fn clone(&self) -> Self {
        ParKeys { ..*self }
    }
}

impl<K: fmt::Debug, V> fmt::Debug for ParKeys<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::key_ref);
        debug_iter_as_list(f, Some("ParKeys"), iter)
    }
}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParKeys<'a, K, V> {
    type Item = &'a K;

    parallel_iterator_methods!(Bucket::key_ref);
}

impl<K: Sync, V: Sync> IndexedParallelIterator for ParKeys<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::key_ref);
}

/// A parallel iterator over the values of a [`IndexMultimap`].
///
/// This `struct` is created by the [`par_values`] method on [`IndexMultimap`].
/// See its documentation for more.
///
/// [`par_values`]: IndexMultimap::par_values
pub struct ParValues<'a, K, V> {
    entries: &'a [Bucket<K, V>],
}

impl<K, V> Clone for ParValues<'_, K, V> {
    fn clone(&self) -> Self {
        ParValues { ..*self }
    }
}

impl<K, V: fmt::Debug> fmt::Debug for ParValues<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::value_ref);
        debug_iter_as_list(f, Some("ParValues"), iter)
    }
}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParValues<'a, K, V> {
    type Item = &'a V;

    parallel_iterator_methods!(Bucket::value_ref);
}

impl<K: Sync, V: Sync> IndexedParallelIterator for ParValues<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::value_ref);
}

impl<K, V, S> IndexMultimap<K, V, S>
where
    K: Send,
    V: Send,
{
    /// Return a parallel iterator over mutable references to the values of the map.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the map is still preserved for operations like `reduce` and `collect`.
    pub fn par_values_mut(&mut self) -> ParValuesMut<'_, K, V> {
        ParValuesMut {
            entries: self.as_mut_pairs(),
        }
    }
}

/// A parallel mutable iterator over the values of a [`IndexMultimap`].
///
/// This `struct` is created by the [`par_values_mut`] method on [`IndexMultimap`].
/// See its documentation for more.
///
/// [`par_values_mut`]: IndexMultimap::par_values_mut
pub struct ParValuesMut<'a, K, V> {
    entries: &'a mut [Bucket<K, V>],
}

impl<K, V: fmt::Debug> fmt::Debug for ParValuesMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::value_ref);
        debug_iter_as_list(f, Some("ParValuesMut"), iter)
    }
}

impl<'a, K: Send, V: Send> ParallelIterator for ParValuesMut<'a, K, V> {
    type Item = &'a mut V;

    parallel_iterator_methods!(Bucket::value_mut);
}

impl<K: Send, V: Send> IndexedParallelIterator for ParValuesMut<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::value_mut);
}

impl<K, V, S> IndexMultimap<K, V, S>
where
    K: Hash + Eq + Send,
    V: Send,
    S: BuildHasher,
{
    /// Sort the map’s key-value pairs in parallel, by the default ordering of the keys.
    pub fn par_sort_keys(&mut self)
    where
        K: Ord,
    {
        self.with_pairs(|entries| {
            entries.par_sort_by(|a, b| K::cmp(&a.key, &b.key));
        });
    }

    /// Sort the map’s key-value pairs in place and in parallel, using the comparison
    /// function `cmp`.
    ///
    /// The comparison function receives two key and value pairs to compare (you
    /// can sort by keys or values or their combination as needed).
    pub fn par_sort_by<F>(&mut self, cmp: F)
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        self.with_pairs(|entries| {
            entries.par_sort_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        });
    }

    /// Sort the key-value pairs of the map in parallel and return a by-value parallel
    /// iterator of the key-value pairs with the result.
    pub fn par_sorted_by<F>(self, cmp: F) -> IntoParIter<K, V>
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        let mut entries = self.into_pairs();
        entries.par_sort_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        IntoParIter { entries }
    }

    /// Sort the map's key-value pairs in parallel, by the default ordering of the keys.
    pub fn par_sort_unstable_keys(&mut self)
    where
        K: Ord,
    {
        self.with_pairs(|entries| {
            entries.par_sort_unstable_by(|a, b| K::cmp(&a.key, &b.key));
        });
    }

    /// Sort the map's key-value pairs in place and in parallel, using the comparison
    /// function `cmp`.
    ///
    /// The comparison function receives two key and value pairs to compare (you
    /// can sort by keys or values or their combination as needed).
    pub fn par_sort_unstable_by<F>(&mut self, cmp: F)
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        self.with_pairs(|entries| {
            entries.par_sort_unstable_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        });
    }

    /// Sort the key-value pairs of the map in parallel and return a by-value parallel
    /// iterator of the key-value pairs with the result.
    pub fn par_sorted_unstable_by<F>(self, cmp: F) -> IntoParIter<K, V>
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        let mut entries = self.into_pairs();
        entries.par_sort_unstable_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        IntoParIter { entries }
    }

    /// Sort the map’s key-value pairs in place and in parallel, using a sort-key extraction
    /// function.
    pub fn par_sort_by_cached_key<T, F>(&mut self, sort_key: F)
    where
        T: Ord + Send,
        F: Fn(&K, &V) -> T + Sync,
    {
        self.with_pairs(move |entries| {
            entries.par_sort_by_cached_key(move |a| sort_key(&a.key, &a.value));
        });
    }
}

impl<K, V, S> FromParallelIterator<(K, V)> for IndexMultimap<K, V, S>
where
    K: Eq + Hash + Send,
    V: Send,
    S: BuildHasher + Default + Send,
{
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = (K, V)>,
    {
        let list = collect(iter);
        let len = list.iter().map(Vec::len).sum();
        // TODO: can we get accurate pairs and keys len?
        let mut map = Self::with_capacity_and_hasher(len, len, S::default());
        for vec in list {
            map.extend(vec);
        }
        map
    }
}

impl<K, V, S> ParallelExtend<(K, V)> for IndexMultimap<K, V, S>
where
    K: Eq + Hash + Send,
    V: Send,
    S: BuildHasher + Send,
{
    fn par_extend<I>(&mut self, iter: I)
    where
        I: IntoParallelIterator<Item = (K, V)>,
    {
        for vec in collect(iter) {
            self.extend(vec);
        }
    }
}

impl<'a, K: 'a, V: 'a, S> ParallelExtend<(&'a K, &'a V)> for IndexMultimap<K, V, S>
where
    K: Copy + Eq + Hash + Send + Sync,
    V: Copy + Send + Sync,
    S: BuildHasher + Send,
{
    fn par_extend<I>(&mut self, iter: I)
    where
        I: IntoParallelIterator<Item = (&'a K, &'a V)>,
    {
        for vec in collect(iter) {
            self.extend(vec);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::multimap::tests::assert_map_eq;

    use super::*;
    use core::panic::AssertUnwindSafe;
    use core::sync::atomic::AtomicU32;
    use std::panic::catch_unwind;
    use std::string::String;

    #[allow(dead_code)]
    fn par_iter() {
        fn test<'a, T: IntoParallelRefIterator<'a>>(t: &'a T) {
            t.par_iter();
        }
        let map: IndexMultimap<u8, u8> = IndexMultimap::new();
        test(&map);
    }

    #[allow(dead_code)]
    fn par_iter_mut() {
        fn test<'a, T: IntoParallelRefMutIterator<'a>>(t: &'a mut T) {
            t.par_iter_mut();
        }
        let mut map: IndexMultimap<u8, u8> = IndexMultimap::new();
        test(&mut map);
    }

    #[test]
    fn insert_order() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5, 3, 17, 19, 22, 23];
        let mut map = IndexMultimap::new();

        for &elt in &insert {
            map.insert_append(elt, ());
        }

        assert_eq!(map.par_keys().count(), map.len_keys());
        assert_eq!(map.par_keys().count(), insert.len());
        insert.par_iter().zip(map.par_keys()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        (0..insert.len())
            .into_par_iter()
            .zip(map.par_keys())
            .for_each(|(i, k)| {
                assert_eq!(map.get_index(i).unwrap().0, k);
            });
    }

    #[test]
    fn partial_eq_and_eq() {
        let mut map_a = IndexMultimap::new();
        map_a.insert_append(1, "1");
        map_a.insert_append(2, "2");
        let mut map_b = map_a.clone();
        assert!(map_a.par_eq(&map_b));
        map_b.swap_remove(&1);
        assert!(!map_a.par_eq(&map_b));
        map_b.insert_append(3, "3");
        assert!(!map_a.par_eq(&map_b));

        let map_c: IndexMultimap<_, String> =
            map_b.into_par_iter().map(|(k, v)| (k, v.into())).collect();
        assert!(!map_a.par_eq(&map_c));
        assert!(!map_c.par_eq(&map_a));
    }

    #[test]
    fn extend() {
        let mut map = IndexMultimap::new();
        map.par_extend(vec![(&1, &2), (&3, &4)]);
        map.par_extend(vec![(5, 6)]);
        assert_eq!(
            map.into_par_iter().collect::<Vec<_>>(),
            vec![(1, 2), (3, 4), (5, 6)]
        );
    }

    #[test]
    fn keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: IndexMultimap<_, _> = vec.into_par_iter().collect();
        let keys: Vec<_> = map.par_keys().copied().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: IndexMultimap<_, _> = vec.into_par_iter().collect();
        let values: Vec<_> = map.par_values().copied().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn values_mut() {
        let vec = vec![(1, 1), (2, 2), (3, 3)];
        let mut map: IndexMultimap<_, _> = vec.into_par_iter().collect();
        map.par_values_mut().for_each(|value| *value *= 2);
        let values: Vec<_> = map.par_values().copied().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&2));
        assert!(values.contains(&4));
        assert!(values.contains(&6));
    }

    #[test]
    fn drain_all() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(..).collect::<Vec<_>>();
        assert_eq!(&drained, &items);
        assert!(map.is_empty());

        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(0..7).collect::<Vec<_>>();
        assert_eq!(&drained, &items);
        assert!(map.is_empty());

        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(0..=6).collect::<Vec<_>>();
        assert_eq!(&drained, &items);
        assert!(map.is_empty());
    }

    #[test]
    fn drain_none() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(items.len()..).collect::<Vec<_>>();
        assert!(drained.is_empty());
        assert_eq!(map.len_pairs(), 7);

        let drained = map.par_drain(3..3).collect::<Vec<_>>();
        assert!(drained.is_empty());
        assert_eq!(map.len_pairs(), 7);
    }

    #[test]
    fn drain_out_of_bounds() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);
        let len = map.len_pairs();
        assert!(catch_unwind(AssertUnwindSafe(|| drop(map.par_drain((len + 1)..)))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| drop(map.par_drain(..(len + 1))))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| drop(map.par_drain(..=len)))).is_err());
    }

    #[test]
    fn drain_start_to_mid() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(..3).collect::<Vec<_>>();
        assert_eq!(&drained, &items[..3]);

        let remaining = &items[3..];
        assert_map_eq(&map, remaining);
    }

    #[test]
    fn drain_mid_to_end() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(3..).collect::<Vec<_>>();
        assert_eq!(&drained, &items[3..]);

        let remaining = &items[..3];
        println!("{map:#?}");
        assert_map_eq(&map, remaining);
    }

    #[test]
    fn drain_mid_to_mid() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(3..6).collect::<Vec<_>>();
        assert_eq!(&drained, &items[3..6]);

        let remaining = [&items[0..3], &items[6..]].concat();
        assert_map_eq(&map, &remaining);
    }

    #[test]
    fn drain_empty() {
        let mut map: IndexMultimap<i32, i32> = IndexMultimap::new();
        let mut map2: IndexMultimap<i32, i32> = map.par_drain(..).collect();
        assert!(map.is_empty());
        assert!(map2.is_empty());
    }

    #[test]
    fn drain_rev() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(2..5).rev().collect::<Vec<_>>();
        let mut expected = items[2..5].to_vec();
        expected.reverse();
        assert_eq!(&drained, &expected);

        let remaining = [&items[..2], &items[5..]].concat();
        assert_map_eq(&map, &remaining);
    }

    // ignore if miri or asan is set but not if ignore_leaks is also set
    #[cfg_attr(
        all(not(run_leaky), any(miri, asan)),
        ignore = "it tests what happens if we leak ParDrain"
    )]
    #[test]
    fn drain_leak() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        ::core::mem::forget(map.par_drain(2..));

        println!("{map:#?}");
        //map.extend(items);
        assert!(map.is_empty());
        assert_eq!(map.len_keys(), 0);
        assert_eq!(map.len_pairs(), 0);
    }

    #[test]
    fn drain_drop_panic() {
        static DROPS: AtomicU32 = AtomicU32::new(0);

        #[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
        struct MaybePanicOnDrop(bool, bool, String);

        impl Drop for MaybePanicOnDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, ::core::sync::atomic::Ordering::SeqCst);

                //println!("dropping {:?}", self);
                if self.0 {
                    self.1 = true;
                    panic!("panic in `Drop`: {}", self.2)
                }
            }
        }

        let items = [
            (1, MaybePanicOnDrop(false, false, String::from("0"))),
            (2, MaybePanicOnDrop(false, false, String::from("1"))),
            (1, MaybePanicOnDrop(false, false, String::from("2"))),
            (3, MaybePanicOnDrop(false, false, String::from("3"))),
            (1, MaybePanicOnDrop(false, false, String::from("4"))),
            (1, MaybePanicOnDrop(true, false, String::from("5"))),
            (1, MaybePanicOnDrop(false, false, String::from("6"))),
            (1, MaybePanicOnDrop(false, false, String::from("7"))),
        ];
        let mut map = IndexMultimap::new();
        map.extend(items);

        catch_unwind(AssertUnwindSafe(|| drop(map.par_drain(4..7)))).ok();
        assert_eq!(DROPS.load(::core::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn drain_drop_panic_consumed() {
        static DROPS: AtomicU32 = AtomicU32::new(0);

        #[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
        struct MaybePanicOnDrop(bool, bool, String);

        impl Drop for MaybePanicOnDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, ::core::sync::atomic::Ordering::SeqCst);

                //println!("dropping {:?}", self);
                if self.0 {
                    self.1 = true;
                    panic!("panic in `Drop`: {}", self.2)
                }
            }
        }

        let items = [
            (1, MaybePanicOnDrop(false, false, String::from("0"))),
            (2, MaybePanicOnDrop(false, false, String::from("1"))),
            (1, MaybePanicOnDrop(false, false, String::from("2"))),
            (3, MaybePanicOnDrop(false, false, String::from("3"))),
            (1, MaybePanicOnDrop(false, false, String::from("4"))),
            (1, MaybePanicOnDrop(true, false, String::from("5"))),
            (1, MaybePanicOnDrop(false, false, String::from("6"))),
            (1, MaybePanicOnDrop(false, false, String::from("7"))),
        ];
        let mut map = IndexMultimap::new();
        map.extend(items);

        catch_unwind(AssertUnwindSafe(|| map.par_drain(4..7).for_each(drop))).ok();
        //println!("{map:#?}");
        assert_eq!(DROPS.load(::core::sync::atomic::Ordering::SeqCst), 3);
    }
}

#![allow(unsafe_code)]

use core::fmt;
use core::iter::FusedIterator;
use core::marker::PhantomData;

use alloc::vec::Vec;

use crate::multimap::core::Unique;
use crate::Bucket;

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
    indices: Unique<I>,
    // What self.pairs really is, constructors should take this to bind the lifetime properly
    _marker: PhantomData<&'a mut [Bucket<K, V>]>,
}

impl<'a, K, V, I> SubsetIterMut<'a, K, V, I>
where
    I: Iterator<Item = usize>,
{
    pub(crate) fn new(pairs: &'a mut [Bucket<K, V>], indices: Unique<I>) -> Self {
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

pub(crate) mod internal {
    pub struct Guard;
    pub trait Sealed {}
    pub struct Bounds<T>(T);
    impl<T> Sealed for Bounds<T> {}
    impl<'a> Sealed for &'a [usize] {}
    impl Sealed for alloc::vec::Vec<usize> {}
}

/// GAT workaround for [`SubsetIndexStorage`] providing an associated `Iter`
/// type that could borrow from self.
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
#[allow(clippy::missing_safety_doc)] // sealed trait, safety is below
pub unsafe trait SubsetIndexStorage
where
    Self: internal::Sealed + for<'a> ToIndexIter<'a> + core::ops::Deref<Target = [usize]> + Default,
{
    // # Safety
    //
    // Methods must behave like the ones on Vec
    type IntoIter: Iterator<Item = usize> + DoubleEndedIterator + ExactSizeIterator + FusedIterator;

    #[doc(hidden)]
    fn into_index_iter(self, _: internal::Guard) -> Self::IntoIter;

    #[doc(hidden)]
    fn index_iter(&self, _: internal::Guard) -> <Self as ToIndexIter<'_>>::Iter;
}

impl<'a> ToIndexIter<'a> for &[usize] {
    type Iter = ::core::iter::Copied<::core::slice::Iter<'a, usize>>;
}

unsafe impl<'a> SubsetIndexStorage for &'a [usize] {
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

unsafe impl SubsetIndexStorage for alloc::vec::Vec<usize> {
    type IntoIter = alloc::vec::IntoIter<usize>;

    fn into_index_iter(self, _: internal::Guard) -> Self::IntoIter {
        core::iter::IntoIterator::into_iter(self)
    }

    fn index_iter(&self, _: internal::Guard) -> <Self as ToIndexIter<'_>>::Iter {
        <[usize]>::iter(self).copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::multimap::core::Unique;
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
        let indices = unsafe { Unique::new_unchecked([0usize, 2, 3, 5].as_slice()) };

        let _iter1 = SubsetIterMut::new(&mut pairs, indices.slice_iter().copied());
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
        let indices = unsafe { Unique::new_unchecked([0usize, 2, 3, 5].as_slice()) };

        let iter1 = SubsetIterMut::new(&mut pairs, indices.slice_iter().copied());
        //println!("{iter1:#?}");
        //iter.clone();
        let items = iter1.collect::<Vec<_>>();
        let values = items.iter().map(|a| *a.2).collect::<Vec<_>>();
        assert!(is_unique(&items));
        assert!(is_unique(&values));

        let mut iter2 = SubsetIterMut::new(&mut pairs, indices.slice_iter().copied());

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

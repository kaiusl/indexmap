#![allow(unsafe_code)]
#![warn(clippy::missing_safety_doc)]

use ::alloc::vec;
use ::alloc::vec::Vec;
use ::core::{mem, ops, slice};
use core::ops::{Range, RangeBounds};
pub(super) use iterators::{UniqueIter, UniqueSortedIter};

use crate::util::{is_sorted_and_unique, try_simplify_range};
use crate::TryReserveError;

/// Unique and sorted set of indices
#[derive(Debug, Clone)]
pub(super) struct Indices {
    // INVARIANTS
    //  * `inner` is unique and sorted
    inner: Vec<usize>,
}

#[allow(dead_code)]
impl Indices {
    #[inline]
    pub(crate) fn one(index: usize) -> Self {
        Self { inner: vec![index] }
    }

    /// Returns the total number of indices self can hold without reallocating.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserve capacity for `additional` more key-value pairs.
    #[inline]
    pub(super) fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    /// Reserve capacity for `additional` more key-value pairs, without over-allocating.
    #[inline]
    pub(super) fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    /// Try to reserve capacity for `additional` more key-value pairs.
    #[inline]
    pub(super) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner
            .try_reserve(additional)
            .map_err(TryReserveError::from_alloc)
    }

    /// Try to reserve capacity for `additional` more key-value pairs, without over-allocating.
    #[inline]
    pub(super) fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner
            .try_reserve_exact(additional)
            .map_err(TryReserveError::from_alloc)
    }

    /// Shrink the capacity of the map with a lower bound
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    #[inline]
    pub(super) fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity);
    }

    /// Shrinks the capacity of self as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator
    /// may still inform the vector that there is space for a few more elements.
    #[inline]
    pub(super) fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }
    #[inline]
    pub(crate) fn as_unique_slice(&self) -> &UniqueSlice<usize> {
        unsafe { UniqueSlice::from_slice_unchecked(&self) }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.inner
    }

    #[inline]
    pub(crate) unsafe fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.inner
    }

    #[inline]
    pub(crate) fn push(&mut self, v: usize) {
        assert!(
            self.inner.is_empty() || v > *self.inner.last().unwrap(),
            "pushed value must be larger than any other value in the vec"
        );
        self.inner.push(v);
    }

    #[inline]
    pub(crate) fn remove(&mut self, index: usize) -> usize {
        // SAFETY: inner.remove preserves the order
        self.inner.remove(index)
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<usize> {
        self.inner.pop()
    }

    #[inline]
    pub(crate) fn retain<F>(&mut self, keep: F)
    where
        F: FnMut(&usize) -> bool,
    {
        // SAFETY: User cannot modify the values and `self.ìnner.retain` preserves the order.
        // Thus `self.inner` remains unique and sorted.
        self.inner.retain(keep);
        debug_assert!(is_sorted_and_unique(self.as_slice()))
    }

    #[inline]
    pub(crate) unsafe fn retain_mut<F>(&mut self, keep: F)
    where
        F: FnMut(&mut usize) -> bool,
    {
        // SAFETY: User cannot modify the values and `self.ìnner.retain` preserves the order.
        // Thus `self.inner` remains unique and sorted.
        self.inner.retain_mut(keep);
        debug_assert!(is_sorted_and_unique(self.as_slice()))
    }

    #[inline]
    pub(crate) fn into_iter(self) -> UniqueSortedIter<alloc::vec::IntoIter<usize>> {
        // SAFETY: vec::IntoIter yields each item in the vector only once and in
        // order. Since our vector (`self.inner`) contains unique items
        // in sorted order, then `self.inner.into_iter()` is an iterator which
        // returns unique items in sorted order.
        unsafe { UniqueSortedIter::new_unchecked(self.inner.into_iter()) }
    }

    /// Replaces an value at `self[old_index]` with `new` and keeps `self` sorted.
    ///
    /// Returns the old value at `self[old_index]`.
    ///
    /// # Panics
    ///
    /// * If `new` already exists. It would violate the uniqueness guarantee of indices.
    /// * If `old_index` is out of bounds for self
    pub(crate) fn replace(&mut self, old_index: usize, new: usize) -> usize {
        if self.len() == 1 {
            return mem::replace(&mut self.inner[old_index], new);
        }

        // SAFETY: we keep the slice sorted and unique
        let slice = unsafe { self.as_mut_slice() };
        let new_sorted_pos = slice.partition_point(|&a| a < new);
        // Since our array is sorted, it will also be unique if the next element
        // returned by partition_point is not our new index
        // Say we want to insert 6. Then in list [1, 2, 4, 6, 8],
        // partition_point will return 3. Then below will panic.
        assert!(
            slice.get(new_sorted_pos).map(|&i| i != new).unwrap_or(true),
            "`new` already exists, it would violate uniqueness"
        );

        let old = mem::replace(&mut slice[old_index], new);
        use core::cmp::Ordering;
        match old_index.cmp(&new_sorted_pos) {
            Ordering::Less => slice[old_index..new_sorted_pos].rotate_left(1),
            Ordering::Equal => {}
            Ordering::Greater => slice[new_sorted_pos..=old_index].rotate_right(1),
        }
        debug_assert!(is_sorted_and_unique(self.as_slice()));
        old
    }

    /// # Safety
    ///
    /// * iter must yield indices larger than anything currently in `self` (larger than self.last())
    ///   and they must be sorted and unique
    pub(crate) unsafe fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = usize>,
    {
        self.inner.extend(iter);
    }

    pub(crate) fn from_range(range: Range<usize>) -> Self {
        Self {
            inner: range.collect(),
        }
    }
}

impl ops::Deref for Indices {
    type Target = [usize];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct UniqueSlice<T> {
    inner: [T],
}

impl<T> ops::Deref for UniqueSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, T> Default for &'a UniqueSlice<T> {
    fn default() -> Self {
        // SAFETY: empty slice is unique
        unsafe { UniqueSlice::from_slice_unchecked(&[]) }
    }
}

impl<T> UniqueSlice<T> {
    pub(crate) unsafe fn from_slice_unchecked(slice: &[T]) -> &Self {
        // SAFETY:
        //  * `Self` is `repr(transparent)` wrapper around `[T]`.
        //  * Reference lifetimes are bound in function signature.
        unsafe { &*(slice as *const [T] as *const Self) }
    }
    pub fn as_slice(&self) -> &[T] {
        &*self
    }

    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<&Self> {
        let range = try_simplify_range(range, self.inner.len())?;
        match self.inner.get(range) {
            Some(inner) => Some(unsafe { UniqueSlice::from_slice_unchecked(inner) }),
            None => None,
        }
    }

    pub fn iter(&self) -> UniqueIter<slice::Iter<'_, T>> {
        // SAFETY: slice::Iter yields each item in the slice only once.
        // Since our slice (`self.inner`) contains unique items,
        // `self.inner.iter()` is an iterator which returns unique items.
        unsafe { UniqueIter::new_unchecked(self.inner.iter()) }
    }

    /// Divides one slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)`
    /// (excluding the index mid itself) and the second will contain all
    /// indices from `[mid, len)`` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub(crate) fn split_at(&self, mid: usize) -> (&Self, &Self) {
        let (left, right) = self.inner.split_at(mid);
        unsafe {
            (
                UniqueSlice::from_slice_unchecked(left),
                UniqueSlice::from_slice_unchecked(right),
            )
        }
    }

    /// Returns the first and all the rest of the elements of the slice, or None if it is empty.
    pub(crate) fn split_first(&self) -> Option<(&T, &Self)> {
        match self.inner.split_first() {
            Some((first, tail)) => {
                Some((first, unsafe { UniqueSlice::from_slice_unchecked(tail) }))
            }
            None => None,
        }
    }

    /// Returns the last and all the rest of the elements of the slice, or None if it is empty
    pub(crate) fn split_last(&self) -> Option<(&T, &Self)> {
        match self.inner.split_last() {
            Some((last, head)) => Some((last, unsafe { UniqueSlice::from_slice_unchecked(head) })),
            None => None,
        }
    }
}

impl<'a, T> IntoIterator for &'a UniqueSlice<T> {
    type Item = &'a T;
    type IntoIter = UniqueIter<slice::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

mod iterators {
    use core::iter::FusedIterator;

    #[derive(Debug, Clone)]
    pub(crate) struct UniqueSortedIter<Inner> {
        inner: Inner,
    }

    impl<Inner> UniqueSortedIter<Inner> {
        /// # Safety
        ///
        /// * items yielded by `ìter` must be unique and sorted
        pub(crate) unsafe fn new_unchecked(iter: Inner) -> Self {
            UniqueSortedIter { inner: iter }
        }
    }

    impl<Inner> Iterator for UniqueSortedIter<Inner>
    where
        Inner: Iterator,
    {
        type Item = Inner::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }

        #[inline]
        fn count(self) -> usize
        where
            Self: Sized,
        {
            self.inner.count()
        }

        #[inline]
        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            self.inner.nth(n)
        }

        #[inline]
        fn collect<B: FromIterator<Self::Item>>(self) -> B
        where
            Self: Sized,
        {
            self.inner.collect()
        }
    }

    impl<I> ExactSizeIterator for UniqueSortedIter<I>
    where
        I: Iterator + ExactSizeIterator,
    {
        #[inline]
        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    impl<I> DoubleEndedIterator for UniqueSortedIter<I>
    where
        I: Iterator + DoubleEndedIterator,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.inner.next_back()
        }
    }

    impl<I> FusedIterator for UniqueSortedIter<I> where I: Iterator + FusedIterator {}

    #[derive(Debug, Clone)]
    pub(crate) struct UniqueIter<Inner> {
        inner: Inner,
    }

    impl<Inner> UniqueIter<Inner> {
        /// # Safety
        ///
        /// * items yielded by `ìter` must be unique
        pub(in super::super) unsafe fn new_unchecked(iter: Inner) -> Self {
            UniqueIter { inner: iter }
        }
    }

    impl<Inner> Iterator for UniqueIter<Inner>
    where
        Inner: Iterator,
    {
        type Item = Inner::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }

        #[inline]
        fn count(self) -> usize
        where
            Self: Sized,
        {
            self.inner.count()
        }

        #[inline]
        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            self.inner.nth(n)
        }

        #[inline]
        fn collect<B: FromIterator<Self::Item>>(self) -> B
        where
            Self: Sized,
        {
            self.inner.collect()
        }
    }

    impl<I> ExactSizeIterator for UniqueIter<I>
    where
        I: Iterator + ExactSizeIterator,
    {
        #[inline]
        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    impl<I> DoubleEndedIterator for UniqueIter<I>
    where
        I: Iterator + DoubleEndedIterator,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.inner.next_back()
        }
    }

    impl<I> FusedIterator for UniqueIter<I> where I: Iterator + FusedIterator {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push() {
        let mut i = Indices::one(5);
        i.push(6);

        i.remove(0);
        i.remove(0);
        assert!(i.is_empty());

        i.push(3);
    }

    #[test]
    fn push_panic() {
        use std::panic::catch_unwind;
        catch_unwind(|| {
            let mut i = Indices::one(5);
            i.push(5);
        })
        .expect_err("pushed value must be larger than any other value in the vec");

        catch_unwind(|| {
            let mut i = Indices::one(5);
            i.push(3);
        })
        .expect_err("pushed value must be larger than any other value in the vec");
    }

    #[test]
    fn replace_unique_sorted_test() {
        // new > old
        let mut i = Indices {
            inner: vec![1, 2, 4, 6, 8, 10],
        };
        i.replace(1, 7);
        assert_eq!(i.as_slice(), [1, 4, 6, 7, 8, 10]);

        // new < old
        let mut i = Indices {
            inner: vec![1, 2, 4, 6, 8, 10],
        };
        i.replace(4, 3);
        assert_eq!(i.as_slice(), [1, 2, 3, 4, 6, 10]);

        // new == old
        let mut i = Indices {
            inner: vec![1, 2, 4, 6, 8, 10],
        };
        i.replace(4, 9);
        assert_eq!(i.as_slice(), [1, 2, 4, 6, 9, 10]);

        // new larger than anything
        let mut i = Indices {
            inner: vec![1, 2, 4, 6, 8, 10],
        };
        i.replace(4, 12);
        assert_eq!(i.as_slice(), [1, 2, 4, 6, 10, 12]);

        // new smaller than anything
        let mut i = Indices {
            inner: vec![1, 2, 4, 6, 8, 10],
        };
        i.replace(4, 0);
        assert_eq!(i.as_slice(), [0, 1, 2, 4, 6, 10]);
    }

    #[test]
    #[should_panic = "`new` already exists, it would violate uniqueness"]
    fn replace_unique_sorted_test_new_exists() {
        // expected replace to panic as `new` already exists
        let mut i = Indices {
            inner: vec![1, 2, 4, 6, 8, 10],
        };
        i.replace(1, 6);
    }

    #[test]
    #[should_panic = "index out of bounds: the len is 6 but the index is 6"]
    fn replace_unique_sorted_test_old_index_out_of_bounds() {
        // expected replace to panic as `old_index` is out of bounds
        let mut i = Indices {
            inner: vec![1, 2, 4, 6, 8, 10],
        };
        i.replace(6, 7);
    }
}

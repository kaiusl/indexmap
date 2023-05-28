#![allow(unsafe_code)]

use ::core::iter::FusedIterator;
use ::core::{iter, mem, ops, slice};

use super::subsets::{internal, SubsetIndexStorage, ToIndexIter};
use super::IndexStorage;

// --- 
// 
// # Safety
// 
// It's important to note that the safety of the structs below
// relies on the fact that the only way to create them is to start with
// `UniqueSorted::one`. That method requires
// that `Inner: IndexStorage` which means that at start `Inner` is some Vec<usize>
// like object. After that it's possible to convert it into iterators or `Unique`.
// Only way to modify the values in `Inner` if through safe `&mut` functions on
// `self` of unsafe ones.
// 
// This means that even though in general if `Inner` type contains some sort of
// interior mutability or is an iterator that yields `&mut something` one could
// modify the inner data through &, which would be unsafe as it could easily 
// mess with our invariants of uniqueness and sortedness. However if one starts
// from `UniqueSorted::one` then it's not possible to have interior mutability 
// and only way to get a mutable iterator would be through unsafe methods
// (which atm we don't even provide).
// 
// ---

/// Wrapper that promises that `Inner` only contains unique and sorted items.
#[derive(Debug, Clone, Default)]
pub(super) struct UniqueSorted<Inner> {
    inner: Inner,
}

// It's OK to Deref as the user cannot modify the slice they get.
// DeferMut on the other hand would be unsafe, for that purpose there is `unsafe fn as_mut_slice`,
impl<Inner> ops::Deref for UniqueSorted<Inner>
where
    Inner: ops::Deref,
{
    type Target = Inner::Target;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Inner> UniqueSorted<Inner> {
    #[inline]
    pub fn as_inner(&self) -> &Inner {
        &self.inner
    }

    #[inline]
    pub fn into_inner(self) -> Inner {
        self.inner
    }

    #[inline]
    pub fn as_slice<T>(&self) -> &[T]
    where
        Inner: ops::Deref<Target = [T]>,
    {
        &self.inner
    }

    /// # Safety
    ///
    /// * The items in the slice must remain unique and sorted
    #[inline]
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T]
    where
        Inner: ops::DerefMut<Target = [T]>,
    {
        &mut self.inner
    }

    #[inline]
    pub fn slice_iter<T>(&self) -> UniqueSorted<slice::Iter<'_, T>>
    where
        Inner: ops::Deref<Target = [T]>,
    {
        UniqueSorted {
            inner: self.inner.iter(),
        }
    }
}

impl<Inner> UniqueSorted<Inner>
where
    Inner: IndexStorage,
{
    #[inline]
    pub(crate) fn one(v: usize) -> Self {
        Self {
            inner: Inner::one(v),
        }
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
        self.inner.remove(index)
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<usize> {
        self.inner.pop()
    }

    #[inline]
    pub(crate) unsafe fn retain<F>(&mut self, keep: F)
    where
        F: FnMut(&mut usize) -> bool,
    {
        self.inner.retain(keep);
    }

    /// Replaces an value at `self[old_index]` with `new` and keeps `self` sorted.
    ///
    /// # Panics
    ///
    /// * If `new` already exists. It would violate the uniqueness guarantee of indices.
    /// * If `old_index` is out of bounds for self
    pub(crate) fn replace(&mut self, old_index: usize, new: usize) -> usize {
        let len = self.len();
        if len == 1 {
            return mem::replace(&mut self.inner[old_index], new);
        }

        // Is this even beneficial? We avoid binary search for position but do bunch of comparisons
        // TODO: benchmark it
        let is_new_larger_than_prev = old_index == 0 || self.inner[old_index - 1] < new;
        let is_new_smaller_than_next = old_index == len - 1 || new < self.inner[old_index + 1];
        if is_new_larger_than_prev && is_new_smaller_than_next {
            return mem::replace(&mut self.inner[old_index], new);
        }

        // SAFETY: we keep the slice sorted and unique
        let slice = unsafe { self.as_mut_slice() };
        let new_sorted_pos = slice.partition_point(|&a| a < new);
        // Since our array is sorted, it will also be unique if the next element
        // returned by partition_point is not our new index
        assert!(
            slice
                .get(new_sorted_pos + 1)
                .map(|&i| i != new)
                .unwrap_or(true),
            "`new` already exists, it would violate uniqueness"
        );

        let old = mem::replace(&mut slice[old_index], new);
        use core::cmp::Ordering;
        match old_index.cmp(&new_sorted_pos) {
            Ordering::Less => slice[old_index..new_sorted_pos].rotate_left(1),
            Ordering::Equal => {}
            Ordering::Greater => slice[new_sorted_pos..=old_index].rotate_right(1),
        }
        old
    }

    #[inline]
    pub(crate) fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    #[inline]
    pub(crate) fn into_iter(self) -> UniqueSorted<Inner::IntoIter> {
        UniqueSorted {
            inner: self.inner.into_iter(),
        }
    }
}

impl<Indices> ops::Index<usize> for UniqueSorted<Indices>
where
    Indices: ops::Index<usize>,
{
    type Output = Indices::Output;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.inner.index(index)
    }
}

impl<I> Iterator for UniqueSorted<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
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
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn collect<B: FromIterator<Self::Item>>(self) -> B
    where
        Self: Sized,
    {
        self.inner.collect()
    }
}

impl<I> ExactSizeIterator for UniqueSorted<I>
where
    I: Iterator + ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<I> DoubleEndedIterator for UniqueSorted<I>
where
    I: Iterator + DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<I> FusedIterator for UniqueSorted<I> where I: Iterator + FusedIterator {}

/// Wrapper that promises that `Inner` only contains unique items.
#[derive(Debug, Clone, Default)]
pub(crate) struct Unique<Inner> {
    inner: Inner,
}

impl<Inner> ops::Deref for Unique<Inner>
where
    Inner: ops::Deref,
{
    type Target = Inner::Target;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Inner> Unique<Inner> {
    #[inline]
    pub fn as_inner(&self) -> &Inner {
        &self.inner
    }
}

#[cfg(test)]
impl<Inner> Unique<Inner> {
    #[inline]
    pub(crate) unsafe fn new_unchecked(indices: Inner) -> Self {
        Self { inner: indices }
    }

    #[inline]
    pub fn slice_iter<T>(&self) -> Unique<slice::Iter<'_, T>>
    where
        Inner: ops::Deref<Target = [T]>,
    {
        Unique {
            inner: self.inner.iter(),
        }
    }
}

impl<Inner> internal::Sealed for Unique<Inner> where Inner: internal::Sealed {}

impl<'a, Inner> ToIndexIter<'a> for Unique<Inner>
where
    Inner: ToIndexIter<'a>,
{
    type Iter = Unique<Inner::Iter>;
}

unsafe impl<Inner> SubsetIndexStorage for Unique<Inner>
where
    Inner: SubsetIndexStorage,
{
    type IntoIter = Unique<Inner::IntoIter>;

    #[inline]
    fn into_index_iter(self, guard: internal::Guard) -> Self::IntoIter {
        Unique {
            inner: self.inner.into_index_iter(guard),
        }
    }

    #[inline]
    fn index_iter(&self, guard: internal::Guard) -> <Self as ToIndexIter<'_>>::Iter {
        Unique {
            inner: self.inner.index_iter(guard),
        }
    }
}

impl<'a, Inner> From<&'a UniqueSorted<Inner>> for Unique<&'a Inner> {
    #[inline]
    fn from(value: &'a UniqueSorted<Inner>) -> Self {
        Self {
            inner: value.as_inner(),
        }
    }
}

impl<'a, Inner> From<&'a UniqueSorted<Inner>> for Unique<&'a [usize]>
where
    Inner: ops::Deref<Target = [usize]>,
{
    #[inline]
    fn from(value: &'a UniqueSorted<Inner>) -> Self {
        Self {
            inner: value.as_slice(),
        }
    }
}

impl<Inner> From<UniqueSorted<Inner>> for Unique<Inner> {
    #[inline]
    fn from(value: UniqueSorted<Inner>) -> Self {
        Self {
            inner: value.into_inner(),
        }
    }
}

impl<I> Unique<I> {
    #[inline]
    pub(crate) fn copied<'a, T: 'a>(self) -> Unique<iter::Copied<I>>
    where
        I: Sized + Iterator<Item = &'a T>,
        T: Copy,
    {
        Unique {
            inner: self.inner.copied(),
        }
    }
}

impl<I> Iterator for Unique<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
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
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn collect<B: FromIterator<Self::Item>>(self) -> B
    where
        Self: Sized,
    {
        self.inner.collect()
    }
}

impl<I> ExactSizeIterator for Unique<I>
where
    I: Iterator + ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<I> DoubleEndedIterator for Unique<I>
where
    I: Iterator + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<I> FusedIterator for Unique<I> where I: Iterator + FusedIterator {}

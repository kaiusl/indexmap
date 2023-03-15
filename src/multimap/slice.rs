use super::{IndexMultimap, IndexStorage};

use crate::map::Slice;
use core::ops::{self, Bound, Index, IndexMut};

// We can't have `impl<I: RangeBounds<usize>> Index<I>` because that conflicts
// both upstream with `Index<usize>` and downstream with `Index<&Q>`.
// Instead, we repeat the implementations for all the core range types.
macro_rules! impl_index {
    ($($range:ty),*) => {$(
        impl<K, V, S, Indices> Index<$range> for IndexMultimap<K, V, S, Indices>
        where
            K: Eq,
            Indices: IndexStorage
        {
            type Output = Slice<K, V>;

            fn index(&self, range: $range) -> &Self::Output {
                Slice::from_slice(&self.core.as_pairs()[range])
            }
        }

        impl<K, V, S, Indices> IndexMut<$range> for IndexMultimap<K, V, S, Indices>
        where
            K: Eq,
            Indices: IndexStorage
        {
            fn index_mut(&mut self, range: $range) -> &mut Self::Output {
                Slice::from_mut_slice(&mut self.core.as_pairs_mut()[range])
            }
        }
    )*}
}
impl_index!(
    ops::Range<usize>,
    ops::RangeFrom<usize>,
    ops::RangeFull,
    ops::RangeInclusive<usize>,
    ops::RangeTo<usize>,
    ops::RangeToInclusive<usize>,
    (Bound<usize>, Bound<usize>)
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn slice_index() {
        fn check(
            vec_slice: &[(i32, i32)],
            map_slice: &Slice<i32, i32>,
            sub_slice: &Slice<i32, i32>,
        ) {
            assert_eq!(map_slice as *const _, sub_slice as *const _);
            itertools::assert_equal(
                vec_slice.iter().copied(),
                map_slice.iter().map(|(&k, &v)| (k, v)),
            );
            itertools::assert_equal(vec_slice.iter().map(|(k, _)| k), map_slice.keys());
            itertools::assert_equal(vec_slice.iter().map(|(_, v)| v), map_slice.values());
        }

        let vec: Vec<(i32, i32)> = (0..10).map(|i| (i, i * i)).collect();
        let map: IndexMultimap<i32, i32> = vec.iter().cloned().collect();
        let slice = map.as_slice();

        // RangeFull
        check(&vec[..], &map[..], slice);

        for i in 0usize..10 {
            // Index
            assert_eq!(vec[i].1, map[i]);
            assert_eq!(vec[i].1, slice[i]);
            assert_eq!(map.get(&(i as i32)).first().unwrap().2, &map[i]);
            assert_eq!(map.get(&(i as i32)).first().unwrap().2, &slice[i]);

            // RangeFrom
            check(&vec[i..], &map[i..], &slice[i..]);

            // RangeTo
            check(&vec[..i], &map[..i], &slice[..i]);

            // RangeToInclusive
            check(&vec[..=i], &map[..=i], &slice[..=i]);

            // (Bound<usize>, Bound<usize>)
            let bounds = (Bound::Excluded(i), Bound::Unbounded);
            check(&vec[i + 1..], &map[bounds], &slice[bounds]);

            for j in i..=10 {
                // Range
                check(&vec[i..j], &map[i..j], &slice[i..j]);
            }

            for j in i..10 {
                // RangeInclusive
                check(&vec[i..=j], &map[i..=j], &slice[i..=j]);
            }
        }
    }

    #[test]
    fn slice_index_mut() {
        fn check_mut(
            vec_slice: &[(i32, i32)],
            map_slice: &mut Slice<i32, i32>,
            sub_slice: &mut Slice<i32, i32>,
        ) {
            assert_eq!(map_slice, sub_slice);
            itertools::assert_equal(
                vec_slice.iter().copied(),
                map_slice.iter_mut().map(|(&k, &mut v)| (k, v)),
            );
            itertools::assert_equal(
                vec_slice.iter().map(|&(_, v)| v),
                map_slice.values_mut().map(|&mut v| v),
            );
        }

        let vec: Vec<(i32, i32)> = (0..10).map(|i| (i, i * i)).collect();
        let mut map: IndexMultimap<i32, i32> = vec.iter().cloned().collect();
        let mut map2 = map.clone();
        let slice = map2.as_mut_slice();

        // RangeFull
        check_mut(&vec[..], &mut map[..], &mut slice[..]);

        for i in 0usize..10 {
            // IndexMut
            assert_eq!(&mut map[i], &mut slice[i]);

            // RangeFrom
            check_mut(&vec[i..], &mut map[i..], &mut slice[i..]);

            // RangeTo
            check_mut(&vec[..i], &mut map[..i], &mut slice[..i]);

            // RangeToInclusive
            check_mut(&vec[..=i], &mut map[..=i], &mut slice[..=i]);

            // (Bound<usize>, Bound<usize>)
            let bounds = (Bound::Excluded(i), Bound::Unbounded);
            check_mut(&vec[i + 1..], &mut map[bounds], &mut slice[bounds]);

            for j in i..=10 {
                // Range
                check_mut(&vec[i..j], &mut map[i..j], &mut slice[i..j]);
            }

            for j in i..10 {
                // RangeInclusive
                check_mut(&vec[i..=j], &mut map[i..=j], &mut slice[i..=j]);
            }
        }
    }
}

use core::ops::{Bound, Range, RangeBounds};

pub(crate) fn third<A, B, C>(t: (A, B, C)) -> C {
    t.2
}

pub(crate) fn simplify_range<R>(range: R, len: usize) -> Range<usize>
where
    R: RangeBounds<usize>,
{
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(&i) if i <= len => i,
        Bound::Excluded(&i) if i < len => i + 1,
        bound => panic!("range start {:?} should be <= length {}", bound, len),
    };
    let end = match range.end_bound() {
        Bound::Unbounded => len,
        Bound::Excluded(&i) if i <= len => i,
        Bound::Included(&i) if i < len => i + 1,
        bound => panic!("range end {:?} should be <= length {}", bound, len),
    };
    if start > end {
        panic!(
            "range start {:?} should be <= range end {:?}",
            range.start_bound(),
            range.end_bound()
        );
    }
    start..end
}

pub(crate) fn try_simplify_range<R>(range: R, len: usize) -> Option<Range<usize>>
where
    R: RangeBounds<usize>,
{
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(&i) if i <= len => i,
        Bound::Excluded(&i) if i < len => i + 1,
        _ => return None,
    };
    let end = match range.end_bound() {
        Bound::Unbounded => len,
        Bound::Excluded(&i) if i <= len => i,
        Bound::Included(&i) if i < len => i + 1,
        _ => return None,
    };
    if start > end {
        return None;
    }
    Some(start..end)
}

/// Replaces an element at index `old_index` with `new_value` and resorts the slice.
/// Return the old value.
///
/// Assumes that the `slice` is sorted to begin with.
pub(crate) fn replace_sorted<T>(slice: &mut [T], old_index: usize, new_value: T) -> T
where
    T: core::cmp::Ord + PartialEq,
{
    let new_sorted_pos = slice.partition_point(|a| a < &new_value);
    let old = core::mem::replace(&mut slice[old_index], new_value);
    use core::cmp::Ordering;
    match old_index.cmp(&new_sorted_pos) {
        Ordering::Less => slice[old_index..new_sorted_pos].rotate_left(1),
        Ordering::Equal => {}
        Ordering::Greater => slice[new_sorted_pos..=old_index].rotate_right(1),
    }
    old
}

pub(crate) fn is_sorted<T: PartialOrd>(slice: &[T]) -> bool {
    slice.windows(2).all(|w| w[0] <= w[1])
}

pub(crate) fn is_unique_sorted<T: PartialOrd>(slice: &[T]) -> bool {
    slice.windows(2).all(|w| w[0] != w[1])
}


/// Checks if the slice contains only unique items.
pub(crate) fn is_unique<T>(slice: &[T]) -> bool
where
    T: PartialEq,
{
    (1..).zip(slice).all(|(i, it)| !slice[i..].contains(it))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_index_test() {
        // new > old
        let mut i = [1, 2, 4, 6, 8, 10];
        replace_sorted(&mut i, 1, 7);
        assert_eq!(i, [1, 4, 6, 7, 8, 10]);

        // new < old
        let mut i = [1, 2, 4, 6, 8, 10];
        replace_sorted(&mut i, 4, 3);
        assert_eq!(i, [1, 2, 3, 4, 6, 10]);

        // new == old
        let mut i = [1, 2, 4, 6, 8, 10];
        replace_sorted(&mut i, 4, 9);
        assert_eq!(i, [1, 2, 4, 6, 9, 10]);
    }

    #[test]
    fn is_sorted_test() {
        assert!(is_sorted(&[1, 2, 3]));
        assert!(is_sorted(&[1, 1, 2, 3]));

        assert!(!is_sorted(&[1, 3, 2]));
        assert!(!is_sorted(&[1, 3, 1, 2]));
    }

    #[test]
    fn is_unique_test() {
        assert!(is_unique(&[1, 3, 2]));
        assert!(!is_unique(&[1, 3, 1, 2]));
    }
}

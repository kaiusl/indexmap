#![allow(clippy::bool_assert_comparison)]

use ::alloc::vec::Vec;
use ::core::fmt::Debug;
use ::core::ops::Bound;
use ::core::panic::AssertUnwindSafe;
use ::core::sync::atomic::AtomicU32;
use ::std::panic::catch_unwind;
use ::std::string::String;

use super::*;

type IndexMultimapVec<K, V> = IndexMultimap<K, V, RandomState>;

fn assert_panics<T>(f: impl FnOnce() -> T) {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev_hook);
    assert!(result.is_err());
}

fn assert_panics_w_msg<T>(f: impl FnOnce() -> T, expected_msg: &str) {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev_hook);
    match result {
        Ok(_) => panic!("expected panic"),
        Err(e) => {
            let mut got_msg = "";
            if let Some(s) = e.downcast_ref::<&str>() {
                got_msg = *s;
            } else if let Some(s) = e.downcast_ref::<String>() {
                got_msg = s;
            }

            assert!(
                got_msg.contains(expected_msg),
                "expected panic with message containing \"{expected_msg}\", got \"{got_msg}\""
            )
        }
    }
}

#[test]
fn it_works() {
    let mut map = IndexMultimapVec::new();
    assert_eq!(map.is_empty(), true);
    map.insert_append(1, 1);
    //println!("{map:#?}");

    map.insert_append(1, 2);
    assert_eq!(map.len_keys(), 1);
    //println!("{map:#?}");

    map.insert_append(2, 21);
    //println!("{map:#?}");
    assert_eq!(map.len_keys(), 2);

    map.insert_append(1, 2);
    assert_eq!(map.len_keys(), 2);
    //println!("{map:#?}");
    assert!(map.get(&1).first().is_some());
    assert_eq!(map.is_empty(), false);
}

/// Assert all lefts to right
#[track_caller]
fn assert_eq_many<T, const N: usize>(lefts: [T; N], right: T)
where
    T: PartialEq + ::core::fmt::Debug,
{
    for (i, left) in lefts.into_iter().enumerate() {
        assert_eq!(left, right, "left_index={}", i);
    }
}

#[test]
fn new() {
    let map = IndexMultimap::<String, String>::new();
    //println!("{:?}", map);
    assert_eq!(map.capacity_keys(), 0);
    assert_eq!(map.len_keys(), 0);
    assert_eq!(map.is_empty(), true);
}

#[test]
fn insert_append() {
    let insert = [0, 4, 2, 12, 8, 7, 11, 5];
    let not_present = [1, 3, 6, 9, 10];
    let mut map = IndexMultimapVec::with_capacity(insert.len(), insert.len());

    for (i, &it) in insert.iter().enumerate() {
        assert_eq_many([map.len_keys(), map.len_pairs()], i);
        let index = map.insert_append(it, it);
        assert_eq!(index, map.len_pairs() - 1);
        assert_eq_many([map.len_keys(), map.len_pairs()], i + 1);
        assert_eq!(&map[index], &it);
    }
    //println!("{:#?}", map);

    for &elt in &not_present {
        assert!(map.get(&elt).first().is_none());
    }

    let present = [0, 12, 2];
    let keys_len = map.len_keys();
    for (i, &it) in (insert.len()..).zip(present.iter()) {
        assert_eq!(map.len_pairs(), i);
        assert_eq!(map.len_keys(), keys_len);
        let index = map.insert_append(it, it);
        assert_eq!(index, map.len_pairs() - 1);
        assert_eq!(map.len_keys(), keys_len);
        assert_eq!(map.len_pairs(), i + 1);
        assert_eq!(&map[index], &it);
    }
    //println!("{:#?}", map);
}

#[test]
fn insert_at() {
    let mut insert = vec![
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert.iter().cloned());
    assert_map_eq(&map, &insert);

    assert!(map.insert_at(insert.len() + 1, 1, 1).is_none());

    let (i, entry) = map.insert_at(0, 1, 11).unwrap();
    assert_eq!(i, 0);
    check_subentries(&entry.as_subset(), &[(0, &1, &11)]);
    insert.insert(0, (1, 11));
    assert_map_eq(&map, &insert);

    let (i, entry) = map.insert_at(3, 4, 44).unwrap();
    assert_eq!(i, 1);
    check_subentries(
        &entry.as_subset(),
        &[(2, &4, &41), (3, &4, &44), (4, &4, &42), (7, &4, &43)],
    );
    insert.insert(3, (4, 44));
    assert_map_eq(&map, &insert);

    let index = insert.len();
    assert_eq!(index, 10);
    let (i, entry) = map.insert_at(index, 5, 55).unwrap();
    assert_eq!(i, 3);
    check_subentries(
        &entry.as_subset(),
        &[(6, &5, &51), (8, &5, &52), (9, &5, &53), (10, &5, &55)],
    );
    insert.insert(10, (5, 55));
    assert_map_eq(&map, &insert);
}

#[test]
fn insert_at_table_must_alloc() {
    let mut map = IndexMultimapVec::new();

    map.insert_at(0, 1, 11).unwrap();
    map.insert_at(0, 2, 11).unwrap();
    map.insert_at(1, 3, 11).unwrap();
    // next call will reallocate the table, and must rehash the keys,
    // there was an issue where the indices in table and pairs indices were offset by one,
    // causing the index to be wrong and rehashing to fail
    map.insert_at(1, 4, 11).unwrap();
}

#[test]
fn extend() {
    let mut map = IndexMultimapVec::new();
    map.extend(vec![(&1, &2), (&3, &4)]);
    map.extend(vec![(5, 6), (5, 7)]);
    assert_eq!(
        map.into_iter().collect::<Vec<_>>(),
        vec![(1, 2), (3, 4), (5, 6), (5, 7)]
    );
}

#[track_caller]
fn check_subentries_full<K, V>(subentries: &Subset<'_, K, V>, expected: &[(usize, &K, &V)])
where
    K: Eq + Debug,
    V: Eq + Debug,
{
    assert_eq!(subentries.len(), expected.len());
    assert_eq!(subentries.first().as_ref(), expected.first());
    assert_eq!(subentries.last().as_ref(), expected.last());
    itertools::assert_equal(subentries.iter(), expected.iter().copied());
    itertools::assert_equal(subentries.indices(), expected.iter().map(|(i, _, _)| i));
    itertools::assert_equal(subentries.keys(), expected.iter().map(|(_, k, _)| *k));
    itertools::assert_equal(subentries.values(), expected.iter().map(|(_, _, v)| *v));
    for i in 0..subentries.len() + 2 {
        assert_eq!(subentries.nth(i).as_ref(), expected.get(i));

        if i < subentries.len() {
            assert_eq!(subentries[i], *expected[i].2);
        } else {
            assert_panics_w_msg(|| &subentries[i], "index out of bounds");
        }

        if i <= subentries.len() {
            let (l, r) = subentries.split_at(i);
            let (l_expected, r_expected) = expected.split_at(i);
            check_subentries(&l, &l_expected);
            check_subentries(&r, &r_expected);
        } else {
            assert_panics_w_msg(|| subentries.split_at(i), "mid > len");
        }
    }

    let split_first = subentries.split_first();
    let split_first_expected = expected.split_first();
    assert_eq!(split_first.is_some(), split_first_expected.is_some());
    if let (Some((f, rest)), Some((f_expected, rest_expected))) =
        (split_first, split_first_expected)
    {
        assert_eq!(f, *f_expected);
        check_subentries(&rest, &rest_expected);
    }

    let split_last = subentries.split_last();
    let split_last_expected = expected.split_last();
    assert_eq!(split_last.is_some(), split_last_expected.is_some());
    if let (Some((last, rest)), Some((last_expected, rest_expected))) =
        (split_last, split_last_expected)
    {
        assert_eq!(last, *last_expected);
        check_subentries(&rest, &rest_expected);
    }

    let ranges = [
        (Bound::Excluded(0), Bound::Excluded(0)),
        (Bound::Included(0), Bound::Included(0)),
        (Bound::Excluded(1), Bound::Excluded(1)),
        (Bound::Included(1), Bound::Included(1)),
        (Bound::Included(0), Bound::Unbounded),
        (Bound::Included(0), Bound::Excluded(subentries.len())),
        (Bound::Included(0), Bound::Included(subentries.len())),
        (Bound::Included(0), Bound::Excluded(2)),
        (Bound::Included(0), Bound::Included(2)),
        (Bound::Unbounded, Bound::Unbounded),
        (Bound::Unbounded, Bound::Excluded(subentries.len())),
        (Bound::Unbounded, Bound::Included(subentries.len())),
        (Bound::Unbounded, Bound::Excluded(2)),
        (Bound::Unbounded, Bound::Included(2)),
        (Bound::Excluded(subentries.len()), Bound::Unbounded),
        (Bound::Included(subentries.len()), Bound::Unbounded),
        (Bound::Included(1), Bound::Included(3)),
        (Bound::Excluded(1), Bound::Included(3)),
        (Bound::Included(3), Bound::Included(1)),
        (Bound::Excluded(3), Bound::Included(1)),
        (Bound::Included(1), Bound::Excluded(3)),
        (Bound::Excluded(1), Bound::Excluded(3)),
        (Bound::Included(3), Bound::Excluded(1)),
    ];
    for r in ranges {
        let sub = subentries.get_range(r);
        let expected = expected.get(r);

        assert_eq!(sub.is_some(), expected.is_some());
        if let (Some(sub), Some(expected)) = (sub, expected) {
            check_subentries(&sub, expected);
        }
    }
}

#[track_caller]
fn check_subentries<K, V>(subentries: &Subset<'_, K, V>, expected: &[(usize, &K, &V)])
where
    K: Eq + Debug,
    V: Eq + Debug,
{
    assert_eq!(subentries.len(), expected.len());
    itertools::assert_equal(subentries.iter(), expected.iter().copied());
}

#[track_caller]
fn check_subentries_mut_full(
    subentries: &mut SubsetMut<'_, i32, i32>,
    expected: &[(usize, &i32, &mut i32)],
) {
    assert_eq!(subentries.len(), expected.len());
    assert_eq!(
        subentries.first().map(|(i, k, v)| (i, *k, *v)),
        expected.first().map(|(i, k, v)| (*i, **k, **v))
    );
    assert_eq!(subentries.first_mut().as_ref(), expected.first());
    assert_eq!(
        subentries.last().map(|(i, k, v)| (i, *k, *v)),
        expected.last().map(|(i, k, v)| (*i, **k, **v))
    );
    assert_eq!(subentries.last_mut().as_ref(), expected.last());

    itertools::assert_equal(
        subentries.iter().map(|(i, k, v)| (i, *k, *v)),
        expected.iter().map(|a| (a.0, *a.1, *a.2)),
    );
    itertools::assert_equal(
        subentries.iter_mut().map(|(i, k, v)| (i, *k, *v)),
        expected.iter().map(|a| (a.0, *a.1, *a.2)),
    );
    itertools::assert_equal(subentries.indices(), expected.iter().map(|(i, _, _)| i));
    itertools::assert_equal(subentries.keys(), expected.iter().map(|(_, k, _)| *k));
    itertools::assert_equal(
        subentries.values().copied(),
        expected.iter().map(|(_, _, v)| **v),
    );
    itertools::assert_equal(
        subentries.values_mut().map(|v| *v),
        expected.iter().map(|(_, _, v)| **v),
    );

    for i in 0..subentries.len() {
        assert_eq!(
            subentries.nth(i).map(|(i, k, v)| (i, *k, *v)),
            expected.get(i).map(|(i, k, v)| (*i, **k, **v))
        );
        assert_eq!(subentries.nth_mut(i).as_ref(), expected.get(i));

        if i < subentries.len() {
            assert_eq!(subentries[i], *expected[i].2);
        } else {
            assert_panics_w_msg(|| &subentries[i], "index out of bounds");
        }

        if i <= subentries.len() {
            let (mut l, mut r) = subentries.split_at_mut(i);
            let (l_expected, r_expected) = expected.split_at(i);
            check_subentries_mut(&mut l, l_expected);
            check_subentries_mut(&mut r, r_expected);
        } else {
            assert_panics_w_msg(|| subentries.split_at(i), "mid > len");
        }
    }

    let split_first = subentries.split_first_mut();
    let split_first_expected = expected.split_first();
    assert_eq!(split_first.is_some(), split_first_expected.is_some());
    if let (Some((f, mut rest)), Some((f_expected, rest_expected))) =
        (split_first, split_first_expected)
    {
        assert_eq!(f, *f_expected);
        check_subentries_mut(&mut rest, &rest_expected);
    }

    let split_last = subentries.split_last_mut();
    let split_last_expected = expected.split_last();
    assert_eq!(split_last.is_some(), split_last_expected.is_some());
    if let (Some((last, mut rest)), Some((last_expected, rest_expected))) =
        (split_last, split_last_expected)
    {
        assert_eq!(last, *last_expected);
        check_subentries_mut(&mut rest, &rest_expected);
    }

    let ranges = [
        (Bound::Excluded(0), Bound::Excluded(0)),
        (Bound::Included(0), Bound::Included(0)),
        (Bound::Excluded(1), Bound::Excluded(1)),
        (Bound::Included(1), Bound::Included(1)),
        (Bound::Included(0), Bound::Unbounded),
        (Bound::Included(0), Bound::Excluded(subentries.len())),
        (Bound::Included(0), Bound::Included(subentries.len())),
        (Bound::Included(0), Bound::Excluded(2)),
        (Bound::Included(0), Bound::Included(2)),
        (Bound::Unbounded, Bound::Unbounded),
        (Bound::Unbounded, Bound::Excluded(subentries.len())),
        (Bound::Unbounded, Bound::Included(subentries.len())),
        (Bound::Unbounded, Bound::Excluded(2)),
        (Bound::Unbounded, Bound::Included(2)),
        (Bound::Excluded(subentries.len()), Bound::Unbounded),
        (Bound::Included(subentries.len()), Bound::Unbounded),
        (Bound::Included(1), Bound::Included(3)),
        (Bound::Excluded(1), Bound::Included(3)),
        (Bound::Included(3), Bound::Included(1)),
        (Bound::Excluded(3), Bound::Included(1)),
        (Bound::Included(1), Bound::Excluded(3)),
        (Bound::Excluded(1), Bound::Excluded(3)),
        (Bound::Included(3), Bound::Excluded(1)),
    ];
    for r in ranges {
        let sub = subentries.get_range_mut(r);
        let expected = expected.get(r);

        assert_eq!(sub.is_some(), expected.is_some());
        if let (Some(mut sub), Some(expected)) = (sub, expected) {
            check_subentries_mut(&mut sub, expected);
        }
    }
}

#[track_caller]
fn check_subentries_mut(
    subentries: &mut SubsetMut<'_, i32, i32>,
    expected: &[(usize, &i32, &mut i32)],
) {
    assert_eq!(subentries.len(), expected.len());
    itertools::assert_equal(
        subentries.iter_mut().map(|(i, k, v)| (i, *k, *v)),
        expected.iter().map(|a| (a.0, *a.1, *a.2)),
    );
}

#[test]
fn get() {
    let insert = [0, 4, 2, 12, 8, 7, 11, 5];
    let not_present = [1, 3, 6, 9, 10];
    let mut map = IndexMultimapVec::with_capacity(insert.len(), insert.len());

    let get = map.get(&5);
    check_subentries_full(&get, &[]);

    let mut get_mut = map.get_mut(&8);
    check_subentries_mut_full(&mut get_mut, &[]);

    assert!(map.get_index(5).is_none());
    assert!(map.get_index_mut(6).is_none());

    for &it in insert.iter() {
        map.insert_append(it, it * 1000 + 1);
    }

    for key in not_present {
        let get = map.get(&key);
        check_subentries_full(&get, &[]);

        let mut get_mut = map.get_mut(&key);
        check_subentries_mut_full(&mut get_mut, &[]);
    }

    for (index, &key) in insert.iter().enumerate() {
        assert!(map.contains_key(&key));

        let mut val = key * 1000 + 1;
        // Test getters
        assert_eq!(&map[index], &val);
        assert_eq!(&mut map[index], &mut val);

        assert_eq!(map.get_index(index), Some((&key, &val)));
        assert_eq!(map.get_index_mut(index), Some((&key, &mut val)));

        let get = map.get(&key);
        check_subentries_full(&get, &[(index, &key, &val)]);

        let mut get_mut = map.get_mut(&key);
        check_subentries_mut_full(&mut get_mut, &[(index, &key, &mut val)]);

        assert!(map.get_indices_of(&key).contains(&index));
    }

    for &elt in &not_present {
        assert!(map.get(&elt).first().is_none());
    }

    let present = [0, 12, 2];
    for &it in present.iter() {
        map.insert_append(it, it * 1000 + 2);
    }

    for (index, &key) in (insert.len()..).zip(present.iter()) {
        assert!(map.contains_key(&key));
        let mut val = key * 1000 + 2;

        assert_eq!(&map[index], &val);
        assert_eq!(&mut map[index], &mut val);

        assert_eq!(map.get_index(index), Some((&key, &val)));
        assert_eq!(map.get_index_mut(index), Some((&key, &mut val)));

        let get = map.get(&key);
        match key {
            0 => {
                check_subentries_full(&get, &[(0, &0, &1), (index, &key, &val)]);
            }
            12 => {
                check_subentries_full(&get, &[(3, &12, &12001), (index, &key, &val)]);
            }
            2 => {
                check_subentries_full(&get, &[(2, &2, &2001), (index, &key, &val)]);
            }
            _ => {}
        }

        let mut get_mut = map.get_mut(&key);
        match key {
            0 => {
                check_subentries_mut_full(
                    &mut get_mut,
                    &[(0, &0, &mut 1), (index, &key, &mut val)],
                );
            }
            12 => {
                check_subentries_mut_full(
                    &mut get_mut,
                    &[(3, &12, &mut 12001), (index, &key, &mut val)],
                );
            }
            2 => {
                check_subentries_mut_full(
                    &mut get_mut,
                    &[(2, &2, &mut 2001), (index, &key, &mut val)],
                );
            }
            _ => {}
        }

        assert!(map.get_indices_of(&key).contains(&index));
    }
}

// Checks that the map yield given items in the given order by index and by key
#[track_caller]
pub(super) fn assert_map_eq<K, V>(map: &IndexMultimapVec<K, V>, expected: &[(K, V)])
where
    K: fmt::Debug + Eq + std::hash::Hash,
    V: fmt::Debug + Eq,
{
    assert_eq!(map.len_pairs(), expected.len());
    itertools::assert_equal(map.iter(), expected.iter().map(|(k, v)| (k, v)));
    for (index, (key, val)) in expected.iter().enumerate() {
        assert_eq!(&map[index], val);
        assert_eq!(map.get_index(index), Some((key, val)));

        let expected_items = expected
            .iter()
            .enumerate()
            .filter(|(_, (k, _))| k == key)
            .map(|(i, (k, v))| (i, k, v));

        itertools::assert_equal(map.get(key).iter(), expected_items);
    }
}

#[test]
fn subset_split_at() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);

    let l_expected = &[];
    let r_expected = &[(1, &4, &41), (2, &4, &42), (5, &4, &43)];
    let set = map.get_mut(&4);
    let (l, r) = set.split_at(0);
    check_subentries(&l, l_expected);
    check_subentries(&r, r_expected);
    let set = map.get(&4);
    let (l, r) = set.split_at(0);
    check_subentries(&l, l_expected);
    check_subentries(&r, r_expected);

    let l_expected = &[(1, &4, &41), (2, &4, &42)];
    let r_expected = &[(5, &4, &43)];
    let set = map.get_mut(&4);
    let (l, r) = set.split_at(2);
    check_subentries(&l, l_expected);
    check_subentries(&r, r_expected);
    let set = map.get(&4);
    let (l, r) = set.split_at(2);
    check_subentries(&l, l_expected);
    check_subentries(&r, r_expected);

    let l_expected = &[(1, &4, &41), (2, &4, &42), (5, &4, &43)];
    let r_expected = &[];
    let set = map.get_mut(&4);
    let (l, r) = set.split_at(3);
    check_subentries(&l, l_expected);
    assert!(r.is_empty());
    check_subentries(&r, r_expected);
    let set = map.get(&4);
    let (l, r) = set.split_at(3);
    check_subentries(&l, l_expected);
    assert!(r.is_empty());
    check_subentries(&r, r_expected);

    let set = map.get_mut(&4);
    assert_panics_w_msg(|| set.split_at(4), "mid > len");
    let set = map.get(&4);
    assert_panics_w_msg(|| set.split_at(4), "mid > len");
}

#[test]
fn subset_mut_split_at_mut() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);

    let mut set = map.get_mut(&4);

    let (mut l, mut r) = set.split_at_mut(0);
    let l_expected = &[];
    let r_expected = &[(1, &4, &mut 41), (2, &4, &mut 42), (5, &4, &mut 43)];

    check_subentries_mut(&mut l, l_expected);
    check_subentries_mut(&mut r, r_expected);
    let (mut l, mut r) = set.split_into(0);
    check_subentries_mut(&mut l, l_expected);
    check_subentries_mut(&mut r, r_expected);

    let mut set = map.get_mut(&4);
    let (mut l, mut r) = set.split_at_mut(2);
    let l_expected = &[(1, &4, &mut 41), (2, &4, &mut 42)];
    let r_expected = &[(5, &4, &mut 43)];
    check_subentries_mut(&mut l, l_expected);
    check_subentries_mut(&mut r, r_expected);
    let (mut l, mut r) = set.split_into(2);
    check_subentries_mut(&mut l, l_expected);
    check_subentries_mut(&mut r, r_expected);

    let mut set = map.get_mut(&4);
    let (mut l, mut r) = set.split_at_mut(3);
    let l_expected = &[(1, &4, &mut 41), (2, &4, &mut 42), (5, &4, &mut 43)];
    let r_expected = &[];
    check_subentries_mut(&mut l, l_expected);
    assert!(r.is_empty());
    check_subentries_mut(&mut r, r_expected);
    let (mut l, mut r) = set.split_into(3);
    check_subentries_mut(&mut l, l_expected);
    assert!(r.is_empty());
    check_subentries_mut(&mut r, r_expected);

    let mut set = map.get_mut(&4);
    assert_panics_w_msg(|| set.split_at_mut(4), "mid > len");
    assert_panics_w_msg(|| set.split_into(4), "mid > len");
}

#[test]
fn subset_split_first() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);

    let l_expected = (1, &4, &41);
    let r_expected = &[(2, &4, &42), (5, &4, &43)];

    let set = map.get(&4);
    let (l, r) = set.split_first().unwrap();
    assert_eq!(l, l_expected);
    check_subentries(&r, r_expected);

    let mut set = map.get_mut(&4);

    let (l, r) = set.split_first().unwrap();
    assert_eq!(l, l_expected);
    check_subentries(&r, r_expected);

    let l_expected = (1, &4, &mut 41);
    let r_expected = &[(2, &4, &mut 42), (5, &4, &mut 43)];
    let (l, mut r) = set.split_first_mut().unwrap();
    assert_eq!(l, l_expected);
    check_subentries_mut(&mut r, r_expected);

    let mut set = map.get_mut(&7);
    assert!(set.split_first().is_none());
    assert!(set.split_first_mut().is_none());
}

#[test]
fn subset_split_last() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);

    let l_expected = (5, &4, &43);
    let r_expected = &[(1, &4, &41), (2, &4, &42)];

    let set = map.get(&4);
    let (l, r) = set.split_last().unwrap();
    assert_eq!(l, l_expected);
    check_subentries(&r, r_expected);

    let mut set = map.get_mut(&4);

    let (l, r) = set.split_last().unwrap();
    assert_eq!(l, l_expected);
    check_subentries(&r, r_expected);

    let l_expected = (5, &4, &mut 43);
    let r_expected = &[(1, &4, &mut 41), (2, &4, &mut 42)];
    let (l, mut r) = set.split_last_mut().unwrap();
    assert_eq!(l, l_expected);
    check_subentries_mut(&mut r, r_expected);

    let mut set = map.get_mut(&7);
    assert!(set.split_last().is_none());
    assert!(set.split_last_mut().is_none());
}

#[test]
fn subset_split_mut_aftermath() {
    /*
    Tests whether any operation on a subset could create aliasing references
    after a split_at_mut or split_into.
    Those methods internally clone a RawSliceMut which could lead to aliasing references
    if not properly handled.

    Previously there were some issues with mixing pointers and references.
    This test under MIRI would have discovered those.
    */

    let insert = [
        0, 4, 2, 4, 12, 8, 7, 4, 4, 11, 8, 4, 4, 8, 5, 3, 17, 4, 19, 22, 23, 0,
    ];
    let mut map = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt * 1000 + 1);
    }

    let mut subset_mut = map.get_mut(&4);

    let subset_mut_ops = vec![
        SubsetMutOp::Len,
        SubsetMutOp::IsEmpty,
        SubsetMutOp::Indices,
        SubsetMutOp::GetRange(0..3),
        SubsetMutOp::GetRangeMut(0..3),
        SubsetMutOp::Iter,
        SubsetMutOp::IterMut,
        SubsetMutOp::Keys,
        SubsetMutOp::Values,
        SubsetMutOp::ValuesMut,
        SubsetMutOp::AsSubset,
        SubsetMutOp::SplitAt(2),
        SubsetMutOp::SplitAtMut(2),
        SubsetMutOp::Nth(2),
        SubsetMutOp::NthMut(2),
        SubsetMutOp::First,
        SubsetMutOp::FirstMut,
        SubsetMutOp::Last,
        SubsetMutOp::LastMut,
        SubsetMutOp::SplitFirst,
        SubsetMutOp::SplitFirstMut,
        SubsetMutOp::SplitLast,
        SubsetMutOp::SplitLastMut,
        SubsetMutOp::TakeFirst,
        SubsetMutOp::TakeLast,
        SubsetMutOp::GetManyMut([1, 2, 0]),
    ];

    for op1 in subset_mut_ops.clone() {
        for op2 in subset_mut_ops.clone() {
            let (mut l, mut r) = subset_mut.split_at_mut(3);

            let mut l = do_subsetmut_op(&mut l, &op1);
            let mut r = do_subsetmut_op(&mut r, &op2);

            do_subset_mut_followup(&mut l);
            do_subset_mut_followup(&mut r);

            // Doing the follow up ops in different order can discover issues
            let (mut l, mut r) = subset_mut.split_at_mut(3);

            let mut l = do_subsetmut_op(&mut l, &op1);
            let mut r = do_subsetmut_op(&mut r, &op2);

            do_subset_mut_followup(&mut r);
            do_subset_mut_followup(&mut l);
        }
    }

    for op1 in subset_mut_ops.clone() {
        for op2 in subset_mut_ops.clone() {
            let subset_mut = map.get_mut(&4);
            let (mut l, mut r) = subset_mut.split_into(3);

            let mut l = do_subsetmut_op(&mut l, &op1);
            let mut r = do_subsetmut_op(&mut r, &op2);

            do_subset_mut_followup(&mut l);
            do_subset_mut_followup(&mut r);

            // Doing the follow up ops in different order can discover issues
            let subset_mut = map.get_mut(&4);
            let (mut l, mut r) = subset_mut.split_into(3);

            let mut l = do_subsetmut_op(&mut l, &op1);
            let mut r = do_subsetmut_op(&mut r, &op2);

            do_subset_mut_followup(&mut r);
            do_subset_mut_followup(&mut l);
        }
    }
}

#[test]
fn subset_split_mut_aftermath_into_ops() {
    let insert = [
        0, 4, 2, 4, 12, 8, 7, 4, 4, 11, 8, 4, 4, 8, 5, 3, 17, 4, 19, 22, 23, 0,
    ];
    let mut map = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt * 1000 + 1);
    }

    let mut subset_mut = map.get_mut(&4);

    let subset_mut_ops = vec![
        SubsetMutOpInto::IntoRange(0..3),
        SubsetMutOpInto::IntoIter,
        SubsetMutOpInto::IntoKeys,
        SubsetMutOpInto::IntoValues,
        SubsetMutOpInto::IntoSubset,
        SubsetMutOpInto::SplitInto(2),
        SubsetMutOpInto::IntoNthMut(2),
        SubsetMutOpInto::IntoFirst,
        SubsetMutOpInto::IntoLast,
        SubsetMutOpInto::IntoManyMut([1, 2, 0]),
    ];

    for op1 in subset_mut_ops.clone() {
        for op2 in subset_mut_ops.clone() {
            let (l, r) = subset_mut.split_at_mut(3);

            let mut l = do_subsetmut_op_into(l, &op1);
            let mut r = do_subsetmut_op_into(r, &op2);

            do_subset_mut_followup(&mut l);
            do_subset_mut_followup(&mut r);

            // Doing the follow up ops in different order can discover issues
            let (l, r) = subset_mut.split_at_mut(3);

            let mut l = do_subsetmut_op_into(l, &op1);
            let mut r = do_subsetmut_op_into(r, &op2);

            do_subset_mut_followup(&mut r);
            do_subset_mut_followup(&mut l);
        }
    }

    for op1 in subset_mut_ops.clone() {
        for op2 in subset_mut_ops.clone() {
            let subset_mut = map.get_mut(&4);
            let (l, r) = subset_mut.split_into(3);

            let mut l = do_subsetmut_op_into(l, &op1);
            let mut r = do_subsetmut_op_into(r, &op2);

            do_subset_mut_followup(&mut l);
            do_subset_mut_followup(&mut r);

            // Doing the follow up ops in different order can discover issues
            let subset_mut = map.get_mut(&4);
            let (l, r) = subset_mut.split_into(3);

            let mut l = do_subsetmut_op_into(l, &op1);
            let mut r = do_subsetmut_op_into(r, &op2);

            do_subset_mut_followup(&mut r);
            do_subset_mut_followup(&mut l);
        }
    }
}

#[derive(Clone, Debug)]
enum SubsetMutOp {
    Len,
    IsEmpty,
    Indices,
    Nth(usize),
    NthMut(usize),
    First,
    FirstMut,
    Last,
    LastMut,
    GetRange(ops::Range<usize>),
    GetRangeMut(ops::Range<usize>),
    SplitAt(usize),
    SplitAtMut(usize),
    AsSubset,
    Iter,
    IterMut,
    Keys,
    Values,
    ValuesMut,
    SplitFirst,
    SplitFirstMut,
    SplitLast,
    SplitLastMut,
    TakeFirst,
    TakeLast,
    GetManyMut([usize; 3]),
}

#[derive(Clone, Debug)]
enum SubsetMutOpInto {
    IntoNthMut(usize),
    IntoFirst,
    IntoLast,
    IntoRange(ops::Range<usize>),
    SplitInto(usize),
    IntoSubset,
    IntoIter,
    IntoKeys,
    IntoValues,
    IntoManyMut([usize; 3]),
}

fn do_subsetmut_op<'b, 'a, K, V>(
    subset: &'b mut SubsetMut<'a, K, V>,
    op: &SubsetMutOp,
) -> SubsetMutOpResult<'b, 'a, K, V> {
    use SubsetMutOp::*;
    match op {
        GetRange(range) => return SubsetMutOpResult::Subset(subset.get_range(range.clone())),
        GetRangeMut(range) => {
            return SubsetMutOpResult::SubsetMut(subset.get_range_mut(range.clone()))
        }
        Iter => return SubsetMutOpResult::Iter(subset.iter()),
        IterMut => return SubsetMutOpResult::IterMut(subset.iter_mut()),
        Keys => return SubsetMutOpResult::Keys(subset.keys()),
        Values => return SubsetMutOpResult::Values(subset.values()),
        ValuesMut => return SubsetMutOpResult::ValuesMut(subset.values_mut()),
        AsSubset => return SubsetMutOpResult::AsSubset(subset.as_subset()),
        SplitAt(range) => return SubsetMutOpResult::SplitAt(subset.split_at(*range)),
        SplitAtMut(range) => return SubsetMutOpResult::SplitAtMut(subset.split_at_mut(*range)),
        SplitFirst => {
            let res = subset.split_first().map(|a| a.1);
            return SubsetMutOpResult::Subset(res);
        }
        SplitFirstMut => {
            let res = subset.split_first_mut().map(|a| a.1);
            return SubsetMutOpResult::SubsetMut(res);
        }
        SplitLast => {
            let res = subset.split_last().map(|a| a.1);
            return SubsetMutOpResult::Subset(res);
        }
        SplitLastMut => {
            let res = subset.split_last_mut().map(|a| a.1);
            return SubsetMutOpResult::SubsetMut(res);
        }
        TakeFirst => {
            subset.take_first();
        }
        TakeLast => {
            subset.take_last();
        }
        Nth(n) => {
            subset.nth(*n);
        }
        NthMut(n) => {
            subset.nth_mut(*n);
        }
        First => {
            subset.first();
        }
        FirstMut => {
            subset.first_mut();
        }
        Last => {
            subset.last();
        }
        LastMut => {
            subset.last_mut();
        }
        Len => {
            subset.len();
        }
        IsEmpty => {
            subset.is_empty();
        }
        Indices => {
            subset.indices();
        }
        GetManyMut(i) => {
            subset.get_many_mut(*i);
        }
    }

    SubsetMutOpResult::SubsetMutRef(subset)
}

fn do_subsetmut_op_into<'b, 'a, K, V>(
    subset: SubsetMut<'a, K, V>,
    op: &SubsetMutOpInto,
) -> SubsetMutOpResult<'b, 'a, K, V> {
    use SubsetMutOpInto::*;
    match op {
        IntoFirst => {
            subset.into_first_mut();
        }
        IntoLast => {
            subset.into_last_mut();
        }
        IntoNthMut(n) => {
            subset.into_nth_mut(*n);
        }
        IntoRange(range) => return SubsetMutOpResult::SubsetMut(subset.into_range(range.clone())),
        IntoSubset => return SubsetMutOpResult::Subset(Some(subset.into_subset())),
        IntoIter => return SubsetMutOpResult::IterMut(subset.into_iter()),
        IntoKeys => return SubsetMutOpResult::Keys(subset.into_keys()),
        IntoValues => return SubsetMutOpResult::ValuesMut(subset.into_values()),
        SplitInto(at) => return SubsetMutOpResult::SplitAtMut(subset.split_into(*at)),
        IntoManyMut(i) => {
            subset.into_many_mut(*i);
        }
    }

    SubsetMutOpResult::None
}

enum SubsetMutOpResult<'a, 'b, K, V> {
    Subset(Option<Subset<'a, K, V>>),
    SubsetMut(Option<SubsetMut<'a, K, V>>),
    SubsetMutRef(&'a mut SubsetMut<'b, K, V>),
    Iter(SubsetIter<'a, K, V>),
    IterMut(SubsetIterMut<'a, K, V>),
    Keys(SubsetKeys<'a, K, V>),
    Values(SubsetValues<'a, K, V>),
    ValuesMut(SubsetValuesMut<'a, K, V>),
    AsSubset(Subset<'a, K, V>),
    SplitAt((Subset<'a, K, V>, Subset<'a, K, V>)),
    SplitAtMut((SubsetMut<'a, K, V>, SubsetMut<'a, K, V>)),
    None,
}

fn do_subset_mut_followup<K, V>(result: &mut SubsetMutOpResult<'_, '_, K, V>) {
    use SubsetMutOpResult as R;
    match result {
        R::Subset(Option::Some(subset)) => {
            let _ = subset.iter().collect::<Vec<_>>();
        }
        R::SubsetMut(Option::Some(subset_mut)) => {
            let _ = subset_mut.iter_mut().collect::<Vec<_>>();
        }
        R::SubsetMutRef(subset_mut_ref) => {
            let _ = subset_mut_ref.iter_mut().collect::<Vec<_>>();
        }
        R::Iter(subset_iter) => {
            let _ = subset_iter.collect::<Vec<_>>();
        }
        R::IterMut(subset_iter_mut) => {
            let _ = subset_iter_mut.collect::<Vec<_>>();
        }
        R::Keys(subset_keys) => {
            let _ = subset_keys.collect::<Vec<_>>();
        }
        R::Values(subset_values) => {
            let _ = subset_values.collect::<Vec<_>>();
        }
        R::ValuesMut(subset_values_mut) => {
            let _ = subset_values_mut.collect::<Vec<_>>();
        }
        R::AsSubset(subset) => {
            let _ = subset.iter().collect::<Vec<_>>();
        }
        R::SplitAt((l, r)) => {
            let _ = l.iter().collect::<Vec<_>>();
            let _ = r.iter().collect::<Vec<_>>();
        }
        R::SplitAtMut((l, r)) => {
            let _ = l.iter_mut().collect::<Vec<_>>();
            let _ = r.iter_mut().collect::<Vec<_>>();
        }
        _ => {}
    }
}

#[test]
fn subset_mut_take_first() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);

    let mut set = map.get_mut(&4);

    let l = set.take_first().unwrap();
    assert_eq!(l, (1, &4, &mut 41));
    check_subentries_mut(&mut set, &[(2, &4, &mut 42), (5, &4, &mut 43)]);

    let mut set = map.get_mut(&7);
    assert!(set.take_first().is_none());
}

#[test]
fn subset_mut_take_last() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);

    let mut set = map.get_mut(&4);

    let l = set.take_last().unwrap();
    assert_eq!(l, (5, &4, &mut 43));
    check_subentries_mut(&mut set, &[(1, &4, &mut 41), (2, &4, &mut 42)]);

    let mut set = map.get_mut(&7);
    assert!(set.take_last().is_none());
}

#[test]
fn subset_get_many_mut() {
    let insert = [
        0, 4, 2, 4, 12, 8, 7, 4, 4, 11, 8, 4, 4, 8, 5, 3, 17, 4, 19, 22, 23, 0,
    ];
    let mut map = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt * 1000 + 1);
    }

    let mut subset_mut = map.get_mut(&4);
    let many = subset_mut.get_many_mut([1, 4, 3]).unwrap();
    let expected = [(3, &4, &mut 4001), (11, &4, &mut 4001), (8, &4, &mut 4001)];
    assert_eq!(many, expected);
    let many = subset_mut.into_many_mut([1, 4, 3]).unwrap();
    assert_eq!(many, expected);

    let mut subset_mut = map.get_mut(&4);
    let duplicate = subset_mut.get_many_mut([1, 2, 3, 2]);
    assert!(duplicate.is_none());
    let duplicate = subset_mut.into_many_mut([1, 2, 3, 2]);
    assert!(duplicate.is_none());

    let mut subset_mut = map.get_mut(&4);
    let out_of_bound = subset_mut.get_many_mut([1, 100]);
    assert!(out_of_bound.is_none());
    let out_of_bound = subset_mut.into_many_mut([1, 100]);
    assert!(out_of_bound.is_none());
}

#[test]
fn subset_into_many_mut() {
    let insert = [
        0, 4, 2, 4, 12, 8, 7, 4, 4, 11, 8, 4, 4, 8, 5, 3, 17, 4, 19, 22, 23, 0,
    ];
    let mut map = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt * 1000 + 1);
    }

    let mut subset_mut = map.get_mut(&4);
    let many = subset_mut.get_many_mut([1, 4, 3]).unwrap();
    assert_eq!(
        many,
        [(3, &4, &mut 4001), (11, &4, &mut 4001), (8, &4, &mut 4001)]
    );

    let duplicate = subset_mut.get_many_mut([1, 2, 3, 2]);
    assert!(duplicate.is_none());

    let out_of_bound = subset_mut.get_many_mut([1, 100]);
    assert!(out_of_bound.is_none());
}

#[test]
fn insert_order() {
    let insert = [0, 4, 2, 4, 12, 8, 7, 11, 8, 8, 5, 3, 17, 4, 19, 22, 23, 0];
    let mut map = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt * 1000 + 1);
    }

    assert_eq_many([map.keys().count(), map.len_pairs()], insert.len());
    for (a, b) in insert.iter().zip(map.keys()) {
        assert_eq!(a, b);
    }
    for (i, k) in (0..insert.len()).zip(map.keys()) {
        assert_eq!(map.get_index(i).unwrap().0, k);
    }
}

#[test]
fn grow() {
    let insert = [0, 4, 2, 12, 8, 7, 11];
    let not_present = [1, 3, 6, 9, 10];
    let mut map = IndexMultimapVec::with_capacity(insert.len(), insert.len());

    for &elt in insert.iter() {
        map.insert_append(elt, elt);
    }
    for &elt in &insert {
        map.insert_append(elt, elt);
    }

    //println!("{:?}", map);
    for &elt in &insert {
        map.insert_append(elt * 10, elt);
    }
    for &elt in &insert {
        map.insert_append(elt * 100, elt);
    }
    for (i, &elt) in insert.iter().cycle().enumerate().take(100) {
        map.insert_append(elt * 100 + i as i32, elt);
    }
    //println!("{:?}", map);
    for &elt in &not_present {
        assert!(map.get(&elt).first().is_none());
    }
}

#[test]
fn reserve() {
    let mut map = IndexMultimapVec::<usize, usize>::new();
    assert_eq!(map.capacity_keys(), 0);
    assert_eq!(map.capacity_pairs(), 0);

    map.reserve(100, 150);
    assert!(map.capacity_keys() >= 100);
    assert!(map.capacity_pairs() >= 150);

    let capacity_keys = map.capacity_keys();
    let capacity_entries = map.capacity_pairs();
    for i in 0..capacity_entries {
        let key = if i >= capacity_keys {
            (i - capacity_keys).clamp(0, capacity_keys - 1)
        } else {
            i
        };
        assert_eq!(map.len_pairs(), i);
        map.insert_append(key, key * key);
        assert_eq!(map.len_pairs(), i + 1);
        assert_eq!(map.capacity_keys(), capacity_keys);
        assert_eq!(map.capacity_pairs(), capacity_entries);
        assert_eq!(map.get(&key).last(), Some((i, &key, &(key * key))));
    }
    assert_eq!(map.len_keys(), capacity_keys);
    assert_eq!(map.len_pairs(), capacity_entries);

    map.insert_append(capacity_entries, usize::MAX);
    assert_eq!(map.len_keys(), capacity_keys + 1);
    assert_eq!(map.len_pairs(), capacity_entries + 1);
    assert!(map.capacity_keys() > capacity_keys);
    assert!(map.capacity_pairs() > capacity_entries);
    assert_eq!(
        map.get(&capacity_entries).first().map(|(_, k, v)| (k, v)),
        Some((&capacity_entries, &usize::MAX))
    );
}

#[test]
fn try_reserve() {
    let mut map = IndexMultimapVec::<usize, usize>::new();
    assert_eq!(map.capacity_keys(), 0);
    assert_eq!(map.capacity_pairs(), 0);
    assert_eq!(map.try_reserve(100, 150), Ok(()));
    assert!(map.capacity_keys() >= 100);
    assert!(map.capacity_pairs() >= 150);
    assert!(map.try_reserve(usize::MAX, 0).is_err());
    assert!(map.try_reserve(0, usize::MAX).is_err());
}

#[test]
fn shrink_to_fit() {
    let mut map = IndexMultimapVec::<usize, usize>::new();
    assert_eq!(map.capacity_keys(), 0);
    assert_eq!(map.capacity_pairs(), 0);

    for i in 0..100 {
        assert_eq!(map.len_pairs(), i);
        map.insert_append(i, i * i);
        assert_eq!(map.len_pairs(), i + 1);
        assert!(map.capacity_pairs() > i);
        assert!(map.capacity_keys() > i);
        assert_eq!(map.get(&i).first(), Some((i, &i, &(i * i))));
        map.shrink_to_fit();
        assert_eq!(map.len_keys(), i + 1);
        assert_eq!(map.len_pairs(), i + 1);
        assert!(map.capacity_keys() > i);
        assert_eq!(map.capacity_pairs(), i + 1);
        assert_eq!(map.get(&i).first(), Some((i, &i, &(i * i))));
    }
}

#[test]
fn swap_remove() {
    // Test remove twice: a) by using the returned iterator and b) by dropping it.
    // Both should remove all the needed pairs.
    let insert = [0, 4, 2, 12, 8, 7, 11, 5, 3, 17, 19, 22, 23];
    let mut map = IndexMultimapVec::new();
    let mut map2 = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt);
        map2.insert_append(elt, elt);
    }

    assert_eq!(map.keys().count(), map.len_keys());
    assert_eq!(map.keys().count(), insert.len());
    for (a, b) in insert.iter().zip(map.keys()) {
        assert_eq!(a, b);
    }

    let remove_fail = [99, 77];
    let remove = [4, 12, 8, 7];

    for &key in &remove_fail {
        assert!(map.swap_remove(&key).is_none());
    }

    for &key in &remove {
        let index = map.get(&key).first().unwrap().0;
        assert_eq!(
            map.swap_remove(&key).unwrap().collect::<Vec<_>>(),
            vec![(index, key, key)]
        );
        assert!(map2.swap_remove(&key).is_some());
    }
    let remaining = [
        (0, 0),
        (23, 23),
        (2, 2),
        (22, 22),
        (19, 19),
        (17, 17),
        (11, 11),
        (5, 5),
        (3, 3),
    ];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());

    for key in &insert {
        assert_eq!(map.get(key).first().is_some(), !remove.contains(key));
    }
    assert_eq!(map.len_keys(), insert.len() - remove.len());
    assert_eq!(map.keys().count(), insert.len() - remove.len());
}

#[test]
fn swap_remove_multientry() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);
    let mut map2 = IndexMultimapVec::from_iter(insert);

    assert_eq!(map.keys().count(), 8);
    assert_eq!(map.values().count(), insert.len());

    let remove_fail = [99, 77];
    for &key in &remove_fail {
        assert!(map.swap_remove(&key).is_none());
    }

    let remove = [4, 5];
    let removed_pairs = map.swap_remove(&remove[0]).unwrap().collect::<Vec<_>>();
    map2.swap_remove(&remove[0]);
    assert_eq!(removed_pairs, [(1, 4, 41), (2, 4, 42), (5, 4, 43)]);
    let remaining = [(0, 0), (5, 53), (5, 52), (3, 3), (5, 51)];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());

    //println!("REMOVED 4: {:#?}", map);

    let removed = map.swap_remove(&remove[1]).unwrap().collect::<Vec<_>>();
    map2.swap_remove(&remove[1]);
    assert_eq!(removed, [(1, 5, 53), (2, 5, 52), (4, 5, 51)]);
    let remaining = [(0, 0), (3, 3)];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());

    for (key, _) in &insert {
        assert_eq!(map.get(key).first().is_some(), !remove.contains(key));
    }
    assert_eq!(map.len_keys(), 2);
    assert_eq!(map.len_pairs(), 2);
}

#[test]
fn swap_remove_multientry2() {
    let insert = [
        (1, 11),
        (0, 1),
        (1, 12),
        (1, 13),
        (0, 2),
        (0, 3),
        (0, 4),
        (2, 21),
        (3, 31),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);
    let mut map2 = IndexMultimapVec::from_iter(insert);
    let removed = map.swap_remove(&0).unwrap().collect::<Vec<_>>();
    map2.swap_remove(&0);
    assert_eq!(removed, [(1, 0, 1), (4, 0, 2), (5, 0, 3), (6, 0, 4)]);
    let remaining = [(1, 11), (3, 31), (1, 12), (1, 13), (2, 21)];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());
}

#[test]
fn shift_remove() {
    let insert = [0, 4, 2, 3];
    let mut map = IndexMultimapVec::new();
    let mut map2 = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt);
        map2.insert_append(elt, elt);
    }

    assert_eq!(map.keys().count(), map.len_keys());
    assert_eq!(map.keys().count(), insert.len());
    for (a, b) in insert.iter().zip(map.keys()) {
        assert_eq!(a, b);
    }

    let remove_fail = [99, 77];
    let remove = [4];

    for &key in &remove_fail {
        assert!(map.shift_remove(&key).is_none());
    }

    for &key in &remove {
        let index = map.get(&key).first().unwrap().0;
        assert_eq!(
            map.shift_remove(&key).unwrap().collect::<Vec<_>>(),
            vec![(index, key, key)]
        );
        assert!(map2.shift_remove(&key).is_some());
    }
    let remaining = [(0, 0), (2, 2), (3, 3)];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());

    for key in &insert {
        assert_eq!(map.get(key).first().is_some(), !remove.contains(key));
    }
    assert_eq!(map.len_keys(), insert.len() - remove.len());
    assert_eq!(map.keys().count(), insert.len() - remove.len());
}

#[test]
fn shift_remove_multientry() {
    let insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (3, 3),
        (5, 51),
        (4, 43),
        (5, 52),
        (5, 53),
        (6, 61),
    ];
    let mut map = IndexMultimapVec::from_iter(insert);
    let mut map2 = IndexMultimapVec::from_iter(insert);

    assert_eq!(map.keys().count(), 9);
    assert_eq!(map.values().count(), insert.len());

    let remove_fail = [99, 77];
    for &key in &remove_fail {
        assert!(map.shift_remove(&key).is_none());
    }

    let remove = [4, 5];
    let removed = map.shift_remove(&4).unwrap().collect::<Vec<_>>();
    assert!(map2.shift_remove(&4).is_some());
    assert_eq!(removed, [(1, 4, 41), (2, 4, 42), (5, 4, 43)]);
    let remaining = [(0, 0), (3, 3), (5, 51), (5, 52), (5, 53), (6, 61)];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());

    let items = map.shift_remove(&5).unwrap().collect::<Vec<_>>();
    assert!(map2.shift_remove(&5).is_some());
    assert_eq!(items, [(2, 5, 51), (3, 5, 52), (4, 5, 53)]);
    let remaining = [(0, 0), (3, 3), (6, 61)];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());

    for (key, _) in &insert {
        assert_eq!(map.get(key).first().is_some(), !remove.contains(key));
    }
    assert_eq!(map.len_keys(), 3);
    assert_eq!(map.len_pairs(), 3);
}

#[test]
fn swap_remove_to_empty() {
    let mut map = indexmultimap! { 0 => 0, 4 => 4, 0=>1, 5 => 5 };
    assert!(map.swap_remove(&5).is_some());
    assert!(map.swap_remove(&4).is_some());
    assert!(map.swap_remove(&0).is_some());
    assert!(map.is_empty());
}

#[test]
fn shift_remove_to_empty() {
    let mut map = indexmultimap! { 0 => 0, 4 => 4, 0=>1, 5 => 5 };
    assert!(map.shift_remove(&5).is_some());
    assert!(map.shift_remove(&4).is_some());
    assert!(map.shift_remove(&0).is_some());
    assert!(map.is_empty());
}

#[test]
fn shift_remove_drop_panics() {
    static DROPS: AtomicU32 = AtomicU32::new(0);

    #[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
    struct DropMayPanic(bool, String);

    impl Drop for DropMayPanic {
        fn drop(&mut self) {
            DROPS.fetch_add(1, ::core::sync::atomic::Ordering::SeqCst);

            if self.0 {
                panic!("panic in `drop`");
            }
        }
    }

    let mut map = IndexMultimapVec::from_iter([
        (1, DropMayPanic(false, String::from("a"))),
        (2, DropMayPanic(false, String::from("a"))),
        (3, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(true, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
    ]);

    catch_unwind(AssertUnwindSafe(|| drop(map.shift_remove(&1)))).ok();
    assert_eq!(DROPS.load(::core::sync::atomic::Ordering::SeqCst), 6);
}

#[test]
fn swap_remove_drop_panics() {
    static DROPS: AtomicU32 = AtomicU32::new(0);

    #[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
    struct DropMayPanic(bool, String);

    impl Drop for DropMayPanic {
        fn drop(&mut self) {
            DROPS.fetch_add(1, ::core::sync::atomic::Ordering::SeqCst);

            if self.0 {
                panic!("panic in `drop`");
            }
        }
    }

    let mut map = IndexMultimapVec::from_iter([
        (1, DropMayPanic(false, String::from("a"))),
        (2, DropMayPanic(false, String::from("a"))),
        (3, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(true, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
    ]);

    catch_unwind(AssertUnwindSafe(|| drop(map.swap_remove(&1)))).ok();
    assert_eq!(DROPS.load(::core::sync::atomic::Ordering::SeqCst), 6);
}

#[test]
fn swap_remove_index() {
    let insert = [0, 4, 2, 12, 8, 4, 19, 4, 4, 7, 11, 5, 3, 17, 19, 22, 23];
    let mut map = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt * 2);
    }

    let mut vector = insert.to_vec();
    let remove_sequence = &[3, 3, 10, 4, 5, 4, 3, 0, 1];

    // check that the same swap remove sequence on vec and map
    // have the same result.
    for &rm in remove_sequence {
        let out_vec = vector.swap_remove(rm);
        let (out_map, _) = map.swap_remove_index(rm).unwrap();
        assert_eq!(out_vec, out_map);
    }
    assert_eq!(vector.len(), map.len_pairs());
    for (a, b) in vector.iter().zip(map.keys()) {
        assert_eq!(a, b);
    }
    let expected = [7, 4, 2, 19, 5, 3, 19, 4].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);
}

#[test]
fn shift_remove_index() {
    let insert = [0, 4, 2, 12, 8, 4, 19, 4, 4, 7, 11, 5, 3, 17, 19, 22, 23];
    let mut map = IndexMultimapVec::new();
    for &elt in &insert {
        map.insert_append(elt, elt * 2);
    }

    let mut vector = insert.to_vec();
    let remove_sequence = &[3, 3, 10, 4, 5, 4, 3, 0, 1];

    // check that the same shift remove sequence on vec and map
    // have the same result.
    for &rm in remove_sequence {
        let out_vec = vector.remove(rm);
        let (out_map, _) = map.shift_remove_index(rm).unwrap();
        assert_eq!(out_vec, out_map);
    }
    assert_eq!(vector.len(), map.len_pairs());
    for (a, b) in vector.iter().zip(map.keys()) {
        assert_eq!(a, b);
    }
    let expected = [4, 7, 11, 5, 17, 19, 22, 23].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);
}

#[test]
fn truncate() {
    let insert = [0, 4, 2, 12, 8, 4, 19, 4, 4, 7, 11, 5, 3, 17, 19, 22, 23];
    let mut map = IndexMultimapVec::new();
    for &elt in &insert {
        map.insert_append(elt, elt * 2);
    }

    map.truncate(5);
    let expected = [0, 4, 2, 12, 8].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.truncate(100);
    let expected = [0, 4, 2, 12, 8].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.truncate(0);
    assert_map_eq(&map, &[]);
    assert!(map.is_empty());
}

#[test]
fn swap_index() {
    let mut insert = [0, 4, 2, 12, 8, 5, 19];
    let mut map = IndexMultimapVec::new();
    for &elt in &insert {
        map.insert_append(elt, elt * 2);
    }

    map.swap_indices(1, 5);
    insert.swap(1, 5);
    let expected = insert.map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.swap_indices(1, 4);
    insert.swap(1, 4);
    let expected = insert.map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.swap_indices(6, 0);
    insert.swap(6, 0);
    let expected = insert.map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.swap_indices(1, 1);
    insert.swap(1, 1);
    let expected = insert.map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.swap_indices(4, 1);
    insert.swap(4, 1);
    let expected = insert.map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);
}

#[test]
fn move_index() {
    let insert = [0, 4, 2, 12, 8, 5, 19];
    let mut map = IndexMultimapVec::new();
    for &elt in &insert {
        map.insert_append(elt, elt * 2);
    }

    map.move_index(1, 4);
    let expected = [0, 2, 12, 8, 4, 5, 19].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.move_index(6, 0);
    let expected = [19, 0, 2, 12, 8, 4, 5].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.move_index(0, 5);
    let expected = [0, 2, 12, 8, 4, 19, 5].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.move_index(1, 1);
    let expected = [0, 2, 12, 8, 4, 19, 5].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.move_index(4, 1);
    let expected = [0, 4, 2, 12, 8, 19, 5].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);
}

#[test]
fn pop() {
    let insert = [0, 4, 2, 12, 8, 4, 19, 4, 4, 7, 11, 5, 3, 17, 19, 22, 23];
    let mut map = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt * 2);
    }

    assert_eq!(map.pop(), Some((23, 46)));
    assert_eq!(map.pop(), Some((22, 44)));

    let expected = [0, 4, 2, 12, 8, 4, 19, 4, 4, 7, 11, 5, 3, 17, 19].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);
}

#[test]
fn retain() {
    let insert = [0, 4, 2, 12, 8, 4, 19, 4, 4, 7, 11, 5, 3, 17, 19, 22, 23].map(|a| (a, a * 2));
    let mut map = IndexMultimapVec::new();
    map.extend(insert);

    map.retain(|k, _| k != &4);
    let expected = [0, 2, 12, 8, 19, 7, 11, 5, 3, 17, 19, 22, 23].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.retain(|k, _| !(k == &0 || k == &23));
    let expected = [2, 12, 8, 19, 7, 11, 5, 3, 17, 19, 22].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);
}

fn test_entry_retain(insert: &[(i32, (i32, bool))]) {
    let mut map = IndexMultimapVec::new();
    map.extend(insert.iter().map(|(k, v)| (*k, *v)));

    let e = map.entry(4);
    let e = match e {
        Entry::Occupied(e) => e,
        Entry::Vacant(_) => unreachable!(),
    };

    let should_remove_all = !insert.iter().any(|(k, (_, remove))| k == &4i32 && !remove);

    let result = e.retain(|_, _, (_, remove)| !*remove);
    if should_remove_all {
        assert!(result.is_none());
    } else {
        let result = result.unwrap();
        let subset = result.as_subset();
        let expected: Vec<_> = insert
            .iter()
            .filter(|(k, (_, remove))| !(k == &4i32 && *remove))
            .enumerate()
            .filter(|(_, (k, _))| k == &4i32)
            .map(|(i, (k, v))| (i, k, v))
            .collect();
        check_subentries_full(&subset, &expected);
    }

    let expected: Vec<_> = insert
        .iter()
        .filter(|(k, (_, remove))| !(k == &4i32 && *remove))
        .map(|(k, v)| (*k, *v))
        .collect();
    assert_map_eq(&map, &expected);
}

#[test]
fn entry_retain_drop_panics() {
    static DROPS: AtomicU32 = AtomicU32::new(0);

    #[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
    struct DropMayPanic(bool, String);

    impl Drop for DropMayPanic {
        fn drop(&mut self) {
            DROPS.fetch_add(1, ::core::sync::atomic::Ordering::SeqCst);

            if self.0 {
                panic!("panic in `drop`");
            }
        }
    }

    let mut map = IndexMultimapVec::from_iter([
        (1, DropMayPanic(false, String::from("a"))),
        (2, DropMayPanic(false, String::from("a"))),
        (3, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("remove1"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(true, String::from("remove2"))),
        (1, DropMayPanic(false, String::from("remove3"))),
        (1, DropMayPanic(false, String::from("a"))),
    ]);

    let e = map.entry(1);
    let e = match e {
        Entry::Occupied(e) => e,
        Entry::Vacant(_) => unreachable!(),
    };

    catch_unwind(AssertUnwindSafe(|| {
        e.retain(|_, _, v| !v.1.starts_with("remove"))
    }))
    .ok();
    assert_eq!(DROPS.load(::core::sync::atomic::Ordering::SeqCst), 2);

    let expected_map_items = [
        (1, DropMayPanic(false, String::from("a"))),
        (2, DropMayPanic(false, String::from("a"))),
        (3, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("a"))),
        (1, DropMayPanic(false, String::from("remove3"))),
        (1, DropMayPanic(false, String::from("a"))),
    ];
    assert_map_eq(&map, &expected_map_items);
}

#[test]
fn entry_retain() {
    let insert = [
        (1, (11, false)),
        (4, (41, false)),
        (2, (21, false)),
        (3, (31, false)),
        (4, (42, true)),
        (2, (22, false)),
        (5, (51, false)),
        (4, (43, false)),
    ];
    test_entry_retain(&insert);

    let insert = [
        (1, (11, false)),
        (4, (41, true)),
        (2, (21, false)),
        (3, (31, false)),
        (4, (42, true)),
        (2, (22, false)),
        (5, (51, false)),
        (4, (43, false)),
    ];
    test_entry_retain(&insert);

    let insert = [
        (1, (11, false)),
        (4, (41, true)),
        (2, (21, false)),
        (3, (31, false)),
        (4, (42, true)),
        (2, (22, false)),
        (5, (51, false)),
        (4, (43, false)),
        (1, (11, false)),
        (4, (44, true)),
        (2, (21, false)),
        (3, (31, false)),
    ];
    test_entry_retain(&insert);
}

#[test]
fn entry_retain_multiple_consecutive() {
    let insert = [
        (1, (11, true)),
        (4, (41, true)),
        (2, (21, false)),
        (3, (31, false)),
        (4, (42, true)),
        (2, (22, false)),
        (5, (51, false)),
        (4, (43, false)),
        (4, (44, true)),
        (4, (45, true)),
        (4, (46, true)),
        (2, (21, false)),
        (3, (31, false)),
    ];
    test_entry_retain(&insert);
}

#[test]
fn entry_retain_first() {
    let insert = [
        (4, (41, true)),
        (2, (21, false)),
        (3, (31, false)),
        (4, (42, true)),
        (2, (22, false)),
        (5, (51, false)),
        (4, (43, true)),
        (4, (44, false)),
        (2, (21, false)),
        (3, (31, false)),
    ];
    test_entry_retain(&insert);
}

#[test]
fn entry_retain_last() {
    let insert = [
        (2, (21, false)),
        (3, (31, false)),
        (4, (42, false)),
        (2, (22, false)),
        (5, (51, false)),
        (4, (43, true)),
    ];
    test_entry_retain(&insert);
}

#[test]
fn entry_retain_none() {
    let insert = [
        (2, (21, false)),
        (3, (31, false)),
        (4, (42, true)),
        (2, (22, false)),
        (5, (51, false)),
        (4, (43, true)),
    ];
    test_entry_retain(&insert);
}

#[test]
fn reverse() {
    let mut insert = [
        (0, 0),
        (4, 41),
        (4, 42),
        (5, 51),
        (5, 52),
        (4, 43),
        (8, 81),
        (4, 44),
    ];
    let mut map = IndexMultimapVec::new();
    map.extend(insert);
    map.reverse();
    insert.reverse();
    assert_map_eq(&map, &insert);
}

#[test]
fn partial_eq_and_eq() {
    let mut map_a = IndexMultimapVec::new();
    map_a.insert_append(1, "1");
    map_a.insert_append(2, "2");
    let mut map_b = map_a.clone();
    assert_eq!(map_a, map_b);
    map_b.swap_remove(&1);
    assert_ne!(map_a, map_b);

    let map_c: IndexMultimapVec<_, String> =
        map_b.into_iter().map(|(k, v)| (k, v.into())).collect();
    assert_ne!(map_a, map_c);
    assert_ne!(map_c, map_a);
}

#[test]
fn entry() {
    let mut map = IndexMultimapVec::new();

    map.insert_append(1, "1");
    map.insert_append(2, "2");
    {
        let e = map.entry(3);
        assert!(matches!(e, Entry::Vacant(..)));
        assert_eq!(e.indices(), &[2]);
        let mut e = e.or_insert("3");
        assert_eq!(e.first_mut(), (2, &3, &mut "3"));
    }

    let e = map.entry(2);
    assert_eq!(e.indices(), &[1]);
    assert_eq!(e.key(), &2);
    match e {
        Entry::Occupied(ref e) => assert_eq!(e.as_subset().first(), Some((1, &2, &"2"))),
        Entry::Vacant(_) => panic!(),
    }
    assert_eq!(e.or_insert("4").first_mut(), (1, &2, &mut "2"));
}

#[test]
fn entry_and_modify() {
    let mut map = IndexMultimapVec::new();

    map.insert_append(1, "1");
    map.entry(1).and_modify(|mut x| {
        let (_, _, x) = x.first_mut().unwrap();
        *x = "2"
    });
    assert_eq!(map.get(&1).first(), Some((0, &1, &"2")));

    map.entry(2).and_modify(|mut x| x[0] = "doesn't exist");
    assert_eq!(map.get(&2).first(), None);
}

#[test]
fn entry_or_default() {
    let mut map = IndexMultimapVec::new();

    #[derive(Debug, PartialEq)]
    enum TestEnum {
        DefaultValue,
        NonDefaultValue,
    }

    impl Default for TestEnum {
        fn default() -> Self {
            TestEnum::DefaultValue
        }
    }

    map.insert_append(1, TestEnum::NonDefaultValue);
    assert_eq!(
        (0, &1, &mut TestEnum::NonDefaultValue),
        map.entry(1).or_default().first_mut()
    );
    assert_eq!(
        (1, &2, &mut TestEnum::DefaultValue),
        map.entry(2).or_default().first_mut()
    );
}

#[test]
fn entry_insert_append() {
    let mut map = IndexMultimapVec::new();
    map.insert_append(1, "1");
    let mut entry = map.entry(1).insert_append("11");
    check_subentries_full(&entry.as_subset(), &[(0, &1, &"1"), (1, &1, &"11")]);
    entry.insert_append("12");
    check_subentries_full(
        &entry.as_subset(),
        &[(0, &1, &"1"), (1, &1, &"11"), (2, &1, &"12")],
    );

    let entry = map.entry(2).insert_append("21");
    check_subentries_full(&entry.as_subset(), &[(3, &2, &"21")]);

    match map.entry(3) {
        Entry::Occupied(_) => unreachable!(),
        Entry::Vacant(entry) => {
            assert_eq!(entry.insert("31"), (4, &3, &mut "31"))
        }
    }
}

#[test]
fn entry_swap_remove() {
    // Test remove twice: a) by using the returned iterator and b) by dropping it.
    // Both should remove all the needed pairs.
    let insert = [0, 4, 2, 12, 8, 7, 11, 5, 3, 17, 19, 22, 23];
    let mut map = IndexMultimapVec::new();
    let mut map2 = IndexMultimapVec::new();

    for &elt in &insert {
        map.insert_append(elt, elt);
        map2.insert_append(elt, elt);
    }

    assert_eq!(map.keys().count(), map.len_keys());
    assert_eq!(map.keys().count(), insert.len());
    for (a, b) in insert.iter().zip(map.keys()) {
        assert_eq!(a, b);
    }

    let remove_fail = [99, 77];
    let remove = [4, 12, 8, 7];

    for &key in &remove_fail {
        match map.entry(key) {
            Entry::Occupied(occupied_entry) => panic!(),
            Entry::Vacant(vacant_entry) => {}
        }
    }

    for &key in &remove {
        let index = map.get(&key).first().unwrap().0;

        match map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                assert_eq!(
                    occupied_entry.swap_remove().collect::<Vec<_>>(),
                    vec![(index, key, key)]
                );
            }
            Entry::Vacant(vacant_entry) => {
                panic!()
            }
        }
        assert!(map2.swap_remove(&key).is_some());
    }
    let remaining = [
        (0, 0),
        (23, 23),
        (2, 2),
        (22, 22),
        (19, 19),
        (17, 17),
        (11, 11),
        (5, 5),
        (3, 3),
    ];
    assert_map_eq(&map, &remaining);
    assert_map_eq(&map2, &remaining);
    assert_eq!(map, map2);
    assert_eq!(map.as_slice(), map2.as_slice());

    for key in &insert {
        assert_eq!(map.get(key).first().is_some(), !remove.contains(key));
    }
    assert_eq!(map.len_keys(), insert.len() - remove.len());
    assert_eq!(map.keys().count(), insert.len() - remove.len());
}

#[test]
fn occupied_entry_key() {
    // These keys match hash and equality, but their addresses are distinct.
    let (k1, k2) = (&mut 1, &mut 1);
    let k1_ptr = k1 as *const i32;
    let k2_ptr = k2 as *const i32;
    assert_ne!(k1_ptr, k2_ptr);

    let mut map = IndexMultimapVec::new();
    map.insert_append(k1, "value");
    match map.entry(k2) {
        Entry::Occupied(ref e) => {
            // `OccupiedEntry::key` should reference the key in the map,
            // not the key that was used to find the entry.
            let ptr = *e.key() as *const i32;
            assert_eq!(ptr, k1_ptr);
            assert_ne!(ptr, k2_ptr);
        }
        Entry::Vacant(_) => panic!(),
    }
}

#[test]
fn indexed_entry_get() {
    let insert = [
        (0, 1),
        (4, 41),
        (4, 42),
        (5, 51),
        (5, 52),
        (4, 43),
        (8, 81),
        (4, 44),
    ];
    let mut map = IndexMultimapVec::new();
    map.extend(insert);

    let e = map.get_index_entry(map.len_pairs());
    assert!(e.is_none());

    let mut e = map.get_index_entry(0).unwrap();
    assert_eq!(e.get(), (0, &0, &1));
    assert_eq!(e.get_mut(), (0, &0, &mut 1));
    assert_eq!(e.key(), &0);
    assert_eq!(e.value(), &1);
    assert_eq!(e.value_mut(), &mut 1);
    assert_eq!(e.index(), 0);
    assert_eq!(e.into_mut(), (0, &0, &mut 1));

    let mut e = map.get_index_entry(2).unwrap();
    assert_eq!(e.get(), (2, &4, &42));

    let get_all = e.get_all();
    check_subentries(
        &get_all,
        &[(1, &4, &41), (2, &4, &42), (5, &4, &43), (7, &4, &44)],
    );

    let expected = &[
        (1, &4, &mut 41),
        (2, &4, &mut 42),
        (5, &4, &mut 43),
        (7, &4, &mut 44),
    ];
    let mut get_all_mut = e.get_all_mut();
    check_subentries_mut(&mut get_all_mut, expected);
    let mut into_all_mut = e.into_all_mut();
    check_subentries_mut(&mut into_all_mut, expected);

    let e = map.get_index_entry(3).unwrap().into_occupied_entry();
    check_subentries(&e.as_subset(), &[(3, &5, &51), (4, &5, &52)]);
}

#[test]
fn indexed_entry_insert() {
    let mut insert = [
        (0, 1),
        (4, 41),
        (4, 42),
        (5, 51),
        (5, 52),
        (4, 43),
        (8, 81),
        (4, 44),
    ];
    let mut map = IndexMultimapVec::new();
    map.extend(insert);

    let mut e = map.get_index_entry(3).unwrap();
    assert_eq!(e.insert(55), 51);

    insert[3].1 = 55;
    assert_map_eq(&map, &insert);
}

#[test]
fn indexed_entry_remove() {
    let mut insert = vec![
        (0, 1),
        (4, 41),
        (4, 42),
        (5, 51),
        (5, 52),
        (4, 43),
        (8, 81),
        (4, 44),
    ];
    let mut map = IndexMultimapVec::new();
    map.extend(insert.iter().cloned());

    let e = map.get_index_entry(3).unwrap();
    assert_eq!(e.swap_remove(), (5, 51));

    insert.swap_remove(3);
    assert_map_eq(&map, &insert);

    let e = map.get_index_entry(3).unwrap();
    assert_eq!(e.shift_remove(), (4, 44));

    insert.remove(3);
    assert_map_eq(&map, &insert);
}

#[test]
fn indexed_entry_move_index() {
    let mut insert = vec![
        (0, 1),
        (4, 41),
        (4, 42),
        (5, 51),
        (5, 52),
        (4, 43),
        (8, 81),
        (4, 44),
    ];
    let mut map = IndexMultimapVec::new();
    map.extend(insert.iter().cloned());

    let e = map.get_index_entry(3).unwrap();
    e.move_index(0);

    let v = insert.remove(3);
    insert.insert(0, v);
    assert_map_eq(&map, &insert);

    let e = map.get_index_entry(3).unwrap();
    e.move_index(1);

    let v = insert.remove(3);
    insert.insert(1, v);
    assert_map_eq(&map, &insert);

    let e = map.get_index_entry(3).unwrap();
    e.move_index(insert.len() - 1);

    let v = insert.remove(3);
    insert.push(v);
    assert_map_eq(&map, &insert);

    assert_panics_w_msg(
        || map.get_index_entry(3).unwrap().move_index(insert.len()),
        "index out of bounds",
    );
}

#[test]
fn indexed_entry_swap_indices() {
    let mut insert = vec![
        (0, 1),
        (4, 41),
        (4, 42),
        (5, 51),
        (5, 52),
        (4, 43),
        (8, 81),
        (4, 44),
    ];
    let mut map = IndexMultimapVec::new();
    map.extend(insert.iter().cloned());

    let e = map.get_index_entry(3).unwrap();
    e.swap_indices(0);

    insert.swap(0, 3);
    assert_map_eq(&map, &insert);

    let e = map.get_index_entry(3).unwrap();
    e.swap_indices(1);

    insert.swap(1, 3);
    assert_map_eq(&map, &insert);

    let e = map.get_index_entry(3).unwrap();
    let other = insert.len() - 1;
    e.swap_indices(other);

    insert.swap(other, 3);
    assert_map_eq(&map, &insert);

    assert_panics_w_msg(
        || map.get_index_entry(3).unwrap().swap_indices(insert.len()),
        "index out of bounds",
    );
}

#[test]
fn iter() {
    let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let map: IndexMultimapVec<_, _> = vec.clone().into_iter().collect();
    let items = map.iter().collect::<Vec<_>>();
    assert_eq!(items.len(), 3);
    itertools::assert_equal(items, vec.iter().map(|(a, b)| (a, b)));
}

#[test]
fn iter_mut() {
    let mut vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let mut map: IndexMultimapVec<_, _> = vec.clone().into_iter().collect();
    let items = map.iter_mut().collect::<Vec<_>>();
    assert_eq!(items.len(), 3);
    itertools::assert_equal(items, vec.iter_mut().map(|(ref a, b)| (a, b)));
}

#[test]
fn keys() {
    let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let map: IndexMultimapVec<_, _> = vec.into_iter().collect();
    let keys: Vec<_> = map.keys().copied().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn into_keys() {
    let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let map: IndexMultimapVec<_, _> = vec.into_iter().collect();
    let keys: Vec<i32> = map.into_keys().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn values() {
    let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let map: IndexMultimapVec<_, _> = vec.into_iter().collect();
    let values: Vec<_> = map.values().copied().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&'a'));
    assert!(values.contains(&'b'));
    assert!(values.contains(&'c'));
}

#[test]
fn values_mut() {
    let vec = vec![(1, 1), (2, 2), (3, 3)];
    let mut map: IndexMultimapVec<_, _> = vec.into_iter().collect();
    for value in map.values_mut() {
        *value *= 2
    }
    let values: Vec<_> = map.values().copied().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&2));
    assert!(values.contains(&4));
    assert!(values.contains(&6));
}

#[test]
fn into_values() {
    let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let map: IndexMultimapVec<_, _> = vec.into_iter().collect();
    let values: Vec<char> = map.into_values().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&'a'));
    assert!(values.contains(&'b'));
    assert!(values.contains(&'c'));
}

#[test]
fn drain_all() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(..).collect::<Vec<_>>();
    assert_eq!(&drained, &items);
    assert!(map.is_empty());

    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(0..7).collect::<Vec<_>>();
    assert_eq!(&drained, &items);
    assert!(map.is_empty());

    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(0..=6).collect::<Vec<_>>();
    assert_eq!(&drained, &items);
    assert!(map.is_empty());
}

#[test]
fn drain_none() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(items.len()..).collect::<Vec<_>>();
    assert!(drained.is_empty());
    assert_eq!(map.len_pairs(), 7);

    let drained = map.drain(3..3).collect::<Vec<_>>();
    assert!(drained.is_empty());
    assert_eq!(map.len_pairs(), 7);
}

#[test]
fn drain_out_of_bounds() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);
    let len = map.len_pairs();
    assert!(catch_unwind(AssertUnwindSafe(|| drop(map.drain((len + 1)..)))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| drop(map.drain(..(len + 1))))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| drop(map.drain(..=len)))).is_err());
}

#[test]
fn drain_start_to_mid() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(..3).collect::<Vec<_>>();
    assert_eq!(&drained, &items[..3]);

    let remaining = &items[3..];
    assert_map_eq(&map, remaining);
}

#[test]
fn drain_mid_to_end() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(3..).collect::<Vec<_>>();
    assert_eq!(&drained, &items[3..]);

    let remaining = &items[..3];
    //println!("{map:#?}");
    assert_map_eq(&map, remaining);
}

#[test]
fn drain_mid_to_mid() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(3..6).collect::<Vec<_>>();
    assert_eq!(&drained, &items[3..6]);

    let remaining = [&items[0..3], &items[6..]].concat();
    assert_map_eq(&map, &remaining);
}

#[test]
fn drain_empty() {
    let mut map: IndexMultimap<i32, i32> = IndexMultimap::new();
    let mut map2: IndexMultimap<i32, i32> = IndexMultimap::new();
    for (k, v) in map.drain(..) {
        map2.insert_append(k, v);
    }
    assert!(map.is_empty());
    assert!(map2.is_empty());
}

#[test]
fn drain_rev() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let drained = map.drain(2..5).rev().collect::<Vec<_>>();
    let mut expected = items[2..5].to_vec();
    expected.reverse();
    assert_eq!(&drained, &expected);

    let remaining = [&items[..2], &items[5..]].concat();
    assert_map_eq(&map, &remaining);
}

// ignore if miri or asan is set but not if ignore_leaks is also set
#[cfg_attr(
    all(not(run_leaky), any(miri, asan)),
    ignore = "it tests what happens if we leak Drain"
)]
#[test]
fn drain_leak() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    ::core::mem::forget(map.drain(2..));

    //map.extend(items);
    assert!(map.is_empty());
    assert_eq!(map.len_keys(), 0);
    assert_eq!(map.len_pairs(), 0);
    //println!("{map:#?}");
}

#[test]
fn drain_drop_panic() {
    static DROPS: AtomicU32 = AtomicU32::new(0);

    #[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
    struct D(bool, bool, String);

    impl Drop for D {
        fn drop(&mut self) {
            DROPS.fetch_add(1, ::core::sync::atomic::Ordering::SeqCst);

            if self.0 {
                self.1 = true;
                panic!("panic in `Drop`")
            }
        }
    }

    let items = [
        (1, D(false, false, String::from("0"))),
        (2, D(false, false, String::from("1"))),
        (1, D(false, false, String::from("2"))),
        (3, D(false, false, String::from("3"))),
        (1, D(false, false, String::from("4"))),
        (1, D(true, false, String::from("5"))),
        (1, D(false, false, String::from("6"))),
        (1, D(false, false, String::from("7"))),
    ];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    catch_unwind(AssertUnwindSafe(|| drop(map.drain(4..7)))).ok();
    //println!("{map:#?}");
    assert_eq!(DROPS.load(::core::sync::atomic::Ordering::SeqCst), 3);
}

#[test]
fn split_off() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let new_map = map.split_off(3);
    assert_map_eq(&new_map, &items[3..]);
    assert_map_eq(&map, &items[..3]);
}

#[test]
fn split_off_start() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let new_map = map.split_off(0);
    assert_map_eq(&new_map, &items[0..]);
    assert_map_eq(&map, &items[..0]);
    assert!(map.is_empty());
}

#[test]
fn split_off_end() {
    let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);

    let new_map = map.split_off(6);
    assert_map_eq(&new_map, &items[6..]);
    assert_map_eq(&map, &items[..6]);
}

#[test]
#[should_panic]
fn split_off_panic() {
    let items = [(0, 0)];
    let mut map = IndexMultimapVec::new();
    map.extend(items);
    let _new_map = map.split_off(2);
}

#[test]
#[cfg(feature = "std")]
fn from_array() {
    let map = IndexMultimapVec::from([(1, 2), (3, 4)]);
    let mut expected = IndexMultimapVec::new();
    expected.insert_append(1, 2);
    expected.insert_append(3, 4);

    assert_eq!(map, expected)
}

#[test]
fn debugs() {
    #[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
    struct Key {
        k1: i32,
        sfasgsg: i32,
    }

    fn key(k: i32) -> Key {
        Key { k1: k, sfasgsg: k }
    }

    let items: [(_, String); 6] = [
        (key(1), "first".into()),
        (key(2), "second".into()),
        (key(1), "first2".into()),
        (key(3), "third".into()),
        (key(3), "third2".into()),
        (key(1), "first3".into()),
    ];
    let mut map = IndexMultimapVec::with_capacity(5, 8);
    map.extend(items);

    println!("map = {map:#?}");

    println!("\nSubset = {:#?}", map.get(&key(1)));
    println!("\nSubsetMut = {:#?}", map.get_mut(&key(3)));

    let mut get_mut = map.get_mut(&key(1));
    println!("\nSubsetIter = {:#?}", get_mut.iter());
    println!("\nSubsetIterMut = {:#?}", get_mut.iter_mut());
    println!("\nSubsetKeys = {:#?}", get_mut.keys());
    println!("\nSubsetValues = {:#?}", get_mut.values());
    println!("\nSubsetValuesMut = {:#?}", get_mut.values_mut());

    let gentry = map.entry(key(1));
    println!("\nEntry(Occupied) = {:#?}", gentry);
    println!(
        "\nOccupiedEntry = {:#?}",
        match gentry {
            Entry::Occupied(o) => o,
            _ => unreachable!(),
        }
    );
    let ventry = map.entry(key(6));
    println!("\nEntry(Vacant) = {:#?}", ventry);
    println!(
        "\nVacantEntry = {:#?}",
        match ventry {
            Entry::Vacant(v) => v,
            _ => unreachable!(),
        }
    );

    let mut map2 = map.clone();
    let mut swpremove = map2.swap_remove(&key(1)).unwrap();
    swpremove.next();
    println!("\nSwapRemove = {:#?}", swpremove);

    let mut map2 = map.clone();
    let mut shiftremove = map2.shift_remove(&key(1)).unwrap();
    shiftremove.next();
    println!("\nShiftRemove = {:#?}", shiftremove);

    let mut map2 = map.clone();
    let drain = map2.drain(1..3);
    println!("\nDrain = {:#?}", drain);
}

#[test]
fn slice_index() {
    fn check(vec_slice: &[(i32, i32)], map_slice: &Slice<i32, i32>, sub_slice: &Slice<i32, i32>) {
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

#[test]
fn test_binary_search_by() {
    // adapted from std's test for binary_search
    let b: IndexMultimap<_, i32> = []
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&5)), Err(0));

    let b: IndexMultimap<_, i32> = [4]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&3)), Err(0));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&4)), Ok(0));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&5)), Err(1));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 6, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&5)), Err(3));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&6)), Ok(3));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&7)), Err(4));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&8)), Ok(4));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 5, 6, 8]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&9)), Err(6));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 6, 7, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&6)), Ok(3));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&5)), Err(3));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&8)), Ok(5));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 5, 6, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&7)), Err(5));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&0)), Err(0));

    let b: IndexMultimap<_, i32> = [1, 3, 3, 3, 7]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&0)), Err(0));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&1)), Ok(0));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&2)), Err(1));
    assert!(match b.binary_search_by(|_, x| x.cmp(&3)) {
        Ok(1..=3) => true,
        _ => false,
    });
    assert!(match b.binary_search_by(|_, x| x.cmp(&3)) {
        Ok(1..=3) => true,
        _ => false,
    });
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&4)), Err(4));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&5)), Err(4));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&6)), Err(4));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&7)), Ok(4));
    assert_eq!(b.binary_search_by(|_, x| x.cmp(&8)), Err(5));
}

#[test]
fn test_binary_search_by_key() {
    // adapted from std's test for binary_search
    let b: IndexMultimap<_, i32> = []
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by_key(&5, |_, &x| x), Err(0));

    let b: IndexMultimap<_, i32> = [4]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by_key(&3, |_, &x| x), Err(0));
    assert_eq!(b.binary_search_by_key(&4, |_, &x| x), Ok(0));
    assert_eq!(b.binary_search_by_key(&5, |_, &x| x), Err(1));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 6, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by_key(&5, |_, &x| x), Err(3));
    assert_eq!(b.binary_search_by_key(&6, |_, &x| x), Ok(3));
    assert_eq!(b.binary_search_by_key(&7, |_, &x| x), Err(4));
    assert_eq!(b.binary_search_by_key(&8, |_, &x| x), Ok(4));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 5, 6, 8]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by_key(&9, |_, &x| x), Err(6));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 6, 7, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by_key(&6, |_, &x| x), Ok(3));
    assert_eq!(b.binary_search_by_key(&5, |_, &x| x), Err(3));
    assert_eq!(b.binary_search_by_key(&8, |_, &x| x), Ok(5));

    let b: IndexMultimap<_, i32> = [1, 2, 4, 5, 6, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by_key(&7, |_, &x| x), Err(5));
    assert_eq!(b.binary_search_by_key(&0, |_, &x| x), Err(0));

    let b: IndexMultimap<_, i32> = [1, 3, 3, 3, 7]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.binary_search_by_key(&0, |_, &x| x), Err(0));
    assert_eq!(b.binary_search_by_key(&1, |_, &x| x), Ok(0));
    assert_eq!(b.binary_search_by_key(&2, |_, &x| x), Err(1));
    assert!(match b.binary_search_by_key(&3, |_, &x| x) {
        Ok(1..=3) => true,
        _ => false,
    });
    assert!(match b.binary_search_by_key(&3, |_, &x| x) {
        Ok(1..=3) => true,
        _ => false,
    });
    assert_eq!(b.binary_search_by_key(&4, |_, &x| x), Err(4));
    assert_eq!(b.binary_search_by_key(&5, |_, &x| x), Err(4));
    assert_eq!(b.binary_search_by_key(&6, |_, &x| x), Err(4));
    assert_eq!(b.binary_search_by_key(&7, |_, &x| x), Ok(4));
    assert_eq!(b.binary_search_by_key(&8, |_, &x| x), Err(5));
}

#[test]
fn test_partition_point() {
    // adapted from std's test for partition_point
    let b: IndexMultimap<_, i32> = []
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.partition_point(|_, &x| x < 5), 0);

    let b: IndexMultimap<_, i32> = [4]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.partition_point(|_, &x| x < 3), 0);
    assert_eq!(b.partition_point(|_, &x| x < 4), 0);
    assert_eq!(b.partition_point(|_, &x| x < 5), 1);

    let b: IndexMultimap<_, i32> = [1, 2, 4, 6, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.partition_point(|_, &x| x < 5), 3);
    assert_eq!(b.partition_point(|_, &x| x < 6), 3);
    assert_eq!(b.partition_point(|_, &x| x < 7), 4);
    assert_eq!(b.partition_point(|_, &x| x < 8), 4);

    let b: IndexMultimap<_, i32> = [1, 2, 4, 5, 6, 8]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.partition_point(|_, &x| x < 9), 6);

    let b: IndexMultimap<_, i32> = [1, 2, 4, 6, 7, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.partition_point(|_, &x| x < 6), 3);
    assert_eq!(b.partition_point(|_, &x| x < 5), 3);
    assert_eq!(b.partition_point(|_, &x| x < 8), 5);

    let b: IndexMultimap<_, i32> = [1, 2, 4, 5, 6, 8, 9]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.partition_point(|_, &x| x < 7), 5);
    assert_eq!(b.partition_point(|_, &x| x < 0), 0);

    let b: IndexMultimap<_, i32> = [1, 3, 3, 3, 7]
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i + 100, x))
        .collect();
    assert_eq!(b.partition_point(|_, &x| x < 0), 0);
    assert_eq!(b.partition_point(|_, &x| x < 1), 0);
    assert_eq!(b.partition_point(|_, &x| x < 2), 1);
    assert_eq!(b.partition_point(|_, &x| x < 3), 1);
    assert_eq!(b.partition_point(|_, &x| x < 4), 4);
    assert_eq!(b.partition_point(|_, &x| x < 5), 4);
    assert_eq!(b.partition_point(|_, &x| x < 6), 4);
    assert_eq!(b.partition_point(|_, &x| x < 7), 4);
    assert_eq!(b.partition_point(|_, &x| x < 8), 5);
}

#[cfg(feature = "rayon")]
mod rayon {
    use ::core::panic::AssertUnwindSafe;
    use ::core::sync::atomic::AtomicU32;
    use ::std::panic::catch_unwind;
    use ::std::string::String;

    use ::rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelDrainRange, ParallelExtend, ParallelIterator,
    };

    use super::*;
    use crate::multimap::tests::assert_map_eq;
    use crate::IndexMultimap;

    #[allow(dead_code)]
    fn impls_into_parallel_ref_iterator() {
        fn test<'a, T: IntoParallelRefIterator<'a>>(t: &'a T) {
            t.par_iter();
        }
        let map: IndexMultimap<u8, u8> = IndexMultimap::new();
        test(&map);
    }

    #[allow(dead_code)]
    fn impls_into_parallel_ref_mut_iterator() {
        fn test<'a, T: IntoParallelRefMutIterator<'a>>(t: &'a mut T) {
            t.par_iter_mut();
        }
        let mut map: IndexMultimap<u8, u8> = IndexMultimap::new();
        test(&mut map);
    }

    #[test]
    fn par_iter_order() {
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
    fn par_partial_eq_and_eq() {
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
    fn par_extend() {
        let mut map = IndexMultimap::new();
        map.par_extend(vec![(&1, &2), (&3, &4)]);
        map.par_extend(vec![(5, 6)]);
        assert_eq!(
            map.into_par_iter().collect::<Vec<_>>(),
            vec![(1, 2), (3, 4), (5, 6)]
        );
    }

    #[test]
    fn par_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: IndexMultimap<_, _> = vec.into_par_iter().collect();
        let keys: Vec<_> = map.par_keys().copied().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn par_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: IndexMultimap<_, _> = vec.into_par_iter().collect();
        let values: Vec<_> = map.par_values().copied().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn par_values_mut() {
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
    fn par_drain_all() {
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
    fn par_drain_none() {
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
    fn par_drain_out_of_bounds() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);
        let len = map.len_pairs();
        assert!(catch_unwind(AssertUnwindSafe(|| drop(map.par_drain((len + 1)..)))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| drop(map.par_drain(..(len + 1))))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| drop(map.par_drain(..=len)))).is_err());
    }

    #[test]
    fn par_drain_start_to_mid() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(..3).collect::<Vec<_>>();
        assert_eq!(&drained, &items[..3]);

        let remaining = &items[3..];
        assert_map_eq(&map, remaining);
    }

    #[test]
    fn par_drain_mid_to_end() {
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
    fn par_drain_mid_to_mid() {
        let items = [(0, 0), (4, 41), (4, 42), (3, 3), (4, 43), (5, 51), (5, 52)];
        let mut map = IndexMultimap::new();
        map.extend(items);

        let drained = map.par_drain(3..6).collect::<Vec<_>>();
        assert_eq!(&drained, &items[3..6]);

        let remaining = [&items[0..3], &items[6..]].concat();
        assert_map_eq(&map, &remaining);
    }

    #[test]
    fn par_drain_empty() {
        let mut map: IndexMultimap<i32, i32> = IndexMultimap::new();
        let mut map2: IndexMultimap<i32, i32> = map.par_drain(..).collect();
        assert!(map.is_empty());
        assert!(map2.is_empty());
    }

    #[test]
    fn par_drain_rev() {
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
    fn par_drain_leak() {
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
    fn par_drain_drop_panic() {
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
    fn par_drain_drop_panic_consumed() {
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

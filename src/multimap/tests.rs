#![allow(clippy::bool_assert_comparison)]

use ::core::fmt::Debug;
use ::core::panic::AssertUnwindSafe;
use ::core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use ::std::panic::catch_unwind;
use ::std::string::String;

use super::*;

type IndexMultimapVec<K, V> = IndexMultimap<K, V, RandomState, Vec<usize>>;

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

    let present = vec![0, 12, 2];
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
fn check_subentries<K, V>(
    subentries: &Subset<'_, K, V, impl SubsetIndexStorage>,
    expected: &[(usize, &K, &V)],
) where
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
    for i in 0..subentries.len() {
        assert_eq!(subentries.nth(i).as_ref(), expected.get(i));
        assert_eq!(subentries[i], *expected[i].2)
    }
}

#[track_caller]
fn check_subentries_mut(
    subentries: &mut SubsetMut<'_, i32, i32, impl SubsetIndexStorage>,
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
        assert_eq!(subentries[i], *expected[i].2)
    }
}

#[test]
fn get() {
    let insert = [0, 4, 2, 12, 8, 7, 11, 5];
    let not_present = [1, 3, 6, 9, 10];
    let mut map = IndexMultimapVec::with_capacity(insert.len(), insert.len());

    let get = map.get(&5);
    check_subentries(&get, &[]);

    let mut get_mut = map.get_mut(&8);
    check_subentries_mut(&mut get_mut, &[]);

    assert!(map.get_index(5).is_none());
    assert!(map.get_index_mut(6).is_none());

    for &it in insert.iter() {
        map.insert_append(it, it * 1000 + 1);
    }

    for key in not_present {
        let get = map.get(&key);
        check_subentries(&get, &[]);

        let mut get_mut = map.get_mut(&key);
        check_subentries_mut(&mut get_mut, &[]);
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
        check_subentries(&get, &[(index, &key, &val)]);

        let mut get_mut = map.get_mut(&key);
        check_subentries_mut(&mut get_mut, &[(index, &key, &mut val)]);

        assert!(map.get_indices_of(&key).contains(&index));
    }

    for &elt in &not_present {
        assert!(map.get(&elt).first().is_none());
    }

    let present = vec![0, 12, 2];
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
                check_subentries(&get, &[(0, &0, &1), (index, &key, &val)]);
            }
            12 => {
                check_subentries(&get, &[(3, &12, &12001), (index, &key, &val)]);
            }
            2 => {
                check_subentries(&get, &[(2, &2, &2001), (index, &key, &val)]);
            }
            _ => {}
        }

        let mut get_mut = map.get_mut(&key);
        match key {
            0 => {
                check_subentries_mut(&mut get_mut, &[(0, &0, &mut 1), (index, &key, &mut val)]);
            }
            12 => {
                check_subentries_mut(
                    &mut get_mut,
                    &[(3, &12, &mut 12001), (index, &key, &mut val)],
                );
            }
            2 => {
                check_subentries_mut(&mut get_mut, &[(2, &2, &mut 2001), (index, &key, &mut val)]);
            }
            _ => {}
        }

        assert!(map.get_indices_of(&key).contains(&index));
    }
}

// Checks that the map yield given items in the given order by index and by key
#[track_caller]
fn assert_map_eq(map: &IndexMultimapVec<i32, i32>, expected: &[(i32, i32)]) {
    assert_eq!(map.len_pairs(), expected.len());
    itertools::assert_equal(map.iter(), expected.iter().map(|(k, v)| (k, v)));
    for (index, &(key, val)) in expected.iter().enumerate() {
        assert_eq!(&map[index], &val);
        assert_eq!(map.get_index(index), Some((&key, &val)));

        let expected_items = expected
            .iter()
            .enumerate()
            .filter(|(_, (k, _))| k == &key)
            .map(|(i, (k, v))| (i, k, v));

        itertools::assert_equal(map.get(&key).iter(), expected_items);
    }
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

    map.insert_append(capacity_entries, std::usize::MAX);
    assert_eq!(map.len_keys(), capacity_keys + 1);
    assert_eq!(map.len_pairs(), capacity_entries + 1);
    assert!(map.capacity_keys() > capacity_keys);
    assert!(map.capacity_pairs() > capacity_entries);
    assert_eq!(
        map.get(&capacity_entries).first().map(|(_, k, v)| (k, v)),
        Some((&capacity_entries, &std::usize::MAX))
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
    let mut insert = [0, 4, 2, 12, 8, 4, 19];
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
}

#[test]
fn move_index() {
    let insert = [0, 4, 2, 12, 8, 4, 19];
    let mut map = IndexMultimapVec::new();
    for &elt in &insert {
        map.insert_append(elt, elt * 2);
    }

    map.move_index(1, 4);
    let expected = [0, 2, 12, 8, 4, 4, 19].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.move_index(6, 0);
    let expected = [19, 0, 2, 12, 8, 4, 4].map(|a| (a, a * 2));
    assert_map_eq(&map, &expected);

    map.move_index(1, 1);
    let expected = [19, 0, 2, 12, 8, 4, 4].map(|a| (a, a * 2));
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
    check_subentries(&entry.as_subset(), &[(0, &1, &"1"), (1, &1, &"11")]);
    entry.insert_append("12");
    check_subentries(
        &entry.as_subset(),
        &[(0, &1, &"1"), (1, &1, &"11"), (2, &1, &"12")],
    );

    let entry = map.entry(2).insert_append("21");
    check_subentries(&entry.as_subset(), &[(3, &2, &"21")]);

    match map.entry(3) {
        Entry::Occupied(_) => unreachable!(),
        Entry::Vacant(entry) => {
            assert_eq!(entry.insert("31"), (4, &3, &mut "31"))
        }
    }
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

#[cfg(not(miri))]
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

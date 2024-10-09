//use indexmap::IndexMultimap;
use std::hash::{BuildHasher, BuildHasherDefault};

use fnv::FnvHasher;
use indexmap::IndexSet;
use itertools::Itertools;
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};

type FnvBuilder = BuildHasherDefault<FnvHasher>;
type IndexMultimapFnv<K, V> = indexmap::IndexMultimap<K, V, FnvBuilder>;
type IndexMultimap<K, V> = indexmap::IndexMultimap<K, V>;

use std::cmp::min;
use std::collections::hash_map::Entry as HEntry;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{self, Bound, Deref};

use indexmap::multimap::{Entry as OEntry, Subset};

fn set<'a, T: 'a, I>(iter: I) -> HashSet<T>
where
    I: IntoIterator<Item = &'a T>,
    T: Copy + Hash + Eq,
{
    iter.into_iter().copied().collect()
}

fn indexmultimap<'a, T: 'a, I>(iter: I) -> IndexMultimap<T, ()>
where
    I: IntoIterator<Item = &'a T>,
    T: Copy + Hash + Eq + std::fmt::Debug,
{
    IndexMultimap::from_iter(iter.into_iter().copied().map(|k| (k, ())))
}

// Helper macro to allow us to use smaller quickcheck limits under miri.
macro_rules! quickcheck_limit {
    (@as_items $($i:item)*) => ($($i)*);
    {
        $(
            $(#[$m:meta])*
            fn $fn_name:ident($($arg_name:ident : $arg_ty:ty),*) -> $ret:ty {
                $($code:tt)*
            }
        )*
    } => (
        quickcheck::quickcheck! {
            @as_items
            $(
                #[test]
                $(#[$m])*
                fn $fn_name() {
                    fn prop($($arg_name: $arg_ty),*) -> $ret {
                        $($code)*
                    }
                    let mut quickcheck = QuickCheck::new()
                    //.gen(Gen::new(100)).tests(100).max_tests(1000)
                    ;
                    if cfg!(miri) {
                        quickcheck = quickcheck
                            .gen(Gen::new(10))
                            .tests(10)
                            .max_tests(100);
                    }

                    quickcheck.quickcheck(prop as fn($($arg_ty),*) -> $ret);
                }
            )*
        }
    )
}

struct TestKeyRemoveResult {
    success: bool,
    map: IndexMultimap<u8, ()>,
    remaining: HashSet<u8>,
}

fn test_key_remove(
    insert: &[u8],
    remove: &[u8],
    op: impl Fn(&mut IndexMultimap<u8, ()>, u8),
) -> TestKeyRemoveResult {
    let mut map = IndexMultimap::new();
    for &key in insert {
        map.insert_append(key, ());
    }
    for &key in remove {
        op(&mut map, key);
    }
    let remaining = &set(insert) - &set(remove);

    // Number of pairs we expect to see in the map for every key
    let mut counts = HashMap::new();
    for r in remaining.iter() {
        let count = insert.iter().filter(|i| *i == r).count();
        counts.insert(r, count);
    }

    let success = map.len_keys() == remaining.len()
        && remaining
            .iter()
            .all(|k| map.get(k).len() == *counts.get(k).unwrap())
        && remove.iter().all(|k| map.get(k).first().is_none());

    TestKeyRemoveResult {
        success,
        map,
        remaining,
    }
}

fn test_key_remove_keep_order(
    insert: &[u8],
    remove: &[u8],
    op: impl Fn(&mut IndexMultimap<u8, ()>, u8),
) -> TestKeyRemoveResult {
    let result = test_key_remove(&insert, &remove, op);
    if !result.success {
        return result;
    }

    let map = result.map;
    let remaining = result.remaining;

    // Check that order is preserved after removals
    let mut iter = map.keys();
    for &key in insert.iter() {
        if remaining.contains(&key) {
            assert_eq!(Some(&key), iter.next());
        }
    }

    TestKeyRemoveResult {
        success: true,
        map,
        remaining,
    }
}

fn test_index_remove(
    insert: &[u8],
    indices: &[usize],
    op: impl Fn(&mut IndexMultimap<u8, usize>, usize) -> Option<(u8, usize)>,
    vec_op: impl Fn(&mut Vec<(u8, usize)>, usize) -> (u8, usize),
) -> TestResult {
    if indices.is_empty() {
        return TestResult::discard();
    }

    let mut map = IndexMultimap::new();
    let mut vec = Vec::new();
    for &key in insert.iter() {
        map.insert_append(key, map.len_pairs());
        vec.push((key, vec.len()));
    }

    for i in indices {
        let i = i % (map.len_pairs() + 5);

        let res = op(&mut map, i);
        let expected = if i < vec.len() {
            Some(vec_op(&mut vec, i))
        } else {
            None
        };

        assert_eq!(res, expected);

        check_map(&map, &vec);
    }

    TestResult::passed()
}

quickcheck_limit! {
    fn contains(insert: Vec<u32>) -> bool {
        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in &insert {
            map.insert_append(key, ());
            vec.push((key, ()));
        }

        check_map(&map, &vec)
    }

    fn contains_not(insert: Vec<u8>, not: Vec<u8>) -> bool {
        let mut map = IndexMultimap::new();
        for &key in &insert {
            map.insert_append(key, ());
        }
        let nots = &set(&not) - &set(&insert);
        nots.iter().all(|&key| map.get(&key).first().is_none())
    }

    fn insert_remove(insert: Vec<u8>, remove: Vec<u8>) -> bool {
        let mut map = IndexMultimap::new();
        for &key in &insert {
            map.insert_append(key, ());
        }
        for &key in &remove {
            map.swap_remove(&key);
        }
        let elements = &set(&insert) - &set(&remove);
        map.len_keys() == elements.len()
        && elements.iter().all(|k| map.get(k).first().is_some())
    }

    fn insertion_order(insert: Vec<u32>) -> bool {
        let mut map = IndexMultimap::new();
        for &key in &insert {
            map.insert_append(key, map.len_pairs());
        }
        itertools::assert_equal(insert.iter().enumerate().map(|(i, k)| (k, i)), map.iter().map(|(k, v)| (k, *v)));
        true
    }

    fn insert_at(insert: Vec<u32>, indices: Vec<usize>) -> bool {
        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for (&key, &index) in insert.iter().zip(indices.iter()) {
            let index = index % (vec.len() + 1);
            map.insert_at(index, key, map.len_pairs());
            vec.insert(index, (key, vec.len()));
        }

        check_map(&map, &vec)
    }

    fn reverse(insert: Vec<u32>) -> bool {
        if insert.is_empty() {
            return true;
        }

        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for key in &insert {
            map.insert_append(key, map.len_pairs());
            vec.push((key, vec.len()));
        }

        map.reverse();
        vec.reverse();

        check_map(&map, &vec)
    }

    fn pop(insert: Vec<u8>) -> bool {
        let mut map = IndexMultimap::new();
        for &key in &insert {
            map.insert_append(key, ());
        }
        let mut pops = Vec::new();
        while let Some((key, _v)) = map.pop() {
            pops.push(key);
        }
        pops.reverse();

        itertools::assert_equal(insert.iter(), &pops);
        true
    }

    fn with_cap_keys(template: Vec<()>) -> bool {
        let cap_keys = template.len();
        let cap_entries = 0;
        let map = IndexMultimap::<u8, u8>::with_capacity(cap_keys, cap_entries);
        println!("wish: {}, got: {} (diff: {})", cap_keys, map.capacity_keys(), map.capacity_keys() as isize - cap_keys as isize);
        map.capacity_keys() >= cap_keys && map.capacity_pairs() == 0
    }

    fn with_cap_entries(template: Vec<()>) -> bool {
        let cap_keys = 0;
        let cap_entries = template.len();
        let map = IndexMultimap::<u8, u8>::with_capacity(cap_keys, cap_entries);
        println!("wish: {}, got: {} (diff: {})", cap_entries, map.capacity_pairs(), map.capacity_pairs() as isize - cap_entries as isize);
        map.capacity_pairs() >= cap_entries && map.capacity_keys() < 10
    }

    fn drain_full(insert: Vec<u8>) -> bool {
        let mut map = IndexMultimap::new();
        for &key in &insert {
            map.insert_append(key, ());
        }
        let mut clone = map.clone();
        let drained = clone.drain(..);
        for (key, _) in drained {
            map.swap_remove(&key);
        }
        map.is_empty() && clone.is_empty()
    }

    fn drain_bounds(insert: Vec<u8>, range: (Bound<usize>, Bound<usize>)) -> TestResult {
        let mut map = IndexMultimap::new();
        for &key in &insert {
            map.insert_append(key, ());
        }

        // First see if `Vec::drain` is happy with this range.
        let result = std::panic::catch_unwind(|| {
            let mut keys: Vec<u8> = map.keys().copied().collect();
            let drained = keys.drain(range).collect::<Vec<_>>();
            (keys, drained)
        });

        if let Ok((keys, drained)) = result {
            let drained_map = map.drain(range).map(|(k, _)| k).collect::<Vec<_>>();
            assert_eq!(drained, drained_map);
            // Check that our `drain` matches the same key order.
            assert!(map.keys().eq(&keys));
            // Check that hash lookups all work too.
            assert!(keys.iter().all(|key| map.contains_key(key)));
            TestResult::passed()
        } else {
            // If `Vec::drain` panicked, so should we.
            TestResult::must_fail(move || { map.drain(range); })
        }
    }

    fn shift_remove(insert: Vec<u8>, remove: Vec<u8>) -> bool {
        test_key_remove_keep_order(&insert, &remove, |map, key| { map.shift_remove(&key); }).success
    }

    fn swap_remove(insert: Vec<u8>, remove: Vec<u8>) -> bool {
        test_key_remove(&insert, &remove, |map, key| { map.swap_remove(&key); }).success
    }

    fn entry_swap_remove(insert: Vec<u8>, remove: Vec<u8>) -> bool {
        test_key_remove(&insert, &remove, |map, key| match map.entry(key) {
            OEntry::Occupied(e) => {
                e.swap_remove();
            }
            OEntry::Vacant(_) => {},
        }).success
    }

    fn entry_shift_remove(insert: Vec<u8>, remove: Vec<u8>) -> bool {
        test_key_remove_keep_order(&insert, &remove, |map, key| match map.entry(key) {
            OEntry::Occupied(e) => {
                e.shift_remove();
            }
            OEntry::Vacant(_) => {},
        }).success
    }

    fn shift_remove_index(insert: Vec<u8>, indices: Vec<usize>) -> TestResult {
        test_index_remove(&insert, &indices, |map, i| map.shift_remove_index(i), |vec, i| vec.remove(i) )
    }


    fn swap_remove_index(insert: Vec<u8>, indices: Vec<usize>) -> TestResult {
        test_index_remove(&insert, &indices, |map, i| map.swap_remove_index(i), |vec, i| vec.swap_remove(i) )
    }

    fn swap_index(insert: Vec<u8>, indices: Vec<usize>) -> TestResult {
        if insert.is_empty() || indices.is_empty() {
            return TestResult::discard()
        }

        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in insert.iter() {
            map.insert_append(key, map.len_pairs());
            vec.push((key, vec.len()));
        }

        for indices in indices.windows(2) {
            let (start, end) = (indices[0], indices[1]);
            let a = start % map.len_pairs();
            let b = end % map.len_pairs();

            map.swap_indices(a, b);
            vec.swap(a, b);

            check_map(&map, &vec);
        }


        TestResult::passed()
    }

    fn move_index(insert: Vec<u8>, indices: Vec<usize>) -> TestResult {
        if insert.is_empty() || indices.is_empty() {
            return TestResult::discard()
         }

        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in insert.iter() {
            map.insert_append(key, map.len_pairs());
            vec.push((key, vec.len()));
        }

        for indices in indices.chunks_exact(2) {
            let (start, end) = (indices[0], indices[1]);
            let a = start % map.len_pairs();
            let b = end % map.len_pairs();

            map.move_index(a, b);
            let v = vec.remove(a);
            vec.insert(b, v);

            check_map(&map, &vec);
        }

        TestResult::passed()
    }

    fn truncate(insert: Vec<u8>, indices: Vec<usize>) -> TestResult {
        if insert.is_empty() || indices.is_empty() {
            return TestResult::discard();
        }

        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in insert.iter() {
            map.insert_append(key, map.len_pairs());
            vec.push((key, vec.len()));
        }

        for index in indices {
            if map.is_empty() {
                break;
            }

            let index = index % (map.len_pairs() + 5);

            map.truncate(index);
            vec.truncate(index);

            check_map(&map, &vec);
        }

        TestResult::passed()
    }

    fn split_off(insert: Vec<u8>, indices: Vec<usize>) -> TestResult {
        if insert.is_empty() || indices.is_empty() {
            return TestResult::discard();
        }


        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in insert.iter() {
            map.insert_append(key, map.len_pairs());
            vec.push((key, vec.len()));
        }

        for index in indices {
            if map.is_empty() {
                break;
            }

            let index = index % (map.len_pairs() + 1);

            let right = map.split_off(index);
            let expected_right = vec.split_off(index);

            check_map(&map, &vec);
            check_map(&right, &expected_right);
        }

        TestResult::passed()
    }

    fn indexing(insert: Vec<u8>) -> bool {
        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in insert.iter() {
            map.insert_append(key, map.len_pairs());
            vec.push((key, vec.len()));
        }


        for (i, &key) in insert.iter().enumerate() {
            assert_eq!(map.get_index(i), Some((&key, &i)));
            assert_eq!(map[i], i);

            *map.get_index_mut(i).unwrap().1 += map.len_pairs();
            vec[i].1 += vec.len();
        }

        check_map(&map, &vec);
        true
    }

    fn entry_insert(insert: Vec<u8>) -> bool {
        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in insert.iter() {
            let value = map.len_pairs();
            // vacant on first this key
            map.entry(key).insert_append(value);
            vec.push((key, value));

            // definitely occupied
            map.entry(key).insert_append(value+1);
            vec.push((key, value + 1));
        }


        check_map(&map, &vec);


        true
    }


    fn entry_insert_many(insert: Vec<u8>) -> bool {
        let mut map = IndexMultimap::new();
        let mut vec = Vec::new();
        for &key in insert.iter() {
            let value = map.len_pairs();
            // vacant on first this key
            map.entry(key).insert_append_many([]).insert_append_many([value, value + 1]).insert_append_many([value + 2, value + 3]);
            vec.push((key, value));
            vec.push((key, value + 1));
            vec.push((key, value + 2));
            vec.push((key, value + 3));
        }


        check_map(&map, &vec);

        true
    }


    fn entry_retain(insert: Vec<(i8, (i8, bool))>) -> bool {
        let mut map = IndexMultimap::new();
        map.extend(insert.iter().map(|(k, v)| (*k, *v)));
        let keys = insert.iter().map(|(k, _)| *k).collect::<HashSet<_>>();

        for key in keys {
            let orig_map = map.clone();
            let entry = map.entry(key);
            let entry = match entry {
                OEntry::Occupied(e) => e,
                OEntry::Vacant(_) => unreachable!(),
            };

            let should_remove_all = !insert.iter().any(|(k, (_, remove))| k == &key && !remove);

            let result = entry.retain(|_, _, (_, remove)| !*remove);
            if should_remove_all {
                if result.is_some() {
                    return false;
                }
            } else {
                let subset = result.unwrap().into_subset();
                let expected: Vec<_> = orig_map
                    .iter()
                    .filter(|(k, (_, remove))| !(*k == &key && *remove))
                    .enumerate()
                    .filter(|(_, (k, _))| *k == &key)
                    .map(|(i, (k, v))| (i, k, v))
                    .collect();
                if !check_subentries(&subset, &expected) {
                    return false;
                }
            }

            let expected_map: Vec<_> = orig_map
                .iter()
                .filter(|(k, (_, remove))| !(*k == &key && *remove))
                .map(|(k, v)| (*k, *v))
                .collect();
            if !check_map(&map, &expected_map) {
                return false;
            }

        }

        true
    }
}

#[cfg(feature = "rayon")]
mod rayon {
    use ::rayon::prelude::*;

    use super::*;
    quickcheck_limit! {
        fn par_drain_full(insert: Vec<u8>) -> bool {
            let mut map = IndexMultimap::new();
            for &key in &insert {
                map.insert_append(key, ());
            }
            let clone = map.clone();
            let new: IndexMultimap<_, _> = map.par_drain(..).collect();
            map.is_empty() && clone == new && clone.as_slice() == new.as_slice()
        }

        fn par_drain_bounds(insert: Vec<u8>, range: (Bound<usize>, Bound<usize>)) -> TestResult {
            // add unique value to each key so that we can distinguish items
            let insert = insert.into_iter().enumerate().map(|(i, k)| (k, i)).collect::<Vec<_>>();
            let mut map = IndexMultimap::new();
            for &(key, value) in &insert {
                map.insert_append(key, value);
            }

            // First see if `Vec::drain` is happy with this range.
            let result = std::panic::catch_unwind(|| {
                let mut insert = insert.clone();
                let drained = insert.drain(range).collect::<Vec<_>>();
                (insert, drained)
            });

            if let Ok((remaining_expected, drained_expected)) = result {
                let drained_map = map.par_drain(range).collect::<Vec<_>>();
                assert_eq!(drained_expected, drained_map);
                itertools::assert_equal(map.iter(), remaining_expected.iter().map(|(k, v)| (k, v)));
                TestResult::passed()
            } else {
                // If `Vec::drain` panicked, so should we.
                TestResult::must_fail(move || { map.drain(range); })
            }
        }
    }
}

fn check_subentries<K, V>(subentries: &Subset<'_, K, V>, expected: &[(usize, &K, &V)]) -> bool
where
    K: Eq + Debug,
    V: Eq + Debug,
{
    let mut result = subentries.len() == expected.len();
    result &= subentries.first().as_ref() == expected.first();
    result &= subentries.last().as_ref() == expected.last();
    result &= itertools::equal(subentries.iter(), expected.iter().copied());
    result &= itertools::equal(subentries.indices(), expected.iter().map(|(i, _, _)| i));
    result &= itertools::equal(subentries.keys(), expected.iter().map(|(_, k, _)| *k));
    result &= itertools::equal(subentries.values(), expected.iter().map(|(_, _, v)| *v));
    for i in 0..subentries.len() {
        result &= subentries.nth(i).as_ref() == expected.get(i);
    }

    result
}

/// Checks that the map yield given items in the given order by index and by key
fn check_map<K, V>(map: &IndexMultimap<K, V>, expected: &[(K, V)]) -> bool
where
    K: core::fmt::Debug + Eq + std::hash::Hash,
    V: core::fmt::Debug + Eq,
{
    let mut result = map.len_pairs() == expected.len();
    result &= itertools::equal(map.iter(), expected.iter().map(|(k, v)| (k, v)));
    for (index, (key, val)) in expected.iter().enumerate() {
        result &= map.get_index(index) == Some((key, val));

        let expected_items = expected
            .iter()
            .enumerate()
            .filter(|(_, (k, _))| k == key)
            .map(|(i, (k, v))| (i, k, v));

        result &= itertools::equal(map.get(key).iter(), expected_items);

        if !result {
            break;
        }
    }
    result
}

use crate::Op::*;
#[derive(Copy, Clone, Debug)]
enum Op<K, V> {
    Add(K, V),
    Remove(K),
    AddEntry(K, V),
    RemoveEntry(K),
}

impl<K, V> Arbitrary for Op<K, V>
where
    K: Arbitrary,
    V: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        match u32::arbitrary(g) % 4 {
            0 => Add(K::arbitrary(g), V::arbitrary(g)),
            1 => AddEntry(K::arbitrary(g), V::arbitrary(g)),
            2 => Remove(K::arbitrary(g)),
            _ => RemoveEntry(K::arbitrary(g)),
        }
    }
}

fn do_ops<K, V, S>(
    ops: &[Op<K, V>],
    a: &mut indexmap::IndexMultimap<K, V, S>,
    b: &mut HashMap<K, Vec<V>>,
) where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
    S: BuildHasher,
{
    for op in ops {
        match *op {
            Add(ref k, ref v) => {
                a.insert_append(k.clone(), v.clone());
                b.entry(k.clone())
                    .and_modify(|a| a.push(v.clone()))
                    .or_insert_with(|| vec![v.clone()]);
                //b.insert(k.clone(), v.clone());
            }
            AddEntry(ref k, ref v) => {
                a.entry(k.clone()).or_insert_with(|| v.clone());
                b.entry(k.clone()).or_insert_with(|| vec![v.clone()]);
            }
            Remove(ref k) => {
                a.swap_remove(k);
                b.remove(k);
            }
            RemoveEntry(ref k) => {
                if let OEntry::Occupied(ent) = a.entry(k.clone()) {
                    ent.swap_remove();
                }
                if let HEntry::Occupied(ent) = b.entry(k.clone()) {
                    ent.remove_entry();
                }
            }
        }
        //println!("{:?}", a);
    }
}

fn assert_maps_equivalent<K, V>(a: &IndexMultimap<K, V>, b: &HashMap<K, Vec<V>>) -> bool
where
    K: Hash + Eq + Debug,
    V: Eq + Debug + Ord,
{
    assert_eq!(a.len_keys(), b.len());
    assert_eq!(a.len_pairs(), b.iter().flat_map(|(_, v)| v).count());
    assert_eq!(
        a.iter().next().is_some(),
        b.iter().flat_map(|(_, v)| v).next().is_some()
    );
    for key in a.keys() {
        assert!(b.contains_key(key), "b does not contain {:?}", key);
    }
    for key in b.keys() {
        assert!(a.get(key).first().is_some(), "a does not contain {:?}", key);
    }
    for key in a.keys() {
        let it = a.get(key).into_iter().map(|(_, _, v)| v).sorted();
        itertools::assert_equal(it, b[key].iter().sorted())
    }
    true
}

quickcheck_limit! {
    fn operations_i8(ops: Large<Vec<Op<i8, i8>>>) -> bool {
        let mut map = IndexMultimap::new();
        let mut reference = HashMap::new();
        do_ops(&ops, &mut map, &mut reference);
        assert_maps_equivalent(&map, &reference)
    }

    fn operations_string(ops: Vec<Op<Alpha, i8>>) -> bool {
        let mut map = IndexMultimap::new();
        let mut reference = HashMap::new();
        do_ops(&ops, &mut map, &mut reference);
        assert_maps_equivalent(&map, &reference)
    }

    fn keys_values(ops: Large<Vec<Op<i8, i8>>>) -> bool {
        let mut map = IndexMultimap::new();
        let mut reference = HashMap::new();
        do_ops(&ops, &mut map, &mut reference);
        let mut visit = IndexMultimap::new();
        for (k, v) in map.keys().zip(map.values()) {
            //assert_eq!(&map[k], v);
            //assert!(!visit.contains_key(k));
            visit.insert_append(*k, *v);
        }
        assert_eq!(visit.len_keys(), reference.len());
        assert_maps_equivalent(&visit, &reference);
        true
    }

    fn keys_values_mut(ops: Large<Vec<Op<i8, i8>>>) -> bool {
        let mut map = IndexMultimap::new();
        let mut reference = HashMap::new();
        do_ops(&ops, &mut map, &mut reference);
        let mut visit = IndexMultimap::new();
        let keys = Vec::from_iter(map.keys().copied());
        for (k, v) in keys.iter().zip(map.values_mut()) {
            //assert_eq!(&reference[k], v);
            //assert!(!visit.contains_key(k));
            visit.insert_append(*k, *v);
        }
        assert_eq!(visit.len_keys(), reference.len());
        assert_maps_equivalent(&visit, &reference);
        true
    }

    fn equality(ops1: Vec<Op<i8, i8>>, removes: Vec<usize>) -> bool {
        let mut map = IndexMultimap::new();
        let mut reference = HashMap::new();
        do_ops(&ops1, &mut map, &mut reference);
        let mut ops2 = ops1.clone();
        for &r in &removes {
            if !ops2.is_empty() {
                let i = r % ops2.len();
                ops2.remove(i);
            }
        }
        let mut map2 = IndexMultimapFnv::default();
        let mut reference2 = HashMap::new();
        do_ops(&ops2, &mut map2, &mut reference2);
        assert_eq!(map == map2, reference == reference2);
        true
    }

    fn retain_ordered(keys: Large<Vec<i8>>, remove: Large<Vec<i8>>) -> () {
        let mut map = indexmultimap(keys.iter());
        let initial_map = map.clone(); // deduplicated in-order input
        let remove_map = indexmultimap(remove.iter());
        let keys_s = set(keys.iter());
        let remove_s = set(remove.iter());
        let answer = &keys_s - &remove_s;
        map.retain(|k, _| !remove_map.contains_key(k));

        // check the values
        assert_eq!(map.len_keys(), answer.len());
        for key in &answer {
            assert!(map.contains_key(key));
        }
        // check the order
        itertools::assert_equal(map.keys(), initial_map.keys().filter(|&k| !remove_map.contains_key(k)));
    }

    fn sort_1(keyvals: Large<Vec<(i8, i8)>>) -> () {
        let mut map = IndexMultimap::from_iter(keyvals.to_vec());
        let mut answer = keyvals.0;
        answer.sort_by_key(|t| t.0);

        map.sort_by(|k1, _, k2, _| Ord::cmp(k1, k2));

        // check the order

        let mapv = Vec::from_iter(map);
        assert_eq!(answer, mapv);

    }

    fn sort_2(keyvals: Large<Vec<(i8, i8)>>) -> () {
        let mut map = IndexMultimap::from_iter(keyvals.to_vec());
        map.sort_by(|_, v1, _, v2| Ord::cmp(v1, v2));
        assert_sorted_by_key(map, |t| t.1);
    }

    fn sort_3(keyvals: Large<Vec<(i8, i8)>>) -> () {
        let mut map = IndexMultimap::from_iter(keyvals.to_vec());
        map.sort_by_cached_key(|&k, _| std::cmp::Reverse(k));
        assert_sorted_by_key(map, |t| std::cmp::Reverse(t.0));
    }
}

fn assert_sorted_by_key<I, Key, X>(iterable: I, key: Key)
where
    I: IntoIterator,
    I::Item: Ord + Clone + Debug,
    Key: Fn(&I::Item) -> X,
    X: Ord,
{
    let input = Vec::from_iter(iterable);
    let mut sorted = input.clone();
    sorted.sort_by_key(key);
    assert_eq!(input, sorted);
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Alpha(String);

impl Deref for Alpha {
    type Target = String;
    fn deref(&self) -> &String {
        &self.0
    }
}

const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

impl Arbitrary for Alpha {
    fn arbitrary(g: &mut Gen) -> Self {
        let len = usize::arbitrary(g) % g.size();
        let len = min(len, 16);
        Alpha(
            (0..len)
                .map(|_| ALPHABET[usize::arbitrary(g) % ALPHABET.len()] as char)
                .collect(),
        )
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new((**self).shrink().map(Alpha))
    }
}

/// quickcheck Arbitrary adaptor -- make a larger vec
#[derive(Clone, Debug)]
struct Large<T>(T);

impl<T> Deref for Large<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Arbitrary for Large<Vec<T>>
where
    T: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        let len = usize::arbitrary(g) % (g.size() * 10);
        Large((0..len).map(|_| T::arbitrary(g)).collect())
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new((**self).shrink().map(Large))
    }
}

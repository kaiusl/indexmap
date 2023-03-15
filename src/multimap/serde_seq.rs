use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::marker::PhantomData;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::IndexMultimap;

use super::IndexStorage;

/// Serializes an [`IndexMultimap`] as an ordered sequence.
///
/// This function may be used in a field attribute for deriving [`Serialize`]:
///
/// ```
/// # use indexmap::IndexMap;
/// # use serde_derive::Serialize;
/// #[derive(Serialize)]
/// struct Data {
///     #[serde(serialize_with = "indexmap::serde_seq::serialize")]
///     map: IndexMap<i32, u64>,
///     // ...
/// }
/// ```
///
/// Requires crate feature `"serde"`
pub fn serialize<K, V, S, Indices, T>(
    map: &IndexMultimap<K, V, S, Indices>,
    serializer: T,
) -> Result<T::Ok, T::Error>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
    Indices: IndexStorage,
    S: BuildHasher,
    T: Serializer,
{
    serializer.collect_seq(map)
}

/// Visitor to deserialize a *sequenced* [`IndexMultimap`]
struct SeqVisitor<K, V, S, Indices>(PhantomData<(K, V, S, Indices)>);

impl<'de, K, V, Indices, S> Visitor<'de> for SeqVisitor<K, V, S, Indices>
where
    K: Deserialize<'de> + Eq + Hash,
    V: Deserialize<'de>,
    Indices: IndexStorage,
    S: Default + BuildHasher,
{
    type Value = IndexMultimap<K, V, S, Indices>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "a sequenced map")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let capacity = seq.size_hint().unwrap_or(0);
        let mut map = IndexMultimap::with_capacity_and_hasher(capacity, capacity, S::default());

        while let Some((key, value)) = seq.next_element()? {
            map.insert_append(key, value);
        }

        Ok(map)
    }
}

/// Deserializes an [`IndexMultimap`] from an ordered sequence.
///
/// This function may be used in a field attribute for deriving [`Deserialize`]:
///
/// ```
/// # use indexmap::IndexMap;
/// # use serde_derive::Deserialize;
/// #[derive(Deserialize)]
/// struct Data {
///     #[serde(deserialize_with = "indexmap::serde_seq::deserialize")]
///     map: IndexMap<i32, u64>,
///     // ...
/// }
/// ```
///
/// Requires crate feature `"serde"`
pub fn deserialize<'de, D, K, V, S, Indices>(
    deserializer: D,
) -> Result<IndexMultimap<K, V, S, Indices>, D::Error>
where
    D: Deserializer<'de>,
    K: Deserialize<'de> + Eq + Hash,
    V: Deserialize<'de>,
    Indices: IndexStorage,
    S: Default + BuildHasher,
{
    deserializer.deserialize_seq(SeqVisitor(PhantomData))
}

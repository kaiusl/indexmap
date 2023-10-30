#[cfg(feature = "rayon")]
pub(crate) mod rayon {
    use ::core::fmt;

    use ::rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
    use ::rayon::prelude::*;

    use crate::util::debug_iter_as_list;
    use crate::vec::Vec;
    use crate::Bucket;

    /// A parallel owning iterator over the entries of a [`IndexMultimap`].
    ///
    /// This `struct` is created by the [`into_par_iter`] method on [`IndexMultimap`]
    /// (provided by [rayon]'s [`IntoParallelIterator`] trait).
    /// See its documentation for more.
    ///
    /// [`into_par_iter`]: crate::IndexMultimap::into_par_iter
    /// [rayon]: https://docs.rs/rayon/1.0/rayon
    /// [`IndexMultimap`]: crate::IndexMultimap
    pub struct IntoParIter<K, V> {
        pub(crate) entries: Vec<Bucket<K, V>>,
    }

    impl<K, V> fmt::Debug for IntoParIter<K, V>
    where
        K: fmt::Debug,
        V: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let iter = self.entries.iter().map(Bucket::refs);
            debug_iter_as_list(f, Some("IntoParIter"), iter)
        }
    }

    impl<K, V> ParallelIterator for IntoParIter<K, V>
    where
        K: Send,
        V: Send,
    {
        type Item = (K, V);

        parallel_iterator_methods!(Bucket::key_value);
    }

    impl<K, V> IndexedParallelIterator for IntoParIter<K, V>
    where
        K: Send,
        V: Send,
    {
        indexed_parallel_iterator_methods!(Bucket::key_value);
    }

    /// A parallel iterator over the entries of a [`IndexMultimap`].
    ///
    /// This `struct` is created by the [`par_iter`] method on [`IndexMultimap`]
    /// (provided by [rayon]'s [`IntoParallelRefIterator`] trait).
    /// See its documentation for more.
    ///
    /// [`par_iter`]: ../struct.IndexMultimap.html#method.par_iter
    /// [rayon]: https://docs.rs/rayon/1.0/rayon
    /// [`IndexMultimap`]: crate::IndexMultimap
    pub struct ParIter<'a, K, V> {
        pub(crate) entries: &'a [Bucket<K, V>],
    }

    impl<K, V> Clone for ParIter<'_, K, V> {
        fn clone(&self) -> Self {
            ParIter { ..*self }
        }
    }

    impl<K, V> fmt::Debug for ParIter<'_, K, V>
    where
        K: fmt::Debug,
        V: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let iter = self.entries.iter().map(Bucket::refs);
            debug_iter_as_list(f, Some("ParIter"), iter)
        }
    }

    impl<'a, K, V> ParallelIterator for ParIter<'a, K, V>
    where
        K: Sync,
        V: Sync,
    {
        type Item = (&'a K, &'a V);

        parallel_iterator_methods!(Bucket::refs);
    }

    impl<K, V> IndexedParallelIterator for ParIter<'_, K, V>
    where
        K: Sync,
        V: Sync,
    {
        indexed_parallel_iterator_methods!(Bucket::refs);
    }

    /// A parallel mutable iterator over the entries of a [`IndexMultimap`].
    ///
    /// This `struct` is created by the [`par_iter_mut`] method on [`IndexMultimap`]
    /// (provided by [rayon]'s [`IntoParallelRefMutIterator`] trait).
    /// See its documentation for more.
    ///
    /// [`par_iter_mut`]: ../struct.IndexMultimap.html#method.par_iter_mut
    /// [rayon]: https://docs.rs/rayon/1.0/rayon
    /// [`IndexMultimap`]: crate::IndexMultimap
    pub struct ParIterMut<'a, K, V> {
        pub(crate) entries: &'a mut [Bucket<K, V>],
    }

    impl<K, V> fmt::Debug for ParIterMut<'_, K, V>
    where
        K: fmt::Debug,
        V: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let iter = self.entries.iter().map(Bucket::refs);
            debug_iter_as_list(f, Some("ParIterMut"), iter)
        }
    }

    impl<'a, K, V> ParallelIterator for ParIterMut<'a, K, V>
    where
        K: Sync + Send,
        V: Send,
    {
        type Item = (&'a K, &'a mut V);

        parallel_iterator_methods!(Bucket::ref_mut);
    }

    impl<K, V> IndexedParallelIterator for ParIterMut<'_, K, V>
    where
        K: Sync + Send,
        V: Send,
    {
        indexed_parallel_iterator_methods!(Bucket::ref_mut);
    }

    /// A parallel iterator over the keys of a [`IndexMultimap`].
    ///
    /// This `struct` is created by the [`par_keys`] method on [`IndexMultimap`].
    /// See its documentation for more.
    ///
    /// [`par_keys`]: crate::IndexMultimap::par_keys
    /// [`IndexMultimap`]: crate::IndexMultimap
    pub struct ParKeys<'a, K, V> {
        pub(crate) entries: &'a [Bucket<K, V>],
    }

    impl<K, V> Clone for ParKeys<'_, K, V> {
        fn clone(&self) -> Self {
            ParKeys { ..*self }
        }
    }

    impl<K, V> fmt::Debug for ParKeys<'_, K, V>
    where
        K: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let iter = self.entries.iter().map(Bucket::key_ref);
            debug_iter_as_list(f, Some("ParKeys"), iter)
        }
    }

    impl<'a, K, V> ParallelIterator for ParKeys<'a, K, V>
    where
        K: Sync,
        V: Sync,
    {
        type Item = &'a K;

        parallel_iterator_methods!(Bucket::key_ref);
    }

    impl<K, V> IndexedParallelIterator for ParKeys<'_, K, V>
    where
        K: Sync,
        V: Sync,
    {
        indexed_parallel_iterator_methods!(Bucket::key_ref);
    }

    /// A parallel iterator over the values of a [`IndexMultimap`].
    ///
    /// This `struct` is created by the [`par_values`] method on [`IndexMultimap`].
    /// See its documentation for more.
    ///
    /// [`par_values`]: crate::IndexMultimap::par_values
    /// [`IndexMultimap`]: crate::IndexMultimap
    pub struct ParValues<'a, K, V> {
        pub(crate) entries: &'a [Bucket<K, V>],
    }

    impl<K, V> Clone for ParValues<'_, K, V> {
        fn clone(&self) -> Self {
            ParValues { ..*self }
        }
    }

    impl<K, V> fmt::Debug for ParValues<'_, K, V>
    where
        V: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let iter = self.entries.iter().map(Bucket::value_ref);
            debug_iter_as_list(f, Some("ParValues"), iter)
        }
    }

    impl<'a, K, V> ParallelIterator for ParValues<'a, K, V>
    where
        K: Sync,
        V: Sync,
    {
        type Item = &'a V;

        parallel_iterator_methods!(Bucket::value_ref);
    }

    impl<K, V> IndexedParallelIterator for ParValues<'_, K, V>
    where
        K: Sync,
        V: Sync,
    {
        indexed_parallel_iterator_methods!(Bucket::value_ref);
    }

    /// A parallel mutable iterator over the values of a [`IndexMultimap`].
    ///
    /// This `struct` is created by the [`par_values_mut`] method on [`IndexMultimap`].
    /// See its documentation for more.
    ///
    /// [`par_values_mut`]: crate::IndexMultimap::par_values_mut
    /// [`IndexMultimap`]: crate::IndexMultimap
    pub struct ParValuesMut<'a, K, V> {
        pub(crate) entries: &'a mut [Bucket<K, V>],
    }

    impl<K, V> fmt::Debug for ParValuesMut<'_, K, V>
    where
        V: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let iter = self.entries.iter().map(Bucket::value_ref);
            debug_iter_as_list(f, Some("ParValuesMut"), iter)
        }
    }

    impl<'a, K, V> ParallelIterator for ParValuesMut<'a, K, V>
    where
        K: Send,
        V: Send,
    {
        type Item = &'a mut V;

        parallel_iterator_methods!(Bucket::value_mut);
    }

    impl<K, V> IndexedParallelIterator for ParValuesMut<'_, K, V>
    where
        K: Send,
        V: Send,
    {
        indexed_parallel_iterator_methods!(Bucket::value_mut);
    }
}

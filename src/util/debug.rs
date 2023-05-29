use core::cell::RefCell;
use core::fmt;
use core::ops::DerefMut;

pub(crate) struct DebugIterAsList<I> {
    iter: RefCell<I>,
    name: Option<&'static str>,
}

impl<I> DebugIterAsList<I> {
    pub(crate) fn new(iter: I) -> Self {
        Self {
            iter: RefCell::new(iter),
            name: None,
        }
    }

    pub(crate) fn with_name(name: &'static str, iter: I) -> Self {
        Self {
            iter: RefCell::new(iter),
            name: Some(name),
        }
    }
}

impl<I, T> fmt::Debug for DebugIterAsList<I>
where
    I: Iterator<Item = T>,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.iter.borrow_mut();
        debug_iter_as_list(f, self.name, iter.deref_mut())
    }
}

pub(crate) fn debug_iter_as_list<I>(
    f: &mut fmt::Formatter<'_>,
    name: Option<&'static str>,
    iter: I,
) -> fmt::Result
where
    I: Iterator,
    I::Item: fmt::Debug,
{
    if let Some(name) = name {
        f.write_str(name)?;
        f.write_str(" ")?;
    }
    f.debug_list().entries(iter).finish()
}

pub(crate) struct DebugIterAsNumberedCompactList<I> {
    iter: RefCell<I>,
    name: Option<&'static str>,
}

impl<I> DebugIterAsNumberedCompactList<I> {
    pub(crate) fn new(iter: I) -> Self {
        Self {
            iter: RefCell::new(iter),
            name: None,
        }
    }

    pub(crate) fn with_name(name: &'static str, iter: I) -> Self {
        Self {
            iter: RefCell::new(iter),
            name: Some(name),
        }
    }
}

impl<I, T> fmt::Debug for DebugIterAsNumberedCompactList<I>
where
    I: Iterator<Item = T>,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.iter.borrow_mut();
        debug_iter_as_numbered_compact_list(f, self.name, iter.deref_mut())
    }
}

pub(crate) fn debug_iter_as_numbered_compact_list<I>(
    f: &mut fmt::Formatter<'_>,
    name: Option<&'static str>,
    iter: I,
) -> fmt::Result
where
    I: Iterator,
    I::Item: fmt::Debug,
{
    if let Some(name) = name {
        f.write_str(name)?;
        f.write_str(" ")?;
    }
    let mut list = f.debug_list();
    for (i, it) in iter.enumerate() {
        list.entry(&format_args!("{i}: {it:?}"));
    }
    list.finish()
}

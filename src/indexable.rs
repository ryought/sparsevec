//!
//! `Indexable`
//!
//! Trait to abstract types that can be used as an index of SparseVec
//!
//! Basically any type convertable from/into usize can satisfy this trait.
//! (e.g. usize itself, petgraph's NodeIndex and EdgeIndex)
//!
//! When accessing vector by ix: Indexable, ix.index(): usize is used.
//!
pub use petgraph::graph::{EdgeIndex, NodeIndex};

///
/// Trait to abstract types that can be used as an index of SparseVec
///
pub trait Indexable: Copy + Eq + std::hash::Hash + std::fmt::Debug {
    ///
    /// from usize
    ///
    fn new(x: usize) -> Self;
    ///
    /// into usize
    ///
    fn index(&self) -> usize;
}

impl Indexable for usize {
    #[inline]
    fn new(x: usize) -> Self {
        x
    }
    #[inline]
    fn index(&self) -> usize {
        *self
    }
}

impl Indexable for NodeIndex {
    #[inline]
    fn new(x: usize) -> Self {
        NodeIndex::new(x)
    }
    #[inline]
    fn index(&self) -> usize {
        // Using this notation to call original method.
        // self.index() will create recursive call.
        NodeIndex::index(*self)
    }
}

impl Indexable for EdgeIndex {
    #[inline]
    fn new(x: usize) -> Self {
        EdgeIndex::new(x)
    }
    #[inline]
    fn index(&self) -> usize {
        // Using this notation to call original method.
        // self.index() will create recursive call.
        EdgeIndex::index(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexable() {
        let x: usize = Indexable::new(11);
        assert_eq!(x.index(), 11);
        let x: NodeIndex = Indexable::new(9);
        assert_eq!(x.index(), 9);
        let x: EdgeIndex = Indexable::new(9);
        assert_eq!(x.index(), 9);
    }
}

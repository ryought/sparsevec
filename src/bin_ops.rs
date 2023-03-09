//!
//! Binary operations between two SparseVecs
//!

use super::*;

//
// Math ops
//

impl<'a, 'b, T, Ix, const N: usize> std::ops::Add<&'a SparseVec<T, Ix, N>>
    for &'b SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + std::ops::Add,
    Ix: Indexable,
{
    type Output = SparseVec<T, Ix, N>;
    fn add(self, other: &'a SparseVec<T, Ix, N>) -> Self::Output {
        unimplemented!();
    }
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math() {}
}

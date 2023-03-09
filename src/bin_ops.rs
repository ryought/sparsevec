//!
//! Binary operations between two SparseVecs
//!
//! # Add
//!
//! Addition between two references of SparseVecs `v+w`
//!
//! * Either of v or w is Dense, return Dense
//! * Both v and w is Sparse, return Sparse containing top N elements
//!
//! # AddAssign
//!
//! `v += w`
//!
//! * v is Dense, straightforward.
//! * v is Sparse
//!     * w is Dense, it will panic.
//!     * w is also Sparse
//!

use super::*;

//
// Math ops
//

impl<'a, 'b, T, Ix, const N: usize> std::ops::Add<&'a SparseVec<T, Ix, N>>
    for &'b SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T>,
    Ix: Indexable,
{
    type Output = SparseVec<T, Ix, N>;
    fn add(self, other: &'a SparseVec<T, Ix, N>) -> Self::Output {
        assert_eq!(self.len(), other.len(), "size is different");
        if self.is_dense() || other.is_dense() {
            let mut ret = SparseVec::new_dense(self.len(), self.default_element());
            for i in 0..self.len() {
                let ix = Ix::new(i);
                ret[ix] = self[ix] + other[ix];
            }
            ret
        } else {
            unimplemented!();
        }
    }
}

impl<'a, T, Ix, const N: usize> std::ops::AddAssign<&'a SparseVec<T, Ix, N>> for SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + std::ops::AddAssign,
    Ix: Indexable,
{
    fn add_assign(&mut self, other: &'a SparseVec<T, Ix, N>) {
        assert_eq!(self.len(), other.len(), "size is different");
        if self.is_dense() {
            for i in 0..self.len() {
                let ix = Ix::new(i);
                self[ix] += other[ix];
            }
        } else {
            unimplemented!();
        }
    }
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let mut a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(vec![5, 4, 3, 1], 0);
        let b: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(vec![1, 1, 3, 1], 0);
        let c = &a + &b;
        assert_eq!(c.to_vec(), vec![6, 5, 6, 2]);

        a += &b;
        assert_eq!(a.to_vec(), vec![6, 5, 6, 2]);
    }
}

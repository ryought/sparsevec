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

impl<'a, 'b, T, Ix, const N: usize> std::ops::Mul<&'a SparseVec<T, Ix, N>>
    for &'b SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + std::ops::Mul<Output = T>,
    Ix: Indexable,
{
    type Output = SparseVec<T, Ix, N>;
    fn mul(self, other: &'a SparseVec<T, Ix, N>) -> Self::Output {
        assert_eq!(self.len(), other.len(), "size is different");
        if self.is_dense() || other.is_dense() {
            let mut ret = SparseVec::new_dense(self.len(), self.default_element());
            for i in 0..self.len() {
                let ix = Ix::new(i);
                ret[ix] = self[ix] * other[ix];
            }
            ret
        } else {
            unimplemented!();
        }
    }
}

impl<T, Ix, const N: usize> std::ops::Div<T> for SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + std::ops::Div<Output = T>,
    Ix: Indexable,
{
    type Output = SparseVec<T, Ix, N>;
    fn div(self, other: T) -> Self::Output {
        match self {
            SparseVec::Dense(mut vec, default_element) => {
                for i in 0..vec.len() {
                    vec[i] = vec[i] / other;
                }
                SparseVec::Dense(vec, default_element)
            }
            SparseVec::Sparse(mut elements, default_element, len) => {
                for i in 0..elements.len() {
                    let (index, value) = elements[i];
                    elements[i] = (index, value / other);
                }

                SparseVec::Sparse(elements, default_element / other, len)
            }
        }
    }
}

impl<T, Ix, const N: usize> std::iter::Sum for SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + std::ops::AddAssign,
    Ix: Indexable,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|mut a, b| {
            a += &b;
            a
        })
        .unwrap()
    }
}

impl<T, Ix, const N: usize> SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd,
    Ix: Indexable,
{
    ///
    ///
    ///
    pub fn sum_stable(vecs: &[SparseVec<T, Ix, N>]) {
        unimplemented!();
    }
}

impl<T, Ix, const N: usize> SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd,
    Ix: Indexable,
{
    ///
    ///
    /// # Example
    ///
    /// ## f64
    ///
    /// * L1 norm of two vecs, f = |a-b|
    /// * L2 norm of two vecs, f = |a-b|^2
    ///
    /// ```
    /// use sparsevec::SparseVec;
    ///
    /// let va = vec![1., 2., 3., 1.];
    /// let a: SparseVec<f64, usize, 2> = SparseVec::dense_from_vec(va, 0.);
    /// let vb = vec![0., 2., 5., 0.];
    /// let b: SparseVec<f64, usize, 2> = SparseVec::sparse_from_slice(&vb, 0.);
    ///
    /// let d1 = a.diff_by(&b, |a, b| (a - b).abs());
    /// assert_eq!(d1, 4.);
    /// let d2 = a.diff_by(&b, |a, b| (a - b).powi(2));
    /// assert_eq!(d2, 6.);
    /// ```
    ///
    ///
    /// ## usize
    ///
    /// ```
    /// use sparsevec::SparseVec;
    ///
    /// let va = vec![1, 2, 3, 1];
    /// let a: SparseVec<usize, usize, 2> = SparseVec::dense_from_vec(va, 0);
    /// let vb = vec![0, 2, 5, 0];
    /// let b: SparseVec<usize, usize, 2> = SparseVec::sparse_from_slice(&vb, 0);
    ///
    /// let d1 = a.diff_by(&b, |a, b| a.abs_diff(b));
    /// assert_eq!(d1, 4);
    /// let d2 = a.diff_by(&b, |a, b| a.abs_diff(b).pow(2));
    /// assert_eq!(d2, 6);
    /// ```
    ///
    pub fn diff_by<F, S>(&self, other: &SparseVec<T, Ix, N>, f: F) -> S
    where
        S: Default + std::ops::AddAssign,
        F: Fn(T, T) -> S,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "diff of two vecs of different sizes cannot be calculated"
        );

        let mut ret = S::default();
        for i in 0..self.len() {
            let index = Ix::new(i);
            ret += f(self[index], other[index]);
        }
        ret
    }
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_mul() {
        {
            let mut a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(vec![5, 4, 3, 1], 0);
            let b: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(vec![1, 1, 3, 1], 0);
            let c = &a + &b;
            assert_eq!(c.to_vec(), vec![6, 5, 6, 2]);

            a += &b;
            assert_eq!(a.to_vec(), vec![6, 5, 6, 2]);
        }

        {
            let a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(vec![5, 4, 3, 1], 0);
            let b: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(vec![1, 1, 3, 1], 0);
            let d = &a * &b;
            println!("d={}", d);
            assert_eq!(d.to_vec(), vec![5, 4, 9, 1]);
        }
    }
    #[test]
    fn sum_and_div() {
        let vs: Vec<_> = vec![
            SparseVec::dense_from_vec(vec![5, 4, 3, 1], 0),
            SparseVec::dense_from_vec(vec![5, 4, 3, 1], 0),
            SparseVec::dense_from_vec(vec![5, 4, 3, 1], 0),
            SparseVec::dense_from_vec(vec![5, 4, 3, 1], 0),
        ];
        let s: SparseVec<u8, usize, 2> = vs.into_iter().sum();
        println!("s={}", s);
        assert_eq!(s.clone().to_vec(), vec![20, 16, 12, 4]);

        let x = s / 4;
        println!("x={}", x);
        assert_eq!(x.clone().to_vec(), vec![5, 4, 3, 1]);
    }
}

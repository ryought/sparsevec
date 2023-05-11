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
    T: Copy + PartialOrd + num_traits::Zero,
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
            let mut ret =
                SparseVec::new_sparse(self.len(), self.default_element() + other.default_element());
            for (index, _) in self.iter() {
                ret[index] = self[index] + other[index];
            }
            for (index, _) in other.iter() {
                ret[index] = self[index] + other[index];
            }
            ret
        }
    }
}

impl<'a, T, Ix, const N: usize> std::ops::AddAssign<&'a SparseVec<T, Ix, N>> for SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + num_traits::Zero + std::ops::AddAssign,
    Ix: Indexable,
{
    fn add_assign(&mut self, other: &'a SparseVec<T, Ix, N>) {
        assert_eq!(self.len(), other.len(), "size is different");
        if self.is_dense() {
            if other.default_element().is_zero() {
                for (ix, x) in other.iter() {
                    self[ix] += x
                }
            } else {
                for i in 0..self.len() {
                    let ix = Ix::new(i);
                    self[ix] += other[ix];
                }
            }
        } else {
            if other.is_dense() {
                panic!("cannot AddAssign dense into sparse")
            }
            let mut default_element = self.default_element();
            default_element += other.default_element();
            let mut ret = SparseVec::new_sparse(self.len(), default_element);
            // self
            for (index, _) in self.iter() {
                let mut x = self[index];
                x += other[index];
                ret[index] = x;
            }
            // other
            for (index, _) in other.iter() {
                let mut x = other[index];
                x += self[index];
                ret[index] = x;
            }
            *self = ret;
        }
    }
}

impl<'a, 'b, T, Ix, const N: usize> std::ops::Mul<&'a SparseVec<T, Ix, N>>
    for &'b SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + num_traits::One,
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
            let mut ret =
                SparseVec::new_sparse(self.len(), self.default_element() * other.default_element());
            for (index, _) in self.iter() {
                ret[index] = self[index] * other[index];
            }
            for (index, _) in other.iter() {
                ret[index] = self[index] * other[index];
            }
            ret
        }
    }
}

impl<'a, T, Ix, const N: usize> std::ops::MulAssign<&'a SparseVec<T, Ix, N>> for SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + num_traits::One + std::ops::MulAssign,
    Ix: Indexable,
{
    fn mul_assign(&mut self, other: &'a SparseVec<T, Ix, N>) {
        assert_eq!(self.len(), other.len(), "size is different");
        if self.is_dense() {
            if other.default_element().is_one() {
                for (ix, x) in other.iter() {
                    self[ix] *= x
                }
            } else {
                for i in 0..self.len() {
                    let ix = Ix::new(i);
                    self[ix] *= other[ix];
                }
            }
        } else {
            if other.is_dense() {
                panic!("cannot MulAssign dense into sparse")
            }
            let mut default_element = self.default_element();
            default_element *= other.default_element();
            let mut ret = SparseVec::new_sparse(self.len(), default_element);
            // self
            for (index, _) in self.iter() {
                let mut x = self[index];
                x *= other[index];
                ret[index] = x;
            }
            // other
            for (index, _) in other.iter() {
                let mut x = other[index];
                x *= self[index];
                ret[index] = x;
            }
            *self = ret;
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
    T: Copy + PartialOrd + num_traits::Zero + std::ops::AddAssign,
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
    #[test]
    fn add_and_addassign_between_sparse() {
        let a: SparseVec<u8, usize, 3> = SparseVec::sparse_from_slice(&[5, 6, 1, 3, 0, 0, 0], 0);
        let b: SparseVec<u8, usize, 3> = SparseVec::sparse_from_slice(&[0, 0, 3, 4, 7, 5, 0], 0);
        let c = &a + &b;
        println!("{}", a);
        println!("{}", b);
        println!("{}", c);
        assert_eq!(a.clone().to_vec(), vec![5, 6, 0, 3, 0, 0, 0]);
        assert_eq!(b.clone().to_vec(), vec![0, 0, 0, 4, 7, 5, 0]);
        assert_eq!(c.clone().to_vec(), vec![0, 0, 0, 7, 7, 5, 0]);

        let mut c: SparseVec<u8, usize, 3> = SparseVec::new_dense(a.len(), 0);
        c += &a;
        c += &b;
        println!("{}", c);
        assert_eq!(c.clone().to_vec(), vec![5, 6, 0, 7, 7, 5, 0]); // top-(N-1)=2 elements are
                                                                   // keeped

        // non-zero default element
        let a: SparseVec<u8, usize, 3> = SparseVec::sparse_from_slice(&[5, 6, 1, 3, 0, 0, 0], 2);
        let b: SparseVec<u8, usize, 3> = SparseVec::sparse_from_slice(&[0, 0, 3, 4, 7, 5, 0], 1);
        let c = &a + &b;
        let mut d: SparseVec<u8, usize, 3> = SparseVec::new_dense(a.len(), 0);
        d += &a;
        d += &b;
        println!("{}", a);
        println!("{}", b);
        println!("{}", c);
        println!("{}", d);
        assert_eq!(a.clone().to_vec(), vec![5, 6, 2, 3, 2, 2, 2]);
        assert_eq!(b.clone().to_vec(), vec![1, 1, 1, 4, 7, 5, 1]);
        assert_eq!(c.clone().to_vec(), vec![3, 3, 3, 7, 9, 7, 3]);
        assert_eq!(d.clone().to_vec(), vec![6, 7, 3, 7, 9, 7, 3]);
    }
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use test::Bencher;
    fn random_vec(is_dense: bool, n: usize, m: usize, seed: u64) -> SparseVec<usize, usize, 10> {
        let mut ret = SparseVec::new(n, 0, is_dense);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        for _ in 0..m {
            let i: usize = rng.gen_range(1..n);
            ret[i] = rng.gen_range(1..10);
        }
        ret
    }
    #[bench]
    fn addassign_benchmark(b: &mut Bencher) {
        let n = 1_000;
        let mut x = random_vec(true, n, 10, 0);
        let y = random_vec(false, n, 10, 1);
        let z = random_vec(true, n, 10, 2);
        println!("x={}", x);
        println!("y={}", y);
        b.iter(|| {
            x += &y;
        });
        println!("x={}", x);
    }
}

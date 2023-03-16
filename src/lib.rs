//!
//! `SparseVec`
//!
//! Sparse vector which stores only top N largest elements
//!

// indexable
pub mod indexable;
pub use indexable::Indexable;

// binary operations
pub mod bin_ops;

use arrayvec::ArrayVec;

///
/// Sparse vector which stores only top N largest elements
///
/// # Features
///
/// * Index access
/// * Iterator on registered element
/// * Vec<T> conversion
/// * Display
/// * Conversion between Dense and Sparse
///
/// # Todo
/// * Binary operations
///     * Math operations (Add Sub)
///     * Calculate diff between two vecs
///
#[derive(Clone, Debug)]
pub enum SparseVec<T: Copy + PartialOrd, Ix: Indexable, const N: usize> {
    Dense(Vec<T>, T),
    Sparse(ArrayVec<(Ix, T), N>, T, usize),
}

///
/// Public functions of SparseVec
///
impl<T: Copy + PartialOrd, Ix: Indexable, const N: usize> SparseVec<T, Ix, N> {
    ///
    /// Construct SparseVec::Dense
    ///
    /// ```
    /// use sparsevec::SparseVec;
    /// let mut a: SparseVec<u8, usize, 2> = SparseVec::new_dense(5, 0);
    /// a[0] = 20;
    /// a[3] = 10;
    /// assert_eq!(a.to_vec(), vec![20, 0, 0, 10, 0]);
    /// ```
    pub fn new_dense(len: usize, default_element: T) -> Self {
        SparseVec::Dense(vec![default_element; len], default_element)
    }
    ///
    /// Construct SparseVec::Sparse
    ///
    /// ```
    /// use sparsevec::SparseVec;
    /// let mut a: SparseVec<u8, usize, 2> = SparseVec::new_sparse(5, 0);
    /// a[0] = 20;
    /// a[3] = 10;
    /// assert_eq!(a.to_vec(), vec![20, 0, 0, 10, 0]);
    ///
    /// let mut b: SparseVec<u8, usize, 3> = SparseVec::new_sparse(5, 1);
    /// b[0] = 20;
    /// b[3] = 10;
    /// b[4] = 5;
    /// b[2] = 8;
    /// assert_eq!(b.to_vec(), vec![20, 1, 8, 10, 1]);
    ///
    /// // Top N-1 element will be preserved in SparseVec.
    /// let mut b: SparseVec<u8, usize, 2> = SparseVec::new_sparse(5, 1);
    /// b[0] = 20;
    /// b[3] = 10;
    /// b[4] = 5;
    /// assert_eq!(b.to_vec(), vec![20, 1, 1, 1, 5]);
    /// ```
    pub fn new_sparse(len: usize, default_element: T) -> Self {
        SparseVec::Sparse(ArrayVec::<(Ix, T), N>::new(), default_element, len)
    }
    ///
    /// Construct SparseVec::Dense or SparseVec::Sparse depending on argument `is_dense`.
    ///
    pub fn new(len: usize, default_element: T, is_dense: bool) -> Self {
        if is_dense {
            SparseVec::new_dense(len, default_element)
        } else {
            SparseVec::new_sparse(len, default_element)
        }
    }
    ///
    /// Construct SparseVec::Dense from Vec<T>
    /// Reuse the vector as inner container of Dense
    ///
    /// ```
    /// use sparsevec::SparseVec;
    /// let v = vec![5, 4, 3, 2];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(v, 0);
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(a.to_vec(), vec![5, 4, 3, 2]);
    /// ```
    pub fn dense_from_vec(vec: Vec<T>, default_element: T) -> Self {
        SparseVec::Dense(vec, default_element)
    }
    ///
    /// Construct SparseVec::Sparse from &[T]
    /// Store top N largest elements in the slice.
    ///
    /// ```
    /// use sparsevec::SparseVec;
    /// let v = vec![5, 2, 0, 4];
    ///
    /// let a: SparseVec<u8, usize, 2> = SparseVec::sparse_from_slice(&v, 0);
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(a.to_vec(), vec![5, 0, 0, 4]);
    ///
    /// let a: SparseVec<u8, usize, 4> = SparseVec::sparse_from_slice(&v, 0);
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(a.to_vec(), vec![5, 2, 0, 4]);
    /// ```
    pub fn sparse_from_slice(slice: &[T], default_element: T) -> Self {
        let mut a = ArrayVec::new();
        for (index, &element) in slice.into_iter().enumerate() {
            if element != default_element {
                if a.len() < N {
                    a.push((Ix::new(index), element));
                } else {
                    // array is full
                    let i = get_min_elem(&a).unwrap();
                    // the minimum element is smaller than current element, swap them.
                    if a[i].1 < element {
                        a[i] = (Ix::new(index), element);
                    }
                }
            }
        }
        SparseVec::Sparse(a, default_element, slice.len())
    }
    ///
    /// Length of vector
    ///
    pub fn len(&self) -> usize {
        match self {
            SparseVec::Dense(vec, _) => vec.len(),
            SparseVec::Sparse(_, _, len) => *len,
        }
    }
    ///
    /// Dense or Sparse?
    ///
    pub fn is_dense(&self) -> bool {
        match self {
            SparseVec::Dense(_, _) => true,
            _ => false,
        }
    }
    ///
    /// Get default element of SparseVec
    ///
    pub fn default_element(&self) -> T {
        match self {
            SparseVec::Dense(_, d) => *d,
            SparseVec::Sparse(_, d, _) => *d,
        }
    }
    ///
    /// Convert to dense
    ///
    /// ```
    /// use sparsevec::SparseVec;
    ///
    /// let v = vec![5, 0, 0, 4];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::sparse_from_slice(&v, 0);
    /// assert_eq!(a.len(), 4);
    /// let b = a.to_dense();
    /// assert_eq!(b.len(), 4);
    /// assert_eq!(b.to_vec(), v);
    /// ```
    pub fn to_dense(self) -> Self {
        match self {
            // if Dense, return as it is
            SparseVec::Dense(v, d) => SparseVec::Dense(v, d),
            // if Sparse, create Vec and fill the elements
            SparseVec::Sparse(elements, default_element, len) => {
                let mut vec = vec![default_element; len];
                for (index, element) in elements {
                    vec[index.index()] = element;
                }
                SparseVec::Dense(vec, default_element)
            }
        }
    }
    ///
    /// Convert to sparse
    ///
    /// If Dense, N biggest elements are stored in the resulting SparseVec.
    ///
    /// ```
    /// use sparsevec::SparseVec;
    ///
    /// let v = vec![5, 4, 7, 2];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(v, 0);
    /// let b = a.to_sparse();
    /// assert_eq!(b.len(), 4);
    /// assert_eq!(b.to_vec(), vec![5, 0, 7, 0]);
    /// ```
    pub fn to_sparse(self) -> Self {
        match self {
            // if Sparse, return as it is
            SparseVec::Sparse(e, d, l) => SparseVec::Sparse(e, d, l),
            // if Dense, pick N biggest elements
            SparseVec::Dense(vec, default_element) => {
                SparseVec::sparse_from_slice(&vec, default_element)
            }
        }
    }
    ///
    /// Eject Vec<T> from SparseVec::Dense
    ///
    /// If Sparse, first converted to Dense
    ///
    /// ```
    /// use sparsevec::SparseVec;
    ///
    /// let v = vec![5, 4, 7, 2];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(v, 0);
    /// assert_eq!(a.to_vec(), vec![5, 4, 7, 2]);
    /// ```
    pub fn to_vec(self) -> Vec<T> {
        match self.to_dense() {
            SparseVec::Dense(v, _) => v,
            _ => unreachable!(),
        }
    }
    ///
    /// Change a type of index (Ix: Indexable)
    ///
    /// ```
    /// use sparsevec::SparseVec;
    /// use petgraph::graph::NodeIndex;
    ///
    /// let v = vec![5, 4, 7, 2];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(v, 0);
    /// assert_eq!(a[0], 5);
    /// let b: SparseVec<u8, NodeIndex, 2> = a.switch_index();
    /// assert_eq!(b[NodeIndex::new(0)], 5);
    /// assert_eq!(b.to_vec(), vec![5, 4, 7, 2]);
    ///
    /// let v = vec![5, 4, 7, 2];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::sparse_from_slice(&v, 0);
    /// assert_eq!(a[0], 5);
    /// let b: SparseVec<u8, NodeIndex, 2> = a.switch_index();
    /// assert_eq!(b[NodeIndex::new(0)], 5);
    /// assert_eq!(b.to_vec(), vec![5, 0, 7, 0]);
    /// ```
    pub fn switch_index<Ix2: Indexable>(self) -> SparseVec<T, Ix2, N> {
        match self {
            SparseVec::Sparse(e, d, l) => {
                let mut e2 = ArrayVec::new();
                for (index, element) in e {
                    e2.push((Ix2::new(index.index()), element));
                }
                SparseVec::Sparse(e2, d, l)
            }
            SparseVec::Dense(v, d) => SparseVec::Dense(v, d),
        }
    }
}

impl<T: Copy + PartialOrd + std::iter::Sum, Ix: Indexable, const N: usize> SparseVec<T, Ix, N> {
    ///
    /// Sum of all elements
    ///
    pub fn sum(&self) -> T {
        (0..self.len()).map(|i| self[Ix::new(i)]).sum()
    }
}

///
/// Find smallest element in ArrayVec<(_, T)>
///
/// If no element is registered, return None.
/// Otherwise, return the index of elements array.
/// Note that this is not an index of global SparseVec.
///
fn get_min_elem<Ix, T, const N: usize>(array: &ArrayVec<(Ix, T), N>) -> Option<usize>
where
    T: PartialOrd + Copy,
{
    if array.len() == 0 {
        None
    } else {
        let mut i_min = 0;
        let (_, mut x_min) = array[0];
        for i in 1..array.len() {
            let (_, x) = array[i];

            // If a smaller element is found, update i_min and x_min.
            if x < x_min {
                i_min = i;
                x_min = x;
            }
        }
        Some(i_min)
    }
}

//
// Index
//

impl<T: Copy + PartialOrd, Ix: Indexable, const N: usize> std::ops::Index<Ix>
    for SparseVec<T, Ix, N>
{
    type Output = T;
    fn index(&self, index: Ix) -> &Self::Output {
        match self {
            SparseVec::Dense(vec, _) => &vec[index.index()],
            SparseVec::Sparse(elements, default_element, _) => {
                assert!(index.index() < self.len(), "index out of size");

                for i in 0..elements.len() {
                    if elements[i].0 == index {
                        return &elements[i].1;
                    }
                }

                // not found
                &default_element
            }
        }
    }
}

impl<T: Copy + PartialOrd, Ix: Indexable, const N: usize> std::ops::IndexMut<Ix>
    for SparseVec<T, Ix, N>
{
    fn index_mut(&mut self, index: Ix) -> &mut Self::Output {
        match self {
            SparseVec::Dense(vec, _) => &mut vec[index.index()],
            SparseVec::Sparse(elements, default_element, len) => {
                assert!(index.index() < *len, "index out of size");
                // search for existing entries
                for i in 0..elements.len() {
                    if elements[i].0 == index {
                        return &mut elements[i].1;
                    }
                }

                // If full, delete smallest element and write onto it.
                if elements.len() == N {
                    let i = get_min_elem(elements).unwrap();
                    elements[i].0 = index;
                    return &mut elements[i].1;
                }

                // add a new entry and return the reference to it
                elements.push((index, *default_element));
                let n = elements.len();
                return &mut elements[n - 1].1;
            }
        }
    }
}

//
// Iterator
//

///
/// Iterator of SparseVec that iterates over the registered elements
///
pub struct SparseVecIterator<'a, T: Copy + PartialOrd, Ix: Indexable, const N: usize> {
    ///
    /// Reference of the original SparseVec
    ///
    sparsevec: &'a SparseVec<T, Ix, N>,
    ///
    /// Index of element to be produced next
    ///
    i: usize,
}

impl<'a, T: Copy + PartialOrd, Ix: Indexable, const N: usize> Iterator
    for SparseVecIterator<'a, T, Ix, N>
{
    type Item = (Ix, T);
    fn next(&mut self) -> Option<Self::Item> {
        match self.sparsevec {
            SparseVec::Dense(v, _) => {
                let i = self.i;
                if i < v.len() {
                    self.i += 1;
                    Some((Ix::new(i), v[i]))
                } else {
                    None
                }
            }
            SparseVec::Sparse(e, _, _) => {
                let i = self.i;
                if i < e.len() {
                    self.i += 1;
                    Some(e[i])
                } else {
                    None
                }
            }
        }
    }
}

impl<T: Copy + PartialOrd, Ix: Indexable, const N: usize> SparseVec<T, Ix, N> {
    ///
    /// Get iterator over registered elements `(Ix, T)`.
    ///
    /// # Dense
    /// (0, v[0]), (1, v[1]), (2, v[2]), ... will be produced.
    ///
    /// ```
    /// use sparsevec::SparseVec;
    /// let v = vec![5, 4, 3, 2];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::dense_from_vec(v, 0);
    /// let w: Vec<(usize, u8)> = a.iter().collect();
    /// assert_eq!(w, vec![(0, 5), (1, 4), (2, 3), (3, 2)]);
    /// ```
    ///
    /// # Sparse
    /// (i, v[i]) for registered index i will be produced.
    ///
    /// ```
    /// use sparsevec::SparseVec;
    /// let v = vec![5, 4, 3, 2];
    /// let a: SparseVec<u8, usize, 2> = SparseVec::sparse_from_slice(&v, 0);
    /// let w: Vec<(usize, u8)> = a.iter().collect();
    /// assert_eq!(w, vec![(0, 5), (1, 4)]);
    /// ```
    pub fn iter<'a>(&'a self) -> SparseVecIterator<'a, T, Ix, N> {
        SparseVecIterator {
            sparsevec: self,
            i: 0,
        }
    }
}

impl<'a, T: Copy + PartialOrd, Ix: Indexable, const N: usize> IntoIterator
    for &'a SparseVec<T, Ix, N>
{
    type Item = (Ix, T);
    type IntoIter = SparseVecIterator<'a, T, Ix, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

//
// Display
//

impl<T, Ix, const N: usize> std::fmt::Display for SparseVec<T, Ix, N>
where
    T: Copy + PartialOrd + std::fmt::Display,
    Ix: Indexable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.len() {
            if i != 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", self[Ix::new(i)])?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let mut v: SparseVec<u8, usize, 2> = SparseVec::new_dense(5, 0);
        v[2] = 100;
        assert!(v.is_dense());
        println!("{}", v);
        for (i, x) in &v {
            println!("i={} x={}", i, x);
        }
        assert_eq!(vec![0, 0, 100, 0, 0], v.to_vec());

        let mut v: SparseVec<u8, usize, 2> = SparseVec::new_sparse(5, 0);
        v[2] = 100;
        assert!(!v.is_dense());
        println!("{}", v);
        for (i, x) in &v {
            println!("i={} x={}", i, x);
        }
        assert_eq!(vec![0, 0, 100, 0, 0], v.to_vec());

        // println!("size={}", std::mem::size_of::<SparseVec<u8, usize, 2>>());
        // println!("size={}", std::mem::size_of::<SparseVec<u8, usize, 10>>());
        // println!("size={}", std::mem::size_of::<SparseVec<u8, usize, 100>>());
    }

    #[test]
    fn conversion() {
        let mut v: SparseVec<u8, usize, 2> = SparseVec::new_dense(5, 0);
        v[2] = 100;
        v[0] = 110;
        v[3] = 120;
        v[4] = 1;
        assert!(v.is_dense());
        println!("{}", v);
        assert_eq!(vec![110, 0, 100, 120, 1], v.clone().to_vec());

        let v = v.to_sparse();
        println!("{}", v);
        // assert_eq!(vec![110, 0, 0, 120, 0], v.clone().to_vec());
    }
}

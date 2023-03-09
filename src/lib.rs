//!
//! `SparseVec`
//!
//! Vector which is memory-efficient when few elements are filled.
//!

// index
pub mod indexable;
pub use indexable::Indexable;

use arrayvec::ArrayVec;

///
/// Maximum number of elements stored in SparseVec::Sparse
///
pub const SIZE: usize = 10;

///
/// For v: SparseVec, ix: Ix, x: T
/// v[ix] = x
///
/// # Features
///
/// * Index access
/// * Conversion between Dense and Sparse
/// * Iterator on registered element
/// * Math operations on SparseVec: Add Sub Sum
/// * Vec<T> conversion
/// * Display
/// * diff
///
pub enum SparseVec<T: Copy + PartialOrd, Ix: Indexable> {
    Dense(Vec<T>, T),
    Sparse(ArrayVec<(Ix, T), SIZE>, T, usize),
}

///
/// Public functions of SparseVec
///
impl<T: Copy + PartialOrd, Ix: Indexable> SparseVec<T, Ix> {
    ///
    /// Construct SparseVec::Dense
    ///
    pub fn new_dense(len: usize, default_element: T) -> Self {
        SparseVec::Dense(vec![default_element; len], default_element)
    }
    ///
    /// Construct SparseVec::Sparse
    ///
    pub fn new_sparse(len: usize, default_element: T) -> Self {
        SparseVec::Sparse(ArrayVec::<(Ix, T), SIZE>::new(), default_element, len)
    }
    ///
    ///
    ///
    pub fn new_sparse_from(vec: &[T], default_element: T) -> Self {
        unimplemented!();
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
    /// Convert to dense
    ///
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
    /// If Dense, SIZE biggest elements are stored in the resulting SparseVec.
    ///
    pub fn to_sparse(self) -> Self {
        match self {
            // if Sparse, return as it is
            SparseVec::Sparse(e, d, l) => SparseVec::Sparse(e, d, l),
            // if Dense, pick SIZE biggest elements
            SparseVec::Dense(v, d) => {
                // SparseVec::Sparse(e, d, l)
                unimplemented!();
            }
        }
    }
}

///
/// Internal functions of SparseVec
///
impl<T: Copy + PartialOrd, Ix: Indexable> SparseVec<T, Ix> {
    ///
    /// Eject Vec<T> from SparseVec::Dense
    ///
    /// If Sparse, first converted to Dense
    ///
    pub fn to_dense_vec(self) -> Vec<T> {
        match self.to_dense() {
            SparseVec::Dense(v, _) => v,
            SparseVec::Sparse(_, _, _) => unreachable!(),
        }
    }
}

///
/// Find smallest element in ArrayVec<(_, T)>
///
/// If no element is registered, return None.
/// Otherwise, return the index of elements array.
/// Note that this is not an index of global SparseVec.
///
fn get_min_elem<Ix, T: PartialOrd + Copy>(array: &ArrayVec<(Ix, T), SIZE>) -> Option<usize> {
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

impl<T: Copy + PartialOrd, Ix: Indexable> std::ops::Index<Ix> for SparseVec<T, Ix> {
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

impl<T: Copy + PartialOrd, Ix: Indexable> std::ops::IndexMut<Ix> for SparseVec<T, Ix> {
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
                if elements.len() == SIZE {
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
pub struct SparseVecIterator<'a, T: Copy + PartialOrd, Ix: Indexable> {
    ///
    /// Reference of the original SparseVec
    ///
    sparsevec: &'a SparseVec<T, Ix>,
    ///
    /// Index of element to be produced next
    ///
    i: usize,
}

impl<'a, T: Copy + PartialOrd, Ix: Indexable> Iterator for SparseVecIterator<'a, T, Ix> {
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

impl<T: Copy + PartialOrd, Ix: Indexable> SparseVec<T, Ix> {
    ///
    /// Get iterator over registered elements `(Ix, T)`.
    ///
    pub fn iter<'a>(&'a self) -> SparseVecIterator<'a, T, Ix> {
        SparseVecIterator {
            sparsevec: self,
            i: 0,
        }
    }
}

impl<'a, T: Copy + PartialOrd, Ix: Indexable> IntoIterator for &'a SparseVec<T, Ix> {
    type Item = (Ix, T);
    type IntoIter = SparseVecIterator<'a, T, Ix>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

//
// Math ops
//

impl<'a, 'b, T, Ix> std::ops::Add<&'a SparseVec<T, Ix>> for &'b SparseVec<T, Ix>
where
    T: Copy + PartialOrd + std::ops::Add,
    Ix: Indexable,
{
    type Output = SparseVec<T, Ix>;
    fn add(self, other: &'a SparseVec<T, Ix>) -> Self::Output {
        unimplemented!();
    }
}

//
// Display
//

impl<T, Ix> std::fmt::Display for SparseVec<T, Ix>
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
        let mut v: SparseVec<u8, usize> = SparseVec::new_dense(5, 0);
        v[2] = 100;
        assert!(v.is_dense());
        println!("{}", v);
        assert_eq!(vec![0, 0, 100, 0, 0], v.to_dense_vec());

        let mut v: SparseVec<u8, usize> = SparseVec::new_sparse(5, 0);
        v[2] = 100;
        assert!(!v.is_dense());
        println!("{}", v);
        assert_eq!(vec![0, 0, 100, 0, 0], v.to_dense_vec());
    }
}

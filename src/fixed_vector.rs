use crate::tree_hash::vec_tree_hash_root;
use crate::Error;
use derivative::Derivative;
use serde_derive::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::SliceIndex;
use tree_hash::Hash256;
use typenum::Unsigned;

pub use typenum;

/// Emulates a SSZ `Vector` (distinct from a Rust `Vec`).
///
/// An ordered, heap-allocated, fixed-length, homogeneous collection of `T`, with `N` values.
///
/// This struct is backed by a Rust `Vec` but constrained such that it must be instantiated with a
/// fixed number of elements and you may not add or remove elements, only modify.
///
/// The length of this struct is fixed at the type-level using
/// [typenum](https://crates.io/crates/typenum).
///
/// ## Note
///
/// Whilst it is possible with this library, SSZ declares that a `FixedVector` with a length of `0`
/// is illegal.
///
/// ## Example
///
/// ```
/// use ssz_types::{FixedVector, typenum};
///
/// let base: Vec<u64> = vec![1, 2, 3, 4];
///
/// // Create a `FixedVector` from a `Vec` that has the expected length (an incorrect length will fail):
/// let exact: FixedVector<_, typenum::U4> = base.clone().try_into().expect("length is valid");
/// assert_eq!(&exact[..], &[1, 2, 3, 4]);
/// ```
#[derive(Serialize, Deserialize, Derivative)]
#[derivative(
    Debug(bound = "T: ::core::fmt::Debug"),
    Clone(bound = "T: ::core::clone::Clone"),
    PartialEq(bound = "T: ::core::cmp::PartialEq"),
    Hash(bound = "T: ::core::hash::Hash")
)]
#[serde(transparent)]
pub struct FixedVector<T, N> {
    vec: Vec<T>,
    _phantom: PhantomData<N>,
}

impl<T, N: Unsigned> FixedVector<T, N> {
    /// Returns `Ok` if the given `vec` equals the fixed length of `Self`. Otherwise returns
    /// `Err`.
    pub fn new(vec: Vec<T>) -> Result<Self, Error> {
        if vec.len() == Self::capacity() {
            Ok(Self {
                vec,
                _phantom: PhantomData,
            })
        } else {
            Err(Error::OutOfBounds {
                i: vec.len(),
                len: Self::capacity(),
            })
        }
    }

    /// Create a new vector filled with clones of `elem`.
    pub fn from_elem(elem: T) -> Self
    where
        T: Clone,
    {
        Self {
            vec: vec![elem; N::to_usize()],
            _phantom: PhantomData,
        }
    }

    /// Identical to `self.capacity`, returns the type-level constant length.
    ///
    /// Exists for compatibility with `Vec`.
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    /// True if the type-level constant length of `self` is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the type-level constant length.
    pub fn capacity() -> usize {
        N::to_usize()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryFromVecError {
    InvalidLength { expected: usize, found: usize },
}

impl<T, N: Unsigned> TryFrom<Vec<T>> for FixedVector<T, N> {
    type Error = TryFromVecError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        if value.len() == N::USIZE {
            Ok(Self {
                vec: value,
                _phantom: PhantomData,
            })
        } else {
            Err(TryFromVecError::InvalidLength {
                expected: N::USIZE,
                found: value.len(),
            })
        }
    }
}

impl<T, N: Unsigned> From<FixedVector<T, N>> for Vec<T> {
    fn from(vector: FixedVector<T, N>) -> Vec<T> {
        vector.vec
    }
}

impl<T: Default, N: Unsigned> Default for FixedVector<T, N> {
    fn default() -> Self {
        Self {
            vec: (0..N::to_usize()).map(|_| T::default()).collect(),
            _phantom: PhantomData,
        }
    }
}

impl<T, N: Unsigned, I: SliceIndex<[T]>> Index<I> for FixedVector<T, N> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.vec, index)
    }
}

impl<T, N: Unsigned, I: SliceIndex<[T]>> IndexMut<I> for FixedVector<T, N> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.vec, index)
    }
}

impl<T, N: Unsigned> Deref for FixedVector<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.vec[..]
    }
}

// This implementation is required to use `get_mut` to access elements.
//
// It's safe because none of the methods on mutable slices allow changing the length
// of the backing vec.
impl<T, N: Unsigned> DerefMut for FixedVector<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.vec[..]
    }
}

impl<'a, T, N: Unsigned> IntoIterator for &'a FixedVector<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, N: Unsigned> tree_hash::TreeHash for FixedVector<T, N>
where
    T: tree_hash::TreeHash,
{
    fn tree_hash_type() -> tree_hash::TreeHashType {
        tree_hash::TreeHashType::Vector
    }

    fn tree_hash_packed_encoding(&self) -> tree_hash::PackedEncoding {
        unreachable!("Vector should never be packed.")
    }

    fn tree_hash_packing_factor() -> usize {
        unreachable!("Vector should never be packed.")
    }

    fn tree_hash_root(&self) -> Hash256 {
        vec_tree_hash_root::<T, N>(&self.vec)
    }
}

impl<T, N: Unsigned> ssz::Encode for FixedVector<T, N>
where
    T: ssz::Encode,
{
    fn is_ssz_fixed_len() -> bool {
        T::is_ssz_fixed_len()
    }

    fn ssz_fixed_len() -> usize {
        if <Self as ssz::Encode>::is_ssz_fixed_len() {
            T::ssz_fixed_len() * N::to_usize()
        } else {
            ssz::BYTES_PER_LENGTH_OFFSET
        }
    }

    fn ssz_bytes_len(&self) -> usize {
        ssz::sequence_ssz_bytes_len(&self.vec)
    }

    fn ssz_append(&self, buf: &mut Vec<u8>) {
        ssz::sequence_ssz_append(&self.vec, buf)
    }
}

impl<T, N: Unsigned> ssz::TryFromIter<T> for FixedVector<T, N> {
    type Error = Error;

    fn try_from_iter<I>(value: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>,
    {
        let n = N::to_usize();
        let iter = value.into_iter();

        let (_, opt_max_len) = iter.size_hint();
        let mut vec =
            Vec::with_capacity(opt_max_len.map_or(n, |max_len| std::cmp::min(n, max_len)));

        for item in iter {
            // Bail out as soon as the length tries to exceed the limit. This guards against
            // memory denial-of-service attacks.
            if vec.len() >= n {
                return Err(Error::OutOfBounds {
                    i: vec.len(),
                    len: n,
                });
            }
            vec.push(item);
        }
        Self::new(vec)
    }
}

impl<T, N: Unsigned> ssz::Decode for FixedVector<T, N>
where
    T: ssz::Decode,
{
    fn is_ssz_fixed_len() -> bool {
        T::is_ssz_fixed_len()
    }

    fn ssz_fixed_len() -> usize {
        if <Self as ssz::Decode>::is_ssz_fixed_len() {
            T::ssz_fixed_len() * N::to_usize()
        } else {
            ssz::BYTES_PER_LENGTH_OFFSET
        }
    }

    fn from_ssz_bytes(bytes: &[u8]) -> Result<Self, ssz::DecodeError> {
        let fixed_len = N::to_usize();

        if bytes.is_empty() {
            Err(ssz::DecodeError::InvalidByteLength {
                len: 0,
                expected: 1,
            })
        } else if T::is_ssz_fixed_len() {
            let num_items = bytes
                .len()
                .checked_div(T::ssz_fixed_len())
                .ok_or(ssz::DecodeError::ZeroLengthItem)?;

            if num_items != fixed_len {
                return Err(ssz::DecodeError::BytesInvalid(format!(
                    "FixedVector of {} items has {} items",
                    num_items, fixed_len
                )));
            }

            bytes
                .chunks(T::ssz_fixed_len())
                .map(|chunk| T::from_ssz_bytes(chunk))
                .collect::<Result<Vec<T>, _>>()
                .and_then(|vec| {
                    Self::new(vec).map_err(|e| {
                        ssz::DecodeError::BytesInvalid(format!(
                            "Wrong number of FixedVector elements: {:?}",
                            e
                        ))
                    })
                })
        } else {
            let vec = ssz::decode_list_of_variable_length_items(bytes, Some(fixed_len))?;
            Self::new(vec).map_err(|e| {
                ssz::DecodeError::BytesInvalid(format!(
                    "Wrong number of FixedVector elements: {:?}",
                    e
                ))
            })
        }
    }
}

#[cfg(feature = "arbitrary")]
impl<'a, T: arbitrary::Arbitrary<'a>, N: 'static + Unsigned> arbitrary::Arbitrary<'a>
    for FixedVector<T, N>
{
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let size = N::to_usize();
        let mut vec: Vec<T> = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(<T>::arbitrary(u)?);
        }
        Ok(Self::new(vec).map_err(|_| arbitrary::Error::IncorrectFormat)?)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ssz::*;
    use tree_hash::{merkle_root, TreeHash};
    use typenum::*;

    #[test]
    fn new() {
        let vec = vec![42; 5];
        let fixed: Result<FixedVector<u64, U4>, _> = FixedVector::new(vec);
        assert!(fixed.is_err());

        let vec = vec![42; 3];
        let fixed: Result<FixedVector<u64, U4>, _> = FixedVector::new(vec);
        assert!(fixed.is_err());

        let vec = vec![42; 4];
        let fixed: Result<FixedVector<u64, U4>, _> = FixedVector::new(vec);
        assert!(fixed.is_ok());
    }

    #[test]
    fn indexing() {
        let vec = vec![1, 2]
            .into_iter()
            .chain(vec![0; 8190])
            .collect::<Vec<_>>();

        let mut fixed: FixedVector<u64, U8192> = vec.clone().try_into().unwrap();

        assert_eq!(fixed[0], 1);
        assert_eq!(&fixed[0..1], &vec[0..1]);
        assert_eq!((fixed[..]).len(), 8192);

        fixed[1] = 3;
        assert_eq!(fixed[1], 3);
    }

    #[test]
    fn length() {
        assert_eq!(
            FixedVector::<u64, U4>::try_from(vec![42; 5]),
            Err(TryFromVecError::InvalidLength {
                expected: 4,
                found: 5
            })
        );

        assert_eq!(
            FixedVector::<u64, U4>::try_from(vec![42; 3]),
            Err(TryFromVecError::InvalidLength {
                expected: 4,
                found: 3
            })
        );

        assert_eq!(
            FixedVector::<u64, U4>::try_from(vec![]),
            Err(TryFromVecError::InvalidLength {
                expected: 4,
                found: 0
            })
        );
    }

    #[test]
    fn deref() {
        let vec = vec![0, 2, 4, 6];
        let fixed: FixedVector<u64, U4> = vec.try_into().unwrap();

        assert_eq!(fixed.first(), Some(&0));
        assert_eq!(fixed.get(3), Some(&6));
        assert_eq!(fixed.get(4), None);
    }

    #[test]
    fn ssz_encode() {
        let vec: FixedVector<u16, U2> = vec![0; 2].try_into().unwrap();
        assert_eq!(vec.as_ssz_bytes(), vec![0, 0, 0, 0]);
        assert_eq!(<FixedVector<u16, U2> as Encode>::ssz_fixed_len(), 4);
    }

    fn ssz_round_trip<T: Encode + Decode + std::fmt::Debug + PartialEq>(item: T) {
        let encoded = &item.as_ssz_bytes();
        assert_eq!(item.ssz_bytes_len(), encoded.len());
        assert_eq!(T::from_ssz_bytes(encoded), Ok(item));
    }

    #[test]
    fn ssz_round_trip_u16_len_8() {
        ssz_round_trip::<FixedVector<u16, U8>>(vec![42; 8].try_into().unwrap());
        ssz_round_trip::<FixedVector<u16, U8>>(vec![0; 8].try_into().unwrap());
    }

    #[test]
    fn tree_hash_u8() {
        let fixed: FixedVector<u8, U0> = vec![].try_into().unwrap();
        assert_eq!(fixed.tree_hash_root(), merkle_root(&[0; 8], 0));

        let fixed: FixedVector<u8, U1> = vec![0; 1].try_into().unwrap();
        assert_eq!(fixed.tree_hash_root(), merkle_root(&[0; 8], 0));

        let fixed: FixedVector<u8, U8> = vec![0; 8].try_into().unwrap();
        assert_eq!(fixed.tree_hash_root(), merkle_root(&[0; 8], 0));

        let fixed: FixedVector<u8, U16> = vec![42; 16].try_into().unwrap();
        assert_eq!(fixed.tree_hash_root(), merkle_root(&[42; 16], 0));

        let source: Vec<u8> = (0..16).collect();
        let fixed: FixedVector<u8, U16> = source.clone().try_into().unwrap();
        assert_eq!(fixed.tree_hash_root(), merkle_root(&source, 0));
    }

    #[derive(Clone, Copy, TreeHash, Default)]
    struct A {
        a: u32,
        b: u32,
    }

    fn repeat(input: &[u8], n: usize) -> Vec<u8> {
        let mut output = vec![];

        for _ in 0..n {
            output.append(&mut input.to_vec());
        }

        output
    }

    #[test]
    fn tree_hash_composite() {
        let a = A { a: 0, b: 1 };

        let fixed: FixedVector<A, U0> = (vec![]).try_into().unwrap();
        assert_eq!(fixed.tree_hash_root(), merkle_root(&[0; 32], 0));

        let fixed: FixedVector<A, U1> = (vec![a]).try_into().unwrap();
        assert_eq!(
            fixed.tree_hash_root(),
            merkle_root(a.tree_hash_root().as_bytes(), 0)
        );

        let fixed: FixedVector<A, U8> = (vec![a; 8]).try_into().unwrap();
        assert_eq!(
            fixed.tree_hash_root(),
            merkle_root(&repeat(a.tree_hash_root().as_bytes(), 8), 0)
        );

        let fixed: FixedVector<A, U13> = (vec![a; 13]).try_into().unwrap();
        assert_eq!(
            fixed.tree_hash_root(),
            merkle_root(&repeat(a.tree_hash_root().as_bytes(), 13), 0)
        );

        let fixed: FixedVector<A, U16> = (vec![a; 16]).try_into().unwrap();
        assert_eq!(
            fixed.tree_hash_root(),
            merkle_root(&repeat(a.tree_hash_root().as_bytes(), 16), 0)
        );
    }
}

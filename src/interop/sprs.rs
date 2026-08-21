//! Zero-copy `sprs` CSR inputs and native `sprs` outputs.
//!
//! The adapter intentionally rejects CSC input: accepting it would require a
//! transpose or a new access pattern and would no longer be a zero-copy CSR
//! integration.

use crate::auto::{AutoSpGemmConfig, AutoSpGemmStats};
use crate::error::{MatrixOperand, SpGemmError};
use crate::matrix::{CsrInput, Scalar};
use sprs::{CsMatI, CsMatViewI, SpIndex};
use std::slice;

/// A checked, zero-copy view of a `sprs` CSR matrix with `i64` values.
#[derive(Clone, Debug)]
pub struct SprsCsrView<'a, I: SpIndex = usize, Iptr: SpIndex = I> {
    matrix: CsMatViewI<'a, Scalar, I, Iptr>,
}

impl<'a, I, Iptr> SprsCsrView<'a, I, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    /// Wraps a borrowed `sprs` matrix, rejecting column-compressed storage.
    pub fn try_new(
        matrix: CsMatViewI<'a, Scalar, I, Iptr>,
        operand: MatrixOperand,
    ) -> Result<Self, SpGemmError> {
        if !matrix.is_csr() {
            return Err(SpGemmError::NonCsrStorage { operand });
        }
        Ok(Self { matrix })
    }

    /// Returns the underlying borrowed `sprs` matrix.
    pub fn as_inner(&self) -> &CsMatViewI<'a, Scalar, I, Iptr> {
        &self.matrix
    }
}

/// Iterator over one borrowed `sprs` row.
#[derive(Clone, Debug)]
pub struct SprsRowIter<'a, I: SpIndex> {
    columns: slice::Iter<'a, I>,
    values: slice::Iter<'a, Scalar>,
}

impl<I: SpIndex> Iterator for SprsRowIter<'_, I> {
    type Item = (usize, Scalar);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let &column = self.columns.next()?;
            let &value = self.values.next()?;
            // Explicit zeros are legal in sprs but not in CsrInput. Skipping
            // them retains zero-copy access and canonical logical rows.
            if value != 0 {
                return Some((column.index(), value));
            }
        }
    }
}

impl<I, Iptr> CsrInput for SprsCsrView<'_, I, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    type RowIter<'a>
        = SprsRowIter<'a, I>
    where
        Self: 'a;

    fn rows(&self) -> usize {
        self.matrix.rows()
    }

    fn cols(&self) -> usize {
        self.matrix.cols()
    }

    fn nnz(&self) -> usize {
        self.matrix
            .data()
            .iter()
            .filter(|&&value| value != 0)
            .count()
    }

    fn row(&self, row: usize) -> Self::RowIter<'_> {
        let range = self.matrix.indptr().outer_inds_sz(row);
        SprsRowIter {
            columns: self.matrix.indices()[range.clone()].iter(),
            values: self.matrix.data()[range].iter(),
        }
    }
}

/// Runs automatic sparse multiplication on zero-copy `sprs` CSR views.
///
/// The result uses the same inner-index and pointer-index types as the input.
/// Both inputs must therefore use the same pair of index types.
pub fn auto_spgemm<I, Iptr>(
    left: CsMatViewI<'_, Scalar, I, Iptr>,
    right: CsMatViewI<'_, Scalar, I, Iptr>,
    config: AutoSpGemmConfig,
) -> Result<(CsMatI<Scalar, I, Iptr>, AutoSpGemmStats), SpGemmError>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    let left = SprsCsrView::try_new(left, MatrixOperand::Left)?;
    let right = SprsCsrView::try_new(right, MatrixOperand::Right)?;
    let (product, stats) = crate::try_auto_spgemm(&left, &right, config)?;

    let row_ptr = product
        .row_ptr
        .into_iter()
        .map(|value| {
            Iptr::try_from_usize(value).ok_or(SpGemmError::IndexOverflow {
                value,
                target: "sprs row-pointer index type",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let col_idx = product
        .col_idx
        .into_iter()
        .map(|value| {
            I::try_from_usize(value).ok_or(SpGemmError::IndexOverflow {
                value,
                target: "sprs column-index type",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let output = CsMatI::try_new(
        (product.rows, product.cols),
        row_ptr,
        col_idx,
        product.values,
    )
    .map_err(|(_, _, _, error)| SpGemmError::InvalidOutputStructure(error.to_string()))?;
    Ok((output, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::CsrInput;
    use sprs::CsMatI;

    #[test]
    fn multiplies_borrowed_csr_and_preserves_index_types() {
        let left = CsMatI::<i64, u16>::new((2, 3), vec![0, 2, 3], vec![0, 2, 1], vec![2, 3, 4]);
        let right =
            CsMatI::<i64, u16>::new((3, 2), vec![0, 1, 2, 4], vec![0, 1, 0, 1], vec![5, 6, 7, 8]);

        let (product, _) = auto_spgemm(left.view(), right.view(), AutoSpGemmConfig::default())
            .expect("compatible CSR inputs");

        assert_eq!(product.to_dense().as_slice().unwrap(), &[31, 24, 0, 24]);
        let _: &CsMatI<i64, u16> = &product;
    }

    #[test]
    fn rejects_csc_and_filters_explicit_zeros() {
        let csc = CsMatI::<i64, usize>::new_csc((2, 2), vec![0, 1, 1], vec![0], vec![1]);
        let error = SprsCsrView::try_new(csc.view(), MatrixOperand::Left).unwrap_err();
        assert_eq!(
            error,
            SpGemmError::NonCsrStorage {
                operand: MatrixOperand::Left
            }
        );

        let csr = CsMatI::<i64, usize>::new((1, 2), vec![0, 2], vec![0, 1], vec![0, 9]);
        let view = SprsCsrView::try_new(csr.view(), MatrixOperand::Left).unwrap();
        assert_eq!(view.nnz(), 1);
        assert_eq!(view.row(0).collect::<Vec<_>>(), vec![(1, 9)]);
    }
}

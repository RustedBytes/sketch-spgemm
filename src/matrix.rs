use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<i64>,
}

impl CsrMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, i64)]) -> Self {
        let mut per_row: Vec<BTreeMap<usize, i64>> = (0..rows).map(|_| BTreeMap::new()).collect();

        for &(r, c, v) in triplets {
            assert!(r < rows, "triplet row {r} out of range {rows}");
            assert!(c < cols, "triplet col {c} out of range {cols}");
            if v == 0 {
                continue;
            }
            *per_row[r].entry(c).or_insert(0) += v;
        }

        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in per_row {
            for (c, v) in row {
                if v != 0 {
                    col_idx.push(c);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn row(&self, r: usize) -> impl Iterator<Item = (usize, i64)> + '_ {
        let start = self.row_ptr[r];
        let end = self.row_ptr[r + 1];
        self.col_idx[start..end]
            .iter()
            .copied()
            .zip(self.values[start..end].iter().copied())
    }

    pub fn to_dense(&self) -> DenseMatrix {
        let mut out = DenseMatrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for (c, v) in self.row(r) {
                out[(r, c)] = v;
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i64>,
}

impl DenseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.iter().filter(|&&x| x != 0).count()
    }

    pub fn to_csr(&self) -> CsrMatrix {
        let mut triplets = Vec::with_capacity(self.nnz());
        for i in 0..self.rows {
            let base = i * self.cols;
            for j in 0..self.cols {
                let v = self.data[base + j];
                if v != 0 {
                    triplets.push((i, j, v));
                }
            }
        }
        CsrMatrix::from_triplets(self.rows, self.cols, &triplets)
    }
}

impl std::ops::Index<(usize, usize)> for DenseMatrix {
    type Output = i64;

    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.data[r * self.cols + c]
    }
}

impl std::ops::IndexMut<(usize, usize)> for DenseMatrix {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.data[r * self.cols + c]
    }
}

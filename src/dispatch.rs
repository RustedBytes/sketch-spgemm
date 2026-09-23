//! Scalar-aware high-level sparse multiplication dispatch.

use crate::auto::{try_auto_spgemm, AutoSpGemmConfig, AutoSpGemmStats};
use crate::error::SpGemmError;
use crate::matrix::{CsrInput, CsrMatrix, SpGemmScalar};
use crate::spgemm::{try_spgemm_hash, SpGemmStats};

/// Statistics produced by the scalar-aware high-level entry point.
#[derive(Clone, Debug)]
pub enum SpGemmExecutionStats {
    /// The `i64` workload used automatic exact/sketch selection.
    Automatic(Box<AutoSpGemmStats>),
    /// The scalar used the representation-independent direct CSR kernel.
    Direct(SpGemmStats),
}

impl SpGemmExecutionStats {
    /// Whether workload analysis and automatic exact/sketch selection ran.
    pub fn used_automatic_path(&self) -> bool {
        matches!(self, Self::Automatic(_))
    }

    /// Candidate products reported by the direct path, when it was selected.
    pub fn direct_candidate_products(&self) -> Option<u128> {
        match self {
            Self::Automatic(_) => None,
            Self::Direct(stats) => Some(stats.candidate_products),
        }
    }
}

/// Scalar-specific policy used by [`try_spgemm`].
///
/// The primitive `i64` implementation runs the complete automatic sketch
/// pipeline. Other numeric primitive implementations use the direct CSR
/// kernel because recovery and fingerprints currently require exact `i64`
/// semantics. Custom scalar types may implement this trait; its default method
/// also selects the direct kernel.
pub trait SpGemmDispatchScalar: SpGemmScalar + Sized {
    /// Multiply two canonical CSR inputs using the policy for this scalar.
    fn try_dispatch<A, B>(
        a: &A,
        b: &B,
        _config: AutoSpGemmConfig,
    ) -> Result<(CsrMatrix<Self>, SpGemmExecutionStats), SpGemmError>
    where
        A: CsrInput<Scalar = Self> + ?Sized,
        B: CsrInput<Scalar = Self> + ?Sized,
    {
        let (product, stats) = try_spgemm_hash(a, b)?;
        Ok((product, SpGemmExecutionStats::Direct(stats)))
    }
}

impl SpGemmDispatchScalar for i64 {
    fn try_dispatch<A, B>(
        a: &A,
        b: &B,
        config: AutoSpGemmConfig,
    ) -> Result<(CsrMatrix<Self>, SpGemmExecutionStats), SpGemmError>
    where
        A: CsrInput<Scalar = Self> + ?Sized,
        B: CsrInput<Scalar = Self> + ?Sized,
    {
        let (product, stats) = try_auto_spgemm(a, b, config)?;
        Ok((product, SpGemmExecutionStats::Automatic(Box::new(stats))))
    }
}

macro_rules! impl_direct_dispatch {
    ($($scalar:ty),+ $(,)?) => {
        $(impl SpGemmDispatchScalar for $scalar {})+
    };
}

impl_direct_dispatch!(i8, i16, i32, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,);

/// Multiply canonical CSR inputs through the best available scalar path.
///
/// For `i64`, `config` controls the existing automatic exact/sketch pipeline.
/// For other built-in numeric scalars, the direct hash-accumulator kernel is
/// selected and `config` is currently ignored.
///
/// # Example
///
/// ```
/// use sketch_spgemm::{try_spgemm, AutoSpGemmConfig, CsrMatrix};
///
/// let a = CsrMatrix::<i32>::from_triplets(1, 2, &[(0, 0, 2), (0, 1, 3)]);
/// let b = CsrMatrix::<i32>::from_triplets(2, 1, &[(0, 0, 5), (1, 0, 7)]);
/// let (product, stats) = try_spgemm(&a, &b, AutoSpGemmConfig::default())?;
/// assert_eq!(product.values, vec![31]);
/// assert_eq!(stats.direct_candidate_products(), Some(2));
/// # Ok::<(), sketch_spgemm::SpGemmError>(())
/// ```
pub fn try_spgemm<A, B>(
    a: &A,
    b: &B,
    config: AutoSpGemmConfig,
) -> Result<(CsrMatrix<A::Scalar>, SpGemmExecutionStats), SpGemmError>
where
    A: CsrInput + ?Sized,
    B: CsrInput<Scalar = A::Scalar> + ?Sized,
    A::Scalar: SpGemmDispatchScalar,
{
    A::Scalar::try_dispatch(a, b, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_uses_automatic_pipeline() {
        let a = CsrMatrix::from_triplets(1, 1, &[(0, 0, 6_i64)]);
        let b = CsrMatrix::from_triplets(1, 1, &[(0, 0, 7_i64)]);

        let (product, stats) = try_spgemm(&a, &b, AutoSpGemmConfig::default()).unwrap();
        assert_eq!(product.values, vec![42]);
        assert!(stats.used_automatic_path());
    }

    #[test]
    fn non_i64_uses_direct_pipeline() {
        let a = CsrMatrix::<i32>::from_triplets(1, 2, &[(0, 0, 2), (0, 1, 3)]);
        let b = CsrMatrix::<i32>::from_triplets(2, 1, &[(0, 0, 5), (1, 0, 7)]);

        let (product, stats) = try_spgemm(&a, &b, AutoSpGemmConfig::default()).unwrap();
        assert_eq!(product.values, vec![31]);
        assert_eq!(stats.direct_candidate_products(), Some(2));
    }

    #[test]
    fn direct_pipeline_preserves_dimension_errors() {
        let a = CsrMatrix::<f32>::zeros(1, 2);
        let b = CsrMatrix::<f32>::zeros(3, 1);

        assert!(matches!(
            try_spgemm(&a, &b, AutoSpGemmConfig::default()),
            Err(SpGemmError::DimensionMismatch { .. })
        ));
    }
}

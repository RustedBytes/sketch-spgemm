use crate::fingerprint::FingerprintConfig;
use crate::matrix::CsrMatrix;
use crate::recovery::{
    nested_spgemm_with_options, MomentConfig, NestedOptions, NestedSpGemmStats, RecoveryBackend,
};
use crate::rect::{adaptive_matmul, RectangularPolicy, RectangularStats};
use crate::spgemm::spgemm_hash;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoChoice {
    Exact,
    Sketch,
    ExactFallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactMethod {
    AdaptiveDense,
    HashSparse,
}

#[derive(Clone, Debug)]
pub struct WorkloadEstimate {
    pub candidate_products: u128,
    /// F / (rows*cols): amplification even if every output cell were nonzero.
    pub structural_amplification: f64,
    pub sampled_rows: usize,
    pub sampled_output_nnz: usize,
    pub sampled_unique_columns: usize,
    pub estimated_output_nnz: usize,
    pub estimated_active_columns: usize,
    pub estimated_avg_nnz_per_active_column: f64,
    pub estimated_output_density: f64,
    pub estimated_rho: f64,
    pub target_q: usize,
    pub estimated_moment_rows: usize,
    pub choose_sketch: bool,
    pub reason: String,
    /// True when the cheap structural prefilter decided before exact row samples.
    pub structural_prefilter_only: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AutoTimingStats {
    pub total: Duration,
    pub analysis_total: Duration,
    pub candidate_count: Duration,
    pub row_sampling: Duration,
    pub nested: Duration,
    pub fingerprint_setup: Duration,
    pub fingerprint_checks: Duration,
    pub exact: Duration,
}

#[derive(Clone, Debug)]
pub struct AutoSpGemmConfig {
    /// Maximum number of exact output rows used by the staged estimator.
    pub sample_rows: usize,
    /// Initial exact row sample. If the classification is far from every
    /// decision boundary, analysis stops here instead of consuming sample_rows.
    pub initial_sample_rows: usize,
    /// Cheap prefilter on F/(rows*cols). Below this level sketching is unlikely
    /// to repay even before output sparsity is estimated, so no exact row sample
    /// is performed.
    pub min_structural_amplification: f64,
    /// Conservative K bound supplied to the nested algorithm. The tighter
    /// sampled estimate is passed separately to the practical q scheduler.
    pub k_safety_factor: f64,
    pub min_estimated_rho: f64,
    pub max_estimated_avg_column_nnz: f64,
    pub max_estimated_output_density: f64,
    pub max_moment_row_ratio: f64,
    pub rectangular_policy: RectangularPolicy,
    pub moment: MomentConfig,
    pub fingerprint: FingerprintConfig,
    /// Maximum total dense cells across A, B and C for the adaptive exact path.
    /// Above this, exact execution stays sparse and uses the CSR hash baseline.
    pub exact_dense_cell_limit: usize,
    pub exact_fallback: bool,
}

impl Default for AutoSpGemmConfig {
    fn default() -> Self {
        Self {
            sample_rows: 8,
            initial_sample_rows: 2,
            min_structural_amplification: 64.0,
            k_safety_factor: 1.5,
            min_estimated_rho: 256.0,
            max_estimated_avg_column_nnz: 64.0,
            max_estimated_output_density: 0.10,
            max_moment_row_ratio: 0.80,
            rectangular_policy: RectangularPolicy::Auto,
            moment: MomentConfig {
                guaranteed_correction: false,
                ..MomentConfig::default()
            },
            fingerprint: FingerprintConfig::default(),
            exact_dense_cell_limit: 16_000_000,
            exact_fallback: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AutoSpGemmStats {
    pub choice: AutoChoice,
    pub estimate: WorkloadEstimate,
    pub k_bound_used: usize,
    pub nested: Option<NestedSpGemmStats>,
    pub exact_stats: Option<RectangularStats>,
    pub exact_method: Option<ExactMethod>,
    pub fallback_reason: Option<String>,
    pub timing: AutoTimingStats,
}

#[derive(Clone, Debug, Default)]
struct AnalysisTiming {
    total: Duration,
    candidate_count: Duration,
    row_sampling: Duration,
}

/// Production-oriented wrapper: perform a cheap structural screen, inspect only
/// as many exact rows as needed to classify the workload confidently, choose
/// exact or moment-sketch execution, and verify a sketch result with an
/// independent bilinear residual fingerprint. It never needs true nnz(A*B).
pub fn auto_spgemm(
    a: &CsrMatrix,
    b: &CsrMatrix,
    config: AutoSpGemmConfig,
) -> (CsrMatrix, AutoSpGemmStats) {
    assert_eq!(a.cols, b.rows, "incompatible matrix dimensions");
    let total_start = Instant::now();
    let (estimate, analysis_timing) = analyze_workload_timed(a, b, &config);
    let mut timing = AutoTimingStats {
        analysis_total: analysis_timing.total,
        candidate_count: analysis_timing.candidate_count,
        row_sampling: analysis_timing.row_sampling,
        ..AutoTimingStats::default()
    };

    if !estimate.choose_sketch || estimate.estimated_output_nnz == 0 {
        let start = Instant::now();
        let (result, exact_stats, exact_method) = exact_dispatch(
            a,
            b,
            config.rectangular_policy,
            config.exact_dense_cell_limit,
        );
        timing.exact = start.elapsed();
        timing.total = total_start.elapsed();
        return (
            result,
            AutoSpGemmStats {
                choice: AutoChoice::Exact,
                estimate,
                k_bound_used: 0,
                nested: None,
                exact_stats,
                exact_method: Some(exact_method),
                fallback_reason: None,
                timing,
            },
        );
    }

    let max_k = a.rows.saturating_mul(b.cols).max(1);
    let k_bound =
        ((estimate.estimated_output_nnz as f64) * config.k_safety_factor.max(1.0)).ceil() as usize;
    let k_bound = k_bound.clamp(1, max_k);

    let mut moment = config.moment.clone();
    moment.guaranteed_correction = false;
    let start = Instant::now();
    let (candidate, nested_stats) = nested_spgemm_with_options(
        a,
        b,
        k_bound,
        RecoveryBackend::Moment(moment),
        NestedOptions {
            rectangular_policy: config.rectangular_policy,
            practical_scheduler: true,
            scheduler_k_hint: Some(estimate.estimated_output_nnz.max(1)),
            masked_residual: true,
            exact_k_bound: false,
            residual_fingerprint: Some(config.fingerprint),
            fingerprint_failure_correction: false,
        },
    );
    timing.nested = start.elapsed();
    timing.fingerprint_setup = nested_stats.fingerprint_setup_time;
    timing.fingerprint_checks = nested_stats.fingerprint_check_time;

    if nested_stats.fingerprint_verified || nested_stats.deterministic_verified {
        timing.total = total_start.elapsed();
        return (
            candidate,
            AutoSpGemmStats {
                choice: AutoChoice::Sketch,
                estimate,
                k_bound_used: k_bound,
                nested: Some(nested_stats),
                exact_stats: None,
                exact_method: None,
                fallback_reason: None,
                timing,
            },
        );
    }

    if config.exact_fallback {
        let reason = Some("sketch execution ended without a residual certificate".to_string());
        let start = Instant::now();
        let (result, exact_stats, exact_method) = exact_dispatch(
            a,
            b,
            config.rectangular_policy,
            config.exact_dense_cell_limit,
        );
        timing.exact = start.elapsed();
        timing.total = total_start.elapsed();
        return (
            result,
            AutoSpGemmStats {
                choice: AutoChoice::ExactFallback,
                estimate,
                k_bound_used: k_bound,
                nested: Some(nested_stats),
                exact_stats,
                exact_method: Some(exact_method),
                fallback_reason: reason,
                timing,
            },
        );
    }

    timing.total = total_start.elapsed();
    (
        candidate,
        AutoSpGemmStats {
            choice: AutoChoice::Sketch,
            estimate,
            k_bound_used: k_bound,
            nested: Some(nested_stats),
            exact_stats: None,
            exact_method: None,
            fallback_reason: Some(
                "returned uncertified sketch result because exact_fallback=false".to_string(),
            ),
            timing,
        },
    )
}

pub fn analyze_workload(
    a: &CsrMatrix,
    b: &CsrMatrix,
    config: &AutoSpGemmConfig,
) -> WorkloadEstimate {
    analyze_workload_timed(a, b, config).0
}

fn analyze_workload_timed(
    a: &CsrMatrix,
    b: &CsrMatrix,
    config: &AutoSpGemmConfig,
) -> (WorkloadEstimate, AnalysisTiming) {
    assert_eq!(a.cols, b.rows);
    let total_start = Instant::now();
    let mut timing = AnalysisTiming::default();

    let start = Instant::now();
    let candidate_products = candidate_product_count(a, b);
    timing.candidate_count = start.elapsed();

    let output_cells = a.rows.saturating_mul(b.cols).max(1);
    let structural_amplification = candidate_products as f64 / output_cells as f64;

    if a.rows == 0 || b.cols == 0 {
        timing.total = total_start.elapsed();
        return (
            empty_estimate(
                candidate_products,
                structural_amplification,
                "empty output domain",
            ),
            timing,
        );
    }

    // Stage 0: no value multiplication at all. If even F/(all output cells) is
    // small, an output-sparse recovery scheme is very unlikely to repay setup.
    if structural_amplification < config.min_structural_amplification {
        let mut e = empty_estimate(
            candidate_products,
            structural_amplification,
            &format!(
                "structural amplification {:.1} < {:.1}",
                structural_amplification, config.min_structural_amplification
            ),
        );
        e.structural_prefilter_only = true;
        timing.total = total_start.elapsed();
        return (e, timing);
    }

    let start = Instant::now();
    let max_samples = config.sample_rows.max(1).min(a.rows);
    let initial_samples = config.initial_sample_rows.max(1).min(max_samples);
    let sample_indices = evenly_spaced_rows(a.rows, max_samples);

    // v0.7.1: exact sampled rows use a dense scratch accumulator plus touched
    // columns, avoiding HashMap hashing/allocation for every candidate product.
    let mut acc = vec![0i64; b.cols];
    let mut stamp = vec![0u32; b.cols];
    let mut generation = 1u32;
    let mut globally_seen = vec![false; b.cols];
    let mut sampled_output_nnz = 0usize;
    let mut sampled_rows = 0usize;
    let mut unique_columns = 0usize;
    let mut current_estimate: Option<WorkloadEstimate> = None;
    let mut touched = Vec::with_capacity(b.cols.min(4096));

    for (pos, &i) in sample_indices.iter().enumerate() {
        touched.clear();
        for (k, av) in a.row(i) {
            for (j, bv) in b.row(k) {
                if stamp[j] != generation {
                    stamp[j] = generation;
                    acc[j] = 0;
                    touched.push(j);
                }
                acc[j] = acc[j].wrapping_add(av.wrapping_mul(bv));
            }
        }
        for &j in &touched {
            if acc[j] != 0 {
                sampled_output_nnz += 1;
                if !globally_seen[j] {
                    globally_seen[j] = true;
                    unique_columns += 1;
                }
            }
        }
        sampled_rows += 1;
        generation = generation.wrapping_add(1);
        if generation == 0 {
            stamp.fill(0);
            generation = 1;
        }

        if sampled_rows >= initial_samples {
            let est = estimate_from_sample(
                a,
                b,
                config,
                candidate_products,
                structural_amplification,
                sampled_rows,
                sampled_output_nnz,
                unique_columns,
            );
            let confident = classification_is_confident(&est, config);
            current_estimate = Some(est);
            if confident || pos + 1 == sample_indices.len() {
                break;
            }
        }
    }
    timing.row_sampling = start.elapsed();
    timing.total = total_start.elapsed();

    let estimate = current_estimate.unwrap_or_else(|| {
        estimate_from_sample(
            a,
            b,
            config,
            candidate_products,
            structural_amplification,
            sampled_rows,
            sampled_output_nnz,
            unique_columns,
        )
    });
    (estimate, timing)
}

fn empty_estimate(
    candidate_products: u128,
    structural_amplification: f64,
    reason: &str,
) -> WorkloadEstimate {
    WorkloadEstimate {
        candidate_products,
        structural_amplification,
        sampled_rows: 0,
        sampled_output_nnz: 0,
        sampled_unique_columns: 0,
        estimated_output_nnz: 0,
        estimated_active_columns: 0,
        estimated_avg_nnz_per_active_column: 0.0,
        estimated_output_density: 0.0,
        estimated_rho: f64::INFINITY,
        target_q: 1,
        estimated_moment_rows: 0,
        choose_sketch: false,
        reason: reason.to_string(),
        structural_prefilter_only: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn estimate_from_sample(
    a: &CsrMatrix,
    b: &CsrMatrix,
    config: &AutoSpGemmConfig,
    candidate_products: u128,
    structural_amplification: f64,
    sampled_rows: usize,
    sampled_output_nnz: usize,
    unique_columns: usize,
) -> WorkloadEstimate {
    let estimated_output_nnz = if sampled_rows == 0 {
        0
    } else {
        ((sampled_output_nnz as u128 * a.rows as u128 + sampled_rows as u128 / 2)
            / sampled_rows as u128) as usize
    }
    .min(a.rows.saturating_mul(b.cols));

    let sample_fraction = if a.rows == 0 {
        1.0
    } else {
        sampled_rows as f64 / a.rows as f64
    };
    let estimated_active_columns = estimate_active_columns(
        unique_columns,
        estimated_output_nnz,
        sample_fraction,
        b.cols,
    );
    let avg_col = if estimated_active_columns == 0 {
        0.0
    } else {
        estimated_output_nnz as f64 / estimated_active_columns as f64
    };
    let output_cells = a.rows.saturating_mul(b.cols).max(1);
    let output_density = estimated_output_nnz as f64 / output_cells as f64;
    let rho = if estimated_output_nnz == 0 {
        f64::INFINITY
    } else {
        candidate_products as f64 / estimated_output_nnz as f64
    };
    let target_q = if avg_col <= 1.0 {
        1
    } else {
        (avg_col.round().max(1.0) as usize)
            .next_power_of_two()
            .min(a.rows.max(1))
    };
    let buckets = (((target_q as f64) * config.moment.oversampling).ceil() as usize)
        .max(config.moment.degree)
        .min(a.rows.max(1));
    let raw_moment_rows = buckets.saturating_mul(3);
    let estimated_moment_rows = if config.moment.identity_fallback && raw_moment_rows >= a.rows {
        a.rows
    } else {
        raw_moment_rows
    };
    let row_ratio = estimated_moment_rows as f64 / a.rows.max(1) as f64;

    let mut reasons = Vec::new();
    if rho < config.min_estimated_rho {
        reasons.push(format!("rho {:.1} < {:.1}", rho, config.min_estimated_rho));
    }
    if avg_col > config.max_estimated_avg_column_nnz {
        reasons.push(format!(
            "avg column nnz {:.1} > {:.1}",
            avg_col, config.max_estimated_avg_column_nnz
        ));
    }
    if output_density > config.max_estimated_output_density {
        reasons.push(format!(
            "output density {:.3} > {:.3}",
            output_density, config.max_estimated_output_density
        ));
    }
    if row_ratio >= config.max_moment_row_ratio {
        reasons.push(format!(
            "moment row ratio {:.3} >= {:.3}",
            row_ratio, config.max_moment_row_ratio
        ));
    }
    if estimated_output_nnz == 0 {
        reasons.push("sampled output is zero".to_string());
    }
    let choose_sketch = reasons.is_empty();
    let reason = if choose_sketch {
        format!(
            "high amplification, sparse columns; estimated q={}, moment rows={}",
            target_q, estimated_moment_rows
        )
    } else {
        reasons.join("; ")
    };

    WorkloadEstimate {
        candidate_products,
        structural_amplification,
        sampled_rows,
        sampled_output_nnz,
        sampled_unique_columns: unique_columns,
        estimated_output_nnz,
        estimated_active_columns,
        estimated_avg_nnz_per_active_column: avg_col,
        estimated_output_density: output_density,
        estimated_rho: rho,
        target_q,
        estimated_moment_rows,
        choose_sketch,
        reason,
        structural_prefilter_only: false,
    }
}

fn classification_is_confident(e: &WorkloadEstimate, config: &AutoSpGemmConfig) -> bool {
    if e.sampled_rows == 0 {
        return true;
    }

    // Active-column inference is underdetermined while every sampled nonzero is
    // in a previously unseen column. Require some repeated column observations
    // before early-accepting Sketch; otherwise continue toward sample_rows.
    let repeats = e
        .sampled_output_nnz
        .saturating_sub(e.sampled_unique_columns);
    let support_model_ready =
        repeats >= 2 && repeats.saturating_mul(10) >= e.sampled_unique_columns.max(1);

    if e.choose_sketch {
        support_model_ready
            && e.estimated_rho >= config.min_estimated_rho * 4.0
            && e.estimated_avg_nnz_per_active_column <= config.max_estimated_avg_column_nnz * 0.5
            && e.estimated_output_density <= config.max_estimated_output_density * 0.5
    } else {
        // A clearly bad workload can be rejected without a precise active-column
        // estimate. This is the useful early-out side of progressive sampling.
        e.estimated_rho < config.min_estimated_rho * 0.5
            || e.estimated_avg_nnz_per_active_column > config.max_estimated_avg_column_nnz * 1.5
            || e.estimated_output_density > config.max_estimated_output_density * 1.5
    }
}

pub fn candidate_product_count(a: &CsrMatrix, b: &CsrMatrix) -> u128 {
    assert_eq!(a.cols, b.rows);
    let b_degree: Vec<usize> = (0..b.rows)
        .map(|k| b.row_ptr[k + 1] - b.row_ptr[k])
        .collect();
    let mut total = 0u128;
    for i in 0..a.rows {
        for (k, _) in a.row(i) {
            total += b_degree[k] as u128;
        }
    }
    total
}

fn exact_dispatch(
    a: &CsrMatrix,
    b: &CsrMatrix,
    policy: RectangularPolicy,
    dense_cell_limit: usize,
) -> (CsrMatrix, Option<RectangularStats>, ExactMethod) {
    let dense_cells = a
        .rows
        .saturating_mul(a.cols)
        .saturating_add(b.rows.saturating_mul(b.cols))
        .saturating_add(a.rows.saturating_mul(b.cols));
    if dense_cells <= dense_cell_limit {
        let ad = a.to_dense();
        let bd = b.to_dense();
        let (dense, stats) = adaptive_matmul(&ad, &bd, policy);
        (dense.to_csr(), Some(stats), ExactMethod::AdaptiveDense)
    } else {
        let (csr, _) = spgemm_hash(a, b);
        (csr, None, ExactMethod::HashSparse)
    }
}

fn evenly_spaced_rows(rows: usize, count: usize) -> Vec<usize> {
    if rows == 0 || count == 0 {
        return Vec::new();
    }
    if count >= rows {
        return (0..rows).collect();
    }
    if count == 1 {
        return vec![rows / 2];
    }
    (0..count).map(|s| s * (rows - 1) / (count - 1)).collect()
}

/// Infer total active columns from the number observed in a row sample. Under a
/// roughly uniform per-column sparsity model, a column with K/C entries is seen
/// with probability 1-(1-f)^(K/C). We choose the C whose expected observation
/// count is closest to the measured union size.
fn estimate_active_columns(
    observed: usize,
    k_est: usize,
    sample_fraction: f64,
    cols: usize,
) -> usize {
    if observed == 0 || k_est == 0 || cols == 0 {
        return 0;
    }
    let lo = observed.min(cols).max(1);
    let mut best = lo;
    let mut best_err = f64::INFINITY;
    for c in lo..=cols {
        let avg = k_est as f64 / c as f64;
        let seen_prob = 1.0 - (1.0 - sample_fraction.clamp(0.0, 1.0)).powf(avg.max(0.0));
        let expected = c as f64 * seen_prob;
        let err = (expected - observed as f64).abs();
        if err < best_err {
            best_err = err;
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spgemm::spgemm_hash;
    use crate::synthetic::{overlap_problem, sparse_output_problem};

    #[test]
    fn auto_analysis_prefers_sparse_output_and_rejects_dense_columns() {
        let sparse = sparse_output_problem(256, 512, 512, 128, 7, 0.75, 512);
        let cfg = AutoSpGemmConfig {
            fingerprint: FingerprintConfig { lanes: 3, seed: 7 },
            ..AutoSpGemmConfig::default()
        };
        let se = analyze_workload(&sparse.a, &sparse.b, &cfg);
        assert!(se.choose_sketch, "{}", se.reason);
        assert!(se.estimated_avg_nnz_per_active_column < 32.0);
        assert!(se.sampled_rows <= cfg.sample_rows);

        let dense_cols = overlap_problem(256, 256, 256, 128, 0.5);
        let de = analyze_workload(&dense_cols.a, &dense_cols.b, &cfg);
        assert!(!de.choose_sketch);
    }

    #[test]
    fn progressive_sampler_waits_for_support_overlap_on_sparse_output() {
        let sparse = sparse_output_problem(256, 512, 512, 128, 7, 0.75, 512);
        let cfg = AutoSpGemmConfig {
            sample_rows: 8,
            initial_sample_rows: 2,
            ..AutoSpGemmConfig::default()
        };
        let e = analyze_workload(&sparse.a, &sparse.b, &cfg);
        assert!(e.choose_sketch, "{}", e.reason);
        // The cyclic synthetic supports do not overlap in the first several
        // sample rows, so the staged estimator correctly keeps sampling until
        // active-column inference becomes identifiable.
        assert_eq!(e.sampled_rows, 8);
        assert_eq!(e.target_q, 8);
    }

    #[test]
    fn structural_prefilter_avoids_sampling_low_amplification() {
        let a = CsrMatrix::from_triplets(32, 32, &(0..32).map(|i| (i, i, 1)).collect::<Vec<_>>());
        let b = a.clone();
        let cfg = AutoSpGemmConfig::default();
        let e = analyze_workload(&a, &b, &cfg);
        assert!(!e.choose_sketch);
        assert!(e.structural_prefilter_only);
        assert_eq!(e.sampled_rows, 0);
    }

    #[test]
    fn auto_spgemm_is_exact_on_sparse_output_without_true_k() {
        let p = sparse_output_problem(128, 256, 128, 32, 5, 0.5, 256);
        let (expected, _) = spgemm_hash(&p.a, &p.b);
        let cfg = AutoSpGemmConfig {
            sample_rows: 8,
            min_structural_amplification: 1.0,
            min_estimated_rho: 16.0,
            max_estimated_avg_column_nnz: 16.0,
            max_estimated_output_density: 0.25,
            fingerprint: FingerprintConfig {
                lanes: 3,
                seed: 123,
            },
            ..AutoSpGemmConfig::default()
        };
        let (actual, stats) = auto_spgemm(&p.a, &p.b, cfg);
        assert_eq!(actual, expected);
        assert!(matches!(
            stats.choice,
            AutoChoice::Sketch | AutoChoice::ExactFallback
        ));
        assert!(stats.timing.total >= stats.timing.analysis_total);
    }
}

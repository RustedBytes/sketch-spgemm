use sketch_spgemm::{
    adaptive_matmul, direct_two_sided_sketch, left_sketch, nested_spgemm_with_policy,
    overlap_problem, paper_schedule, right_sketch, sparse_output_problem, spgemm_hash, GuvConfig,
    CsrMatrix, RecoveryBackend, RectangularPolicy, SignatureConfig, SketchMap,
};
use std::env;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Config {
    rows: usize,
    inner: usize,
    cols: usize,
    hubs: usize,
    cancel: f64,
    synthetic: String,
    active_cols: usize,
    output_col_nnz: usize,
    amplification_width: usize,
    h_buckets: usize,
    g_buckets: usize,
    degree: usize,
    repeats: usize,
    nested_backend: String,
    recovery_degree: usize,
    recovery_oversampling: f64,
    guv_alpha: f64,
    guv_epsilon: f64,
    identity_fallback: bool,
    guaranteed_correction: bool,
    rectangular_policy: RectangularPolicy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rows: 256,
            inner: 256,
            cols: 256,
            hubs: 64,
            cancel: 0.50,
            synthetic: "overlap".to_string(),
            active_cols: 128,
            output_col_nnz: 8,
            amplification_width: 0,
            h_buckets: 64,
            g_buckets: 64,
            degree: 3,
            repeats: 3,
            nested_backend: "guv".to_string(),
            recovery_degree: 5,
            recovery_oversampling: 2.0,
            guv_alpha: 1.0,
            guv_epsilon: 1.0 / 12.0,
            identity_fallback: true,
            guaranteed_correction: false,
            rectangular_policy: RectangularPolicy::Auto,
        }
    }
}

fn main() {
    let cfg = parse_args();
    println!("SketchSpGEMM cached nested-recovery prototype v0.5");
    println!("config: {cfg:?}\n");

    let problem = match cfg.synthetic.as_str() {
        "overlap" => overlap_problem(cfg.rows, cfg.inner, cfg.cols, cfg.hubs, cfg.cancel),
        "sparse-output" | "sparse" => {
            let width = if cfg.amplification_width == 0 {
                largest_power_of_two_leq(cfg.inner)
            } else {
                cfg.amplification_width
            };
            sparse_output_problem(
                cfg.rows,
                cfg.inner,
                cfg.cols,
                cfg.active_cols,
                cfg.output_col_nnz,
                cfg.cancel,
                width,
            )
        }
        other => panic!("unknown synthetic generator: {other}; expected overlap|sparse-output"),
    };
    println!(
        "A: {}x{}, nnz={} | B: {}x{}, nnz={}",
        problem.a.rows,
        problem.a.cols,
        problem.a.nnz(),
        problem.b.rows,
        problem.b.cols,
        problem.b.nnz()
    );
    println!("synthetic generator: {}", problem.generator);
    println!("synthetic canceled columns: {}", problem.canceled_columns);
    println!("synthetic active output columns: {}", problem.active_output_columns);
    if let Some(nnz) = problem.target_nnz_per_active_column {
        println!("target nnz / active output column: {nnz}");
    }
    println!("amplification width: {}", problem.amplification_width);

    let ((c, stats), baseline_t) = timed_best(cfg.repeats, || spgemm_hash(&problem.a, &problem.b));
    assert_eq!(stats.candidate_products, problem.expected_candidate_products);

    let rho = if c.nnz() == 0 {
        f64::INFINITY
    } else {
        stats.candidate_products as f64 / c.nnz() as f64
    };
    let gamma_denom = problem.a.nnz() + problem.b.nnz() + c.nnz();
    let gamma = stats.candidate_products as f64 / gamma_denom.max(1) as f64;

    println!("C exact nnz={}", c.nnz());
    let geometry = output_geometry(&c);
    println!(
        "C output geometry: nonzero_cols={}, nnz/active_col min={} avg={:.2} max={}",
        geometry.nonzero_columns,
        geometry.min_nnz_per_nonzero_column,
        geometry.avg_nnz_per_nonzero_column,
        geometry.max_nnz_per_nonzero_column,
    );
    println!("candidate scalar products F={}", stats.candidate_products);
    println!("rho=F/nnz(C)={rho:.2}");
    println!("gamma=F/(nnz(A)+nnz(B)+nnz(C))={gamma:.2}");
    println!("baseline hash SpGEMM: {}", fmt_duration(baseline_t));

    let ((exact_dense, exact_rect_stats), adaptive_exact_t) = timed_best(cfg.repeats, || {
        let ad = problem.a.to_dense();
        let bd = problem.b.to_dense();
        adaptive_matmul(&ad, &bd, cfg.rectangular_policy)
    });
    assert_eq!(exact_dense, c.to_dense(), "adaptive exact baseline disagrees with hash SpGEMM");
    println!(
        "baseline adaptive exact: {} [{}; mults={}]",
        fmt_duration(adaptive_exact_t),
        exact_rect_stats.kernel.unwrap(),
        exact_rect_stats.scalar_multiplications,
    );

    // Keep the original cheap hash-sketch probe. It answers whether the
    // compressed rectangular kernel is attractive independently of recovery.
    let h = SketchMap::new(problem.a.rows, cfg.h_buckets, cfg.degree, 0xA11CE);
    let g = SketchMap::new(problem.b.cols, cfg.g_buckets, cfg.degree, 0xB0B);

    let (ha, left_t) = timed_best(cfg.repeats, || left_sketch(&problem.a, &h));
    let (bgt, right_t) = timed_best(cfg.repeats, || right_sketch(&problem.b, &g));
    let ((compressed, probe_rect_stats), gemm_t) = timed_best(cfg.repeats, || {
        adaptive_matmul(&ha, &bgt, cfg.rectangular_policy)
    });
    let (oracle, direct_t) = timed_best(cfg.repeats, || direct_two_sided_sketch(&c, &h, &g));

    assert_eq!(compressed, oracle, "two-sided sketch identity failed");

    println!("\nKernel-only two-sided sketch probe:");
    println!(
        "  H: {}x{} implicit binary degree {}",
        cfg.h_buckets, problem.a.rows, cfg.degree
    );
    println!(
        "  G: {}x{} implicit binary degree {}",
        cfg.g_buckets, problem.b.cols, cfg.degree
    );
    println!("  HA: {}x{}, nnz={}", ha.rows, ha.cols, ha.nnz());
    println!("  BG^T: {}x{}, nnz={}", bgt.rows, bgt.cols, bgt.nnz());
    println!(
        "  W=(HA)(BG^T): {}x{}, nnz={}",
        compressed.rows,
        compressed.cols,
        compressed.nnz()
    );
    println!("  H*A:             {}", fmt_duration(left_t));
    println!("  B*G^T:           {}", fmt_duration(right_t));
    println!(
        "  compressed GEMM: {} [{}; HA density={:.3}, BG^T density={:.3}, mults={}]",
        fmt_duration(gemm_t),
        probe_rect_stats.kernel.unwrap(),
        probe_rect_stats.a_density,
        probe_rect_stats.b_density,
        probe_rect_stats.scalar_multiplications,
    );
    println!("  direct H*C*G^T:  {} (oracle only)", fmt_duration(direct_t));
    println!("  identity check:  PASS");

    let compressed_kernel_t = left_t + right_t + gemm_t;
    println!("\nGo/no-go kernel comparison (recovery excluded):");
    println!("  compressed kernels total: {}", fmt_duration(compressed_kernel_t));
    println!(
        "  compressed/hash-baseline: {:.3}x",
        compressed_kernel_t.as_secs_f64() / baseline_t.as_secs_f64().max(1e-12)
    );
    println!(
        "  compressed/adaptive-exact: {:.3}x",
        compressed_kernel_t.as_secs_f64() / adaptive_exact_t.as_secs_f64().max(1e-12)
    );

    println!("\nGraia Algorithm-1 capacity schedule using K=nnz(C):");
    println!("  i    t       q       p       Q_after");
    for r in paper_schedule(problem.a.rows, problem.b.cols, c.nnz()) {
        println!(
            "  {:<4} {:<7} {:<7} {:<7} {}",
            r.i, r.t, r.q, r.p, r.q_accumulated_after
        );
    }

    if cfg.nested_backend != "off" {
        let backend = match cfg.nested_backend.as_str() {
            "identity" => RecoveryBackend::Identity,
            "signature" => RecoveryBackend::Signature(SignatureConfig {
                degree: cfg.recovery_degree,
                oversampling: cfg.recovery_oversampling,
                seed: 0x5A17_9EED_D15C_A11E,
                identity_fallback: cfg.identity_fallback,
                guaranteed_correction: cfg.guaranteed_correction,
            }),
            "guv" => RecoveryBackend::Guv(GuvConfig {
                alpha: cfg.guv_alpha,
                epsilon: cfg.guv_epsilon,
                identity_fallback: cfg.identity_fallback,
                guaranteed_correction: cfg.guaranteed_correction,
            }),
            other => panic!("unknown nested backend: {other}"),
        };

        let start = Instant::now();
        let (recovered, nested_stats) = nested_spgemm_with_policy(
            &problem.a,
            &problem.b,
            c.nnz(),
            backend,
            cfg.rectangular_policy,
        );
        let nested_t = start.elapsed();
        let exact = recovered == c;

        println!("\nNested residual recovery ({})", cfg.nested_backend);
        println!("  total: {}", fmt_duration(nested_t));
        println!("  exact product check: {}", if exact { "PASS" } else { "FAIL" });
        println!("  output nnz: {}", recovered.nnz());
        println!("  rounds:");
        println!("    i  q     p     H(rows/kind)          G(rows/kind)          outer inner D_nnz");
        for s in &nested_stats.rounds {
            println!(
                "    {:<2} {:<5} {:<5} {:<5}/{:<14} {:<5}/{:<14} {:<5} {:<5} {}",
                s.params.i,
                s.params.q,
                s.params.p,
                s.h_rows,
                s.h_kind,
                s.g_rows,
                s.g_kind,
                s.outer_recovered,
                s.inner_updates,
                s.d_nnz_after,
            );
            println!(
                "       kernel={} + {} + {} [{}], HDG^T/sub={}, decode={}, W_nnz={}",
                fmt_duration(s.left_time),
                fmt_duration(s.right_time),
                fmt_duration(s.rectangular_time),
                s.rectangular_kernel,
                fmt_duration(s.residual_measure_time),
                fmt_duration(s.decode_time),
                s.w_nnz,
            );
            println!(
                "       HA dens={:.3}, BGT dens={:.3}, sparse candidates={}, mults={}, matrix-cache(H/G)={}/{}, factor-cache(HA/BGT)={}/{}, HABG-cache={}, W-cache={}",
                s.ha_density,
                s.bgt_density,
                s.rectangular_candidate_products,
                s.rectangular_scalar_multiplications,
                if s.h_matrix_cache_hit { "hit" } else { "miss" },
                if s.g_matrix_cache_hit { "hit" } else { "miss" },
                if s.ha_cache_hit { "hit" } else { "miss" },
                if s.bgt_cache_hit { "hit" } else { "miss" },
                if s.product_cache_hit { "hit" } else { "miss" },
                if s.residual_cache_hit { "hit" } else { "miss" },
            );
        }
        if nested_stats.terminated_early {
            println!(
                "  early termination: {}",
                nested_stats.termination_reason.as_deref().unwrap_or("yes")
            );
        }
        if let Some(correction) = &nested_stats.correction_pass {
            println!(
                "  exact correction fallback: columns={}, nnz={}, time={}",
                correction.residual_columns,
                correction.residual_nnz,
                fmt_duration(correction.elapsed)
            );
        }

        if !exact && cfg.guaranteed_correction {
            panic!("guaranteed correction was enabled but nested result was not exact");
        }
    }

    println!("\nScope note:");
    println!("  guv backend = explicit Parvaresh-Vardy/GUV neighbor construction +");
    println!("                Bennett A⊗_rB Reduce/Recovery decoder.");
    println!("  signature backend = deterministic hash graph for practical comparison.");
    println!("  identity backend = exact Algorithm-1 recovery logic, no compression.");
    println!("  With --identity-fallback true, literal GUV constants often select I_N;");
    println!("  this is expected and follows the recovery theorem's smaller-row fallback.");
    println!("  --guaranteed-correction true adds a validation-only exact residual pass.");
    println!("  v0.5 caches repeated HA, BG^T, HABG^T and W while D is unchanged.");
    println!("  --rect-kernel auto dispatches dense/sparse rectangular multiplication per round.");
}


#[derive(Clone, Copy, Debug, Default)]
struct OutputGeometry {
    nonzero_columns: usize,
    min_nnz_per_nonzero_column: usize,
    max_nnz_per_nonzero_column: usize,
    avg_nnz_per_nonzero_column: f64,
}

fn output_geometry(c: &CsrMatrix) -> OutputGeometry {
    let mut counts = vec![0usize; c.cols];
    for i in 0..c.rows {
        for (j, _) in c.row(i) {
            counts[j] += 1;
        }
    }
    let nonzero: Vec<usize> = counts.into_iter().filter(|&x| x != 0).collect();
    if nonzero.is_empty() {
        return OutputGeometry::default();
    }
    let total: usize = nonzero.iter().sum();
    OutputGeometry {
        nonzero_columns: nonzero.len(),
        min_nnz_per_nonzero_column: *nonzero.iter().min().unwrap(),
        max_nnz_per_nonzero_column: *nonzero.iter().max().unwrap(),
        avg_nnz_per_nonzero_column: total as f64 / nonzero.len() as f64,
    }
}

fn timed_best<T, F>(repeats: usize, mut f: F) -> (T, Duration)
where
    F: FnMut() -> T,
{
    let repeats = repeats.max(1);
    let mut best = Duration::MAX;
    let mut last = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let value = f();
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
        last = Some(value);
    }
    (last.unwrap(), best)
}

fn fmt_duration(d: Duration) -> String {
    if d.as_secs_f64() >= 1.0 {
        format!("{:.3} s", d.as_secs_f64())
    } else if d.as_secs_f64() >= 1e-3 {
        format!("{:.3} ms", d.as_secs_f64() * 1e3)
    } else {
        format!("{:.3} us", d.as_secs_f64() * 1e6)
    }
}


fn largest_power_of_two_leq(x: usize) -> usize {
    assert!(x > 0, "inner dimension must be positive");
    1usize << (usize::BITS as usize - 1 - x.leading_zeros() as usize)
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            print_help();
            std::process::exit(0);
        }
        let value = args.next().unwrap_or_else(|| panic!("missing value for {flag}"));
        match flag.as_str() {
            "--rows" => cfg.rows = value.parse().unwrap(),
            "--inner" => cfg.inner = value.parse().unwrap(),
            "--cols" => cfg.cols = value.parse().unwrap(),
            "--hubs" => cfg.hubs = value.parse().unwrap(),
            "--cancel" => cfg.cancel = value.parse().unwrap(),
            "--synthetic" => cfg.synthetic = value,
            "--active-cols" => cfg.active_cols = value.parse().unwrap(),
            "--output-col-nnz" => cfg.output_col_nnz = value.parse().unwrap(),
            "--amplification-width" => cfg.amplification_width = value.parse().unwrap(),
            "--h-buckets" => cfg.h_buckets = value.parse().unwrap(),
            "--g-buckets" => cfg.g_buckets = value.parse().unwrap(),
            "--degree" => cfg.degree = value.parse().unwrap(),
            "--repeats" => cfg.repeats = value.parse().unwrap(),
            "--nested-backend" => cfg.nested_backend = value,
            "--recovery-degree" => cfg.recovery_degree = value.parse().unwrap(),
            "--recovery-oversampling" => cfg.recovery_oversampling = value.parse().unwrap(),
            "--guv-alpha" => cfg.guv_alpha = value.parse().unwrap(),
            "--guv-epsilon" => cfg.guv_epsilon = value.parse().unwrap(),
            "--identity-fallback" => cfg.identity_fallback = parse_bool(&value),
            "--guaranteed-correction" => cfg.guaranteed_correction = parse_bool(&value),
            "--rect-kernel" => cfg.rectangular_policy = value.parse().unwrap(),
            _ => panic!("unknown flag: {flag}"),
        }
    }
    cfg
}

fn parse_bool(value: &str) -> bool {
    match value {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => panic!("expected boolean 0/1/true/false, got {value}"),
    }
}

fn print_help() {
    println!("Usage: cargo run --release -- [options]");
    println!("  --rows N                    A rows (default 256)");
    println!("  --inner N                   shared dimension (default 256)");
    println!("  --cols N                    B columns (default 256)");
    println!("  --hubs N                    overlap-generator shared hubs (default 64)");
    println!("  --cancel X                  cancellation fraction 0..1 (default 0.5)");
    println!("  --synthetic MODE            overlap|sparse-output (default overlap)");
    println!("  --active-cols N             sparse-output active C columns (default 128)");
    println!("  --output-col-nnz N          sparse-output nnz per active C column (default 8)");
    println!("  --amplification-width N     sparse-output Hadamard width; 0=auto (default 0)");
    println!("  --h-buckets N               kernel-probe H rows (default 64)");
    println!("  --g-buckets N               kernel-probe G rows (default 64)");
    println!("  --degree N                  kernel-probe sketch degree (default 3)");
    println!("  --repeats N                 timing repetitions (default 3)");
    println!("  --nested-backend MODE       guv|signature|identity|off (default guv)");
    println!("  --recovery-degree N         hash-signature degree (default 5)");
    println!("  --recovery-oversampling X   hash buckets/capacity (default 2.0)");
    println!("  --guv-alpha X               GUV alpha constant (default 1.0)");
    println!("  --guv-epsilon X             GUV expansion epsilon (default 1/12)");
    println!("  --identity-fallback BOOL    use I_N when smaller (default true)");
    println!("  --guaranteed-correction BOOL exact final residual pass (default false)");
    println!("  --rect-kernel MODE          auto|dense|sparse-left|sparse-right|sparse-sparse (default auto)");
}

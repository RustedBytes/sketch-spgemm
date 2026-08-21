use std::hint::black_box;
use std::time::{Duration, Instant};

pub struct Measurement {
    iterations: u64,
    elapsed: Duration,
}

impl Measurement {
    fn ops_per_second(&self) -> f64 {
        self.iterations as f64 / self.elapsed.as_secs_f64()
    }
}

pub fn duration_from_args() -> Duration {
    let seconds = std::env::args()
        .nth(1)
        .map(|value| {
            value
                .parse::<f64>()
                .expect("benchmark duration must be a number of seconds")
        })
        .unwrap_or(1.0);
    assert!(
        seconds.is_finite() && seconds > 0.0,
        "duration must be positive"
    );
    Duration::from_secs_f64(seconds)
}

pub fn measure<F, R>(duration: Duration, mut operation: F) -> Measurement
where
    F: FnMut() -> R,
{
    // One complete warm-up prevents first-use allocation and dispatch costs
    // from dominating short benchmark runs.
    black_box(operation());

    let start = Instant::now();
    let mut iterations = 0u64;
    loop {
        black_box(operation());
        iterations += 1;
        if start.elapsed() >= duration {
            break;
        }
    }

    Measurement {
        iterations,
        elapsed: start.elapsed(),
    }
}

pub fn print_comparison(standard_name: &str, standard: &Measurement, sketch: &Measurement) {
    let standard_rate = standard.ops_per_second();
    let sketch_rate = sketch.ops_per_second();
    println!("{:<28} {:>14}", "implementation", "ops/sec");
    println!("{:<28} {:>14.2}", standard_name, standard_rate);
    println!("{:<28} {:>14.2}", "sketch-spgemm", sketch_rate);
    println!(
        "{:<28} {:>13.2}x",
        "sketch / standard",
        sketch_rate / standard_rate
    );
}

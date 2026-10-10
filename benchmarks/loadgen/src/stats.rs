//! What a run measured: counts, achieved rate and latency percentiles.

use serde_json::{Value, json};

pub struct Outcome {
    pub latency_ms: f64,
    pub error: Option<String>,
}

pub struct Report {
    pub target_rps: f64,
    pub sent: usize,
    pub ok: usize,
    pub dropped: usize,
    pub achieved_rps: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub first_error: Option<String>,
}

/// The value below which `fraction` of the sorted latencies lie (nearest rank).
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (fraction * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

impl Report {
    pub fn new(outcomes: &[Outcome], dropped: usize, target_rps: f64, duration: f64) -> Self {
        let mut sorted: Vec<f64> = outcomes
            .iter()
            .filter(|outcome| outcome.error.is_none())
            .map(|outcome| outcome.latency_ms)
            .collect();
        sorted.sort_by(f64::total_cmp);
        Self {
            target_rps,
            sent: outcomes.len(),
            ok: sorted.len(),
            dropped,
            achieved_rps: sorted.len() as f64 / duration,
            p50: percentile(&sorted, 0.50),
            p95: percentile(&sorted, 0.95),
            p99: percentile(&sorted, 0.99),
            max: sorted.last().copied().unwrap_or(f64::NAN),
            first_error: outcomes.iter().find_map(|outcome| outcome.error.clone()),
        }
    }

    /// True if every request was sent and answered.
    pub const fn clean(&self) -> bool {
        self.ok == self.sent && self.dropped == 0
    }

    pub fn summary(&self) -> String {
        let text = format!(
            "target {:.0} req/s; achieved {:.0} req/s; {}/{} ok; {} dropped; \
             p50 {:.2}ms; p95 {:.2}ms; p99 {:.2}ms; max {:.2}ms",
            self.target_rps,
            self.achieved_rps,
            self.ok,
            self.sent,
            self.dropped,
            self.p50,
            self.p95,
            self.p99,
            self.max
        );
        match &self.first_error {
            Some(error) => format!("{text}\nfirst error: {error}"),
            None => text,
        }
    }

    /// One JSON object; NaN (no successful request) becomes null.
    pub fn json(&self) -> Value {
        let number = |value: f64| {
            if value.is_finite() {
                json!(value)
            } else {
                Value::Null
            }
        };
        json!({
            "target_rps": self.target_rps,
            "achieved_rps": self.achieved_rps,
            "sent": self.sent,
            "ok": self.ok,
            "dropped": self.dropped,
            "p50_ms": number(self.p50),
            "p95_ms": number(self.p95),
            "p99_ms": number(self.p99),
            "max_ms": number(self.max),
            "first_error": self.first_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(latency_ms: f64) -> Outcome {
        Outcome {
            latency_ms,
            error: None,
        }
    }

    #[test]
    fn percentiles_use_the_nearest_rank() {
        let sorted: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((percentile(&sorted, 0.5) - 50.0).abs() < f64::EPSILON);
        assert!((percentile(&sorted, 0.99) - 99.0).abs() < f64::EPSILON);
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn failed_requests_are_counted_but_not_timed() {
        let outcomes = [
            ok(1.0),
            ok(3.0),
            Outcome {
                latency_ms: 900.0,
                error: Some("timeout".into()),
            },
        ];

        let report = Report::new(&outcomes, 0, 10.0, 1.0);

        assert_eq!((report.sent, report.ok), (3, 2));
        assert!((report.max - 3.0).abs() < f64::EPSILON);
        assert!(!report.clean());
    }

    #[test]
    fn dropped_requests_make_a_run_unclean() {
        assert!(!Report::new(&[ok(1.0)], 1, 10.0, 1.0).clean());
    }
}

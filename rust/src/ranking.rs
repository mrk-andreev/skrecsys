//! Top-k selection over a dense score matrix.
//!
//! Selecting k of n beats sorting all n, which is what `recommend` needs: k is a
//! handful and n is the catalog. Each row keeps a k-sized heap of the best candidates
//! seen so far, and rows are independent, so they run in parallel.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

/// Rank of a candidate: higher score first, then lower column index.
#[derive(Clone, Copy)]
struct Candidate {
    score: f64,
    index: usize,
}

impl Candidate {
    /// Ordering by rank, worst first, so a max-heap keeps the worst on top.
    fn worst_first(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then(self.index.cmp(&other.index))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.worst_first(other)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

/// Per row of `scores`, return the column indices of the `k` best eligible entries,
/// best first. Ties go to the lower column index.
///
/// `scores` and `eligible` are row-major with `n_cols` columns. Returns an error with
/// the offending row when it has fewer than `k` eligible entries.
pub fn top_k_per_row(
    scores: &[f64],
    eligible: &[bool],
    n_cols: usize,
    k: usize,
) -> Result<Vec<usize>, TooFewEligible> {
    if n_cols == 0 || k == 0 {
        return if k == 0 {
            Ok(Vec::new())
        } else {
            Err(TooFewEligible { row: 0, found: 0 })
        };
    }
    scores
        .par_chunks(n_cols)
        .zip(eligible.par_chunks(n_cols))
        .enumerate()
        .map(|(row, (row_scores, row_eligible))| {
            let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
            for (index, (&score, &is_eligible)) in
                row_scores.iter().zip(row_eligible.iter()).enumerate()
            {
                if !is_eligible {
                    continue;
                }
                let candidate = Candidate { score, index };
                if heap.len() < k {
                    heap.push(candidate);
                } else if let Some(worst) = heap.peek()
                    && candidate.worst_first(worst) == Ordering::Less
                {
                    heap.pop();
                    heap.push(candidate);
                }
            }
            if heap.len() < k {
                return Err(TooFewEligible {
                    row,
                    found: heap.len(),
                });
            }
            let mut best = heap.into_vec();
            best.sort_unstable_by(|a, b| a.worst_first(b));
            Ok(best.into_iter().map(|c| c.index).collect::<Vec<usize>>())
        })
        .collect::<Result<Vec<Vec<usize>>, TooFewEligible>>()
        .map(|rows| rows.concat())
}

/// A row had fewer eligible entries than the requested `k`.
pub struct TooFewEligible {
    pub row: usize,
    pub found: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select(scores: &[f64], eligible: &[bool], n_cols: usize, k: usize) -> Vec<usize> {
        top_k_per_row(scores, eligible, n_cols, k).unwrap_or_else(|_| panic!("too few eligible"))
    }

    #[test]
    fn ranks_by_score_then_by_index() {
        let scores = [0.5, 2.0, 2.0, -1.0];
        let eligible = [true; 4];
        assert_eq!(select(&scores, &eligible, 4, 4), vec![1, 2, 0, 3]);
        assert_eq!(select(&scores, &eligible, 4, 2), vec![1, 2]);
    }

    #[test]
    fn skips_ineligible_entries() {
        let scores = [3.0, 2.0, 1.0];
        assert_eq!(select(&scores, &[false, true, true], 3, 2), vec![1, 2]);
    }

    #[test]
    fn handles_several_rows() {
        let scores = [1.0, 3.0, 2.0, 9.0, 0.0, 8.0];
        let eligible = [true; 6];
        assert_eq!(select(&scores, &eligible, 3, 2), vec![1, 2, 0, 2]);
    }

    #[test]
    fn reports_the_row_with_too_few_eligible() {
        let scores = [1.0, 2.0, 3.0, 4.0];
        let eligible = [true, true, true, false];
        let error = top_k_per_row(&scores, &eligible, 2, 2).expect_err("should fail");
        assert_eq!((error.row, error.found), (1, 1));
    }

    #[test]
    fn matches_a_full_sort() {
        let n_cols = 500;
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let scores: Vec<f64> = (0..40 * n_cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Few distinct values, so ties are common.
                (state % 20) as f64
            })
            .collect();
        let eligible: Vec<bool> = (0..scores.len()).map(|i| !i.is_multiple_of(7)).collect();
        let k = 12;
        let got = top_k_per_row(&scores, &eligible, n_cols, k).unwrap_or_else(|_| unreachable!());
        for row in 0..40 {
            let mut all: Vec<Candidate> = (0..n_cols)
                .filter(|&i| eligible[row * n_cols + i])
                .map(|i| Candidate {
                    score: scores[row * n_cols + i],
                    index: i,
                })
                .collect();
            all.sort_by(|a, b| a.worst_first(b));
            let expected: Vec<usize> = all.iter().take(k).map(|c| c.index).collect();
            assert_eq!(&got[row * k..(row + 1) * k], &expected[..], "row {row}");
        }
    }
}

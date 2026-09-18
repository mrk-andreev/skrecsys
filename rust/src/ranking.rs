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
pub struct Candidate {
    pub score: f64,
    pub index: usize,
}

impl Candidate {
    /// Ordering by rank, worst first, so a max-heap keeps the worst on top.
    pub fn worst_first(&self, other: &Self) -> Ordering {
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

/// Column indices excluded from each row, in CSR layout.
///
/// A row's indices must be strictly increasing, which lets the selection walk them
/// alongside the scores instead of testing a dense mask. `recommend` hands over the
/// items a query has already interacted with, so this is `nnz` numbers rather than the
/// `n_queries * n_items` booleans a mask would cost.
pub struct Excluded<'a> {
    pub indptr: &'a [usize],
    pub indices: &'a [i64],
}

impl Excluded<'_> {
    /// The excluded column indices of `row`, ascending.
    pub fn row(&self, row: usize) -> &[i64] {
        &self.indices[self.indptr[row]..self.indptr[row + 1]]
    }
}

/// Per row of `scores`, return the column indices of the `k` best entries that are not
/// excluded, best first. Ties go to the lower column index.
///
/// `scores` is row-major with `n_cols` columns. Returns an error with the offending row
/// when it has fewer than `k` entries left.
pub fn top_k_per_row(
    scores: &[f64],
    excluded: &Excluded<'_>,
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
    // This is the predict path, so the result is written straight into one flat buffer:
    // a `Vec` per row plus a concatenating copy would allocate once per query.
    let n_rows = scores.len() / n_cols;
    let mut selected = vec![0usize; n_rows * k];
    let outcome = selected
        .par_chunks_mut(k)
        .zip(scores.par_chunks(n_cols))
        .enumerate()
        .try_for_each(|(row, (out, row_scores))| {
            let skip = excluded.row(row);
            let mut next_skipped = 0usize;
            let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
            for (index, &score) in row_scores.iter().enumerate() {
                // The excluded indices ascend, so one cursor keeps up with the scan.
                if next_skipped < skip.len() && skip[next_skipped] == index as i64 {
                    next_skipped += 1;
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
            for (slot, candidate) in out.iter_mut().zip(&best) {
                *slot = candidate.index;
            }
            Ok(())
        });
    outcome.map(|()| selected)
}

/// A row had fewer eligible entries than the requested `k`.
pub struct TooFewEligible {
    pub row: usize,
    pub found: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build CSR exclusions from one list of excluded columns per row.
    fn excluded(rows: &[Vec<usize>]) -> (Vec<usize>, Vec<i64>) {
        let mut indptr = vec![0usize];
        let mut indices: Vec<i64> = Vec::new();
        for row in rows {
            indices.extend(row.iter().map(|&c| c as i64));
            indptr.push(indices.len());
        }
        (indptr, indices)
    }

    fn select_excluding(
        scores: &[f64],
        rows: &[Vec<usize>],
        n_cols: usize,
        k: usize,
    ) -> Vec<usize> {
        let (indptr, indices) = excluded(rows);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        top_k_per_row(scores, &excluded, n_cols, k).unwrap_or_else(|_| panic!("too few eligible"))
    }

    fn select(scores: &[f64], n_cols: usize, k: usize) -> Vec<usize> {
        let rows = vec![Vec::new(); scores.len() / n_cols];
        select_excluding(scores, &rows, n_cols, k)
    }

    #[test]
    fn ranks_by_score_then_by_index() {
        let scores = [0.5, 2.0, 2.0, -1.0];
        assert_eq!(select(&scores, 4, 4), vec![1, 2, 0, 3]);
        assert_eq!(select(&scores, 4, 2), vec![1, 2]);
    }

    #[test]
    fn skips_excluded_entries() {
        let scores = [3.0, 2.0, 1.0];
        assert_eq!(select_excluding(&scores, &[vec![0]], 3, 2), vec![1, 2]);
        assert_eq!(select_excluding(&scores, &[vec![0, 1]], 3, 1), vec![2]);
    }

    #[test]
    fn excludes_per_row() {
        let scores = [1.0, 3.0, 2.0, 9.0, 0.0, 8.0];
        let rows = vec![vec![1], vec![0, 2]];
        assert_eq!(select_excluding(&scores, &rows, 3, 1), vec![2, 1]);
    }

    #[test]
    fn handles_several_rows() {
        let scores = [1.0, 3.0, 2.0, 9.0, 0.0, 8.0];
        assert_eq!(select(&scores, 3, 2), vec![1, 2, 0, 2]);
    }

    #[test]
    fn reports_the_row_with_too_few_eligible() {
        let scores = [1.0, 2.0, 3.0, 4.0];
        let (indptr, indices) = excluded(&[vec![], vec![1]]);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        let error = top_k_per_row(&scores, &excluded, 2, 2).expect_err("should fail");
        assert_eq!((error.row, error.found), (1, 1));
    }

    #[test]
    fn matches_a_full_sort() {
        let n_cols: usize = 500;
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
        let rows: Vec<Vec<usize>> = (0..40)
            .map(|_| (0..n_cols).filter(|&i| i.is_multiple_of(7)).collect())
            .collect();
        let eligible: Vec<bool> = (0..scores.len())
            .map(|i| !(i % n_cols).is_multiple_of(7))
            .collect();
        let k = 12;
        let got = select_excluding(&scores, &rows, n_cols, k);
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

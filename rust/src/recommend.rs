//! Fused scoring and top-k selection for the item-item recommenders.
//!
//! `recommend` used to materialize a dense `(n_queries, n_items)` score matrix, rank it
//! and throw all but `k` columns per row away. Here a query row is accumulated with the
//! SMMP accumulator, has its seen items removed and is reduced to its `k` best entries
//! before the next one starts, so the dense row never outlives the query and never
//! leaves the native side.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::ranking::{Candidate, Excluded, TooFewEligible};
use crate::sparse::{Accumulator, Csr};

/// For each row `q` of `users`, the `k` best candidates of `users * similarity`, best
/// first, with ties going to the lower candidate position.
///
/// `candidates` holds the item index of each candidate position, ascending, and
/// `position[item]` is its inverse, `-1` for an item nobody may be recommended. The
/// excluded columns are candidate positions, as in [`crate::ranking::top_k_per_row`].
///
/// Returns the candidate positions and their scores, both `(n_rows, k)` row-major.
pub fn top_k_from_similarity(
    users: &Csr,
    similarity: &Csr,
    candidates: &[usize],
    position: &[i64],
    excluded: &Excluded<'_>,
    k: usize,
) -> Result<(Vec<usize>, Vec<f64>), TooFewEligible> {
    let n_candidates = candidates.len();
    let mut order = vec![0usize; users.n_rows * k];
    let mut scores = vec![0.0f64; users.n_rows * k];
    if k == 0 {
        return Ok((order, scores));
    }

    let outcome = order
        .par_chunks_mut(k)
        .zip(scores.par_chunks_mut(k))
        .enumerate()
        .try_for_each_init(
            || {
                (
                    Accumulator::new(similarity.n_cols),
                    vec![false; n_candidates],
                )
            },
            |(acc, dropped), (row, (out_order, out_scores))| {
                let skip = excluded.row(row);
                let available = n_candidates - skip.len();
                if available < k {
                    return Err(TooFewEligible {
                        row,
                        found: available,
                    });
                }
                for &p in skip {
                    dropped[p as usize] = true;
                }
                for (item, weight) in users.row(row) {
                    for (j, w) in similarity.row(item) {
                        acc.add(j, weight * w);
                    }
                }

                let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
                for &j in acc.touched() {
                    let p = position[j];
                    if p < 0 || dropped[p as usize] {
                        continue;
                    }
                    let candidate = Candidate {
                        score: acc.get(j),
                        index: p as usize,
                    };
                    if heap.len() < k {
                        heap.push(candidate);
                    } else if let Some(worst) = heap.peek()
                        && candidate.worst_first(worst) == Ordering::Less
                    {
                        heap.pop();
                        heap.push(candidate);
                    }
                }
                let mut scored = heap.into_vec();
                scored.sort_unstable_by(|a, b| a.worst_first(b));

                // A candidate the product never reached scores exactly zero, which the
                // dense path ranked alongside the rest. That can only matter when fewer
                // than `k` were reached, or when the worst score kept is not positive.
                let zeros = if scored.len() < k || scored[scored.len() - 1].score <= 0.0 {
                    let mut zeros: Vec<Candidate> = Vec::with_capacity(k);
                    for (p, &j) in candidates.iter().enumerate() {
                        if zeros.len() == k {
                            break;
                        }
                        if !dropped[p] && !acc.is_touched(j) {
                            zeros.push(Candidate {
                                score: 0.0,
                                index: p,
                            });
                        }
                    }
                    zeros
                } else {
                    Vec::new()
                };

                acc.reset();
                for &p in skip {
                    dropped[p as usize] = false;
                }
                merge(out_order, out_scores, &scored, &zeros);
                Ok(())
            },
        );
    outcome.map(|()| (order, scores))
}

/// Fill one row of the output from the two ranked runs, best first.
fn merge(order: &mut [usize], scores: &mut [f64], scored: &[Candidate], zeros: &[Candidate]) {
    let (mut next_scored, mut next_zero) = (0usize, 0usize);
    for slot in 0..order.len() {
        let take_scored = match (scored.get(next_scored), zeros.get(next_zero)) {
            (Some(a), Some(b)) => a.worst_first(b) == Ordering::Less,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => return,
        };
        let pick = if take_scored {
            next_scored += 1;
            scored[next_scored - 1]
        } else {
            next_zero += 1;
            zeros[next_zero - 1]
        };
        order[slot] = pick.index;
        scores[slot] = pick.score;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::{Owned, pseudo_random_dense};

    struct Case {
        users: Owned,
        similarity: Owned,
        candidates: Vec<usize>,
        position: Vec<i64>,
        indptr: Vec<usize>,
        indices: Vec<i64>,
    }

    impl Case {
        fn new(users: &[Vec<f64>], similarity: &[Vec<f64>], candidates: Vec<usize>) -> Self {
            let n_items = similarity[0].len();
            let mut position = vec![-1i64; n_items];
            for (p, &j) in candidates.iter().enumerate() {
                position[j] = p as i64;
            }
            Self {
                users: Owned::from_dense(users),
                similarity: Owned::from_dense(similarity),
                candidates,
                position,
                indptr: vec![0; users.len() + 1],
                indices: Vec::new(),
            }
        }

        fn exclude(mut self, rows: &[Vec<usize>]) -> Self {
            self.indptr = vec![0];
            self.indices = Vec::new();
            for row in rows {
                self.indices.extend(row.iter().map(|&c| c as i64));
                self.indptr.push(self.indices.len());
            }
            self
        }

        fn run(&self, k: usize) -> (Vec<usize>, Vec<f64>) {
            top_k_from_similarity(
                &self.users.csr(),
                &self.similarity.csr(),
                &self.candidates,
                &self.position,
                &Excluded {
                    indptr: &self.indptr,
                    indices: &self.indices,
                },
                k,
            )
            .unwrap_or_else(|_| panic!("too few eligible"))
        }
    }

    /// The dense product this kernel replaces, restricted to the candidates.
    fn dense_scores(
        users: &[Vec<f64>],
        similarity: &[Vec<f64>],
        candidates: &[usize],
    ) -> Vec<Vec<f64>> {
        users
            .iter()
            .map(|u| {
                candidates
                    .iter()
                    .map(|&j| (0..u.len()).map(|i| u[i] * similarity[i][j]).sum())
                    .collect()
            })
            .collect()
    }

    fn rank(scores: &[f64], excluded: &[usize], k: usize) -> (Vec<usize>, Vec<f64>) {
        let mut all: Vec<Candidate> = scores
            .iter()
            .enumerate()
            .filter(|(p, _)| !excluded.contains(p))
            .map(|(index, &score)| Candidate { score, index })
            .collect();
        all.sort_by(|a, b| a.worst_first(b));
        all.truncate(k);
        (
            all.iter().map(|c| c.index).collect(),
            all.iter().map(|c| c.score).collect(),
        )
    }

    #[test]
    fn matches_the_dense_product_and_ranking() {
        let users = pseudo_random_dense(40, 25);
        let similarity = pseudo_random_dense(25, 25);
        let candidates: Vec<usize> = (0..25).collect();
        let case = Case::new(&users, &similarity, candidates.clone());
        let k = 6;
        let (order, scores) = case.run(k);
        let dense = dense_scores(&users, &similarity, &candidates);
        for row in 0..40 {
            let (want_order, want_scores) = rank(&dense[row], &[], k);
            assert_eq!(&order[row * k..(row + 1) * k], &want_order[..], "row {row}");
            for (got, want) in scores[row * k..(row + 1) * k].iter().zip(&want_scores) {
                assert!((got - want).abs() < 1e-12, "row {row}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn falls_back_to_untouched_candidates() {
        // User 0 reaches item 1 only, so the other candidates score zero and rank by
        // position; a negative weight must rank below those zeros.
        let users = vec![vec![1.0, 0.0, 0.0]];
        let similarity = vec![vec![0.0, -2.0, 0.0], vec![0.0; 3], vec![0.0; 3]];
        let case = Case::new(&users, &similarity, vec![0, 1, 2]);
        let (order, scores) = case.run(3);
        assert_eq!(order, vec![0, 2, 1]);
        assert_eq!(scores, vec![0.0, 0.0, -2.0]);
    }

    #[test]
    fn honours_exclusions_and_candidate_subsets() {
        let users = vec![vec![1.0, 1.0, 0.0, 0.0]];
        let similarity = vec![
            vec![0.0, 1.0, 5.0, 2.0],
            vec![0.0, 0.0, 3.0, 9.0],
            vec![0.0; 4],
            vec![0.0; 4],
        ];
        // Candidates are items 2 and 3, whose positions are 0 and 1; exclude item 3.
        let case = Case::new(&users, &similarity, vec![2, 3]).exclude(&[vec![1]]);
        let (order, scores) = case.run(1);
        assert_eq!((order, scores), (vec![0], vec![8.0]));
    }

    #[test]
    fn reports_a_query_with_too_few_candidates() {
        let users = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let similarity = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        let case = Case::new(&users, &similarity, vec![0, 1]).exclude(&[vec![], vec![0]]);
        let error = top_k_from_similarity(
            &case.users.csr(),
            &case.similarity.csr(),
            &case.candidates,
            &case.position,
            &Excluded {
                indptr: &case.indptr,
                indices: &case.indices,
            },
            2,
        )
        .expect_err("should fail");
        assert_eq!((error.row, error.found), (1, 1));
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let users = pseudo_random_dense(120, 40);
        let similarity = pseudo_random_dense(40, 40);
        let case = Case::new(&users, &similarity, (0..40).collect());
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| case.run(9))
        };
        assert_eq!(run(1), run(4));
    }
}

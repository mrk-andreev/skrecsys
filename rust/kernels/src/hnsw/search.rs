//! Traversal of an HNSW graph, and the top-k it produces.
//!
//! The layer primitives are shared with construction: an insertion is a search for the
//! node being inserted, so there is one implementation of "walk this level and keep the
//! `ef` best", used once per level by a build and once per query by a search.
//!
//! Exclusions and candidate filters are applied **at admission, not by over-fetching**.
//! A filtered-out item still carries the graph's connectivity -- the traversal walks
//! through it -- but never enters the result set. So `ef_search` is sized against `k`
//! alone, with no over-fetch factor to tune and no cliff for a user with a long history.
//! When the filter is selective enough that the result set never fills, the stop test
//! never fires and the walk continues; that is bounded by the catalog, and the Python
//! side keeps it from being reached by falling back to exact scoring for small candidate
//! sets.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use rayon::prelude::*;

use super::graph::{GraphView, Links};
use super::visited::Visited;
use crate::ranking::{Candidate, Excluded, TooFewEligible};
use crate::vectors::{Items, Queries, Scatter};

/// Scratch one worker reuses across the queries it handles.
pub struct Workspace {
    visited: Visited,
    neighbours: Vec<u32>,
    scatter: Option<Scatter>,
    dropped: Vec<bool>,
}

impl Workspace {
    pub fn new(n_nodes: usize, n_candidates: usize, scatter_dim: Option<usize>) -> Self {
        Self {
            visited: Visited::new(n_nodes),
            neighbours: Vec::new(),
            scatter: scatter_dim.map(Scatter::new),
            dropped: vec![false; n_candidates],
        }
    }
}

/// Greedy descent at one level: step to the best neighbour until none is better.
///
/// Used for the levels above the one an insertion or a search does its work on, where a
/// single best entry point is all the next level down needs.
pub fn greedy(
    score: &impl Fn(usize) -> f64,
    links: &impl Links,
    level: usize,
    entry: Candidate,
    neighbours: &mut Vec<u32>,
) -> Candidate {
    let mut best = entry;
    loop {
        neighbours.clear();
        links.neighbors(best.index, level, neighbours);
        let mut improved = false;
        for &index in neighbours.iter() {
            let index = index as usize;
            let candidate = Candidate {
                score: score(index),
                index,
            };
            if candidate.worst_first(&best) == Ordering::Less {
                best = candidate;
                improved = true;
            }
        }
        if !improved {
            return best;
        }
    }
}

/// Walk `level` from `entries`, returning the `ef` best admitted nodes, best first.
///
/// `admit` decides what may be *returned*; everything reachable is still walked.
#[allow(clippy::too_many_arguments)]
pub fn search_layer(
    score: &impl Fn(usize) -> f64,
    admit: &impl Fn(usize) -> bool,
    links: &impl Links,
    level: usize,
    entries: &[Candidate],
    ef: usize,
    visited: &mut Visited,
    neighbours: &mut Vec<u32>,
) -> Vec<Candidate> {
    // `results` orders worst first, so a max-heap evicts the weakest kept; `frontier`
    // wraps the same order in `Reverse`, so it pops the most promising left to expand.
    let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);
    let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef + 1);
    for &entry in entries {
        if !visited.insert(entry.index) {
            continue;
        }
        frontier.push(Reverse(entry));
        if admit(entry.index) {
            keep(&mut results, entry, ef);
        }
    }

    while let Some(Reverse(current)) = frontier.pop() {
        if results.len() >= ef
            && let Some(worst) = results.peek()
            && current.worst_first(worst) == Ordering::Greater
        {
            break;
        }
        neighbours.clear();
        links.neighbors(current.index, level, neighbours);
        for &index in neighbours.iter() {
            let index = index as usize;
            if !visited.insert(index) {
                continue;
            }
            let candidate = Candidate {
                score: score(index),
                index,
            };
            // Only prune the frontier once the results are full: while a selective
            // filter keeps them short there is no bound to prune against, and stopping
            // would return fewer than the caller asked for.
            if results.len() >= ef
                && results
                    .peek()
                    .is_some_and(|worst| candidate.worst_first(worst) == Ordering::Greater)
            {
                continue;
            }
            frontier.push(Reverse(candidate));
            if admit(index) {
                keep(&mut results, candidate, ef);
            }
        }
    }

    let mut best = results.into_vec();
    best.sort_unstable_by(|a, b| a.worst_first(b));
    best
}

/// Push `candidate` into an `ef`-bounded result heap, evicting the weakest.
fn keep(results: &mut BinaryHeap<Candidate>, candidate: Candidate, ef: usize) {
    results.push(candidate);
    if results.len() > ef {
        results.pop();
    }
}

/// Walk the whole graph down to `level 0` and return the `ef` best admitted nodes.
///
/// Takes the entry point and its level rather than a [`GraphView`], so that the build
/// can walk its own half-finished graph with the same code a query uses.
#[allow(clippy::too_many_arguments)]
pub fn search_graph(
    score: &impl Fn(usize) -> f64,
    admit: &impl Fn(usize) -> bool,
    links: &impl Links,
    entry_point: usize,
    top_level: usize,
    ef: usize,
    visited: &mut Visited,
    neighbours: &mut Vec<u32>,
) -> Vec<Candidate> {
    let mut current = Candidate {
        score: score(entry_point),
        index: entry_point,
    };
    for level in (1..=top_level).rev() {
        current = greedy(score, links, level, current, neighbours);
    }
    // The upper levels only navigated; level 0 is where the marks start mattering.
    visited.clear();
    search_layer(score, admit, links, 0, &[current], ef, visited, neighbours)
}

/// The `k` best candidate positions of each query under the graph, and their scores.
///
/// `candidates` holds the item index of each candidate position, ascending, and
/// `position` is its inverse over the catalog with `-1` for items that are not
/// candidates -- the convention [`crate::recommend::top_k_from_similarity`] already
/// uses. Excluded columns are candidate positions, as in [`crate::ranking`].
///
/// A query the traversal leaves short is completed by an exact scan of its eligible
/// items, so the result is always `k` items the caller may show. That costs one exact
/// query and happens when the graph is disconnected or the filter was selective, which
/// is rare and bounded -- the alternative is a short list, and `recommend` promises `k`.
#[allow(clippy::too_many_arguments)]
pub fn top_k(
    items: &impl Items,
    graph: &GraphView<'_>,
    queries: &Queries<'_>,
    candidates: &[usize],
    position: &[i64],
    excluded: &Excluded<'_>,
    k: usize,
    ef: usize,
) -> Result<(Vec<usize>, Vec<f64>), TooFewEligible> {
    let n_queries = queries.len();
    let mut selected = vec![0usize; n_queries * k];
    let mut scores = vec![0.0f64; n_queries * k];
    if k == 0 || n_queries == 0 {
        return Ok((selected, scores));
    }

    let outcome = selected
        .par_chunks_mut(k)
        .zip(scores.par_chunks_mut(k))
        .enumerate()
        .try_for_each_init(
            || Workspace::new(graph.len(), candidates.len(), queries.scatter_dim()),
            |work, (row, (out_index, out_score))| {
                // Destructured so that the admission mask can be borrowed immutably by
                // the probe closure while the traversal scratch is still borrowed
                // mutably: they are disjoint fields, and this is what shows it.
                let Workspace {
                    visited,
                    neighbours,
                    scatter,
                    dropped,
                } = work;
                let skip = excluded.row(row);
                for &p in skip {
                    dropped[p as usize] = true;
                }
                let eligible = candidates.len() - skip.len();
                let result = if eligible < k {
                    Err(TooFewEligible {
                        row,
                        found: eligible,
                    })
                } else {
                    let mask: &[bool] = dropped;
                    let found = queries.with_probe(row, scatter, |probe| {
                        let score = |item: usize| items.score(item, probe);
                        let admit = |item: usize| {
                            let p = position[item];
                            p >= 0 && !mask[p as usize]
                        };
                        let best = search_graph(
                            &score,
                            &admit,
                            graph,
                            graph.entry_point,
                            graph.top_level,
                            ef.max(k),
                            visited,
                            neighbours,
                        );
                        if best.len() < k {
                            exhaustive(&score, &admit, candidates, k)
                        } else {
                            best
                        }
                    });
                    for (slot, candidate) in out_index.iter_mut().zip(&found) {
                        *slot = position[candidate.index] as usize;
                    }
                    for (slot, candidate) in out_score.iter_mut().zip(&found) {
                        *slot = candidate.score;
                    }
                    Ok(())
                };
                for &p in skip {
                    dropped[p as usize] = false;
                }
                result
            },
        );
    outcome.map(|()| (selected, scores))
}

/// Every eligible candidate, scored, reduced to the `k` best.
fn exhaustive(
    score: &impl Fn(usize) -> f64,
    admit: &impl Fn(usize) -> bool,
    candidates: &[usize],
    k: usize,
) -> Vec<Candidate> {
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
    for &item in candidates {
        if !admit(item) {
            continue;
        }
        keep(
            &mut heap,
            Candidate {
                score: score(item),
                index: item,
            },
            k,
        );
    }
    let mut best = heap.into_vec();
    best.sort_unstable_by(|a, b| a.worst_first(b));
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::build::tests::{pseudo_vectors, unit_vectors};
    use crate::hnsw::graph::Params;
    use crate::sparse::testing::Owned;
    use crate::vectors::{DenseItems, SparseItems};

    struct Case {
        graph: crate::hnsw::graph::Graph,
        candidates: Vec<usize>,
        position: Vec<i64>,
    }

    fn build_case(items: &impl Items, m: usize) -> Case {
        let graph = crate::hnsw::build::build(
            items,
            &Params {
                m,
                ef_construction: 100,
                seed: 4242,
            },
        );
        let candidates: Vec<usize> = (0..items.len()).collect();
        let position: Vec<i64> = (0..items.len()).map(|i| i as i64).collect();
        Case {
            graph,
            candidates,
            position,
        }
    }

    impl Case {
        fn view(&self) -> GraphView<'_> {
            GraphView::new(
                &self.graph.node_level,
                &self.graph.links_indptr,
                &self.graph.links_indices,
                self.graph.entry_point,
            )
            .expect("a built graph describes itself")
        }
    }

    /// Empty exclusions for `n_queries` rows.
    fn no_exclusions(n_queries: usize) -> (Vec<usize>, Vec<i64>) {
        (vec![0; n_queries + 1], Vec::new())
    }

    /// The exact `k` best candidate positions of each query, by a full scan.
    fn brute_force(
        items: &impl Items,
        queries: &Queries<'_>,
        candidates: &[usize],
        dropped: &dyn Fn(usize, usize) -> bool,
        k: usize,
    ) -> Vec<Vec<usize>> {
        (0..queries.len())
            .map(|q| {
                let mut scatter = queries.scatter_dim().map(Scatter::new);
                let mut scored: Vec<Candidate> = queries.with_probe(q, &mut scatter, |probe| {
                    candidates
                        .iter()
                        .enumerate()
                        .filter(|&(position, _)| !dropped(q, position))
                        .map(|(position, &item)| Candidate {
                            score: items.score(item, probe),
                            index: position,
                        })
                        .collect()
                });
                scored.sort_unstable_by(|a, b| a.worst_first(b));
                scored.into_iter().take(k).map(|c| c.index).collect()
            })
            .collect()
    }

    fn recall(found: &[usize], exact: &[Vec<usize>], k: usize) -> f64 {
        let hits: usize = exact
            .iter()
            .enumerate()
            .map(|(row, want)| {
                let got = &found[row * k..(row + 1) * k];
                want.iter().filter(|w| got.contains(w)).count()
            })
            .sum();
        hits as f64 / (exact.len() * k) as f64
    }

    #[test]
    fn recall_against_brute_force_on_dense_vectors() {
        let vectors = unit_vectors(3000, 16);
        let items = DenseItems {
            vectors: &vectors,
            dim: 16,
        };
        let case = build_case(&items, 16);
        let query_data = unit_vectors(200, 16);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 16,
        };
        let (indptr, indices) = no_exclusions(200);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };

        let exact = brute_force(&items, &queries, &case.candidates, &|_, _| false, 10);
        let (found, scores) = top_k(
            &items,
            &case.view(),
            &queries,
            &case.candidates,
            &case.position,
            &excluded,
            10,
            64,
        )
        .expect("every item is eligible");

        let recall = recall(&found, &exact, 10);
        assert!(recall >= 0.95, "recall@10 was {recall}");
        // Whatever it found, it must have scored correctly and ranked best first.
        for row in 0..200 {
            let row_scores = &scores[row * 10..(row + 1) * 10];
            assert!(
                row_scores.windows(2).all(|w| w[0] >= w[1]),
                "row {row} not descending"
            );
        }
    }

    #[test]
    fn recall_improves_with_ef_search() {
        let vectors = unit_vectors(2000, 16);
        let items = DenseItems {
            vectors: &vectors,
            dim: 16,
        };
        let case = build_case(&items, 8);
        let query_data = unit_vectors(150, 16);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 16,
        };
        let (indptr, indices) = no_exclusions(150);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        let exact = brute_force(&items, &queries, &case.candidates, &|_, _| false, 10);

        let measured: Vec<f64> = [10usize, 32, 128]
            .into_iter()
            .map(|ef| {
                let (found, _) = top_k(
                    &items,
                    &case.view(),
                    &queries,
                    &case.candidates,
                    &case.position,
                    &excluded,
                    10,
                    ef,
                )
                .expect("every item is eligible");
                recall(&found, &exact, 10)
            })
            .collect();
        assert!(
            measured.windows(2).all(|w| w[1] >= w[0]),
            "recall did not grow with ef_search: {measured:?}"
        );
        assert!(
            measured[2] >= 0.95,
            "recall@10 at ef=128 was {}",
            measured[2]
        );
    }

    #[test]
    fn recall_against_brute_force_on_sparse_vectors() {
        // The item-item shape: short sparse rows over a catalog-sized dimension, and
        // queries that are themselves sparse interaction rows.
        let dense: Vec<Vec<f64>> = crate::sparse::testing::pseudo_random_dense(800, 800);
        let owned = Owned::from_dense(&dense);
        let items = SparseItems {
            matrix: owned.csr(),
        };
        let case = build_case(&items, 16);
        let query_rows = crate::sparse::testing::pseudo_random_dense(100, 800);
        let query_owned = Owned::from_dense(&query_rows);
        let queries = Queries::Scattered(query_owned.csr());
        let (indptr, indices) = no_exclusions(100);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };

        let exact = brute_force(&items, &queries, &case.candidates, &|_, _| false, 10);
        let (found, _) = top_k(
            &items,
            &case.view(),
            &queries,
            &case.candidates,
            &case.position,
            &excluded,
            10,
            64,
        )
        .expect("every item is eligible");
        let recall = recall(&found, &exact, 10);
        assert!(recall >= 0.90, "sparse recall@10 was {recall}");
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let vectors = unit_vectors(1200, 16);
        let items = DenseItems {
            vectors: &vectors,
            dim: 16,
        };
        let case = build_case(&items, 8);
        let query_data = unit_vectors(64, 16);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 16,
        };
        let (indptr, indices) = no_exclusions(64);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };

        let run = || {
            top_k(
                &items,
                &case.view(),
                &queries,
                &case.candidates,
                &case.position,
                &excluded,
                10,
                48,
            )
            .expect("every item is eligible")
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a pool");
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("a pool");
        assert_eq!(one.install(run).0, many.install(run).0);
    }

    #[test]
    fn excluded_items_are_never_returned() {
        let vectors = unit_vectors(1000, 8);
        let items = DenseItems {
            vectors: &vectors,
            dim: 8,
        };
        let case = build_case(&items, 8);
        let query_data = unit_vectors(40, 8);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 8,
        };

        // Exclude each query's own exact best twenty, which is the worst case: every
        // exclusion is somewhere the traversal wants to go.
        let exact = brute_force(&items, &queries, &case.candidates, &|_, _| false, 20);
        let mut indptr = vec![0usize];
        let mut indices: Vec<i64> = Vec::new();
        for row in &exact {
            let mut row = row.clone();
            row.sort_unstable();
            indices.extend(row.iter().map(|&p| p as i64));
            indptr.push(indices.len());
        }
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };

        let (found, _) = top_k(
            &items,
            &case.view(),
            &queries,
            &case.candidates,
            &case.position,
            &excluded,
            10,
            64,
        )
        .expect("990 items remain eligible");
        for (row, banned) in exact.iter().enumerate() {
            for position in &found[row * 10..(row + 1) * 10] {
                assert!(
                    !banned.contains(position),
                    "row {row} returned excluded {position}"
                );
            }
        }
    }

    #[test]
    fn only_candidates_are_returned() {
        let vectors = unit_vectors(600, 8);
        let items = DenseItems {
            vectors: &vectors,
            dim: 8,
        };
        let case = build_case(&items, 8);
        // Every third item, so the filter is selective enough to matter.
        let candidates: Vec<usize> = (0..600).step_by(3).collect();
        let mut position = vec![-1i64; 600];
        for (p, &item) in candidates.iter().enumerate() {
            position[item] = p as i64;
        }
        let query_data = unit_vectors(30, 8);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 8,
        };
        let (indptr, indices) = no_exclusions(30);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };

        let (found, _) = top_k(
            &items,
            &case.view(),
            &queries,
            &candidates,
            &position,
            &excluded,
            10,
            64,
        )
        .expect("200 candidates remain eligible");
        assert!(found.iter().all(|&p| p < candidates.len()));
        let exact = brute_force(&items, &queries, &candidates, &|_, _| false, 10);
        assert!(recall(&found, &exact, 10) >= 0.90);
    }

    #[test]
    fn a_disconnected_graph_still_returns_k_items() {
        // The exhaustive completion: a graph with no links at all finds nothing by
        // walking, and `recommend` still promises k eligible items.
        let vectors = unit_vectors(200, 8);
        let items = DenseItems {
            vectors: &vectors,
            dim: 8,
        };
        let node_level = vec![0i32; 200];
        let links_indptr = vec![0i64; 201];
        let links_indices: Vec<i32> = Vec::new();
        let view = GraphView::new(&node_level, &links_indptr, &links_indices, 0)
            .expect("an edgeless graph is still a graph");
        let candidates: Vec<usize> = (0..200).collect();
        let position: Vec<i64> = (0..200).map(|i| i as i64).collect();
        let query_data = unit_vectors(20, 8);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 8,
        };
        let (indptr, indices) = no_exclusions(20);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };

        let (found, _) = top_k(
            &items,
            &view,
            &queries,
            &candidates,
            &position,
            &excluded,
            10,
            32,
        )
        .expect("every item is eligible");
        let exact = brute_force(&items, &queries, &candidates, &|_, _| false, 10);
        // Falling back to a full scan means the answer is not approximate at all.
        assert_eq!(recall(&found, &exact, 10), 1.0);
    }

    #[test]
    fn too_few_eligible_items_is_an_error_naming_the_query() {
        let vectors = unit_vectors(50, 8);
        let items = DenseItems {
            vectors: &vectors,
            dim: 8,
        };
        let case = build_case(&items, 8);
        let query_data = unit_vectors(3, 8);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 8,
        };
        // Row 1 has every item but two excluded, and is asked for ten.
        let mut indptr = vec![0usize];
        let mut indices: Vec<i64> = Vec::new();
        for row in 0..3 {
            if row == 1 {
                indices.extend((0..48).map(|p| p as i64));
            }
            indptr.push(indices.len());
        }
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        let error = top_k(
            &items,
            &case.view(),
            &queries,
            &case.candidates,
            &case.position,
            &excluded,
            10,
            32,
        )
        .expect_err("row 1 has two eligible items");
        assert_eq!(error.row, 1);
        assert_eq!(error.found, 2);
    }

    #[test]
    fn ties_break_by_candidate_position() {
        // Every item identical, so every score ties and only the tie-break decides.
        // The exact path resolves by fitted item order, and so must this one.
        let vectors = vec![1.0; 40 * 4];
        let items = DenseItems {
            vectors: &vectors,
            dim: 4,
        };
        let case = build_case(&items, 8);
        let query_data = vec![1.0; 4];
        let queries = Queries::Dense {
            data: &query_data,
            dim: 4,
        };
        let (indptr, indices) = no_exclusions(1);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        let (found, _) = top_k(
            &items,
            &case.view(),
            &queries,
            &case.candidates,
            &case.position,
            &excluded,
            5,
            32,
        )
        .expect("every item is eligible");
        assert_eq!(found, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn an_empty_batch_and_a_zero_k_return_nothing() {
        let vectors = unit_vectors(20, 4);
        let items = DenseItems {
            vectors: &vectors,
            dim: 4,
        };
        let case = build_case(&items, 8);
        let (indptr, indices) = no_exclusions(0);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        let empty = Queries::Dense { data: &[], dim: 4 };
        let (found, scores) = top_k(
            &items,
            &case.view(),
            &empty,
            &case.candidates,
            &case.position,
            &excluded,
            5,
            32,
        )
        .expect("nothing to rank");
        assert!(found.is_empty() && scores.is_empty());
    }

    #[test]
    fn raw_inner_product_still_ranks_what_it_finds() {
        // Vectors with heterogeneous norms are the awkward case for an inner-product
        // graph: a low-norm item loses to its neighbours for every query, so it is
        // dominated rather than merely unlinked. Recall is lower than on the sphere,
        // and this pins how much lower rather than pretending otherwise.
        let vectors = pseudo_vectors(2000, 16);
        let items = DenseItems {
            vectors: &vectors,
            dim: 16,
        };
        let case = build_case(&items, 16);
        let query_data = pseudo_vectors(150, 16);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 16,
        };
        let (indptr, indices) = no_exclusions(150);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        let exact = brute_force(&items, &queries, &case.candidates, &|_, _| false, 10);
        let (found, _) = top_k(
            &items,
            &case.view(),
            &queries,
            &case.candidates,
            &case.position,
            &excluded,
            10,
            64,
        )
        .expect("every item is eligible");
        let recall = recall(&found, &exact, 10);
        assert!(recall >= 0.90, "raw inner-product recall@10 was {recall}");
    }
}

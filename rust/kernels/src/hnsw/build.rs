//! Inserting nodes into an HNSW graph.
//!
//! An insertion is a search: the node being inserted is its own query, and the `ef`
//! best nodes each level turns up are what it links to. So construction and retrieval
//! share [`search_layer`], and the only thing construction adds is which of those
//! candidates to keep.
//!
//! Nodes are inserted in descending level, then ascending index. The tallest node is
//! therefore in place before anything else arrives, which makes it the entry point for
//! the whole build and means no lock is needed to maintain one.

use rayon::prelude::*;

use super::graph::{BuildLinks, Graph, Links, Params};
use super::search::{greedy, search_graph, search_layer};
use super::visited::Visited;
use crate::ranking::Candidate;
use crate::vectors::Items;

/// Build a hierarchical navigable small-world graph over `items`.
///
/// Runs on the rayon pool the caller installed. On one thread the result is a function
/// of `items` and `params` alone; on more, concurrent insertions see different partial
/// graphs and the links vary between runs, though the levels never do.
pub fn build(items: &impl Items, params: &Params) -> Graph {
    let n_nodes = items.len();
    let node_level: Vec<usize> = (0..n_nodes).map(|node| params.level_of(node)).collect();
    let links = BuildLinks::new(node_level.clone(), params);
    if n_nodes == 0 {
        return links.freeze(0);
    }

    let mut order: Vec<usize> = (0..n_nodes).collect();
    order.sort_unstable_by(|&a, &b| node_level[b].cmp(&node_level[a]).then(a.cmp(&b)));
    let entry = order[0];
    let entry_level = node_level[entry];

    let insert = |scratch: &mut (Visited, Vec<u32>), &node: &usize| {
        let (visited, neighbours) = scratch;
        insert_one(
            node,
            items,
            &links,
            params,
            entry,
            entry_level,
            visited,
            neighbours,
        );
    };
    // The tallest node seeds the graph with no links of its own; everything else
    // attaches to what is already there.
    let rest = &order[1..];
    if rayon::current_num_threads() == 1 {
        let mut scratch = (Visited::new(n_nodes), Vec::new());
        for node in rest {
            insert(&mut scratch, node);
        }
    } else {
        rest.par_iter()
            .for_each_init(|| (Visited::new(n_nodes), Vec::new()), insert);
    }
    repair(items, &links, params, entry, entry_level, n_nodes);
    links.freeze(entry)
}

/// Link any node the entry point cannot reach back into the graph.
///
/// Pruning is one-sided: a node keeps its own links when a neighbour drops it, so a
/// cluster can end up with edges only among itself. On a catalog whose items lie near a
/// one-dimensional manifold that is not rare -- two dropped edges are enough to strand a
/// whole arc -- and under an inner product the low-norm items lose their in-links to the
/// long ones. An unreachable item is one `recommend` can never return, at any score,
/// which would show up as missing catalog coverage and nothing else.
///
/// Each pass walks level 0 from the entry point, covers what it missed with one
/// representative per stranded component, and links each representative to its nearest
/// reachable node in both directions. The searches are the expensive part and are read
/// only, so they run in parallel against one snapshot of what is reachable; the links
/// are then applied on one thread, which is what keeps one repair from evicting the
/// edge another just relied on.
fn repair(
    items: &impl Items,
    links: &BuildLinks,
    params: &Params,
    entry: usize,
    entry_level: usize,
    n_nodes: usize,
) {
    let mut reachable = vec![false; n_nodes];
    let mut stack = Vec::new();
    let mut neighbours = Vec::new();

    // A pass can only shrink the stranded set, and one that finds nothing ends the
    // repair; the bound is there so that a pathological catalog cannot spin.
    for _ in 0..MAX_REPAIR_PASSES {
        reachable.iter_mut().for_each(|seen| *seen = false);
        mark_component(links, entry, &mut reachable, &mut stack, &mut neighbours);

        // One representative per stranded component: attaching it brings everything it
        // reaches along with it, so the searches are per component and not per node.
        let mut covered = reachable.clone();
        let mut representatives = Vec::new();
        for node in 0..n_nodes {
            if covered[node] {
                continue;
            }
            representatives.push(node);
            mark_component(links, node, &mut covered, &mut stack, &mut neighbours);
        }
        if representatives.is_empty() {
            return;
        }

        let hosts: Vec<Vec<Candidate>> = representatives
            .par_iter()
            .map_init(
                || (Visited::new(n_nodes), Vec::new()),
                |(visited, neighbours), &node| {
                    let score = |other: usize| items.similarity(node, other);
                    let admit = |other: usize| reachable[other];
                    let mut found = search_graph(
                        &score,
                        &admit,
                        links,
                        entry,
                        entry_level,
                        params.ef_construction,
                        visited,
                        neighbours,
                    );
                    found.truncate(HOSTS_CONSIDERED);
                    found
                },
            )
            .collect();

        let cap = params.m0();
        for (&node, candidates) in representatives.iter().zip(&hosts) {
            attach(items, links, node, candidates, cap);
        }
    }
}

/// How many times [`repair`] will re-check its own work before giving up.
const MAX_REPAIR_PASSES: usize = 8;

/// How many reachable neighbours a stranded node will settle for, best first.
///
/// Only the first with room is used, and one of a handful almost always has some; the
/// rest are there so that the serial linking pass is not forced to evict an edge
/// merely because the single best host happened to be full.
const HOSTS_CONSIDERED: usize = 16;

/// Link `node` to the first of `candidates` with room, in both directions.
fn attach(
    items: &impl Items,
    links: &BuildLinks,
    node: usize,
    candidates: &[Candidate],
    cap: usize,
) {
    // The in-edge is the one that matters: without it nothing can walk to `node`.
    let host = candidates
        .iter()
        .find(|candidate| links.link_if_room(candidate.index, 0, node as u32, cap))
        .or_else(|| {
            // Nothing near has room, so the nearest gives up its weakest link. Rare,
            // and the pass that follows picks up anything this strands.
            let best = candidates.first()?;
            links.link(best.index, 0, &[node as u32], cap, |current| {
                prune_keeping(current, best.index, node as u32, cap, items)
            });
            Some(best)
        });
    if let Some(host) = host {
        links.link(node, 0, &[host.index as u32], cap, |current| {
            prune_keeping(current, node, host.index as u32, cap, items)
        });
    }
}

/// Mark every node reachable from `start` along level 0.
fn mark_component(
    links: &BuildLinks,
    start: usize,
    reachable: &mut [bool],
    stack: &mut Vec<usize>,
    neighbours: &mut Vec<u32>,
) {
    stack.clear();
    stack.push(start);
    reachable[start] = true;
    while let Some(node) = stack.pop() {
        neighbours.clear();
        links.neighbors(node, 0, neighbours);
        for &next in neighbours.iter() {
            let next = next as usize;
            if !reachable[next] {
                reachable[next] = true;
                stack.push(next);
            }
        }
    }
}

/// Prune `owner`'s links to `cap`, whatever else goes, keeping `required`.
fn prune_keeping(
    current: &[u32],
    owner: usize,
    required: u32,
    cap: usize,
    items: &impl Items,
) -> Vec<u32> {
    let mut kept = prune(current, owner, cap, items);
    if !kept.contains(&required) {
        kept.pop();
        kept.push(required);
    }
    kept
}

#[allow(clippy::too_many_arguments)]
fn insert_one(
    node: usize,
    items: &impl Items,
    links: &BuildLinks,
    params: &Params,
    entry: usize,
    entry_level: usize,
    visited: &mut Visited,
    neighbours: &mut Vec<u32>,
) {
    let level = links.level_of(node);
    let score = |other: usize| items.similarity(node, other);
    let admit = |_: usize| true;

    let mut current = Candidate {
        score: score(entry),
        index: entry,
    };
    // Above this node's own top level there is nothing to link, only navigation.
    for l in (level + 1..=entry_level).rev() {
        current = greedy(&score, links, l, current, neighbours);
    }

    let mut entries = vec![current];
    for l in (0..=level.min(entry_level)).rev() {
        visited.clear();
        let found = search_layer(
            &score,
            &admit,
            links,
            l,
            &entries,
            params.ef_construction,
            visited,
            neighbours,
        );
        let cap = if l == 0 { params.m0() } else { params.m };
        let selected = select_neighbours(&found, cap, items);
        links.link(node, l, &selected, cap, |current| {
            prune(current, node, cap, items)
        });
        for &neighbour in &selected {
            let neighbour = neighbour as usize;
            links.link(neighbour, l, &[node as u32], cap, |current| {
                prune(current, neighbour, cap, items)
            });
        }
        entries = found;
    }
}

/// Re-select `owner`'s neighbours from `current` once the list has overflowed.
fn prune(current: &[u32], owner: usize, cap: usize, items: &impl Items) -> Vec<u32> {
    let mut candidates: Vec<Candidate> = current
        .iter()
        .map(|&n| Candidate {
            score: items.similarity(owner, n as usize),
            index: n as usize,
        })
        .collect();
    candidates.sort_unstable_by(|a, b| a.worst_first(b));
    select_neighbours(&candidates, cap, items)
}

/// Keep up to `cap` of `candidates`, spread over directions rather than piled up.
///
/// The paper's heuristic, and Qdrant's: walking the candidates best first, keep one only
/// when it is nearer the node being linked than it is to everything already kept. A
/// plain top-`cap` would fill a node's list with one tight cluster, and a query arriving
/// from any other direction would have no edge to follow.
///
/// Rejected candidates are not thrown away. If the heuristic leaves room -- and in a
/// dense neighbourhood it usually does, since every candidate after the first is close
/// to something already kept -- the best of them fill it. Half-empty link lists would
/// cost far more recall than the redundancy costs, which is the paper's
/// `keepPrunedConnections` and Qdrant's default.
///
/// `candidates` must be ordered best first.
fn select_neighbours(candidates: &[Candidate], cap: usize, items: &impl Items) -> Vec<u32> {
    let mut kept: Vec<u32> = Vec::with_capacity(cap);
    let mut rejected: Vec<u32> = Vec::new();
    for candidate in candidates {
        if kept.len() >= cap {
            break;
        }
        let diverse = kept
            .iter()
            .all(|&other| items.similarity(candidate.index, other as usize) < candidate.score);
        if diverse {
            kept.push(candidate.index as u32);
        } else {
            rejected.push(candidate.index as u32);
        }
    }
    let room = cap.saturating_sub(kept.len());
    kept.extend(rejected.into_iter().take(room));
    kept
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::hnsw::graph::{GraphView, MAX_LEVEL};
    use crate::vectors::DenseItems;

    fn params(m: usize) -> Params {
        Params {
            m,
            ef_construction: 64,
            seed: 20_260_920,
        }
    }

    /// Points on a circle, so "near" is an angle and the geometry is easy to reason about.
    fn circle(n: usize) -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let theta = std::f64::consts::TAU * i as f64 / n as f64;
                [theta.cos(), theta.sin()]
            })
            .collect()
    }

    fn view(graph: &Graph) -> GraphView<'_> {
        GraphView::new(
            &graph.node_level,
            &graph.links_indptr,
            &graph.links_indices,
            graph.entry_point,
        )
        .expect("a built graph describes itself")
    }

    #[test]
    fn an_empty_catalog_builds_an_empty_graph() {
        let items = DenseItems {
            vectors: &[],
            dim: 2,
        };
        let graph = build(&items, &params(8));
        assert_eq!(graph.entry_point, -1);
        assert!(graph.links_indices.is_empty());
    }

    #[test]
    fn a_single_item_builds_a_graph_with_no_links() {
        let vectors = [1.0, 0.0];
        let graph = build(
            &DenseItems {
                vectors: &vectors,
                dim: 2,
            },
            &params(8),
        );
        assert_eq!(graph.entry_point, 0);
        assert!(graph.links_indices.is_empty());
    }

    #[test]
    fn degrees_never_exceed_the_caps() {
        let vectors = circle(400);
        let p = params(6);
        let graph = build(
            &DenseItems {
                vectors: &vectors,
                dim: 2,
            },
            &p,
        );
        let view = view(&graph);
        let mut neighbours = Vec::new();
        for node in 0..400 {
            for level in 0..=view.level_of(node) {
                neighbours.clear();
                view.neighbors(node, level, &mut neighbours);
                let cap = if level == 0 { p.m0() } else { p.m };
                assert!(
                    neighbours.len() <= cap,
                    "node {node} level {level}: {}",
                    neighbours.len()
                );
                assert!(
                    !neighbours.contains(&(node as u32)),
                    "node {node} links to itself"
                );
            }
        }
    }

    #[test]
    fn the_entry_point_is_the_tallest_node() {
        let vectors = circle(300);
        let graph = build(
            &DenseItems {
                vectors: &vectors,
                dim: 2,
            },
            &params(8),
        );
        let top = *graph.node_level.iter().max().expect("a node");
        assert_eq!(graph.node_level[graph.entry_point as usize], top);
        // Ties go to the lowest index, which is what makes the entry deterministic.
        let first = graph
            .node_level
            .iter()
            .position(|&l| l == top)
            .expect("a node");
        assert_eq!(graph.entry_point, first as i64);
        assert!(top as usize <= MAX_LEVEL);
    }

    #[test]
    fn a_single_threaded_build_is_reproducible() {
        let vectors = circle(256);
        let items = DenseItems {
            vectors: &vectors,
            dim: 2,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a pool");
        let (first, second) =
            pool.install(|| (build(&items, &params(8)), build(&items, &params(8))));
        assert_eq!(first.node_level, second.node_level);
        assert_eq!(first.links_indptr, second.links_indptr);
        assert_eq!(first.links_indices, second.links_indices);
        assert_eq!(first.entry_point, second.entry_point);
    }

    #[test]
    fn levels_do_not_depend_on_the_thread_count() {
        // The links may differ between builds; the shape of the graph may not.
        let vectors = circle(256);
        let items = DenseItems {
            vectors: &vectors,
            dim: 2,
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a pool");
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("a pool");
        let sequential = one.install(|| build(&items, &params(8)));
        let parallel = many.install(|| build(&items, &params(8)));
        assert_eq!(sequential.node_level, parallel.node_level);
        assert_eq!(sequential.entry_point, parallel.entry_point);
    }

    #[test]
    fn the_heuristic_spreads_links_where_a_plain_top_m_would_not() {
        // Four points on the unit circle. Ranked by score alone the two best are 1 and
        // 2, which sit almost on top of each other: a query arriving from below has no
        // edge to follow. The heuristic rejects 2 -- it is nearer 1 than it is to the
        // node being linked -- and reaches past it for 3 on the other side. That is the
        // whole difference between the two rules.
        let angles = [0.0, 0.2, 0.25, -0.5];
        let vectors: Vec<f64> = angles
            .iter()
            .flat_map(|&a| [f64::cos(a), f64::sin(a)])
            .collect();
        let items = DenseItems {
            vectors: &vectors,
            dim: 2,
        };
        let mut candidates: Vec<Candidate> = (1..4)
            .map(|index| Candidate {
                score: items.similarity(0, index),
                index,
            })
            .collect();
        candidates.sort_unstable_by(|a, b| a.worst_first(b));
        assert_eq!(
            candidates.iter().map(|c| c.index).collect::<Vec<_>>(),
            vec![1, 2, 3],
            "by score alone the crowded pair leads"
        );

        assert_eq!(select_neighbours(&candidates, 2, &items), vec![1, 3]);
    }

    #[test]
    fn a_rejected_candidate_comes_back_when_there_is_room_for_it() {
        // Same fixture with room for three: the heuristic still only wants two, so the
        // one it rejected fills the last slot rather than leaving the node short.
        let angles = [0.0, 0.2, 0.25, -0.5];
        let vectors: Vec<f64> = angles
            .iter()
            .flat_map(|&a| [f64::cos(a), f64::sin(a)])
            .collect();
        let items = DenseItems {
            vectors: &vectors,
            dim: 2,
        };
        let mut candidates: Vec<Candidate> = (1..4)
            .map(|index| Candidate {
                score: items.similarity(0, index),
                index,
            })
            .collect();
        candidates.sort_unstable_by(|a, b| a.worst_first(b));
        assert_eq!(select_neighbours(&candidates, 3, &items), vec![1, 3, 2]);
    }

    #[test]
    fn identical_points_still_link_to_each_other() {
        // Two points on top of each other: each is exactly as near the other as it is to
        // the node being linked, so the strict heuristic keeps neither past the first.
        // Without the top-up this node would leave with one link instead of two.
        let vectors = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let items = DenseItems {
            vectors: &vectors,
            dim: 2,
        };
        let candidates = vec![
            Candidate {
                score: 1.0,
                index: 1,
            },
            Candidate {
                score: 1.0,
                index: 2,
            },
        ];
        assert_eq!(select_neighbours(&candidates, 2, &items), vec![1, 2]);
    }

    #[test]
    fn nothing_to_choose_from_keeps_nothing() {
        let vectors = [1.0, 0.0];
        let items = DenseItems {
            vectors: &vectors,
            dim: 2,
        };
        assert!(select_neighbours(&[], 4, &items).is_empty());
    }

    #[test]
    fn every_node_is_reachable_from_the_entry_point() {
        // An unreachable node can never be recommended, whatever its score, so this is
        // the one graph property a recall threshold would not catch.
        //
        // The vectors are unit-norm here on purpose. Under a raw inner product a
        // low-norm item scores below its neighbours for *every* query, so it is
        // dominated rather than merely unlinked, and full reachability stops being
        // either achievable or meaningful; recall is the invariant that survives, and
        // `search.rs` is where it is tested. On the sphere the two agree.
        for (name, vectors, n, dim) in [
            ("circle", circle(512), 512usize, 2usize),
            ("sphere", unit_vectors(3000, 8), 3000, 8),
            ("sphere-wide", unit_vectors(2000, 32), 2000, 32),
        ] {
            for m in [4usize, 8, 16] {
                let graph = build(
                    &DenseItems {
                        vectors: &vectors,
                        dim,
                    },
                    &Params {
                        m,
                        ef_construction: 64,
                        seed: 11,
                    },
                );
                let view = view(&graph);
                let mut seen = vec![false; n];
                let mut queue = vec![view.entry_point];
                seen[view.entry_point] = true;
                let mut neighbours = Vec::new();
                while let Some(node) = queue.pop() {
                    neighbours.clear();
                    view.neighbors(node, 0, &mut neighbours);
                    for &next in &neighbours {
                        if !seen[next as usize] {
                            seen[next as usize] = true;
                            queue.push(next as usize);
                        }
                    }
                }
                let stranded = seen.iter().filter(|s| !**s).count();
                assert_eq!(stranded, 0, "{name} at m={m} stranded {stranded} of {n}");
            }
        }
    }

    /// Reproducible pseudo-random vectors, without depending on an RNG crate.
    pub(crate) fn pseudo_vectors(n: usize, dim: usize) -> Vec<f64> {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        (0..n * dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 2000) as f64 / 1000.0 - 1.0
            })
            .collect()
    }

    /// The same vectors scaled onto the unit sphere, where an inner product ranks the
    /// same way an angle does.
    pub(crate) fn unit_vectors(n: usize, dim: usize) -> Vec<f64> {
        let mut vectors = pseudo_vectors(n, dim);
        for row in vectors.chunks_mut(dim) {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            row.iter_mut().for_each(|v| *v /= norm);
        }
        vectors
    }
}

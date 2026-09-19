//! Layered link storage for an HNSW graph.
//!
//! A node lives on levels `0..=node_level[node]`, and keeps a neighbour list on each of
//! them. The lists are one CSR over `(node, level)` slots ordered node-major, so a slot
//! is `slot_start[node] + level` and needs no search. Levels are geometric with ratio
//! `1/m`, so the slots total about `n * m / (m - 1)` -- seven percent over one per node
//! at the default `m = 16`, which is why the simple layout is also the small one.
//!
//! The frozen form is four flat arrays and nothing else. That is what crosses PyO3, what
//! a fitted estimator pickles, and what a search in a fresh process runs from; there is
//! no Rust-side object that outlives a call.

use std::sync::RwLock;

/// Highest level a node may reach, so one unlucky draw cannot make the graph tall.
pub const MAX_LEVEL: usize = 32;

/// Construction parameters.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Neighbours kept per node above level 0.
    pub m: usize,
    /// Width of the candidate list an insertion keeps at each level.
    pub ef_construction: usize,
    /// Seed the levels are drawn from.
    pub seed: u64,
}

impl Params {
    /// Neighbours kept at level 0, where the graph carries every node and must stay
    /// connected; the paper and Qdrant both double the cap there.
    pub fn m0(&self) -> usize {
        2 * self.m
    }

    /// The `1 / ln(m)` multiplier the level draw is scaled by.
    fn level_scale(&self) -> f64 {
        1.0 / (self.m.max(2) as f64).ln()
    }

    /// The level of `node`, drawn from the seed and the node alone.
    ///
    /// Deliberately not drawn from a shared stream: a level that depended on the order
    /// nodes were inserted in would make a parallel build's *shape* vary as well as its
    /// links, and the shape is the part worth pinning in a test.
    pub fn level_of(&self, node: usize) -> usize {
        let bits = splitmix64(self.seed ^ (node as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        // Uniform in (0, 1]: never zero, so the logarithm is always finite.
        let uniform = ((bits >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
        let level = (-uniform.ln() * self.level_scale()) as usize;
        level.min(MAX_LEVEL)
    }
}

/// One round of SplitMix64, which is how a seed and a node index become a draw.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Where each node's level-0 slot starts; the last entry is the number of slots.
pub fn slot_starts(levels: impl IntoIterator<Item = usize>, n_nodes: usize) -> Vec<usize> {
    let mut starts = Vec::with_capacity(n_nodes + 1);
    let mut at = 0;
    for level in levels {
        starts.push(at);
        at += level + 1;
    }
    starts.push(at);
    starts
}

/// Somewhere neighbour lists can be read from, during a build or after one.
pub trait Links: Sync {
    /// Append the neighbours of `node` at `level` to `out`, which the caller clears.
    fn neighbors(&self, node: usize, level: usize, out: &mut Vec<u32>);

    /// The highest level `node` lives on.
    fn level_of(&self, node: usize) -> usize;
}

/// Neighbour lists under construction, one lock per slot.
///
/// A lock per slot rather than one over the graph: insertions touch the slots of the
/// node being inserted and of the handful it links to, so contention is proportional to
/// `m` and not to the number of workers. The locks are what keeps the build free of
/// `unsafe` while still letting insertions run in parallel.
pub struct BuildLinks {
    node_level: Vec<usize>,
    slot_start: Vec<usize>,
    slots: Vec<RwLock<Vec<u32>>>,
}

impl BuildLinks {
    pub fn new(node_level: Vec<usize>, params: &Params) -> Self {
        let n_nodes = node_level.len();
        let slot_start = slot_starts(node_level.iter().copied(), n_nodes);
        let slots = (0..n_nodes)
            .flat_map(|node| {
                (0..=node_level[node]).map(move |level| {
                    let cap = if level == 0 { params.m0() } else { params.m };
                    RwLock::new(Vec::with_capacity(cap))
                })
            })
            .collect();
        Self {
            node_level,
            slot_start,
            slots,
        }
    }

    #[inline]
    fn slot(&self, node: usize, level: usize) -> usize {
        self.slot_start[node] + level
    }

    /// Replace `node`'s neighbours at `level`, for building a fixture by hand.
    #[cfg(test)]
    pub fn set(&self, node: usize, level: usize, neighbours: &[u32]) {
        let mut slot = self.slots[self.slot(node, level)]
            .write()
            .expect("hnsw link lock poisoned");
        slot.clear();
        slot.extend_from_slice(neighbours);
    }

    /// Add links to `node` at `level`, pruning with `prune` when they overflow `cap`.
    ///
    /// Adding rather than replacing, even for a node's own links: under a parallel build
    /// a node can receive back-links from its neighbours *before* it inserts itself, and
    /// overwriting those would cut it out of the graph it was just attached to.
    ///
    /// `prune` receives the current neighbours and returns the ones to keep. It runs
    /// under the write lock, which is the price of not copying the list out and racing
    /// another insertion to put it back.
    pub fn link(
        &self,
        node: usize,
        level: usize,
        new: &[u32],
        cap: usize,
        prune: impl Fn(&[u32]) -> Vec<u32>,
    ) {
        let mut slot = self.slots[self.slot(node, level)]
            .write()
            .expect("hnsw link lock poisoned");
        let before = slot.len();
        for &neighbour in new {
            if neighbour as usize != node && !slot.contains(&neighbour) {
                slot.push(neighbour);
            }
        }
        if slot.len() > cap && slot.len() != before {
            *slot = prune(&slot);
        }
    }

    /// Add one link only if `node` has room for it at `level`, reporting whether it
    /// did. Lets the repair pass attach a stranded node without evicting an edge that
    /// something else is relying on to stay reachable.
    pub fn link_if_room(&self, node: usize, level: usize, new: u32, cap: usize) -> bool {
        let mut slot = self.slots[self.slot(node, level)]
            .write()
            .expect("hnsw link lock poisoned");
        if slot.contains(&new) {
            return true;
        }
        if new as usize == node || slot.len() >= cap {
            return false;
        }
        slot.push(new);
        true
    }

    /// Freeze the lists into the flat arrays that cross the boundary.
    pub fn freeze(self, entry_point: usize) -> Graph {
        let n_nodes = self.node_level.len();
        let mut links_indptr = Vec::with_capacity(self.slots.len() + 1);
        let mut links_indices = Vec::new();
        links_indptr.push(0);
        for slot in &self.slots {
            let neighbours = slot.read().expect("hnsw link lock poisoned");
            links_indices.extend(neighbours.iter().map(|&n| n as i32));
            links_indptr.push(links_indices.len() as i64);
        }
        Graph {
            node_level: self.node_level.iter().map(|&l| l as i32).collect(),
            links_indptr,
            links_indices,
            entry_point: if n_nodes == 0 { -1 } else { entry_point as i64 },
        }
    }
}

impl Links for BuildLinks {
    fn neighbors(&self, node: usize, level: usize, out: &mut Vec<u32>) {
        if level > self.node_level[node] {
            return;
        }
        let slot = self.slots[self.slot(node, level)]
            .read()
            .expect("hnsw link lock poisoned");
        out.extend_from_slice(&slot);
    }

    fn level_of(&self, node: usize) -> usize {
        self.node_level[node]
    }
}

/// A finished graph, as the flat arrays that are its only representation.
///
/// `links_indices` holds node ids as `i32`: it is much the largest array -- `2 * m` per
/// node at level 0 alone -- and a catalog that needs more than two billion items has
/// problems an index will not fix. Everything else keeps the `i64` layout numpy hands
/// the other kernels.
pub struct Graph {
    pub node_level: Vec<i32>,
    pub links_indptr: Vec<i64>,
    pub links_indices: Vec<i32>,
    /// The node a search starts from, or `-1` when the graph has no nodes.
    pub entry_point: i64,
}

/// A borrowed view of a finished graph, as a search receives it back from Python.
pub struct GraphView<'a> {
    node_level: &'a [i32],
    slot_start: Vec<usize>,
    links_indptr: &'a [i64],
    links_indices: &'a [i32],
    pub entry_point: usize,
    pub top_level: usize,
}

impl<'a> GraphView<'a> {
    /// Borrow a graph's arrays, rebuilding the slot offsets from the levels.
    ///
    /// The offsets are a prefix sum over the nodes, so they cost one pass over the
    /// catalog per call -- against `ef_search` node visits per query, which is why they
    /// are recomputed rather than carried around and validated.
    ///
    /// Returns `None` when the arrays do not describe a graph: they may have come from a
    /// pickle, so an inconsistent one has to be an error rather than a panic.
    pub fn new(
        node_level: &'a [i32],
        links_indptr: &'a [i64],
        links_indices: &'a [i32],
        entry_point: i64,
    ) -> Option<Self> {
        let n_nodes = node_level.len();
        if node_level.iter().any(|&l| l < 0 || l as usize > MAX_LEVEL) {
            return None;
        }
        let slot_start = slot_starts(node_level.iter().map(|&l| l as usize), n_nodes);
        if links_indptr.len() != slot_start[n_nodes] + 1 || links_indptr.first() != Some(&0) {
            return None;
        }
        if links_indptr.windows(2).any(|w| w[0] > w[1])
            || links_indptr.last() != Some(&(links_indices.len() as i64))
        {
            return None;
        }
        if links_indices
            .iter()
            .any(|&n| n < 0 || n as usize >= n_nodes)
        {
            return None;
        }
        let entry = usize::try_from(entry_point).ok()?;
        if entry >= n_nodes {
            return None;
        }
        Some(Self {
            node_level,
            slot_start,
            links_indptr,
            links_indices,
            entry_point: entry,
            top_level: node_level[entry] as usize,
        })
    }

    pub fn len(&self) -> usize {
        self.node_level.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Links for GraphView<'_> {
    fn neighbors(&self, node: usize, level: usize, out: &mut Vec<u32>) {
        if level > self.node_level[node] as usize {
            return;
        }
        let slot = self.slot_start[node] + level;
        let range = self.links_indptr[slot] as usize..self.links_indptr[slot + 1] as usize;
        out.extend(self.links_indices[range].iter().map(|&n| n as u32));
    }

    fn level_of(&self, node: usize) -> usize {
        self.node_level[node] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params {
            m: 16,
            ef_construction: 100,
            seed: 7,
        }
    }

    #[test]
    fn levels_depend_only_on_the_seed_and_the_node() {
        let p = params();
        let once: Vec<usize> = (0..64).map(|n| p.level_of(n)).collect();
        let again: Vec<usize> = (0..64).rev().map(|n| p.level_of(n)).collect();
        assert!(once.iter().rev().eq(again.iter()));
        let other = Params { seed: 8, ..p };
        assert!((0..256).any(|n| other.level_of(n) != p.level_of(n)));
    }

    #[test]
    fn levels_are_geometric_and_mostly_zero() {
        // At m = 16 roughly one node in sixteen reaches level 1, so a few thousand
        // draws must be overwhelmingly level 0 and must still reach above it.
        let p = params();
        let levels: Vec<usize> = (0..4096).map(|n| p.level_of(n)).collect();
        let zeros = levels.iter().filter(|&&l| l == 0).count();
        assert!(zeros > 3500, "{zeros} of 4096 at level 0");
        assert!(levels.iter().any(|&l| l >= 1));
        assert!(levels.iter().all(|&l| l <= MAX_LEVEL));
    }

    #[test]
    fn slots_are_one_per_node_per_level() {
        assert_eq!(slot_starts([0, 2, 0, 1], 4), vec![0, 1, 4, 5, 7]);
        assert_eq!(slot_starts([], 0), vec![0]);
    }

    #[test]
    fn freezing_keeps_what_was_linked() {
        let links = BuildLinks::new(vec![1, 0, 0], &params());
        links.set(0, 0, &[1, 2]);
        links.set(0, 1, &[]);
        links.set(1, 0, &[0]);
        links.set(2, 0, &[0]);
        let graph = links.freeze(0);
        assert_eq!(graph.node_level, vec![1, 0, 0]);
        assert_eq!(graph.links_indptr, vec![0, 2, 2, 3, 4]);
        assert_eq!(graph.links_indices, vec![1, 2, 0, 0]);

        let view = GraphView::new(
            &graph.node_level,
            &graph.links_indptr,
            &graph.links_indices,
            graph.entry_point,
        )
        .expect("the frozen graph describes itself");
        let mut out = Vec::new();
        view.neighbors(0, 0, &mut out);
        assert_eq!(out, vec![1, 2]);
        out.clear();
        view.neighbors(1, 1, &mut out);
        assert!(out.is_empty(), "level 1 is above node 1");
        assert_eq!(view.top_level, 1);
    }

    #[test]
    fn a_view_rejects_arrays_that_do_not_describe_a_graph() {
        // These arrive from a pickle, so every one of them has to be an error rather
        // than a panic or an out-of-bounds read.
        assert!(GraphView::new(&[0, 0], &[0, 1, 2], &[1, 0], 0).is_some());
        assert!(
            GraphView::new(&[0, 0], &[0, 1], &[1], 0).is_none(),
            "short indptr"
        );
        assert!(
            GraphView::new(&[0, 0], &[0, 1, 2], &[9, 0], 0).is_none(),
            "link out of range"
        );
        assert!(
            GraphView::new(&[0, 0], &[0, 2, 1], &[1, 0], 0).is_none(),
            "indptr descends"
        );
        assert!(
            GraphView::new(&[0, 0], &[0, 1, 2], &[1, 0], 5).is_none(),
            "entry out of range"
        );
        assert!(
            GraphView::new(&[-1, 0], &[0, 1, 2], &[1, 0], 0).is_none(),
            "negative level"
        );
        assert!(
            GraphView::new(&[0, 0], &[1, 1, 2], &[1, 0], 0).is_none(),
            "indptr does not start at 0"
        );
    }

    #[test]
    fn link_prunes_only_when_the_cap_is_passed() {
        let links = BuildLinks::new(vec![0, 0, 0, 0, 0], &params());
        links.set(0, 0, &[1, 2]);
        links.link(0, 0, &[3], 3, |_| unreachable!("still within the cap"));
        let mut out = Vec::new();
        links.neighbors(0, 0, &mut out);
        assert_eq!(out, vec![1, 2, 3]);

        // Nothing new, so nothing to prune, even though the list is already full.
        links.link(0, 0, &[3, 1, 2], 3, |_| unreachable!("already linked"));
        // A node never links to itself, whatever the caller passes.
        links.link(0, 0, &[0], 3, |_| unreachable!("self-links are dropped"));

        links.link(0, 0, &[4], 3, |current| {
            assert_eq!(current, [1, 2, 3, 4]);
            current[..3].to_vec()
        });
        out.clear();
        links.neighbors(0, 0, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn link_keeps_back_links_a_node_received_before_it_inserted() {
        // The parallel-build hazard: node 0 is linked to by 1 and 2 while it is still
        // waiting its turn, then inserts and links to 3. All three must survive.
        let links = BuildLinks::new(vec![0, 0, 0, 0], &params());
        links.link(0, 0, &[1], 8, |c| c.to_vec());
        links.link(0, 0, &[2], 8, |c| c.to_vec());
        links.link(0, 0, &[3], 8, |c| c.to_vec());
        let mut out = Vec::new();
        links.neighbors(0, 0, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }
}

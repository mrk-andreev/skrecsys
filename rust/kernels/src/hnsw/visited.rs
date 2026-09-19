//! Visited marks for one graph traversal.
//!
//! A traversal visits a few hundred of a catalog's items, so clearing a catalog-sized
//! flag array between queries would cost more than the search. Each mark carries the
//! generation that wrote it instead, and a new traversal simply bumps the generation.

/// Generation-stamped visited marks over a fixed number of nodes.
pub struct Visited {
    stamps: Vec<u32>,
    generation: u32,
}

impl Visited {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            stamps: vec![0; n_nodes],
            generation: 0,
        }
    }

    /// Forget every mark, in constant time.
    pub fn clear(&mut self) {
        // Zero is the "never visited" stamp, so a wrap has to rewrite the marks rather
        // than hand out a generation that stale entries would match. Once every four
        // billion traversals, which is why it is a plain pass and not something clever.
        if let Some(next) = self.generation.checked_add(1) {
            self.generation = next;
        } else {
            self.stamps.iter_mut().for_each(|stamp| *stamp = 0);
            self.generation = 1;
        }
    }

    /// Mark `node` visited, returning whether it was not already.
    #[inline]
    pub fn insert(&mut self, node: usize) -> bool {
        let seen = self.stamps[node] == self.generation;
        self.stamps[node] = self.generation;
        !seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_is_new_once_per_generation() {
        let mut visited = Visited::new(4);
        visited.clear();
        assert!(visited.insert(2));
        assert!(!visited.insert(2));
        visited.clear();
        assert!(visited.insert(2));
    }

    #[test]
    fn a_wrapped_generation_does_not_resurrect_old_marks() {
        let mut visited = Visited::new(4);
        visited.generation = u32::MAX - 1;
        visited.clear();
        assert!(visited.insert(1));
        // The next clear wraps, which must leave every node unvisited again.
        visited.clear();
        assert!(visited.insert(1));
        assert!(visited.insert(3));
    }
}

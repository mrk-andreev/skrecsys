"""Fixtures shared by the index tests."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def interactions():
    """A catalog big enough for an index to be used rather than fallen back from.

    Structured rather than uniform: users belong to one of eight tastes and draw mostly
    from that taste's slice of the catalog. Uniform noise has no geometry for a
    nearest-neighbour graph to exploit, so it measures nothing but the pathological
    case; real interaction data has structure, and so does this.
    """
    rng = np.random.default_rng(0)
    n_users, n_items, n_tastes = 600, 5000, 8
    slice_size = n_items // n_tastes
    rows = []
    for user in range(n_users):
        taste = user % n_tastes
        n = rng.integers(20, 60)
        near = rng.integers(taste * slice_size, (taste + 1) * slice_size, int(n * 0.8))
        far = rng.integers(0, n_items, n - len(near))
        rows.extend((user, int(item)) for item in np.concatenate([near, far]))
    return np.asarray(rows)

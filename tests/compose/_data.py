"""Synthetic interactions with a pattern a ranker can learn and a generator cannot."""

import numpy as np

N_USERS = 80
N_ITEMS = 30
#: Items every user's *latest* interaction comes from.
TRENDING = np.arange(25, 30)
#: Items the older interactions favour ten to one, so that they stay the most popular
#: even counting the trending rows: popularity alone recommends the head, not the trend.
HEAD = np.arange(10)
HISTORY = 5


def trending_interactions(seed: int = 0) -> np.ndarray:
    """Integer (user, item) rows, each user's in time order and the last one trending.

    A popularity generator cannot tell trending items from the rest; an item feature
    saying which items trend lets a ranker put them first.
    """
    rng = np.random.default_rng(seed)
    weights = np.where(np.isin(np.arange(N_ITEMS), HEAD), 10.0, 1.0)
    weights /= weights.sum()
    rows = []
    for user in range(N_USERS):
        older = rng.choice(N_ITEMS, size=HISTORY - 1, replace=False, p=weights)
        latest = rng.choice(np.setdiff1d(TRENDING, older))
        rows.extend([user, item] for item in [*older, latest])
    return np.array(rows, dtype=np.int64)


def trending_table() -> np.ndarray:
    """Item id and a 0/1 trending flag."""
    items = np.arange(N_ITEMS)
    return np.column_stack([items, np.isin(items, TRENDING)]).astype(float)

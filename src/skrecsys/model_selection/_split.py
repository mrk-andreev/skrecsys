"""Cross-validation splitters for interaction data."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from skrecsys._typing import override
from skrecsys.utils.validation import check_interactions

__all__ = ["WarmStartKFold"]

_MIN_SPLITS = 2


class WarmStartKFold(BaseCrossValidator):
    """K-fold splitter keeping every test user and item in the training fold.

    The first interaction of each user and of each item (in the possibly shuffled
    order) is pinned to the training set of every split. The remaining interactions
    are distributed round-robin over ``n_splits`` disjoint test folds. Every user and
    item in a test fold therefore also occurs in its training fold.

    Random interaction splits ignore time. They are invalid when the production
    decision is time ordered.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle interactions before pinning and fold assignment.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling when ``shuffle=True``.

    Raises
    ------
    ValueError
        From ``split`` when fewer unpinned interactions than ``n_splits`` exist, so
        some test fold would be empty.

    Examples
    --------
    >>> from skrecsys.model_selection import WarmStartKFold
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u1", "c"], ["u2", "c"]]
    >>> [test.tolist() for _, test in WarmStartKFold(n_splits=2).split(X)]
    [[3], [5]]
    """

    def __init__(
        self, n_splits: int = 5, *, shuffle: bool = False, random_state: Any = None
    ) -> None:
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < _MIN_SPLITS:
            raise ValueError(f"n_splits must be an integer >= 2, got {n_splits!r}.")
        if not shuffle and random_state is not None:
            raise ValueError("Setting random_state has no effect since shuffle is False.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @override
    def _iter_test_indices(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[NDArray[np.intp]]:
        if X is None:
            raise ValueError("WarmStartKFold.split requires X.")
        users, items, _ = check_interactions(X)
        n_samples = len(users)
        order = np.arange(n_samples)
        if self.shuffle:
            order = check_random_state(self.random_state).permutation(n_samples)

        pinned = np.zeros(n_samples, dtype=bool)
        for ids in (users, items):
            _, first = np.unique(ids[order], return_index=True)
            pinned[order[first]] = True

        free = order[~pinned[order]]
        if len(free) < self.n_splits:
            raise ValueError(
                f"Cannot build {self.n_splits} warm-start folds: only {len(free)} "
                "interactions remain after keeping one interaction per user and item "
                "in training."
            )
        folds = np.arange(len(free)) % self.n_splits
        for fold in range(self.n_splits):
            yield np.sort(free[folds == fold])

    @override
    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits

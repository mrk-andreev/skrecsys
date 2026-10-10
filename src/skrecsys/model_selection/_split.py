"""Cross-validation splitters for interaction data."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from skrecsys.typing import RandomStateLike, override
from skrecsys.utils._param_validation import check_int, check_real
from skrecsys.utils.validation import check_interactions, factorize

__all__ = ["ColdStartSplit", "LatestInteractionsSplit", "WarmStartKFold"]

_MIN_SPLITS = 2


def positions_in_user(
    codes: NDArray[np.intp], times: NDArray[np.generic] | None = None
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Where each row stands among the rows of its user, and how many rows each user has.

    ``codes`` names the user of each row, as :func:`~skrecsys.utils.validation.factorize`
    does. A user's rows are numbered from 0 in the order of ``times``, ties in row order,
    or in row order alone without times; so the latest ``n`` rows of a user with ``count``
    rows are those at ``position >= count - n``.
    """
    order = np.argsort(codes, kind="stable") if times is None else np.lexsort((times, codes))
    counts = np.bincount(codes).astype(np.intp)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    position = np.empty(len(codes), dtype=np.intp)
    position[order] = np.arange(len(codes)) - np.repeat(starts, counts)
    return position, counts


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
        self, n_splits: int = 5, *, shuffle: bool = False, random_state: RandomStateLike = None
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
    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


class ColdStartSplit(BaseCrossValidator):
    """One train/test split holding out cold users and the latest rows of warm ones.

    A random ``cold_users`` fraction of the users goes to the test set with every one of
    their interactions, so a model fitted on the training set has never seen them. Every
    other user keeps its first rows for training and gives its last ``test_size`` fraction,
    rounded down, to the test set. Row order is read as time, as :class:`Cascade`'s
    ``split`` reads it, so sort ``X`` by timestamp when you have one.

    The test set therefore asks both questions a production recommender faces: what to
    show a user it knows, and what to show one it has never seen.

    Parameters
    ----------
    cold_users : float, default=0.1
        Fraction of users held out entirely, in (0, 1), rounded down.
    test_size : float, default=0.2
        Fraction of every other user's interactions held out, in [0, 1). Zero keeps the
        warm users wholly in training, so the test set is the cold users alone.
    random_state : int, RandomState instance or None, default=None
        Controls which users are cold.

    Examples
    --------
    >>> from skrecsys.model_selection import ColdStartSplit
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "c"], ["u3", "b"], ["u3", "c"]]
    >>> train, test = next(ColdStartSplit(cold_users=0.34, test_size=0.5, random_state=0).split(X))
    >>> test.tolist()
    [1, 3, 4, 5]
    """

    def __init__(
        self,
        cold_users: float = 0.1,
        test_size: float = 0.2,
        *,
        random_state: RandomStateLike = None,
    ) -> None:
        check_real(
            cold_users,
            "cold_users",
            min_value=0,
            max_value=1,
            min_inclusive=False,
            max_inclusive=False,
        )
        check_real(test_size, "test_size", min_value=0, max_value=1, max_inclusive=False)
        self.cold_users = cold_users
        self.test_size = test_size
        self.random_state = random_state

    @override
    def _iter_test_indices(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[NDArray[np.intp]]:
        if X is None:
            raise ValueError("ColdStartSplit.split requires X.")
        users, _, _ = check_interactions(X)
        distinct, codes = factorize(users)
        n_cold = int(np.floor(self.cold_users * len(distinct)))
        if n_cold == 0 or n_cold == len(distinct):
            raise ValueError(
                f"cold_users={self.cold_users} of {len(distinct)} users leaves "
                f"{n_cold} cold and {len(distinct) - n_cold} warm; both must be non-empty."
            )
        rng = check_random_state(self.random_state)
        cold = np.zeros(len(distinct), dtype=bool)
        cold[rng.choice(len(distinct), size=n_cold, replace=False)] = True

        position, counts = positions_in_user(codes)
        n_held = np.floor(self.test_size * counts).astype(np.intp)
        latest = position >= (counts - n_held)[codes]
        yield np.flatnonzero(cold[codes] | latest)

    @override
    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of splitting iterations, which is always one."""
        return 1


class LatestInteractionsSplit(BaseCrossValidator):
    """One train/test split holding out the latest rows of every user, or of the latest users.

    Every user gives the last ``test_size`` fraction of its interactions, rounded down, to
    the test set and keeps the rest for training, which is what a float ``split`` of a
    :class:`~skrecsys.compose.Cascade` does. ``max_users`` holds out from that many users
    only: a ranker gets one group of candidates per user with held-out rows, and its fit
    time grows with their number, while the generator is better for every row it keeps.
    Row order is read as time, as :class:`ColdStartSplit` reads it, so sort ``X`` by
    timestamp when you have one.

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction of a user's interactions held out, in (0, 1). A user with too few for
        that to be a whole row keeps them all.
    max_users : int, default=None
        The most users to hold out from: those whose last row comes latest among the
        users with something to hold out. Everyone else stays wholly in training.
        ``None`` holds out from every user.

    Examples
    --------
    >>> from skrecsys.model_selection import LatestInteractionsSplit
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "c"], ["u3", "b"], ["u3", "c"]]
    >>> train, test = next(LatestInteractionsSplit(test_size=0.5).split(X))
    >>> test.tolist()
    [1, 3, 5]
    >>> train, test = next(LatestInteractionsSplit(test_size=0.5, max_users=2).split(X))
    >>> test.tolist()
    [3, 5]
    """

    def __init__(self, test_size: float = 0.2, *, max_users: int | None = None) -> None:
        check_real(
            test_size,
            "test_size",
            min_value=0,
            max_value=1,
            min_inclusive=False,
            max_inclusive=False,
        )
        check_int(max_users, "max_users", min_value=1, allow_none=True)
        self.test_size = test_size
        self.max_users = max_users

    @override
    def _iter_test_indices(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[NDArray[np.intp]]:
        if X is None:
            raise ValueError("LatestInteractionsSplit.split requires X.")
        users, _, _ = check_interactions(X)
        _, codes = factorize(users)
        position, counts = positions_in_user(codes)
        n_held = np.floor(self.test_size * counts).astype(np.intp)
        chosen = n_held > 0
        if self.max_users is not None and chosen.sum() > self.max_users:
            # The last row of each user says how recently it was active.
            last_row = np.zeros(len(counts), dtype=np.intp)
            np.maximum.at(last_row, codes, np.arange(len(codes)))
            candidates = np.flatnonzero(chosen)
            latest = candidates[np.argsort(-last_row[candidates])[: self.max_users]]
            chosen = np.zeros(len(counts), dtype=bool)
            chosen[latest] = True
        yield np.flatnonzero(chosen[codes] & (position >= (counts - n_held)[codes]))

    @override
    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of splitting iterations, which is always one."""
        return 1

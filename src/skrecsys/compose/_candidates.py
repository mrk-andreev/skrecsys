"""Candidate lists as plain arrays, and the two operations every multi-stage model needs.

Between stages, candidates travel as three arrays and nothing else:

``pairs`` : ndarray of shape (n_pairs, 2)
    ``[:, 0]`` names a query, ``[:, 1]`` an item -- the layout of the ``X`` of ``fit``.
    Rows are contiguous per query, in the order the generator ranked them. A recommender
    constructed with ``time=True`` hands its features ``(n_pairs, 3)`` pairs, the third
    column being the time the query is ranked as of (see :func:`with_time`).
``scores`` : ndarray of float64 of shape (n_pairs,) or (n_pairs, n_generators)
    What the generator scored each pair; with several generators, one column each.
``groups`` : ndarray of int64 of shape (n_groups,)
    How many rows each query has, in order: the convention of LightGBM and XGBoost.

Groups differ in size: a query gets as many candidates as it has eligible items, up to
the number asked for, because ``recommend`` raises rather than padding.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import FittedRecommender, PairScorer
from skrecsys.base import predict_pairs, serves_unknown_users
from skrecsys.utils.validation import factorize, lookup_ids


def stack_pairs(users: NDArray[np.generic], items: NDArray[np.generic]) -> NDArray[np.generic]:
    """Column-stack identifiers without letting numpy coerce one namespace into the other.

    ``np.column_stack`` of integer users and string items would silently turn the users
    into strings, which no longer match the fitted ones; an object array keeps both.
    """
    if users.dtype == items.dtype:
        return np.column_stack([users, items])
    pairs = np.empty((len(users), 2), dtype=object)
    pairs[:, 0] = users
    pairs[:, 1] = items
    return pairs


def stack_columns(columns: list[NDArray[np.generic]]) -> NDArray[np.generic]:
    """Columns side by side, each keeping what it holds.

    Columns of one dtype stack as they are. Otherwise the result is ``object``, as in
    :func:`stack_pairs`, and a ``datetime64`` column is stored as ``datetime`` objects --
    ``None`` where missing -- because numpy would store its values as bare integers.
    """
    if all(c.dtype == columns[0].dtype for c in columns):
        return np.column_stack(columns)
    out = np.empty((len(columns[0]), len(columns)), dtype=object)
    for j, column in enumerate(columns):
        out[:, j] = (
            column.astype("datetime64[us]").astype(object) if column.dtype.kind == "M" else column
        )
    return out


def with_time(pairs: NDArray[np.generic], times: NDArray[np.generic]) -> NDArray[np.generic]:
    """``pairs`` with a third column: the time each is ranked as of, missing for "latest"."""
    return stack_columns([pairs[:, 0], pairs[:, 1], times])


def retrieve(
    generator: FittedRecommender,
    queries: NDArray[np.generic],
    *,
    n_retrieved: int,
    min_retrieved: int,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    exclude_interactions: ArrayLike | None = None,
    first_query: int = 0,
) -> tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.int64], NDArray[np.intp]]:
    """Ask ``generator`` for up to ``n_retrieved`` items per query.

    Each query gets ``min(n_retrieved, eligible)`` candidates, where ``eligible`` is
    what ``recommend`` could return to it under the same filters. A query with fewer than
    ``min_retrieved`` raises, as ``recommend`` would; with ``min_retrieved=0`` a query
    left with nothing is dropped instead.

    Returns
    -------
    pairs, scores, groups
        The candidates, as described in the module docstring.
    kept : ndarray of shape (n_groups,)
        Positions in ``queries`` of the queries that have a group, in order.
    """
    eligible = generator._count_eligible(
        queries,
        candidates=candidates,
        exclude_seen=exclude_seen,
        exclude_interactions=exclude_interactions,
    )
    short = np.flatnonzero(eligible < min_retrieved)
    if short.size:
        row = int(short[0])
        raise ValueError(
            f"Cannot recommend {min_retrieved} items: query {row + first_query} has only "
            f"{eligible[row]} eligible items."
        )
    wanted = np.minimum(eligible, n_retrieved)
    kept = np.flatnonzero(wanted > 0)
    groups = wanted[kept].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(groups)[:-1]]).astype(np.int64)
    n_pairs = int(groups.sum())

    users = np.empty(n_pairs, dtype=queries.dtype)
    items = np.empty(n_pairs, dtype=generator.item_ids_.dtype)
    scores = np.empty(n_pairs, dtype=np.float64)
    # One generator call per distinct group size: nearly every query has the full
    # `n_retrieved`, and only the few with a short history need a call of their own.
    for size in np.unique(groups):
        members = np.flatnonzero(groups == size)
        found, found_scores = generator.recommend(
            queries[kept[members]],
            n_recommendations=int(size),
            candidates=candidates,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
        )
        rows = (starts[members][:, None] + np.arange(size)).ravel()
        users[rows] = np.repeat(queries[kept[members]], size)
        items[rows] = found.ravel()
        scores[rows] = found_scores.ravel()
    return stack_pairs(users, items), scores, groups, kept


def concat_ids(parts: list[NDArray[np.generic]]) -> NDArray[np.generic]:
    """Concatenate identifier arrays, as an object array when their dtypes differ.

    The reason is that of :func:`stack_pairs`: ``np.concatenate`` of integer and string
    identifiers would turn the integers into strings.
    """
    if len({part.dtype for part in parts}) <= 1:
        return np.concatenate(parts)
    return np.concatenate([part.astype(object) for part in parts])


def score_pairs(generator: FittedRecommender, pairs: NDArray[np.generic]) -> NDArray[np.float64]:
    """What ``generator`` scores each pair, NaN where it cannot.

    A generator cannot score a pair whose user or item it was not fitted on, and cannot
    score any pair without ``predict``.
    """
    out = np.full(len(pairs), np.nan)
    if not isinstance(generator, PairScorer) or not len(pairs):
        return out
    known = lookup_ids(pairs[:, 0], generator.user_ids_, name="user")[1]
    known &= lookup_ids(pairs[:, 1], generator.item_ids_, name="item")[1]
    if known.any():
        out[known] = predict_pairs(generator, pairs[known])
    return out


def retrieve_union(
    generators: list[FittedRecommender],
    queries: NDArray[np.generic],
    *,
    n_retrieved: int,
    min_retrieved: int,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    exclude_interactions: ArrayLike | None = None,
    first_query: int = 0,
) -> tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.int64], NDArray[np.intp]]:
    """Merge the candidates of several generators into up to ``n_retrieved`` per query.

    Every generator retrieves up to ``n_retrieved`` items per query, as :func:`retrieve`
    does; a generator that does not serve unknown users is asked only about the queries
    it was fitted on. The lists are interleaved round-robin -- every generator's first
    item, then every generator's second -- an item already taken is skipped, and each
    query keeps the first ``n_retrieved``. A query left with fewer items has all that any
    generator could give it, because every generator was asked for the full budget.

    Returns what :func:`retrieve` does, except that ``scores`` has one column per
    generator: the score a generator gave a candidate it retrieved, and otherwise its
    ``predict`` of the pair, NaN where it cannot score it (see :func:`score_pairs`).
    """
    n_queries = len(queries)
    rows, ranks, sources, items, scores = [], [], [], [], []
    for source, generator in enumerate(generators):
        if serves_unknown_users(generator):
            served = np.arange(n_queries)
        else:
            served = np.flatnonzero(lookup_ids(queries, generator.user_ids_, name="user")[1])
        if not len(served):
            continue
        pairs, found, groups, kept = retrieve(
            generator,
            queries[served],
            n_retrieved=n_retrieved,
            min_retrieved=0,
            candidates=candidates,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
        )
        starts = np.repeat(np.cumsum(groups) - groups, groups)
        rows.append(np.repeat(served[kept], groups))
        ranks.append(np.arange(len(pairs)) - starts)
        sources.append(np.full(len(pairs), source, dtype=np.intp))
        items.append(pairs[:, 1])
        scores.append(found)

    if rows:
        row, rank, source = np.concatenate(rows), np.concatenate(ranks), np.concatenate(sources)
        item, score = concat_ids(items), np.concatenate(scores)
    else:
        row = rank = source = np.empty(0, dtype=np.intp)
        item, score = np.empty(0, dtype=queries.dtype), np.empty(0, dtype=np.float64)
    item_ids, item_codes = factorize(item)
    keys = row.astype(np.int64) * max(len(item_ids), 1) + item_codes
    # lexsort sorts by the last key first: query, then rank, then generator.
    order = np.lexsort((source, rank, row))
    distinct, first, inverse = np.unique(keys[order], return_index=True, return_inverse=True)
    taken = order[np.sort(first)]
    counts = np.bincount(row[taken], minlength=n_queries)
    position = np.arange(len(taken)) - np.repeat(np.cumsum(counts) - counts, counts)
    taken = taken[position < n_retrieved]

    counts = np.bincount(row[taken], minlength=n_queries)
    short = np.flatnonzero(counts < min_retrieved)
    if short.size:
        query = int(short[0])
        raise ValueError(
            f"Cannot recommend {min_retrieved} items: query {query + first_query} has only "
            f"{counts[query]} eligible items."
        )
    kept = np.flatnonzero(counts > 0)
    groups = counts[kept].astype(np.int64)
    pairs = stack_pairs(queries[row[taken]], item[taken])

    # Every raw row lands on the output row of its (query, item), if that row was kept.
    out_row = np.full(len(distinct), -1, dtype=np.intp)
    out_row[np.searchsorted(distinct, keys[taken])] = np.arange(len(taken))
    target = np.empty(len(order), dtype=np.intp)
    target[order] = out_row[np.ravel(inverse)]
    merged = np.full((len(taken), len(generators)), np.nan)
    retrieved = np.zeros(merged.shape, dtype=bool)
    hit = target >= 0
    merged[target[hit], source[hit]] = score[hit]
    retrieved[target[hit], source[hit]] = True
    for column, generator in enumerate(generators):
        missing = np.flatnonzero(~retrieved[:, column])
        if missing.size:
            merged[missing, column] = score_pairs(generator, pairs[missing])
    return pairs, merged, groups, kept


def rank_within_groups(
    scores: NDArray[np.floating],
    groups: NDArray[np.int64],
    item_positions: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Row indices putting each group's rows best first, groups kept in order.

    Ranks by descending score and breaks ties by ``item_positions`` -- the fitted item
    order -- which is the tie rule ``recommend`` promises.
    """
    group_of_row = np.repeat(np.arange(len(groups)), groups)
    # lexsort sorts by the last key first: group, then score descending, then item.
    return np.lexsort((item_positions, -np.asarray(scores, dtype=np.float64), group_of_row))


def top_k_per_group(
    scores: NDArray[np.floating],
    groups: NDArray[np.int64],
    item_positions: NDArray[np.intp],
    k: int,
) -> NDArray[np.intp]:
    """Row indices of the ``k`` best-scored rows of each group, shape ``(n_groups, k)``.

    Ordered as :func:`rank_within_groups` orders them. Every group must have at least
    ``k`` rows.
    """
    order = rank_within_groups(scores, groups, item_positions)
    starts = np.concatenate([[0], np.cumsum(groups)[:-1]])
    return order[starts[:, None] + np.arange(k)]

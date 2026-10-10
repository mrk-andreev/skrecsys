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

from skrecsys._tracing import active_tracer, span
from skrecsys.base import predict_pairs, serves_unknown_users
from skrecsys.typing import FittedRecommender, PairScorer
from skrecsys.utils.validation import factorize, lookup_ids, stack_columns, stack_pairs


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
    source: str = "generator",
) -> tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.int64], NDArray[np.intp]]:
    """Ask ``generator`` for up to ``n_retrieved`` items per query.

    Each query gets ``min(n_retrieved, eligible)`` candidates, where ``eligible`` is
    what ``recommend`` could return to it under the same filters. A query with fewer than
    ``min_retrieved`` raises, as ``recommend`` would; with ``min_retrieved=0`` a query
    left with nothing is dropped instead. ``source`` names the generator to a tracer.

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
    with span(source):
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
    pairs = stack_pairs(users, items)
    if (tracer := active_tracer()) is not None:
        tracer.candidates(source, pairs, scores, groups)
    return pairs, scores, groups, kept


def served_queries(
    recommender: FittedRecommender, queries: NDArray[np.generic]
) -> NDArray[np.intp]:
    """Positions of the queries ``recommender`` can answer.

    All of them when it serves users it has never seen (see
    :func:`skrecsys.base.serves_unknown_users`), otherwise those of the users it was
    fitted on.
    """
    if serves_unknown_users(recommender):
        return np.arange(len(queries))
    if len(recommender.user_ids_) == 0:
        return np.empty(0, dtype=np.intp)
    return np.flatnonzero(lookup_ids(queries, recommender.user_ids_, name="user")[1])


def retrieve_lists(  # pylint: disable=too-many-arguments
    recommender: FittedRecommender,
    queries: NDArray[np.generic],
    n_items: int | NDArray[np.int64],
    *,
    candidates: NDArray[np.generic] | None = None,
    exclude_seen: bool = True,
    exclude_interactions: ArrayLike | None = None,
    source: str = "generator",
) -> tuple[NDArray[np.intp], NDArray[np.generic]]:
    """The best items ``recommender`` has for each query, as many as ``n_items`` or fewer.

    For the composites that put lists together rather than rank candidates. ``n_items``
    is one length for all queries or one per query; a query asked for none, one the
    recommender cannot answer (see :func:`served_queries`) and one it has nothing for
    get nothing, which is not an error. Of ``candidates``, which must be validated
    identifiers, the recommender is given those it knows: the parts of such a composite
    need not know the same items.

    Returns
    -------
    rows : ndarray of shape (n_found,)
        The position in ``queries`` of the query each item is for, ascending.
    items : ndarray of shape (n_found,)
        The items, best first within a query.
    """
    limits = np.broadcast_to(np.asarray(n_items, dtype=np.int64), queries.shape)
    asked = np.flatnonzero(limits > 0)
    served = asked[served_queries(recommender, queries[asked])]
    if candidates is not None:
        candidates = candidates[lookup_ids(candidates, recommender.item_ids_, name="item")[1]]
        if len(candidates) == 0:
            served = served[:0]
    if len(served) == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=recommender.item_ids_.dtype)
    pairs, _, groups, kept = retrieve(
        recommender,
        queries[served],
        n_retrieved=int(limits[served].max()),
        min_retrieved=0,
        candidates=candidates,
        exclude_seen=exclude_seen,
        exclude_interactions=exclude_interactions,
        source=source,
    )
    rows = np.repeat(served[kept], groups)
    keep = ranks_in_rows(rows, len(queries)) < limits[rows]
    return rows[keep], pairs[keep, 1]


def ranks_in_rows(rows: NDArray[np.intp], n_rows: int) -> NDArray[np.intp]:
    """Where each entry stands among the entries of its row, from 0; ``rows`` ascending."""
    counts = np.bincount(rows, minlength=n_rows)
    return np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)


def not_among(
    rows: NDArray[np.intp],
    items: NDArray[np.generic],
    taken_rows: NDArray[np.intp],
    taken_items: NDArray[np.generic],
) -> NDArray[np.bool_]:
    """Which ``(row, item)`` entries are not among the taken ones.

    What keeps a list put together from several sources free of repeats. Entries are
    matched as integer keys over the items both sides name, as :func:`retrieve_union`
    matches them.
    """
    if len(rows) == 0 or len(taken_rows) == 0:
        return np.ones(len(rows), dtype=bool)
    item_ids, codes = factorize(concat_ids([items, taken_items]))
    keys = rows.astype(np.int64) * len(item_ids) + codes[: len(rows)]
    taken = taken_rows.astype(np.int64) * len(item_ids) + codes[len(rows) :]
    return ~np.isin(keys, taken)


def concat_ids(parts: list[NDArray[np.generic]]) -> NDArray[np.generic]:
    """Concatenate identifier arrays, as an object array when their dtypes differ.

    The reason is that of :func:`~skrecsys.utils.validation.stack_pairs`:
    ``np.concatenate`` of integer and string identifiers would turn the integers into
    strings.
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
    if not isinstance(generator, PairScorer) or len(pairs) == 0:
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
    names: list[str] | None = None,
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
    ``names`` names the generators to a tracer.
    """
    n_queries = len(queries)
    rows, ranks, sources, items, scores = [], [], [], [], []
    for source, generator in enumerate(generators):
        served = served_queries(generator, queries)
        if len(served) == 0:
            continue
        pairs, found, groups, kept = retrieve(
            generator,
            queries[served],
            n_retrieved=n_retrieved,
            min_retrieved=0,
            candidates=candidates,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
            source=f"generator{source}" if names is None else names[source],
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

"""Why an item was recommended to a query, or why it was not."""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from skrecsys._typing import FittedRecommender, override
from skrecsys.base import uses_time
from skrecsys.inspection._trace import (
    CandidatesStep,
    FusionStep,
    PostprocessStep,
    QueryTrace,
    RankerStep,
    RouteStep,
    jsonable,
    trace,
)
from skrecsys.utils.validation import check_ids, check_queries, lookup_ids

__all__ = ["Explanation", "LeafReasons", "RankerDecision", "Retrieval", "explain"]

#: What became of an item, from the most final stage back to the first.
Status: TypeAlias = Literal[
    "served",
    "unknown_item",
    "excluded",
    "dropped_by_postprocess",
    "ranked_out",
    "not_retrieved",
]


@dataclass(frozen=True)
class _Filters:
    """The filters of the ``recommend`` call being explained."""

    candidates: ArrayLike | None
    exclude_seen: bool
    exclude_interactions: ArrayLike | None


@dataclass(frozen=True)
class Retrieval:
    """Where a generator placed the item among the candidates it retrieved.

    ``rank`` is 1-based, None when the generator did not retrieve the item; ``score`` is
    the generator's, None for the merged ``"union"`` list, which has one per generator.
    """

    path: str
    source: str
    rank: int | None
    score: float | None
    n_retrieved: int


@dataclass(frozen=True)
class RankerDecision:
    """What the ranker made of the item among the query's candidates.

    ``position`` is 1-based among ``n_candidates``. ``contributions`` maps the features
    that weighed most -- and ``"bias"``, what no feature owns -- to what they added to
    ``score``, in the ranker's own space; None when the ranker cannot say.
    """

    path: str
    score: float
    position: int
    n_candidates: int
    contributions: dict[str, float] | None


@dataclass(frozen=True)
class LeafReasons:
    """Why the leaf recommender at ``path`` scored the item as it did.

    ``reasons`` are ``(history item, weight)``, the largest weight first. With ``exact``,
    the weights and ``rest`` add up to the leaf's score; see
    :mod:`skrecsys._attribution` for ``kind``.
    """

    path: str
    kind: str
    exact: bool
    reasons: tuple[tuple[object, float], ...]
    rest: float | None


@dataclass(frozen=True)
class Explanation:
    """Why ``item`` was, or was not, recommended to ``query``.

    Attributes
    ----------
    query, item : hashable
    status : str
        What became of the item: ``"served"``; or, if it was not, the first stage that
        let it go, ``"unknown_item"`` (the model was not fitted on it), ``"excluded"``
        (``exclude_seen``, ``exclude_interactions`` or ``candidates`` removed it before
        scoring), ``"not_retrieved"`` (no generator proposed it),
        ``"dropped_by_postprocess"``, or ``"ranked_out"`` (scored, but below the cut).
    excluded_by : {"candidates", "exclude_interactions", "exclude_seen"} or None
        With ``"excluded"``, the filter that removed the item.
    position : int or None
        1-based position in the served list.
    score : float or None
        The served score, or for an item that was not served what ``predict`` scores the
        pair, when the estimator can.
    cutoff : float or None
        The score of the last item served.
    routes : tuple of RouteStep
        The branches the query was routed down.
    retrieval : tuple of Retrieval
        Where each generator placed the item.
    ranker : RankerDecision or None
        What the ranker of the stage whose list was served made of it, when that stage
        has one and the item reached it.
    reasons : tuple of LeafReasons
        Why each leaf recommender the query reached scored the item as it did.
    trace : QueryTrace
        Everything the call did for the query.
    """

    query: object
    item: object
    status: Status
    excluded_by: str | None
    position: int | None
    score: float | None
    cutoff: float | None
    routes: tuple[RouteStep, ...]
    retrieval: tuple[Retrieval, ...]
    ranker: RankerDecision | None
    reasons: tuple[LeafReasons, ...]
    trace: QueryTrace = field(repr=False, compare=False)

    @property
    def detail(self) -> str:
        """The status in a sentence."""
        ranker = self.ranker
        ranked = None if ranker is None else f"ranked #{ranker.position} of {ranker.n_candidates}"
        if self.status == "served":
            return f"served #{self.position}"
        if self.status == "excluded":
            return f"filtered out before scoring by {self.excluded_by}"
        if self.status == "not_retrieved":
            return self._missed()
        if self.status == "dropped_by_postprocess":
            return f"{ranked or 'scored'}, then removed by postprocess"
        if self.status == "ranked_out":
            return self._below(ranked)
        return "the model was not fitted on this item"

    def _missed(self) -> str:
        proposed = [r for r in self.retrieval if r.rank is not None and r.source != "union"]
        merged = [r for r in self.retrieval if r.source == "union"]
        if proposed and merged:
            by = ", ".join(f"{r.source} #{r.rank}" for r in proposed)
            return f"proposed by {by}, but cut from the merged top {merged[0].n_retrieved}"
        sources = ", ".join(f"{r.source} top {r.n_retrieved}" for r in self.retrieval)
        return f"no generator retrieved it ({sources})" if sources else "no generator retrieved it"

    def _below(self, ranked: str | None) -> str:
        if ranked is not None:
            return f"a candidate, {ranked} by the ranker, below the served list"
        if self.score is not None and self.cutoff is not None:
            return f"scored {self.score:.4g}, below the cut-off {self.cutoff:.4g}"
        return "scored below the served list"

    def to_dict(self) -> dict[str, object]:
        """The explanation as JSON-compatible values, without the trace."""
        return {
            "query": jsonable(self.query),
            "item": jsonable(self.item),
            "status": self.status,
            "excluded_by": self.excluded_by,
            "detail": self.detail,
            "position": self.position,
            "score": jsonable(self.score),
            "cutoff": jsonable(self.cutoff),
            "routes": [
                {"path": r.path, "condition": r.condition, "branch": r.branch} for r in self.routes
            ],
            "retrieval": [jsonable(vars(r)) for r in self.retrieval],
            "ranker": None if self.ranker is None else jsonable(vars(self.ranker)),
            "reasons": [
                {
                    "path": r.path,
                    "kind": r.kind,
                    "exact": r.exact,
                    "reasons": [[jsonable(i), jsonable(w)] for i, w in r.reasons],
                    "rest": jsonable(r.rest),
                }
                for r in self.reasons
            ],
        }

    @override
    def __str__(self) -> str:
        # A leaf's ranked-out sentence already gives the score.
        stated = self.status == "ranked_out" and self.ranker is None
        score = "" if self.score is None or stated else f" (score {self.score:.4g})"
        lines = [f"query {self.query!r}, item {self.item!r}: {self.detail}{score}"]
        lines += [f"  route: {r.condition} -> {r.branch}" for r in self.routes]
        if any(r.rank is not None for r in self.retrieval):
            placed = ", ".join(_placed(r) for r in self.retrieval)
            lines.append(f"  retrieved: {placed}")
        if self.ranker is not None:
            ranker = self.ranker
            line = f"  ranker: #{ranker.position} of {ranker.n_candidates} ({ranker.score:.4g})"
            if ranker.contributions:
                terms = ", ".join(f"{k} {v:+.4g}" for k, v in ranker.contributions.items())
                line += f"; {terms}"
            lines.append(line)
        for leaf in self.reasons:
            exact = "exact" if leaf.exact else "approximate"
            terms = ", ".join(f"{i!r} {w:+.4g}" for i, w in leaf.reasons) or "no history item"
            if leaf.kind == "popularity" and leaf.rest is not None:
                terms, rest = f"popularity {leaf.rest:.4g}", ""
            else:
                rest = "" if leaf.rest is None else f"; rest {leaf.rest:+.4g}"
            lines.append(f"  {leaf.path.rsplit('/', 1)[-1]} ({leaf.kind}, {exact}): {terms}{rest}")
        return "\n".join(lines)


def explain(
    estimator: FittedRecommender,
    X: ArrayLike,
    *,
    items: ArrayLike | None = None,
    n_recommendations: int = 10,
    n_reasons: int = 5,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    exclude_interactions: ArrayLike | None = None,
    as_of: ArrayLike | None = None,
) -> list[Explanation]:
    """Explain what ``estimator.recommend`` serves each query, and why not ``items``.

    ``recommend`` is called once, traced (see :func:`trace`), and every served item of
    every query is explained: the route the query took, where each generator placed the
    item, what the ranker made of it -- and which features weighed most, when the ranker
    can say -- and which history items made each leaf recommender score it as it did.

    Parameters
    ----------
    estimator : recommender
        A fitted recommender or pipeline.
    X : array-like of shape (n_queries,) or (n_queries, 1 + n_context)
        The queries, with their context when ``X`` is a matrix, as ``recommend``
        takes them.
    items : array-like of shape (n_items,), default=None
        Items to explain for every query on top of those served, typically ones you
        expected to see. Each gets the stage that let it go: see
        :attr:`Explanation.status`.
    n_recommendations : int, default=10
        Passed to ``recommend``.
    n_reasons : int, default=5
        How many history items and ranker features to give per explanation.
    candidates, exclude_seen, exclude_interactions
        Passed to ``recommend``; an item they remove is ``"excluded"``.
    as_of : scalar or array-like of shape (n_queries,), default=None
        Passed to ``recommend``, for an estimator constructed with ``time=True``.

    Returns
    -------
    explanations : list of Explanation
        For each distinct query in order, its served items best first, then ``items``
        it was not served.

    Examples
    --------
    >>> from skrecsys.inspection import explain
    >>> from skrecsys.recommendation import ItemKNNRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "b"], ["u2", "c"], ["u3", "a"],
    ...      ["u4", "a"], ["u4", "c"], ["u4", "d"]]
    >>> rec = ItemKNNRecommender().fit(X)
    >>> for explanation in explain(rec, ["u3"], items=["a", "b"], n_recommendations=1):
    ...     print(explanation)
    query 'u3', item 'd': served #1 (score 0.5774)
      ItemKNNRecommender (history, exact): 'a' +0.5774; rest +0
    query 'u3', item 'a': filtered out before scoring by exclude_seen (score 0)
      ItemKNNRecommender (history, exact): no history item; rest +0
    query 'u3', item 'b': scored 0.4082, below the cut-off 0.5774
      ItemKNNRecommender (history, exact): 'a' +0.4082; rest +0
    """
    # The queries go to recommend as they came, context included.
    queries, _ = check_queries(X)
    extra = [] if items is None else check_ids(items, name="items").tolist()
    filters = _Filters(candidates, exclude_seen, exclude_interactions)
    with trace(level="full", n_reasons=n_reasons) as traced:
        if uses_time(estimator):
            estimator.recommend(
                X,
                n_recommendations=n_recommendations,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
                as_of=as_of,
            )
        elif as_of is not None:
            raise ValueError("as_of needs an estimator constructed with time=True.")
        else:
            estimator.recommend(
                X,
                n_recommendations=n_recommendations,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )
    out = []
    for query in dict.fromkeys(queries.tolist()):
        query_trace = traced.query(query)
        final = query_trace.final
        served = [] if final is None else final.items.tolist()
        wanted = served + [item for item in dict.fromkeys(extra) if item not in served]
        out.extend(_explain(estimator, query_trace, item, n_reasons, filters) for item in wanted)
    return out


def _explain(
    estimator: FittedRecommender,
    query_trace: QueryTrace,
    item: Hashable,
    n_reasons: int,
    filters: _Filters,
) -> Explanation:
    query = query_trace.query
    final = query_trace.final
    position = score = cutoff = None
    if final is not None and len(final.items):
        cutoff = float(final.scores[-1])
        served = np.flatnonzero(final.items == item)
        if len(served):
            position = int(served[0]) + 1
            score = float(final.scores[served[0]])
    if score is None:
        score = _predict(estimator, query, item)
    status, excluded_by = _status(estimator, query_trace, item, position, filters)
    return Explanation(
        query=query,
        item=item,
        status=status,
        excluded_by=excluded_by,
        position=position,
        score=score,
        cutoff=cutoff,
        routes=tuple(query_trace.routes),
        retrieval=tuple(_retrieval(step, item) for step in query_trace.candidates),
        ranker=_ranker_decision(query_trace, item, n_reasons),
        reasons=tuple(_leaf_reasons(estimator, query_trace, item, n_reasons)),
        trace=query_trace,
    )


def _status(
    estimator: FittedRecommender,
    query_trace: QueryTrace,
    item: Hashable,
    position: int | None,
    filters: _Filters,
) -> tuple[Status, str | None]:
    """The first stage, from the end of the pipeline back, that let ``item`` go.

    With ``"excluded"``, also the filter that did.
    """
    if position is not None:
        return "served", None
    if not lookup_ids(_ids(item), estimator.item_ids_, name="item")[1][0]:
        return "unknown_item", None
    excluded_by = _exclusion(estimator, query_trace.query, item, filters)
    if excluded_by is not None:
        return "excluded", excluded_by
    return _lost_at(query_trace, item), None


def _lost_at(query_trace: QueryTrace, item: Hashable) -> Status:
    """Where an eligible item that was not served fell out of the pipeline.

    Asked of the stage whose list was served only: an item it never received is
    ``"not_retrieved"``, whatever the stages nested in it made of the item.
    """
    path = _decision_path(query_trace)
    for step in query_trace.postprocess:
        if step.path != path:
            continue
        if item in step.before_items.tolist() and item not in step.after_items.tolist():
            return "dropped_by_postprocess"
    ranker = _final_ranker(query_trace)
    fusion = [step for step in query_trace.fusion if step.path == path]
    candidates = [step for step in query_trace.candidates if step.path == path]
    if ranker is not None:
        reached = item in ranker.items.tolist()
    elif fusion:
        reached = any(item in step.items.tolist() for step in fusion)
    elif candidates:
        reached = any(item in step.items.tolist() for step in candidates)
    else:
        reached = True
    return "ranked_out" if reached else "not_retrieved"


_DECISIONS = (CandidatesStep, FusionStep, RankerStep, PostprocessStep)


def _decision_path(query_trace: QueryTrace) -> str:
    """The path of the stage whose list was served to the query.

    That is the outermost stage that retrieves, fuses, ranks or postprocesses; a stage
    that only delegates -- a Switch to the branch it routed the query to, a search to its
    best estimator -- is followed down to the one part that served the query.
    """
    path = query_trace.path
    while not any(isinstance(s, _DECISIONS) and s.path == path for s in query_trace.steps):
        depth = path.count("/") + 2
        children = {
            step.path
            for step in query_trace.served
            if step.path.startswith(f"{path}/") and step.path.count("/") == depth
        }
        if len(children) != 1:
            break
        (path,) = children
    return path


def _exclusion(
    estimator: FittedRecommender, query: object, item: Hashable, filters: _Filters
) -> str | None:
    """The filter of the call that leaves the query no way to be served ``item``, if any.

    Asked of the estimator itself, with ``item`` as the only candidate: whatever the
    pipeline, what it counts as eligible is what its filters let through. Each filter is
    then lifted in turn to name the one that removed the item.
    """
    candidates = filters.candidates
    if candidates is not None and item not in check_ids(candidates, name="candidates").tolist():
        return "candidates"
    exclude_seen = filters.exclude_seen
    excluded = filters.exclude_interactions
    if _eligible(estimator, query, item, exclude_seen=exclude_seen, excluded=excluded):
        return None
    if excluded is not None and _eligible(
        estimator, query, item, exclude_seen=exclude_seen, excluded=None
    ):
        return "exclude_interactions"
    return "exclude_seen" if exclude_seen else "exclude_interactions"


def _eligible(
    estimator: FittedRecommender,
    query: object,
    item: Hashable,
    *,
    exclude_seen: bool,
    excluded: ArrayLike | None,
) -> bool:
    """Whether ``item`` is eligible for ``query`` under these filters; True if unsure."""
    try:
        eligible = estimator._count_eligible(
            _ids(query),
            candidates=_ids(item),
            exclude_seen=exclude_seen,
            exclude_interactions=excluded,
        )
    except (ValueError, TypeError):
        return True
    return int(eligible[0]) > 0


def _final_ranker(query_trace: QueryTrace) -> RankerStep | None:
    """The ranker whose order was served: that of the stage whose list was served.

    None when that stage has no ranker, as a fusion has none, even if a ranker is
    nested in one of its branches.
    """
    path = _decision_path(query_trace)
    return next((step for step in query_trace.ranker if step.path == path), None)


def _ranker_decision(
    query_trace: QueryTrace, item: Hashable, n_reasons: int
) -> RankerDecision | None:
    step = _final_ranker(query_trace)
    if step is None:
        return None
    row = np.flatnonzero(step.items == item)
    if not len(row):
        return None
    ranked = step.ranked().tolist()
    contributions = None
    if step.contributions is not None:
        names = next(
            (f.names for f in query_trace.features if f.path == step.path and f.names), None
        )
        values = step.contributions[row[0]]
        names = (*(names or [f"x{j}" for j in range(len(values) - 1)]), "bias")
        top = np.argsort(-np.abs(np.nan_to_num(values)), kind="stable")[: n_reasons + 1]
        contributions = {names[j]: float(values[j]) for j in top}
    return RankerDecision(
        step.path,
        float(step.scores[row[0]]),
        ranked.index(item) + 1,
        len(ranked),
        contributions,
    )


def _retrieval(step: CandidatesStep, item: Hashable) -> Retrieval:
    rank = step.rank_of(item)
    score = None
    if rank is not None and step.scores.ndim == 1:
        score = float(step.scores[rank])
    return Retrieval(
        step.path, step.source, None if rank is None else rank + 1, score, len(step.items)
    )


def _leaf_reasons(
    estimator: FittedRecommender, query_trace: QueryTrace, item: Hashable, n_reasons: int
) -> list[LeafReasons]:
    """What each leaf the query reached says about ``item``, asked afresh.

    Asked rather than read from the trace, which holds reasons only for the items each
    leaf served, and ``item`` may be one it did not.
    """
    out = []
    paths = dict.fromkeys(step.path for step in query_trace.steps if step.path)
    for path in paths:
        leaf = _resolve(estimator, path)
        attribute = getattr(leaf, "_attribute", None)
        if attribute is None:
            continue
        try:
            found = attribute(_ids(query_trace.query), _ids(item), n_reasons)
        except (ValueError, TypeError):  # the leaf does not know the user or the item
            continue
        if found is None:
            continue
        reasons = tuple(
            (reason, float(weight))
            for reason, weight in zip(found.items[0], found.weights[0], strict=True)
            if reason is not None
        )
        rest = float(found.rest[0]) if found.exact else None
        out.append(LeafReasons(path, found.kind, found.exact, reasons, rest))
    return out


def _resolve(root: object, path: str) -> object | None:
    """The fitted part of ``root`` that ``path`` names, or None if it names none.

    A path alternates the role of a part in its parent -- ``on_true``,
    ``best_estimator``, a generator's name -- with the part's class name.
    """
    parts = path.split("/")
    if parts[0] != type(root).__name__:
        return None
    node = root
    for role, name in zip(parts[1::2], parts[2::2], strict=False):
        node = _child(node, role)
        if node is None or type(node).__name__ != name:
            return None
    return node if len(parts) % 2 else None


def _child(node: object, role: str) -> object | None:
    fitted = getattr(node, f"{role}_", None)
    if fitted is not None:
        return fitted
    for attribute in ("generators_", "recommenders_"):
        named = getattr(node, attribute, None)
        if named:
            return dict(named).get(role)
    return None


def _predict(estimator: FittedRecommender, query: object, item: Hashable) -> float | None:
    """What ``estimator.predict`` scores the pair, when it can."""
    predict = getattr(estimator, "predict", None)
    if predict is None:
        return None
    pair = np.empty((1, 2), dtype=object)
    pair[0] = [query, item]
    try:
        return float(np.asarray(predict(pair))[0])
    except (ValueError, TypeError):
        return None


def _placed(retrieval: Retrieval) -> str:
    if retrieval.rank is None:
        return f"{retrieval.source} not in top {retrieval.n_retrieved}"
    score = "" if retrieval.score is None else f" ({retrieval.score:.4g})"
    return f"{retrieval.source} #{retrieval.rank}{score}"


def _ids(value: object) -> np.ndarray:
    """One identifier as a 1-element object array, whatever its type."""
    out = np.empty(1, dtype=object)
    out[0] = value
    return out

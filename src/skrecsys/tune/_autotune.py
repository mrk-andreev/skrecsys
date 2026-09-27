"""A recommender that tunes the hyperparameters of another before fitting it."""

from collections.abc import Mapping, Sequence
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.utils.validation import check_is_fitted

from skrecsys._tracing import span, traced_recommend, untraced
from skrecsys._typing import (
    CrossValidator,
    FittedRecommender,
    RandomStateLike,
    Recommender,
    clone_as,
    override,
)
from skrecsys.base import RecommenderMixin, fit_clone, is_recommender, predict_pairs, uses_time
from skrecsys.base import serves_unknown_users as _serves_unknown_users
from skrecsys.metrics import get_scorer
from skrecsys.metrics._named import Scorer
from skrecsys.model_selection import WarmStartKFold
from skrecsys.tune._space import Categorical, Distribution, Float, Int, search_space
from skrecsys.tune._study import Study, Trial
from skrecsys.utils._param_validation import check_component, check_int

__all__ = ["AutoTune"]


class AutoTune(RecommenderMixin, BaseEstimator):
    """Tune a recommender's hyperparameters by cross-validation, then fit the best.

    ``fit`` runs a :class:`Study` over the parameters ``estimator`` declares tunable (see
    :func:`search_space`), scoring each candidate configuration by cross-validation on
    the training interactions alone, and refits the best configuration on all of them.
    The first trial is always ``estimator`` as configured, so the tuned model never
    scores below it in cross-validation. ``recommend`` and ``predict`` are then those of
    the refitted model.

    Parameters
    ----------
    estimator : recommender
        The model to tune; it is cloned, never fitted itself.
    search_space : dict of str to distribution, default=None
        Distributions by ``set_params`` name, added to what ``estimator`` declares and
        overriding it where both name a parameter. ``None`` uses the declared space
        alone.
    freeze : sequence of str, default=None
        Parameters to hold at the value ``estimator`` already has, by ``set_params``
        name. Each must be tunable, declared or in ``search_space``; freezing all of
        them leaves nothing to search and raises.
    scoring : str, metric or callable, default=None
        What to maximize: a metric name such as ``"recall@20"``, a metric such as
        ``Recall(20)``, or any ``scorer(estimator, X, y) -> float``, higher being
        better, such as :func:`skrecsys.metrics.make_recommender_scorer` makes. See
        :func:`skrecsys.metrics.get_scorer`. ``None`` is NDCG@10.
    cv : int, cross-validation generator or iterable, default=None
        How the training interactions are split for scoring. ``None`` is a shuffled
        :class:`~skrecsys.model_selection.WarmStartKFold` with 3 folds, and an integer
        is that many such folds; a splitter must keep test users known to the training
        fold, since a recommender cannot rank for a user it has not seen.
    n_trials : int, default=50
        Configurations evaluated, the initial one included.
    sampler : {"tpe", "random"}, default="tpe"
        See :class:`Study`.
    n_startup_trials : int, default=10
        Random trials before TPE takes over.
    random_state : int, RandomState instance or None, default=None
        Seeds the default ``cv`` and the sampler.

    Attributes
    ----------
    best_estimator_ : recommender
        ``estimator`` with ``best_params_``, fitted on all of ``X``.
    best_params_ : dict
        The tuned parameter values.
    best_score_ : float
        Their mean cross-validation score.
    study_ : Study
        Every trial, with its parameters and score.
    user_ids_, item_ids_ : ndarray
        Those of ``best_estimator_``.
    n_users_, n_items_ : int

    Examples
    --------
    >>> from skrecsys.recommendation import BM25Recommender
    >>> from skrecsys.tune import AutoTune
    >>> X = [[f"u{u}", f"i{(u * 7 + k) % 40}"] for u in range(30) for k in range(8)]
    >>> tuned = AutoTune(BM25Recommender(), n_trials=5, random_state=0).fit(X)
    >>> sorted(tuned.best_params_)
    ['b', 'k1', 'n_neighbors']

    Parameters set on the instance can be held out of the search:

    >>> held = AutoTune(BM25Recommender(k1=0.5), freeze=["k1"], n_trials=5, random_state=0)
    >>> held.fit(X).best_estimator_.k1, sorted(held.best_params_)
    (0.5, ['b', 'n_neighbors'])

    The metric to maximize can be named:

    >>> AutoTune(BM25Recommender(), scoring="recall@5", n_trials=3).fit(X).best_score_ >= 0
    True
    """

    def __init__(
        self,
        estimator: Recommender,
        *,
        search_space: Mapping[str, Distribution] | None = None,
        freeze: Sequence[str] | None = None,
        scoring: str | Scorer | None = None,
        cv: int | CrossValidator | None = None,
        n_trials: int = 50,
        sampler: Literal["tpe", "random"] = "tpe",
        n_startup_trials: int = 10,
        random_state: RandomStateLike = None,
    ) -> None:
        self.estimator = estimator
        self.search_space = search_space
        self.freeze = freeze
        self.scoring = scoring
        self.cv = cv
        self.n_trials = n_trials
        self.sampler = sampler
        self.n_startup_trials = n_startup_trials
        self.random_state = random_state

    @property
    def _serves_unknown_users(self) -> bool:
        """Read by :func:`skrecsys.base.serves_unknown_users`: as the tuned model does."""
        return _serves_unknown_users(self.estimator)

    @property
    def time(self) -> bool:
        """Read by :func:`skrecsys.base.uses_time`: as the tuned model does."""
        return uses_time(self.estimator)

    def _space(self) -> dict[str, Distribution]:
        """The declared space, with ``search_space`` applied and ``freeze`` removed."""
        space = search_space(self.estimator) | self._checked_search_space()
        unknown = sorted(set(space) - set(self.estimator.get_params(deep=True)))
        if unknown:
            raise ValueError(
                f"search_space names {unknown}, which are not parameters of "
                f"{type(self.estimator).__name__}."
            )
        if not space:
            raise ValueError(
                f"{type(self.estimator).__name__} declares no tunable parameters; annotate "
                "its __init__ parameters with skrecsys.tune distributions or pass "
                "search_space."
            )
        frozen = self._checked_freeze()
        not_tunable = sorted(frozen - set(space))
        if not_tunable:
            raise ValueError(
                f"freeze names {not_tunable}, which are not tunable parameters of "
                f"{type(self.estimator).__name__}; tunable: {sorted(space)}."
            )
        space = {name: dist for name, dist in space.items() if name not in frozen}
        if not space:
            raise ValueError("freeze holds every tunable parameter; nothing is left to search.")
        return space

    def _checked_search_space(self) -> dict[str, Distribution]:
        if self.search_space is None:
            return {}
        if not isinstance(self.search_space, Mapping):
            raise TypeError(
                f"search_space must be a mapping, got {type(self.search_space).__name__}."
            )
        for name, dist in self.search_space.items():
            if not isinstance(dist, Float | Int | Categorical):
                raise TypeError(
                    f"search_space[{name!r}] must be a Float, Int or Categorical, got {dist!r}."
                )
        return dict(self.search_space)

    def _checked_freeze(self) -> set[str]:
        if self.freeze is None:
            return set()
        # A bare string is a sequence of one-letter names, never what was meant.
        if isinstance(self.freeze, str) or not all(isinstance(n, str) for n in self.freeze):
            raise TypeError(f"freeze must be a sequence of parameter names, got {self.freeze!r}.")
        return set(self.freeze)

    @override
    @untraced()
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Tune ``estimator`` on the interactions ``X`` and fit the best configuration.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers.
        y : array-like of shape (n_interactions,), default=None
            Interaction values; ``None`` gives every interaction weight 1.

        Returns
        -------
        self : object
        """
        check_component(self.estimator, "estimator", is_recommender, "a recommender")
        n_trials = check_int(self.n_trials, "n_trials", min_value=1)
        space = self._space()
        scoring = get_scorer(self.scoring)
        cv = self.cv
        if cv is None or isinstance(cv, int):
            cv = WarmStartKFold(
                3 if cv is None else cv, shuffle=True, random_state=self.random_state
            )
        # The folds are drawn once, so every trial is scored on the same splits and the
        # comparison between trials is of the parameters alone.
        splits = list(check_cv(cv).split(X, y))

        study = Study(
            sampler=self.sampler,
            n_startup_trials=self.n_startup_trials,
            random_state=self.random_state,
        )
        study.enqueue(self.estimator.get_params(deep=True))

        def objective(trial: Trial) -> float:
            params = {name: trial.suggest(name, dist) for name, dist in space.items()}
            candidate = clone_as(self.estimator).set_params(**params)
            scores = cross_val_score(
                candidate,
                X,
                y,
                scoring=scoring,
                cv=splits,
                error_score="raise",
            )
            return float(np.mean(scores))

        study.optimize(objective, n_trials)

        self.study_ = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        best: FittedRecommender = fit_clone(
            clone_as(self.estimator).set_params(**self.best_params_), X, y
        )
        self.best_estimator_ = best
        self.user_ids_ = best.user_ids_
        self.item_ids_ = best.item_ids_
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        return self

    @override
    @traced_recommend
    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = 10,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
        as_of: ArrayLike | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Return the recommendations of ``best_estimator_``.

        Parameters are those of :meth:`skrecsys.base.RecommenderMixin.recommend`, and
        ``as_of``, the time to rank as of, for an estimator constructed with ``time=True``.
        """
        check_is_fitted(self)
        best = self.best_estimator_
        if as_of is not None and not uses_time(best):
            raise ValueError("as_of needs an estimator constructed with time=True.")
        with span("best_estimator"):
            if uses_time(best):
                return best.recommend(
                    X,
                    n_recommendations=n_recommendations,
                    candidates=candidates,
                    exclude_seen=exclude_seen,
                    exclude_interactions=exclude_interactions,
                    as_of=as_of,
                )
            return best.recommend(
                X,
                n_recommendations=n_recommendations,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )

    @override
    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        check_is_fitted(self)
        return self.best_estimator_._count_eligible(
            X,
            candidates=candidates,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
        )

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs with ``best_estimator_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            User-item pairs.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        return predict_pairs(self.best_estimator_, X)

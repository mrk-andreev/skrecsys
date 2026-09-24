"""Hyperparameter tuning: an Optuna-style study with a native TPE sampler.

Estimators declare the range worth searching for each tunable parameter as an
annotation on ``__init__``, for example ``k1: Annotated[float, Float(0.05, 5.0)]``;
:func:`search_space` reads them back. :class:`Study` is the ask-and-tell loop for any
objective, and :class:`AutoTune` wraps a recommender so that ``fit`` tunes it by
cross-validation first.
"""

from skrecsys.tune._autotune import AutoTune
from skrecsys.tune._space import Categorical, Distribution, Float, Int, search_space
from skrecsys.tune._study import Study, Trial

__all__ = [
    "AutoTune",
    "Categorical",
    "Distribution",
    "Float",
    "Int",
    "Study",
    "Trial",
    "search_space",
]

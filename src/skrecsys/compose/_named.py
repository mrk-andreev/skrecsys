"""Lists of named components, as :class:`ConcatFeatures` and :class:`BlendRanker` hold them.

Both composites take a list of components -- bare, or as ``(name, component)`` tuples --
and expose each component's parameters as ``name__param``, the way scikit-learn's
``FeatureUnion`` does. The rules are the same whatever the components are, so they live
here once, parameterized by what a component must be.
"""

import sys
from collections.abc import Callable, Sequence
from typing import ClassVar, Generic, Self, TypeAlias, TypeVar

from sklearn.base import BaseEstimator

from skrecsys._typing import Estimator, override

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

C = TypeVar("C", bound=Estimator)

#: A component list as a caller writes it: all bare components or all named ones.
ComponentList: TypeAlias = list[C] | list[tuple[str, C]]


def name_components(items: Sequence[C | tuple[str, C]], param: str) -> list[tuple[str, C]]:
    """``items`` as (name, component) pairs, naming what the caller did not.

    An unnamed component is named after its class in lower case, numbered when a class
    repeats.
    """
    explicit = [item for item in items if isinstance(item, tuple)]
    bare = [item for item in items if not isinstance(item, tuple)]
    if explicit and bare:
        raise ValueError(f"{param} must be all components or all (name, component) tuples.")
    if explicit:
        names = [name for name, _ in explicit]
        if len(set(names)) != len(names):
            raise ValueError(f"Names in {param} must be unique, got {names}.")
        return [(str(name), component) for name, component in explicit]
    base = [type(item).__name__.lower() for item in bare]
    counts = {name: base.count(name) for name in base}
    seen: dict[str, int] = {}
    named: list[tuple[str, C]] = []
    for name, component in zip(base, bare, strict=True):
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
        label = f"{name}-{seen[name]}" if counts[name] > 1 else name
        named.append((label, component))
    return named


def check_component_list(
    value: object, param: str, is_component: Callable[[object], TypeIs[C]], kind: str
) -> ComponentList[C]:
    """A component list set through ``set_params``, validated like ``fit`` would."""
    if not isinstance(value, list):
        raise TypeError(f"{param} must be a list, got {type(value).__name__}.")
    bare: list[C] = []
    named: list[tuple[str, C]] = []
    for item in value:
        match item:
            case tuple((name, component)) if is_component(component):
                named.append((name, component))
            case tuple((_, component)):
                raise TypeError(f"{type(component).__name__} is not {kind}.")
            case _ if is_component(item):
                bare.append(item)
            case _:
                raise TypeError(f"{type(item).__name__} is not {kind}.")
    if bare and named:
        raise ValueError(f"{param} must be all components or all (name, component) tuples.")
    return named or bare


def nested_params(named: Sequence[tuple[str, C]]) -> dict[str, object]:
    """Every component by name, and each of its parameters as ``name__param``."""
    params: dict[str, object] = {}
    for name, component in named:
        params[name] = component
        for key, value in component.get_params(deep=True).items():
            params[f"{name}__{key}"] = value
    return params


def set_nested_params(
    owner: str,
    named: list[tuple[str, C]],
    params: dict[str, object],
    is_component: Callable[[object], TypeIs[C]],
    kind: str,
) -> bool:
    """Apply ``name__param`` and ``name`` parameters to ``named`` in place.

    Returns whether a component was replaced outright, which the caller has to write
    back to its own list.
    """
    names = [name for name, _ in named]
    replaced = False
    for key, value in params.items():
        name, _, rest = key.partition("__")
        if name not in names:
            raise ValueError(f"Invalid parameter {key!r} for {owner}.")
        position = names.index(name)
        if rest:
            named[position][1].set_params(**{rest: value})
        elif is_component(value):
            named[position] = (name, value)
            replaced = True
        else:
            raise TypeError(f"{type(value).__name__} is not {kind}.")
    return replaced


class NamedComponentsEstimator(BaseEstimator, Generic[C]):
    """An estimator holding one list of named components, whose parameters it exposes.

    ``get_params`` lists every component by name and each of its parameters as
    ``name__param``; ``set_params`` takes them back, the way :class:`BlendRanker` and
    ``FeatureUnion`` do. A subclass names the parameter holding the list in
    ``_components_param``, returns its value from :meth:`_components`, and says what a
    component must be in ``_component_kind`` and :meth:`_is_component`. The list may be
    ``None`` where the subclass allows it.
    """

    _components_param: ClassVar[str]
    _component_kind: ClassVar[str]

    def _is_component(self, value: object) -> TypeIs[C]:
        raise NotImplementedError

    def _components(self) -> ComponentList[C] | None:
        """The value of the ``_components_param`` parameter."""
        raise NotImplementedError

    def _named(self) -> list[tuple[str, C]]:
        """The components as (name, component) pairs, naming what the caller did not."""
        items = self._components()
        return [] if items is None else name_components(items, self._components_param)

    @override
    def get_params(self, deep: bool = True) -> dict[str, object]:
        params = dict[str, object](super().get_params(deep=False))
        if not deep:
            return params
        return params | nested_params(self._named())

    @override
    def set_params(self, **params: object) -> Self:
        param, kind = self._components_param, self._component_kind
        if param in params:
            value = params.pop(param)
            if value is not None:
                value = check_component_list(value, param, self._is_component, kind)
            setattr(self, param, value)
        own_names = set(super().get_params(deep=False))
        own = {key: params.pop(key) for key in list(params) if key in own_names}
        super().set_params(**own)
        if not params:
            return self
        named = self._named()
        if set_nested_params(type(self).__name__, named, params, self._is_component, kind):
            items = self._components()
            explicit = items is not None and isinstance(items[0], tuple)
            setattr(self, param, named if explicit else [component for _, component in named])
        return self

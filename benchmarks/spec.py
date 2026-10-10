"""What the benchmarks run, read from ``benchmarks/config/*.json``.

Each benchmark has a config file listing its entries -- a model or an index, named by
``package`` and ``cls`` with its ``params`` -- and the datasets it runs on, with the
settings each one is timed under. ``datasets.json`` defines the datasets themselves,
since the leaderboard and the index report share them.

Every entry carries a ``version`` that is bumped by hand, and a result is keyed on it
*and* on a hash of everything the config says about the run. So an edit to ``params``
or to a dataset's settings re-runs exactly the results it touches without anyone having
to remember to bump anything, while ``version`` is kept for the changes a config cannot
see -- a rewritten kernel, a fixed bug -- which would otherwise leave an old number
looking current. ``comment`` and ``caption`` are prose and change nothing.

The format is deliberately strict. An unknown key is an error rather than something to
ignore, because the likeliest unknown key is a misspelled known one, and ``"parms"``
silently building a model at its defaults is exactly the kind of wrong number this whole
arrangement exists to prevent.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing_extensions import TypeAliasType

from skrecsys.typing import override

CONFIG_DIR = Path(__file__).resolve().parent / "config"

#: The config format this module reads. A result records it, so a format change that
#: alters what a key means invalidates everything keyed under the old one.
SCHEMA = 1

#: Benchmarks with a config file, in the order a report lists them.
BENCHMARKS = ("leaderboard", "sequential", "reranking", "candidates", "indexes")

#: Stands for every model of the ``models_from`` config, in its order.
WILDCARD = "*"


#: A value as ``json.loads`` returns it.
JSON = TypeAliasType("JSON", "str | int | float | bool | list[JSON] | dict[str, JSON] | None")


class Warmup(TypedDict):
    """Untimed calls made before each operation is sampled."""

    fit: int
    rank: int


class Settings(TypedDict):
    """A dataset's settings, once :func:`check_settings` has accepted them.

    Which keys are present depends on the benchmark; see :data:`SETTINGS`.
    """

    k: NotRequired[int]
    cutoffs: NotRequired[list[int]]
    repeat: int
    rank_repeat: NotRequired[int]
    budget: float
    warmup: Warmup
    latency_batch: NotRequired[list[int]]
    latency_repeat: NotRequired[int]
    catalog_scale: NotRequired[list[float]]
    n_retrieved: NotRequired[list[int]]


class Split(Protocol):
    """What a benchmark reads of a loaded dataset: the interactions and one split."""

    data: NDArray[np.generic]
    target: NDArray[np.float64]
    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]


#: The attributes of a :class:`Split`, which a loaded dataset is checked for.
_SPLIT_FIELDS = ("data", "target", "train_indices", "test_indices")


class Loader(Protocol):
    """How a dataset named in ``datasets.json`` is loaded, such as
    :func:`skrecsys.datasets.fetch_movielens_100k`: its ``subset`` and ``load`` keys."""

    def __call__(self, *, subset: str, **kwargs: JSON) -> object: ...


class ConfigError(ValueError):
    """A config file that does not describe a runnable benchmark."""


def _check_keys(
    where: str, found: Mapping[str, JSON], required: set[str], optional: set[str]
) -> None:
    missing = required - set(found)
    if missing:
        raise ConfigError(f"{where}: missing {sorted(missing)}.")
    unknown = set(found) - required - optional
    if unknown:
        raise ConfigError(
            f"{where}: unknown {sorted(unknown)}; expected some of {sorted(required | optional)}."
        )


def parse_json(text: str) -> JSON:
    """``json.loads``, typed as what it returns: typeshed says ``Any``."""
    return cast(JSON, json.loads(text))


def _object(where: str, value: JSON) -> dict[str, JSON]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be an object, got {value!r}.")
    return value


def _list(where: str, value: JSON) -> list[JSON]:
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list, got {value!r}.")
    return value


def _string(where: str, value: JSON) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string, got {value!r}.")
    return value


def _ints(where: str, value: JSON) -> list[int]:
    items = _list(where, value)
    ints = [item for item in items if isinstance(item, int)]
    if len(ints) != len(items):
        raise ConfigError(f"{where} must list integers, got {value!r}.")
    return ints


def _read(path: Path) -> dict[str, JSON]:
    try:
        data = parse_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path} does not exist.") from None
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a JSON object.")
    if data.get("schema") != SCHEMA:
        raise ConfigError(f"{path}: schema must be {SCHEMA}, got {data.get('schema')!r}.")
    return data


#: What :func:`build_object` is asked to build.
T = TypeVar("T")


def canonical(value: JSON) -> str:
    """One spelling of ``value``, so equal configs hash equal whatever their key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def key(inputs: Mapping[str, JSON]) -> str:
    """The cache key of a result: a short hash of everything that determined it."""
    return hashlib.sha256(canonical(dict(inputs)).encode()).hexdigest()[:16]


def resolve(dotted: str) -> object:
    """The object ``package.name`` names, importing ``package`` to find it."""
    package, _, name = dotted.rpartition(".")
    if not package:
        raise ConfigError(f"{dotted!r} is not a dotted path.")
    return getattr(importlib.import_module(package), name)


def build_object(package: str, cls: str, params: Mapping[str, JSON], kind: type[T]) -> T:
    """Construct ``package.cls(**params)``, checked to be a ``kind``.

    A parameter whose value is itself ``{"package": ..., "cls": ..., "params": ...}`` is
    constructed the same way, as a scikit-learn estimator, so an estimator can take an
    index -- or another estimator -- as a parameter without the config format knowing
    what either of them is.
    """
    built = {name: _build_value(value) for name, value in params.items()}
    instance = getattr(importlib.import_module(package), cls)(**built)
    if not isinstance(instance, kind):
        raise ConfigError(f"{package}.{cls} is not a {kind.__name__}.")
    return instance


def _build_value(value: JSON) -> JSON | BaseEstimator:
    if isinstance(value, dict) and "cls" in value:
        _check_keys("nested object", value, {"package", "cls"}, {"params"})
        package, cls, params = value["package"], value["cls"], value.get("params", {})
        if not isinstance(package, str) or not isinstance(cls, str):
            raise ConfigError(f"nested object: package and cls must be strings, got {value!r}.")
        if not isinstance(params, dict):
            raise ConfigError(f"nested object: params must be an object, got {params!r}.")
        return build_object(package, cls, params, BaseEstimator)
    return value


@dataclass(frozen=True)
class Dial:
    """The query-time parameter an index is swept on, and the values it is swept over."""

    param: str
    label: str
    values: tuple[int, ...]

    def setting(self, value: int) -> str:
        """How one point of the sweep is named in a table cell."""
        return f"{self.label}={value}"

    def middle(self) -> int:
        """The representative point the single-setting tables run at."""
        return self.values[len(self.values) // 2]


@dataclass(frozen=True)
class Entry:
    """One thing a benchmark runs: a class, its parameters and a version."""

    name: str
    package: str
    cls: str
    params: Mapping[str, JSON]
    version: str
    comment: str = ""

    def spec(self) -> dict[str, JSON]:
        """The part of the entry that determines a measurement, for the cache key."""
        return {
            "package": self.package,
            "cls": self.cls,
            "params": copy.deepcopy(dict(self.params)),
            "version": self.version,
        }

    def build(self, kind: type[T]) -> T:
        """A fresh, unfitted instance, checked to be a ``kind``: any recommender for a
        leaderboard row, a ``BaseRecommender`` where it is indexed as well."""
        return build_object(self.package, self.cls, self.params, kind)


@dataclass(frozen=True, kw_only=True)
class IndexEntry(Entry):
    """An index: an entry with ``label`` and ``short`` for the tables and a sweep dial."""

    label: str
    short: str
    dial: Dial

    @override
    def spec(self) -> dict[str, JSON]:
        return super().spec() | {
            "dial": {"param": self.dial.param, "values": list(self.dial.values)}
        }

    def with_values(self, values: Sequence[int]) -> IndexEntry:
        """The same index swept over other values, as a dataset may ask for."""
        return replace(self, dial=replace(self.dial, values=tuple(values)))


_ENTRY_REQUIRED = {"name", "package", "cls", "params", "version"}


def _entry(where: str, raw: JSON, extra: frozenset[str] = frozenset()) -> Entry:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: an entry must be an object, got {raw!r}.")
    _check_keys(where, raw, _ENTRY_REQUIRED | extra, {"comment"})
    name = _string(f"{where} name", raw["name"])
    where = f"{where} {name!r}"
    params, version = raw["params"], raw["version"]
    if not isinstance(params, dict):
        raise ConfigError(f"{where}: params must be an object.")
    if not isinstance(version, str) or not version:
        raise ConfigError(f"{where}: version must be a non-empty string.")
    return Entry(
        name=name,
        package=_string(f"{where} package", raw["package"]),
        cls=_string(f"{where} cls", raw["cls"]),
        params=params,
        version=version,
        comment=_string(f"{where} comment", raw.get("comment", "")),
    )


def _index_entry(where: str, raw: JSON) -> IndexEntry:
    entry = _entry(where, raw, frozenset({"label", "short", "dial"}))
    raw = _object(where, raw)
    where = f"{where} {entry.name!r}"
    raw_dial = _object(f"{where} dial", raw["dial"])
    _check_keys(f"{where} dial", raw_dial, {"param", "label", "values"}, set())
    values = raw_dial["values"]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, int) and v >= 1 for v in values)
    ):
        raise ConfigError(f"{where}: dial values must be a non-empty list of integers >= 1.")
    return IndexEntry(
        name=entry.name,
        package=entry.package,
        cls=entry.cls,
        params=entry.params,
        version=entry.version,
        comment=entry.comment,
        label=_string(f"{where} label", raw["label"]),
        short=_string(f"{where} short", raw["short"]),
        dial=Dial(
            _string(f"{where} dial param", raw_dial["param"]),
            _string(f"{where} dial label", raw_dial["label"]),
            tuple(_ints(f"{where} dial values", values)),
        ),
    )


@dataclass(frozen=True)
class DatasetDef:
    """A dataset as ``datasets.json`` defines it: how to load it and what to call it."""

    name: str
    loader: str
    subset: str
    caption: str
    load_kwargs: Mapping[str, JSON] = field(default_factory=dict)
    max_eval_users: int | None = None
    #: A splitter, as ``{"package", "cls", "params"}``, that replaces the subset's split.
    split: Mapping[str, JSON] | None = None

    def spec(self) -> dict[str, JSON]:
        """What determines the data a result was measured on. The caption does not.

        ``split`` is there only when set, so that a dataset without one keeps the key
        its results were stored under before datasets could have one.
        """
        found: dict[str, JSON] = {
            "loader": self.loader,
            "subset": self.subset,
            "load": dict(self.load_kwargs),
            "max_eval_users": self.max_eval_users,
        }
        if self.split is not None:
            found["split"] = copy.deepcopy(dict(self.split))
        return found

    def title(self) -> str:
        """The caption with its placeholders filled in."""
        return self.caption.format(subset=self.subset, **self.load_kwargs)

    def load(self) -> Split:
        """The split, as a Bunch carrying ``train_indices`` and ``test_indices``."""
        loader = resolve(self.loader)
        if not callable(loader):
            raise ConfigError(f"{self.loader} is not a dataset loader.")
        # Imported by name, so only the call below can show it takes these arguments.
        loaded = cast(Loader, loader)(subset=self.subset, **self.load_kwargs)
        if self.split is not None:
            _apply_split(loaded, self.split)
        # Not `isinstance(loaded, Split)`: since Python 3.12 a protocol check looks
        # attributes up statically, and a Bunch's are dictionary keys it cannot see.
        if not all(hasattr(loaded, name) for name in _SPLIT_FIELDS):
            raise ConfigError(f"{self.loader}(subset={self.subset!r}) returned no split.")
        return cast(Split, loaded)


class Splitter(Protocol):
    """A scikit-learn splitter, such as :class:`skrecsys.model_selection.ColdStartSplit`."""

    def split(
        self, X: NDArray[np.generic]
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...


def _apply_split(loaded: object, split: Mapping[str, JSON]) -> None:
    """Set ``train_indices`` and ``test_indices`` from the first split ``split`` makes."""
    package, cls = split["package"], split["cls"]
    if not isinstance(package, str) or not isinstance(cls, str):
        raise ConfigError(f"split: package and cls must be strings, got {dict(split)!r}.")
    params = _object("split params", split.get("params", {}))
    splitter = cast(Splitter, build_object(package, cls, params, object))
    # A loader returns a Bunch, which is a dict whose keys are also attributes.
    if not isinstance(loaded, dict) or "data" not in loaded:
        raise ConfigError(f"split: the loaded dataset has no data to split with {cls}.")
    train, test = next(iter(splitter.split(loaded["data"])))
    loaded["train_indices"] = np.asarray(train, dtype=np.intp)
    loaded["test_indices"] = np.asarray(test, dtype=np.intp)


def load_datasets() -> dict[str, DatasetDef]:
    """Every dataset ``datasets.json`` defines."""
    path = CONFIG_DIR / "datasets.json"
    data = _read(path)
    _check_keys(str(path), data, {"schema", "datasets"}, set())
    datasets: dict[str, DatasetDef] = {}
    for name, value in _object(f"{path} datasets", data["datasets"]).items():
        where = f"{path} dataset {name!r}"
        raw = _object(where, value)
        _check_keys(
            where, raw, {"loader", "subset", "caption"}, {"load", "max_eval_users", "split"}
        )
        max_eval_users = raw.get("max_eval_users")
        if max_eval_users is not None and not _positive_int(max_eval_users):
            raise ConfigError(f"{where}: max_eval_users must be an integer >= 1 or null.")
        datasets[name] = DatasetDef(
            name=name,
            loader=_string(f"{where} loader", raw["loader"]),
            subset=_string(f"{where} subset", raw["subset"]),
            caption=_string(f"{where} caption", raw["caption"]),
            load_kwargs=_object(f"{where} load", raw.get("load", {})),
            max_eval_users=max_eval_users if isinstance(max_eval_users, int) else None,
            split=_split_spec(where, raw.get("split")),
        )
    return datasets


def _split_spec(where: str, value: JSON) -> dict[str, JSON] | None:
    """A dataset's ``split`` key, checked for shape; it is built when the dataset loads."""
    if value is None:
        return None
    raw = _object(f"{where} split", value)
    _check_keys(f"{where} split", raw, {"package", "cls"}, {"params"})
    _string(f"{where} split package", raw["package"])
    _string(f"{where} split cls", raw["cls"])
    _object(f"{where} split params", raw.get("params", {}))
    return raw


@dataclass(frozen=True)
class Target:
    """One dataset as one benchmark runs it: the data, the settings and the exceptions."""

    definition: DatasetDef
    settings: Settings
    #: The same settings as the config states them, which is what a result's key hashes.
    raw_settings: dict[str, JSON]
    #: Models left out, and why, in the order the caption lists them.
    skip: Mapping[str, str]
    #: Per-index dial values that replace the index's own, for an expensive dataset.
    dials: Mapping[str, tuple[int, ...]]

    @property
    def name(self) -> str:
        return self.definition.name


#: The settings each benchmark reads, all of them required: a setting a run needs is
#: caught missing when the config loads rather than minutes into the run that needs it.
SETTINGS = {
    "leaderboard": ("k", "repeat", "rank_repeat", "budget", "warmup"),
    "sequential": ("cutoffs", "repeat", "rank_repeat", "budget", "warmup"),
    "reranking": ("k", "latency_repeat", "budget", "warmup"),
    "candidates": ("n_retrieved", "latency_repeat", "budget", "warmup"),
    "indexes": (
        "k",
        "repeat",
        "rank_repeat",
        "budget",
        "warmup",
        "latency_batch",
        "latency_repeat",
        "catalog_scale",
    ),
}


def _positive_int(value: JSON) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def check_settings(benchmark: str, where: str, settings: Mapping[str, JSON]) -> Settings:
    """Refuse settings a run could not use, naming the one that is wrong.

    Every timed operation samples at least once, which is what lets a result always
    have a median; a cap of zero would leave a model fitted zero times.
    """
    _check_keys(f"{where} settings", settings, set(SETTINGS[benchmark]), set())
    for name in ("k", "repeat", "rank_repeat", "latency_repeat"):
        if name in settings and not _positive_int(settings[name]):
            raise ConfigError(f"{where}: {name} must be an integer >= 1, got {settings[name]!r}.")
    budget = settings["budget"]
    if isinstance(budget, bool) or not isinstance(budget, int | float) or budget <= 0:
        raise ConfigError(f"{where}: budget must be a number of seconds > 0, got {budget!r}.")
    warmup = settings["warmup"]
    if not isinstance(warmup, Mapping):
        raise ConfigError(f"{where}: warmup must be an object, got {warmup!r}.")
    _check_keys(f"{where} warmup", warmup, {"fit", "rank"}, set())
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in warmup.values()):
        raise ConfigError(f"{where}: warm-up counts must be integers >= 0, got {dict(warmup)}.")
    _check_lists(where, settings)
    # Every key the benchmark reads was checked above, and no other key is allowed in.
    return cast(Settings, dict(settings))


def _check_lists(where: str, settings: Mapping[str, JSON]) -> None:
    """The list-valued settings of :func:`check_settings`."""
    if "cutoffs" in settings:
        cutoffs = settings["cutoffs"]
        if not isinstance(cutoffs, list) or not cutoffs or not all(map(_positive_int, cutoffs)):
            raise ConfigError(f"{where}: cutoffs must be a non-empty list of integers >= 1.")
    if "n_retrieved" in settings:
        budgets = settings["n_retrieved"]
        if not isinstance(budgets, list) or not budgets or not all(map(_positive_int, budgets)):
            raise ConfigError(f"{where}: n_retrieved must be a non-empty list of integers >= 1.")
    if "latency_batch" in settings:
        batches = settings["latency_batch"]
        if not isinstance(batches, list) or not all(map(_positive_int, batches)):
            raise ConfigError(f"{where}: latency_batch must list integers >= 1.")
    if "catalog_scale" in settings:
        shares = settings["catalog_scale"]
        if not isinstance(shares, list) or not all(
            isinstance(share, int | float) and 0 < share <= 1 for share in shares
        ):
            raise ConfigError(f"{where}: catalog_scale must list shares in (0, 1].")


def _merge(base: Mapping[str, JSON], override: Mapping[str, JSON]) -> dict[str, JSON]:
    """``override`` laid over ``base``, one level of objects deep (``warmup`` and so on)."""
    merged = copy.deepcopy(dict(base))
    for name, value in override.items():
        current = merged.get(name)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[name] = {**current, **value}
        else:
            merged[name] = copy.deepcopy(value)
    return merged


@dataclass(frozen=True)
class Config:
    """One benchmark's config, with ``models_from`` and ``*`` already resolved."""

    benchmark: str
    models: tuple[Entry, ...]
    indexes: tuple[IndexEntry, ...]
    targets: Mapping[str, Target]

    def model(self, name: str) -> Entry:
        for entry in self.models:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def active_models(self, dataset: str) -> list[Entry]:
        """The models run on ``dataset``: every entry its skip list does not name."""
        skip = self.targets[dataset].skip
        return [entry for entry in self.models if entry.name not in skip]

    def active_indexes(self, dataset: str) -> list[IndexEntry]:
        """The indexes swept on ``dataset``, with any per-dataset dial values applied."""
        dials = self.targets[dataset].dials
        return [
            entry.with_values(dials[entry.name]) if entry.name in dials else entry
            for entry in self.indexes
        ]


_CONFIG_REQUIRED = {"schema", "settings", "datasets", "models"}
_CONFIG_OPTIONAL = {"models_from", "indexes"}


def load(benchmark: str) -> Config:
    """The config of ``benchmark``, validated and resolved."""
    if benchmark not in BENCHMARKS:
        raise ConfigError(f"unknown benchmark {benchmark!r}; choose from {list(BENCHMARKS)}.")
    path = CONFIG_DIR / f"{benchmark}.json"
    data = _read(path)
    _check_keys(str(path), data, _CONFIG_REQUIRED, _CONFIG_OPTIONAL)

    models = _models(path, benchmark, data)
    indexes = tuple(
        _index_entry(f"{path} indexes[{position}]", raw)
        for position, raw in enumerate(_list(f"{path} indexes", data.get("indexes", [])))
    )
    _check_unique(path, "index", [entry.name for entry in indexes])
    targets = _targets(path, benchmark, data, models, indexes)
    return Config(benchmark, models, indexes, targets)


def _models(path: Path, benchmark: str, data: Mapping[str, JSON]) -> tuple[Entry, ...]:
    """The config's models, with names and ``*`` resolved against ``models_from``."""
    inherited: tuple[Entry, ...] = ()
    if "models_from" in data:
        source = _string(f"{path} models_from", data["models_from"])
        if source == benchmark:
            raise ConfigError(f"{path}: models_from cannot name the config itself.")
        inherited = load(source).models

    models: list[Entry] = []
    for position, raw in enumerate(_list(f"{path} models", data["models"])):
        where = f"{path} models[{position}]"
        if raw == WILDCARD:
            if not inherited:
                raise ConfigError(f"{where}: {WILDCARD!r} needs a models_from to expand.")
            models.extend(inherited)
        elif isinstance(raw, str):
            matches = [entry for entry in inherited if entry.name == raw]
            if not matches:
                raise ConfigError(
                    f"{where}: {raw!r} is not a model of "
                    f"{data.get('models_from', 'any models_from')}."
                )
            models.extend(matches)
        else:
            models.append(_entry(where, raw))
    _check_unique(path, "model", [entry.name for entry in models])
    return tuple(models)


def _targets(
    path: Path,
    benchmark: str,
    data: Mapping[str, JSON],
    models: Sequence[Entry],
    indexes: Sequence[IndexEntry],
) -> dict[str, Target]:
    """The config's datasets, each with its settings merged over the config's own."""
    definitions = load_datasets()
    model_names = {entry.name for entry in models}
    index_names = {entry.name for entry in indexes}
    targets: dict[str, Target] = {}
    for name, value in _object(f"{path} datasets", data["datasets"]).items():
        where = f"{path} dataset {name!r}"
        if name not in definitions:
            raise ConfigError(f"{where}: not defined in datasets.json.")
        raw = _object(where, value)
        _check_keys(where, raw, set(), {"settings", "skip", "dials"})
        skip = {
            model: _string(f"{where} skip {model!r}", why)
            for model, why in _object(f"{where} skip", raw.get("skip", {})).items()
        }
        unknown = [model for model in skip if model not in model_names]
        if unknown:
            raise ConfigError(f"{where}: skip names {unknown}, which are not models here.")
        dials = {
            index: _ints(f"{where} dials {index!r}", values)
            for index, values in _object(f"{where} dials", raw.get("dials", {})).items()
        }
        unknown = [index for index in dials if index not in index_names]
        if unknown:
            raise ConfigError(f"{where}: dials name {unknown}, which are not indexes here.")
        settings = _merge(
            _object(f"{path} settings", data["settings"]),
            _object(f"{where} settings", raw.get("settings", {})),
        )
        targets[name] = Target(
            definition=definitions[name],
            settings=check_settings(benchmark, where, settings),
            raw_settings=settings,
            skip=skip,
            dials={index: tuple(values) for index, values in dials.items()},
        )
    return targets


def _check_unique(path: Path, what: str, names: Sequence[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ConfigError(f"{path}: {what} {name!r} is listed twice.")
        seen.add(name)

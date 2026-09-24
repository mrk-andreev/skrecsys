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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from skrecsys._typing import override

CONFIG_DIR = Path(__file__).resolve().parent / "config"

#: The config format this module reads. A result records it, so a format change that
#: alters what a key means invalidates everything keyed under the old one.
SCHEMA = 1

#: Benchmarks with a config file, in the order a report lists them.
BENCHMARKS = ("leaderboard", "sequential", "indexes")

#: Stands for every model of the ``models_from`` config, in its order.
WILDCARD = "*"


class ConfigError(ValueError):
    """A config file that does not describe a runnable benchmark."""


def _check_keys(
    where: str, found: Mapping[str, Any], required: set[str], optional: set[str]
) -> None:
    missing = required - set(found)
    if missing:
        raise ConfigError(f"{where}: missing {sorted(missing)}.")
    unknown = set(found) - required - optional
    if unknown:
        raise ConfigError(
            f"{where}: unknown {sorted(unknown)}; expected some of {sorted(required | optional)}."
        )


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path} does not exist.") from None
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a JSON object.")
    if data.get("schema") != SCHEMA:
        raise ConfigError(f"{path}: schema must be {SCHEMA}, got {data.get('schema')!r}.")
    return data


def canonical(value: Any) -> str:
    """One spelling of ``value``, so equal configs hash equal whatever their key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def key(inputs: Mapping[str, Any]) -> str:
    """The cache key of a result: a short hash of everything that determined it."""
    return hashlib.sha256(canonical(inputs).encode()).hexdigest()[:16]


def resolve(dotted: str) -> Any:
    """The object ``package.name`` names, importing ``package`` to find it."""
    package, _, name = dotted.rpartition(".")
    if not package:
        raise ConfigError(f"{dotted!r} is not a dotted path.")
    return getattr(importlib.import_module(package), name)


def build_object(package: str, cls: str, params: Mapping[str, Any]) -> Any:
    """Construct ``package.cls(**params)``, building nested object specs first.

    A parameter whose value is itself ``{"package": ..., "cls": ..., "params": ...}`` is
    constructed the same way, so an estimator can take an index -- or anything else --
    as a parameter without the config format knowing what either of them is.
    """
    built = {name: _build_value(value) for name, value in params.items()}
    return getattr(importlib.import_module(package), cls)(**built)


def _build_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "cls" in value:
        _check_keys("nested object", value, {"package", "cls"}, {"params"})
        return build_object(value["package"], value["cls"], value.get("params", {}))
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
    params: Mapping[str, Any]
    version: str
    comment: str = ""

    def spec(self) -> dict[str, Any]:
        """The part of the entry that determines a measurement, for the cache key."""
        return {
            "package": self.package,
            "cls": self.cls,
            "params": copy.deepcopy(dict(self.params)),
            "version": self.version,
        }

    def build(self) -> Any:
        """A fresh, unfitted instance."""
        return build_object(self.package, self.cls, self.params)


@dataclass(frozen=True, kw_only=True)
class IndexEntry(Entry):
    """An index: an entry with ``label`` and ``short`` for the tables and a sweep dial."""

    label: str
    short: str
    dial: Dial

    @override
    def spec(self) -> dict[str, Any]:
        return super().spec() | {
            "dial": {"param": self.dial.param, "values": list(self.dial.values)}
        }

    def with_values(self, values: Sequence[int]) -> IndexEntry:
        """The same index swept over other values, as a dataset may ask for."""
        return replace(self, dial=replace(self.dial, values=tuple(values)))


_ENTRY_REQUIRED = {"name", "package", "cls", "params", "version"}


def _entry(where: str, raw: Any, extra: frozenset[str] = frozenset()) -> Entry:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: an entry must be an object, got {raw!r}.")
    _check_keys(where, raw, _ENTRY_REQUIRED | extra, {"comment"})
    where = f"{where} {raw['name']!r}"
    if not isinstance(raw["params"], Mapping):
        raise ConfigError(f"{where}: params must be an object.")
    if not isinstance(raw["version"], str) or not raw["version"]:
        raise ConfigError(f"{where}: version must be a non-empty string.")
    return Entry(
        name=raw["name"],
        package=raw["package"],
        cls=raw["cls"],
        params=raw["params"],
        version=raw["version"],
        comment=raw.get("comment", ""),
    )


def _index_entry(where: str, raw: Any) -> IndexEntry:
    entry = _entry(where, raw, frozenset({"label", "short", "dial"}))
    where = f"{where} {raw['name']!r}"
    raw_dial = raw["dial"]
    _check_keys(f"{where} dial", raw_dial, {"param", "label", "values"}, set())
    values = raw_dial["values"]
    if not values or not all(isinstance(v, int) and v >= 1 for v in values):
        raise ConfigError(f"{where}: dial values must be a non-empty list of integers >= 1.")
    return IndexEntry(
        **{field.name: getattr(entry, field.name) for field in fields(entry)},
        label=raw["label"],
        short=raw["short"],
        dial=Dial(raw_dial["param"], raw_dial["label"], tuple(values)),
    )


@dataclass(frozen=True)
class DatasetDef:
    """A dataset as ``datasets.json`` defines it: how to load it and what to call it."""

    name: str
    loader: str
    subset: str
    caption: str
    load_kwargs: Mapping[str, Any] = field(default_factory=dict)
    max_eval_users: int | None = None

    def spec(self) -> dict[str, Any]:
        """What determines the data a result was measured on. The caption does not."""
        return {
            "loader": self.loader,
            "subset": self.subset,
            "load": dict(self.load_kwargs),
            "max_eval_users": self.max_eval_users,
        }

    def title(self) -> str:
        """The caption with its placeholders filled in."""
        return self.caption.format(subset=self.subset, **self.load_kwargs)

    def load(self) -> Any:
        """The split, as a Bunch carrying ``train_indices`` and ``test_indices``."""
        return resolve(self.loader)(subset=self.subset, **self.load_kwargs)


def load_datasets() -> dict[str, DatasetDef]:
    """Every dataset ``datasets.json`` defines."""
    path = CONFIG_DIR / "datasets.json"
    data = _read(path)
    _check_keys(str(path), data, {"schema", "datasets"}, set())
    datasets: dict[str, DatasetDef] = {}
    for name, raw in data["datasets"].items():
        where = f"{path} dataset {name!r}"
        _check_keys(where, raw, {"loader", "subset", "caption"}, {"load", "max_eval_users"})
        datasets[name] = DatasetDef(
            name=name,
            loader=raw["loader"],
            subset=raw["subset"],
            caption=raw["caption"],
            load_kwargs=raw.get("load", {}),
            max_eval_users=raw.get("max_eval_users"),
        )
    return datasets


@dataclass(frozen=True)
class Target:
    """One dataset as one benchmark runs it: the data, the settings and the exceptions."""

    definition: DatasetDef
    settings: Mapping[str, Any]
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


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def check_settings(benchmark: str, where: str, settings: Mapping[str, Any]) -> None:
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
    if "cutoffs" in settings:
        cutoffs = settings["cutoffs"]
        if not cutoffs or not all(_positive_int(c) for c in cutoffs):
            raise ConfigError(f"{where}: cutoffs must be a non-empty list of integers >= 1.")
    if "latency_batch" in settings and not all(_positive_int(b) for b in settings["latency_batch"]):
        raise ConfigError(f"{where}: latency_batch must list integers >= 1.")
    if "catalog_scale" in settings and not all(
        isinstance(share, int | float) and 0 < share <= 1 for share in settings["catalog_scale"]
    ):
        raise ConfigError(f"{where}: catalog_scale must list shares in (0, 1].")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``override`` laid over ``base``, one level of objects deep (``warmup`` and so on)."""
    merged = copy.deepcopy(dict(base))
    for name, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(name), Mapping):
            merged[name] = {**merged[name], **value}
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
        for position, raw in enumerate(data.get("indexes", []))
    )
    _check_unique(path, "index", [entry.name for entry in indexes])
    targets = _targets(path, benchmark, data, models, indexes)
    return Config(benchmark, models, indexes, targets)


def _models(path: Path, benchmark: str, data: Mapping[str, Any]) -> tuple[Entry, ...]:
    """The config's models, with names and ``*`` resolved against ``models_from``."""
    inherited: tuple[Entry, ...] = ()
    if "models_from" in data:
        source = data["models_from"]
        if source == benchmark:
            raise ConfigError(f"{path}: models_from cannot name the config itself.")
        inherited = load(source).models

    models: list[Entry] = []
    for position, raw in enumerate(data["models"]):
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
    data: Mapping[str, Any],
    models: Sequence[Entry],
    indexes: Sequence[IndexEntry],
) -> dict[str, Target]:
    """The config's datasets, each with its settings merged over the config's own."""
    definitions = load_datasets()
    model_names = {entry.name for entry in models}
    index_names = {entry.name for entry in indexes}
    targets: dict[str, Target] = {}
    for name, raw in data["datasets"].items():
        where = f"{path} dataset {name!r}"
        if name not in definitions:
            raise ConfigError(f"{where}: not defined in datasets.json.")
        _check_keys(where, raw, set(), {"settings", "skip", "dials"})
        skip = raw.get("skip", {})
        unknown = [model for model in skip if model not in model_names]
        if unknown:
            raise ConfigError(f"{where}: skip names {unknown}, which are not models here.")
        dials = raw.get("dials", {})
        unknown = [index for index in dials if index not in index_names]
        if unknown:
            raise ConfigError(f"{where}: dials name {unknown}, which are not indexes here.")
        settings = _merge(data["settings"], raw.get("settings", {}))
        check_settings(benchmark, where, settings)
        targets[name] = Target(
            definition=definitions[name],
            settings=settings,
            skip=dict(skip),
            dials={index: tuple(values) for index, values in dials.items()},
        )
    return targets


def _check_unique(path: Path, what: str, names: Sequence[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ConfigError(f"{path}: {what} {name!r} is listed twice.")
        seen.add(name)

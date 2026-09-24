import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import spec
import store

from skrecsys.datasets import fetch_movielens_100k


@pytest.fixture(scope="session")
def movielens_100k_ua():
    """MovieLens 100K with the official ``ua`` split: 10 held-out ratings per user."""
    return fetch_movielens_100k(subset="ua")


def synthetic_dataset():
    """40 users over 12 items: each user likes one contiguous band of the catalog.

    The last two interactions of every user are held out. Small enough to fit every
    model a sandbox runs in well under a second, and structured enough that a
    neighbourhood model and a popularity baseline still come out differently.
    """
    rows, targets = [], []
    for user in range(40):
        start = user % 6
        for item in range(start, start + 6):
            rows.append([f"u{user}", f"i{item}"])
            targets.append(5.0)
    data = np.array(rows)
    target = np.array(targets)
    indices = np.arange(len(data))
    return SimpleNamespace(
        data=data,
        target=target,
        train_indices=indices[(indices % 6) < 4],
        test_indices=indices[(indices % 6) >= 4],
    )


@pytest.fixture
def dataset():
    return synthetic_dataset()


#: The smallest configs that exercise every benchmark: two cheap models, an HNSW index,
#: and timings short enough that measuring everything takes a second or two.
SANDBOX_CONFIGS = {
    "datasets": {
        "schema": 1,
        "datasets": {
            "toy": {"loader": "sandbox.load", "subset": "bands", "caption": "Toy `{subset}` split"}
        },
    },
    "leaderboard": {
        "schema": 1,
        "settings": {
            "k": 3,
            "repeat": 1,
            "rank_repeat": 1,
            "budget": 0.001,
            "warmup": {"fit": 0, "rank": 0},
        },
        "datasets": {"toy": {}},
        "models": [
            {
                "name": "MostPopular",
                "package": "skrecsys.recommendation",
                "cls": "MostPopularRecommender",
                "params": {},
                "version": "v1",
            },
            {
                "name": "ItemKNN",
                "package": "skrecsys.recommendation",
                "cls": "ItemKNNRecommender",
                "params": {"n_neighbors": 5},
                "version": "v1",
                "comment": "a comment is prose and keys nothing",
            },
        ],
    },
    "sequential": {
        "schema": 1,
        "models_from": "leaderboard",
        "settings": {
            "cutoffs": [1, 3],
            "repeat": 1,
            "rank_repeat": 1,
            "budget": 0.001,
            "warmup": {"fit": 0, "rank": 0},
        },
        "datasets": {"toy": {}},
        "models": ["*"],
    },
    "indexes": {
        "schema": 1,
        "models_from": "leaderboard",
        "settings": {
            "k": 3,
            "repeat": 1,
            "rank_repeat": 1,
            "budget": 0.001,
            "warmup": {"fit": 0, "rank": 0},
            "latency_batch": [1],
            "latency_repeat": 1,
            "catalog_scale": [1.0],
        },
        "datasets": {"toy": {}},
        "models": ["ItemKNN"],
        "indexes": [
            {
                "name": "hnsw",
                "label": "hnsw(m=4)",
                "short": "hnsw",
                "package": "skrecsys.indexing",
                "cls": "HNSW",
                "params": {"m": 4, "ef_construction": 8, "min_index_size": 1},
                "dial": {"param": "ef_search", "label": "ef", "values": [2, 4, 8]},
                "version": "v1",
            }
        ],
    },
}


class Sandbox:
    """Configs and results in a temporary directory, and a dataset that needs no download."""

    def __init__(self, root: Path) -> None:
        self.config_dir = root / "config"
        self.results_dir = root / "results"
        self.config_dir.mkdir()
        for name, config in SANDBOX_CONFIGS.items():
            self.write(name, config)

    def read(self, name: str) -> dict[str, Any]:
        return json.loads((self.config_dir / f"{name}.json").read_text(encoding="utf-8"))

    def write(self, name: str, config: dict[str, Any]) -> None:
        (self.config_dir / f"{name}.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    def edit(self, name: str, change) -> None:
        """Apply ``change`` to a config file in place, the way a person would."""
        config = self.read(name)
        change(config)
        self.write(name, config)


def stored(unit: store.Unit) -> dict[str, Any]:
    """The stored result of ``unit``, which a test has just arranged to exist."""
    result = store.read(unit)
    assert result is not None, f"{unit.label} has no stored result"
    return result


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    box = Sandbox(tmp_path)
    monkeypatch.setattr(spec, "CONFIG_DIR", box.config_dir)
    monkeypatch.setattr(store, "RESULTS_DIR", box.results_dir)
    monkeypatch.setattr(spec.DatasetDef, "load", lambda self: synthetic_dataset())
    return box

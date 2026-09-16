import shutil
import zipfile
from pathlib import Path

import pytest

from skrecsys.datasets import _movielens

GENRES = ["unknown", "Action", "Comedy"]

# user, item, rating, timestamp
RATINGS = [
    (1, 10, 5, 300),
    (1, 20, 3, 100),
    (1, 30, 4, 100),
    (2, 10, 1, 200),
    (2, 30, 2, 250),
    (3, 20, 4, 50),
]
USERS = ["1|24|M|technician|85711", "2|53|F|other|T8H1N", "3|23|M|writer|32067"]
ITEMS = [
    "10|Toy Story (1995)|01-Jan-1995||http://imdb/10|0|0|1",
    "20|Amélie (2001)|25-Apr-2001||http://imdb/20|0|1|1",
    "30|unknown||||1|0|0",
]


def _lines(rows):
    return "".join("\t".join(map(str, row)) + "\n" for row in rows)


def make_archive(path):
    """Write a tiny archive in the MovieLens 100K layout."""
    folds = {
        "u1": (RATINGS[:4], RATINGS[4:]),
        "ua": (RATINGS[1:], RATINGS[:1]),
    }
    for name in ("u2", "u3", "u4", "u5", "ub"):
        folds[name] = folds["u1"]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml-100k/u.data", _lines(RATINGS))
        archive.writestr("ml-100k/u.user", "\n".join(USERS) + "\n")
        archive.writestr("ml-100k/u.item", ("\n".join(ITEMS) + "\n").encode("latin-1"))
        archive.writestr("ml-100k/u.genre", "".join(f"{g}|{i}\n" for i, g in enumerate(GENRES)))
        for name, (base, test) in folds.items():
            archive.writestr(f"ml-100k/{name}.base", _lines(base))
            archive.writestr(f"ml-100k/{name}.test", _lines(test))
    return path


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    """Replace the network download with a copy of a tiny local archive.

    Returns the list of download calls.
    """
    source = make_archive(tmp_path / "source.zip")
    calls = []

    def fetch_remote(remote, dirname, n_retries=3, delay=1):
        calls.append(remote)
        target = Path(dirname) / remote.filename
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr(_movielens, "_fetch_remote", fetch_remote)
    return calls


@pytest.fixture
def data_home(tmp_path):
    return tmp_path / "data_home"

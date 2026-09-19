import shutil
import zipfile
from pathlib import Path

import pytest

from skrecsys.datasets import _amazon, _movielens

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


# Amazon Books: reviewer_id,asin,rating,timestamp.
#
# Users ``u1``-``u5`` rate items ``i1``-``i5``, so both clear the 5-interaction floor
# and survive. Users ``d1``-``d5`` rate ``x1``-``x4``, which also clear the floor, plus
# one item of their own that does not: those rows go, which leaves each ``d`` user with
# four interactions and drops them too. ``x1``-``x4`` are therefore encoded, as the
# reference preprocessing encodes before it drops short users, but no interaction of
# theirs remains.
AMAZON_USERS = [f"u{i}" for i in range(1, 6)]
AMAZON_ITEMS = [f"i{i}" for i in range(1, 6)]
AMAZON_DROPPED_USERS = [f"d{i}" for i in range(1, 6)]
AMAZON_ORPHAN_ITEMS = [f"x{i}" for i in range(1, 5)]
#: Every asin that reaches the encoding, in the sorted order it is encoded in.
AMAZON_ENCODED_ITEMS = sorted(AMAZON_ITEMS + AMAZON_ORPHAN_ITEMS)
AMAZON_RATINGS = (
    [
        # later users rate earlier, and each user rates i1..i5 in reverse time order
        (user, item, float(u_index + 1), 1000 - 10 * u_index - i_index)
        for u_index, user in enumerate(AMAZON_USERS)
        for i_index, item in enumerate(AMAZON_ITEMS)
    ]
    + [
        (user, item, 3.0, 500 + i_index)
        for user in AMAZON_DROPPED_USERS
        for i_index, item in enumerate(AMAZON_ORPHAN_ITEMS)
    ]
    + [(user, f"rare-{user}", 2.0, 600) for user in AMAZON_DROPPED_USERS]
)


@pytest.fixture
def fake_amazon_download(tmp_path, monkeypatch):
    """Replace the network download with a copy of a tiny local ratings CSV.

    Returns the list of download calls.
    """
    source = tmp_path / "ratings.csv"
    source.write_text("".join(f"{u},{i},{r},{t}\n" for u, i, r, t in AMAZON_RATINGS))
    calls = []

    def fetch_remote(remote, dirname, n_retries=3, delay=1):
        calls.append(remote)
        target = Path(dirname) / remote.filename
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr(_amazon, "_fetch_remote", fetch_remote)
    return calls


# MovieLens 1M: user::item::rating::timestamp, and two metadata tables beside it.
# Three users rate the same three movies, latest first in the file, so the loader has
# something to reorder; `m3` is in the movie table without ever being rated.
ML1M_USERS = ["1::F::25::4::02138", "2::M::35::7::94043", "3::M::18::0::55455"]
ML1M_MOVIES = [
    "1::Toy Story (1995)::Animation|Children's|Comedy",
    "2::Amélie (2001)::Comedy|Romance",
    "3::Untitled::Drama",
]
ML1M_RATINGS = [
    (user, item, 5 - item, 1000 - 10 * user - item) for user in (1, 2, 3) for item in (1, 2)
]


@pytest.fixture
def fake_ml1m_download(tmp_path, monkeypatch):
    """Replace the network download with a copy of a tiny local ml-1m archive.

    Returns the list of download calls.
    """
    source = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(source, "w") as archive:
        rows = "".join(f"{u}::{i}::{r}::{t}\n" for u, i, r, t in ML1M_RATINGS)
        archive.writestr("ml-1m/ratings.dat", rows)
        archive.writestr("ml-1m/users.dat", "\n".join(ML1M_USERS) + "\n")
        archive.writestr("ml-1m/movies.dat", ("\n".join(ML1M_MOVIES) + "\n").encode("latin-1"))
    calls = []

    def fetch_remote(remote, dirname, n_retries=3, delay=1):
        calls.append(remote)
        target = Path(dirname) / remote.filename
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr(_movielens, "_fetch_remote", fetch_remote)
    return calls

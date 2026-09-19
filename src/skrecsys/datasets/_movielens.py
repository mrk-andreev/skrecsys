"""MovieLens ratings datasets.

https://grouplens.org/datasets/movielens/
"""

import csv
import io
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from sklearn.utils import Bunch

from skrecsys.datasets._base import (
    get_data_home,
    import_pandas,
    leave_one_out,
    load_descr,
    tail_rows,
)

__all__ = ["fetch_movielens_1m", "fetch_movielens_100k"]

ARCHIVE = RemoteFileMetadata(
    filename="ml-100k.zip",
    url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    checksum="50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229",
)
ARCHIVE_1M = RemoteFileMetadata(
    filename="ml-1m.zip",
    url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    checksum="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
)
_CACHE_NAME = "movielens_100k.joblib"
_CACHE_NAME_1M = "movielens_1m.joblib"
_ARCHIVE_ROOT_1M = "ml-1m"
_SEQUENTIAL_SUBSETS = ("all", "leave-one-out")
#: ``ratings.dat``, ``users.dat`` and ``movies.dat`` are all ``::``-separated.
_SEPARATOR_1M = "::"
_N_RATING_FIELDS_1M = 4  # user_id, item_id, rating, timestamp
_N_MOVIE_FIELDS_1M = 3  # item_id, title, genres
_ARCHIVE_ROOT = "ml-100k"
_SUBSETS = ("all", "u1", "u2", "u3", "u4", "u5", "ua", "ub")
_MONTHS = {
    name: number
    for number, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
    )
}  # parsed explicitly since strptime's %b depends on the locale
_N_ITEM_FIELDS = 5  # item_id, title, release_date, video_release_date, imdb_url

Subset: TypeAlias = Literal["all", "u1", "u2", "u3", "u4", "u5", "ua", "ub"]


@overload
def fetch_movielens_100k(
    *,
    data_home: str | os.PathLike[str] | None = ...,
    subset: Subset = ...,
    download_if_missing: bool = ...,
    return_X_y: Literal[False] = ...,
    as_frame: bool = ...,
    n_retries: int = ...,
    delay: float = ...,
) -> Bunch: ...


@overload
def fetch_movielens_100k(
    *,
    data_home: str | os.PathLike[str] | None = ...,
    subset: Literal["all"] = ...,
    download_if_missing: bool = ...,
    return_X_y: Literal[True],
    as_frame: bool = ...,
    n_retries: int = ...,
    delay: float = ...,
) -> tuple[Any, Any]: ...


def fetch_movielens_100k(
    *,
    data_home: str | os.PathLike[str] | None = None,
    subset: Subset = "all",
    download_if_missing: bool = True,
    return_X_y: bool = False,
    as_frame: bool = False,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch | tuple[Any, Any]:
    """Load the MovieLens 100K ratings dataset, downloading it if necessary.

    =================   ==============
    Ratings             100,000
    Users               943
    Items               1,682
    Rating scale        1-5 (integers)
    =================   ==============

    Read more in ``DESCR``.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Download and cache folder. By default data is stored in ``~/skrecsys_data``
        (or ``SKRECSYS_DATA``); see :func:`skrecsys.datasets.get_data_home`.

    subset : {"all", "u1", "u2", "u3", "u4", "u5", "ua", "ub"}, default="all"
        Official split to expose. Every subset returns all ratings; a subset other
        than ``"all"`` adds ``train_indices`` and ``test_indices`` into ``data``.
        ``u1``-``u5`` are the folds of a 5-fold cross-validation; ``ua`` and ``ub``
        hold out exactly 10 ratings per user.

    download_if_missing : bool, default=True
        If False, raise an ``OSError`` when the data is not cached locally instead of
        downloading it.

    return_X_y : bool, default=False
        If True, return ``(data, target)`` instead of a Bunch. Not allowed with a
        subset other than ``"all"``, since the split indices would be lost.

    as_frame : bool, default=False
        If True, ``data``, ``user_info`` and ``item_info`` are pandas DataFrames, ``target``
        and ``timestamps`` are Series, and ``frame`` combines all rating columns.
        Requires pandas.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        data : ndarray of int64 of shape (100000, 2)
            ``[user_id, item_id]`` pairs, sorted by user, timestamp and item.
        target : ndarray of float64 of shape (100000,)
            Ratings.
        timestamps : ndarray of int64 of shape (100000,)
            Rating times in Unix seconds.
        user_info : Bunch or DataFrame
            ``user_id``, ``age``, ``gender``, ``occupation``, ``zip_code`` of 943 users.
        item_info : Bunch or DataFrame
            ``item_id``, ``title``, ``release_date`` (datetime64[D], NaT if unknown),
            ``imdb_url`` and ``genres`` (bool array of shape (1682, 19), one column per
            genre in ``genre_names``) of 1682 movies. As a DataFrame, genres are bool
            columns named after the genres.
        genre_names : list of str
        feature_names : list of str
            ``["user_id", "item_id"]``.
        target_names : list of str
            ``["rating"]``.
        frame : DataFrame
            Only when ``as_frame=True``: ``user_id``, ``item_id``, ``rating``, ``timestamp``.
        train_indices, test_indices : ndarray of intp
            Only for a subset other than ``"all"``: row positions in ``data``. Use
            ``cv=[(train_indices, test_indices)]`` in scikit-learn model selection tools.
        DESCR : str
            Description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    Examples
    --------
    >>> from skrecsys.datasets import fetch_movielens_100k
    >>> X, y = fetch_movielens_100k(return_X_y=True)  # doctest: +SKIP
    >>> X.shape  # doctest: +SKIP
    (100000, 2)
    """
    if subset not in _SUBSETS:
        raise ValueError(f"subset must be one of {_SUBSETS}, got {subset!r}.")
    if return_X_y and subset != "all":
        raise ValueError(
            "return_X_y=True is only supported with subset='all'; use the returned Bunch "
            "to access train_indices and test_indices."
        )
    pd = import_pandas() if as_frame else None

    data_home = get_data_home(data_home)
    cache_path = data_home / _CACHE_NAME
    if not cache_path.exists():
        if not download_if_missing:
            raise OSError(f"Data not found in {data_home} and download_if_missing is False.")
        archive_path = _fetch_remote(ARCHIVE, dirname=data_home, n_retries=n_retries, delay=delay)
        parsed = _parse_archive(Path(archive_path))
        joblib.dump(parsed, cache_path, compress=6)
        Path(archive_path).unlink()
    cached = joblib.load(cache_path)

    ratings = cached["ratings"]
    data = np.column_stack([ratings["user_id"], ratings["item_id"]])
    target = ratings["rating"].astype(np.float64)
    timestamps = ratings["timestamp"]
    users = Bunch(**cached["users"])
    items = Bunch(**cached["items"])
    genre_names = list(cached["genre_names"])
    feature_names = ["user_id", "item_id"]

    bunch = Bunch(
        genre_names=genre_names,
        feature_names=feature_names,
        target_names=["rating"],
        DESCR=load_descr("movielens_100k.rst"),
    )
    if pd is not None:
        data = pd.DataFrame(data, columns=feature_names)
        target = pd.Series(target, name="rating")
        timestamps = pd.Series(timestamps, name="timestamp")
        genres = items.pop("genres")
        items = pd.concat(
            [pd.DataFrame(dict(items)), pd.DataFrame(genres, columns=genre_names)], axis=1
        )
        users = pd.DataFrame(dict(users))
        bunch.frame = pd.concat([data, target, timestamps], axis=1)
    bunch.update(data=data, target=target, timestamps=timestamps, user_info=users, item_info=items)

    if subset != "all":
        bunch.train_indices, bunch.test_indices = cached["splits"][subset]

    if return_X_y:
        return data, target
    return bunch


def _parse_archive(path: Path) -> dict[str, Any]:
    """Parse the MovieLens 100K zip archive into NumPy arrays."""
    with zipfile.ZipFile(path) as archive:
        raw = _load_ratings(archive, "u.data")
        order = np.lexsort((raw[:, 1], raw[:, 3], raw[:, 0]))
        raw = raw[order]
        ratings = {
            "user_id": raw[:, 0],
            "item_id": raw[:, 1],
            "rating": raw[:, 2],
            "timestamp": raw[:, 3],
        }
        keys = _pair_keys(raw[:, 0], raw[:, 1])
        if len(np.unique(keys)) != len(keys):
            raise ValueError("MovieLens 100K contains duplicate user-item pairs.")
        key_order = np.argsort(keys)
        splits = {
            name: tuple(
                _locate(keys, key_order, _load_ratings(archive, f"{name}.{part}"), name)
                for part in ("base", "test")
            )
            for name in _SUBSETS[1:]
        }
        genre_names = [row[0] for row in _read_rows(archive, "u.genre") if row]
        users = _parse_users(_read_rows(archive, "u.user"))
        items = _parse_items(_read_rows(archive, "u.item"), n_genres=len(genre_names))
    return {
        "ratings": ratings,
        "users": users,
        "items": items,
        "genre_names": genre_names,
        "splits": splits,
    }


def _open(archive: zipfile.ZipFile, name: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(archive.open(f"{_ARCHIVE_ROOT}/{name}"), encoding="latin-1")


def _read_rows(archive: zipfile.ZipFile, name: str) -> Iterator[list[str]]:
    with _open(archive, name) as f:
        yield from csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)


def _load_ratings(archive: zipfile.ZipFile, name: str) -> NDArray[np.int64]:
    with _open(archive, name) as f:
        return np.loadtxt(f, dtype=np.int64, delimiter="\t", ndmin=2)


def _pair_keys(users: NDArray[np.int64], items: NDArray[np.int64]) -> NDArray[np.int64]:
    return users * (1 << 32) + items


def _locate(
    keys: NDArray[np.int64], key_order: NDArray[np.intp], rows: NDArray[np.int64], name: str
) -> NDArray[np.intp]:
    """Return positions in ``keys`` of the user-item pairs in ``rows``."""
    wanted = _pair_keys(rows[:, 0], rows[:, 1])
    sorted_keys = keys[key_order]
    pos = np.minimum(np.searchsorted(sorted_keys, wanted), len(keys) - 1)
    if np.any(sorted_keys[pos] != wanted):
        raise ValueError(f"Split {name!r} contains ratings missing from u.data.")
    return np.sort(key_order[pos]).astype(np.intp)


def _parse_users(rows: Iterator[list[str]]) -> dict[str, NDArray[Any]]:
    user_id, age, gender, occupation, zip_code = zip(*(row for row in rows if row), strict=True)
    return {
        "user_id": np.array(user_id, dtype=np.int64),
        "age": np.array(age, dtype=np.int64),
        "gender": np.array(gender),
        "occupation": np.array(occupation),
        "zip_code": np.array(zip_code),
    }


def _parse_items(rows: Iterator[list[str]], *, n_genres: int) -> dict[str, NDArray[Any]]:
    records = [row for row in rows if row]
    for row in records:
        if len(row) != _N_ITEM_FIELDS + n_genres:
            raise ValueError(f"Malformed u.item row: {'|'.join(row)!r}.")
    return {
        "item_id": np.array([row[0] for row in records], dtype=np.int64),
        "title": np.array([row[1] for row in records]),
        "release_date": np.array([_parse_date(row[2]) for row in records], dtype="datetime64[D]"),
        "imdb_url": np.array([row[4] for row in records]),
        "genres": np.array([row[_N_ITEM_FIELDS:] for row in records], dtype=np.int8).astype(bool),
    }


def _parse_date(value: str) -> np.datetime64:
    if not value:
        return np.datetime64("NaT", "D")
    day, month, year = value.split("-")
    return np.datetime64(f"{year}-{_MONTHS[month]:02d}-{int(day):02d}", "D")


Subset1M: TypeAlias = Literal["all", "leave-one-out"]


@overload
def fetch_movielens_1m(
    *,
    data_home: str | os.PathLike[str] | None = ...,
    subset: Subset1M = ...,
    max_sequence_length: int | None = ...,
    download_if_missing: bool = ...,
    return_X_y: Literal[False] = ...,
    as_frame: bool = ...,
    n_retries: int = ...,
    delay: float = ...,
) -> Bunch: ...


@overload
def fetch_movielens_1m(
    *,
    data_home: str | os.PathLike[str] | None = ...,
    subset: Literal["all"] = ...,
    max_sequence_length: int | None = ...,
    download_if_missing: bool = ...,
    return_X_y: Literal[True],
    as_frame: bool = ...,
    n_retries: int = ...,
    delay: float = ...,
) -> tuple[Any, Any]: ...


def fetch_movielens_1m(
    *,
    data_home: str | os.PathLike[str] | None = None,
    subset: Subset1M = "all",
    max_sequence_length: int | None = None,
    download_if_missing: bool = True,
    return_X_y: bool = False,
    as_frame: bool = False,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch | tuple[Any, Any]:
    """Load the MovieLens 1M ratings dataset, downloading it if necessary.

    =================   ==============
    Ratings             1,000,209
    Users               6,040
    Items rated         3,706
    Item identifiers    3,952
    Rating scale        1-5 (integers)
    =================   ==============

    With ``subset="leave-one-out"`` and ``max_sequence_length=200`` this is the
    ``ml-1m-l200`` dataset the sequential-recommendation literature reports on, which is
    what :class:`skrecsys.nn.HSTU` is benchmarked against. Unlike the Amazon loader
    nothing is filtered or renumbered: every user in this dataset already has at least
    20 ratings, and the identifiers are the ones MovieLens ships.

    Read more in ``DESCR``.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Download and cache folder. By default data is stored in ``~/skrecsys_data``
        (or ``SKRECSYS_DATA``); see :func:`skrecsys.datasets.get_data_home`.

    subset : {"all", "leave-one-out"}, default="all"
        Split to expose. Both return every interaction; ``"leave-one-out"`` adds
        ``train_indices`` and ``test_indices`` into ``data``, holding out the last
        rating of each user.

    max_sequence_length : int or None, default=None
        Keep only the last this many ratings of each user, plus the held-out one under
        ``subset="leave-one-out"``. None, the default, keeps all 1,000,209 ratings; the
        sequential benchmark passes 200.

    download_if_missing : bool, default=True
        If False, raise an ``OSError`` when the data is not cached locally instead of
        downloading it.

    return_X_y : bool, default=False
        If True, return ``(data, target)`` instead of a Bunch. Not allowed with
        ``subset="leave-one-out"``, since the split indices would be lost.

    as_frame : bool, default=False
        If True, ``data``, ``user_info`` and ``item_info`` are pandas DataFrames,
        ``target`` and ``timestamps`` are Series, and ``frame`` combines all rating
        columns. Requires pandas.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        data : ndarray of int64 of shape (n_ratings, 2)
            ``[user_id, item_id]`` pairs, sorted by user, timestamp and item.
        target : ndarray of float64 of shape (n_ratings,)
            Ratings.
        timestamps : ndarray of int64 of shape (n_ratings,)
            Rating times in Unix seconds.
        user_info : Bunch or DataFrame
            ``user_id``, ``gender``, ``age``, ``occupation``, ``zip_code`` of 6040 users.
            ``age`` and ``occupation`` are the coded values MovieLens ships, described
            in ``DESCR``.
        item_info : Bunch or DataFrame
            ``item_id``, ``title``, ``year`` (0 when the title carries none) and
            ``genres`` (bool array, one column per genre in ``genre_names``) of the 3883
            movies, rated or not. As a DataFrame, genres are bool columns.
        genre_names : list of str
        feature_names : list of str
            ``["user_id", "item_id"]``.
        target_names : list of str
            ``["rating"]``.
        frame : DataFrame
            Only when ``as_frame=True``: ``user_id``, ``item_id``, ``rating``, ``timestamp``.
        train_indices, test_indices : ndarray of intp
            Only for ``subset="leave-one-out"``: row positions in ``data``. Use
            ``cv=[(train_indices, test_indices)]`` in scikit-learn model selection tools.
        DESCR : str
            Description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    Examples
    --------
    >>> from skrecsys.datasets import fetch_movielens_1m
    >>> X, y = fetch_movielens_1m(return_X_y=True)  # doctest: +SKIP
    >>> X.shape  # doctest: +SKIP
    (1000209, 2)
    """
    if subset not in _SEQUENTIAL_SUBSETS:
        raise ValueError(f"subset must be one of {_SEQUENTIAL_SUBSETS}, got {subset!r}.")
    if return_X_y and subset != "all":
        raise ValueError(
            "return_X_y=True is only supported with subset='all'; use the returned Bunch "
            "to access train_indices and test_indices."
        )
    if max_sequence_length is not None and max_sequence_length < 1:
        raise ValueError(f"max_sequence_length must be >= 1 or None, got {max_sequence_length!r}.")
    pd = import_pandas() if as_frame else None

    home = get_data_home(data_home)
    cache_path = home / _CACHE_NAME_1M
    if not cache_path.exists():
        if not download_if_missing:
            raise OSError(f"Data not found in {home} and download_if_missing is False.")
        archive_path = _fetch_remote(ARCHIVE_1M, dirname=home, n_retries=n_retries, delay=delay)
        joblib.dump(_parse_archive_1m(Path(archive_path)), cache_path, compress=6)
        Path(archive_path).unlink()
    cached = joblib.load(cache_path)

    ratings = cached["ratings"]
    held_out = subset == "leave-one-out"
    rows = tail_rows(ratings["user_id"], max_sequence_length, held_out=held_out)

    data = np.column_stack([ratings["user_id"][rows], ratings["item_id"][rows]])
    target = ratings["rating"][rows].astype(np.float64)
    timestamps = ratings["timestamp"][rows]
    users = Bunch(**cached["users"])
    items = Bunch(**cached["items"])
    genre_names = list(cached["genre_names"])
    feature_names = ["user_id", "item_id"]

    bunch = Bunch(
        genre_names=genre_names,
        feature_names=feature_names,
        target_names=["rating"],
        DESCR=load_descr("movielens_1m.rst"),
    )
    if pd is not None:
        data = pd.DataFrame(data, columns=feature_names)
        target = pd.Series(target, name="rating")
        timestamps = pd.Series(timestamps, name="timestamp")
        genres = items.pop("genres")
        items = pd.concat(
            [pd.DataFrame(dict(items)), pd.DataFrame(genres, columns=genre_names)], axis=1
        )
        users = pd.DataFrame(dict(users))
        bunch.frame = pd.concat([data, target, timestamps], axis=1)
    bunch.update(data=data, target=target, timestamps=timestamps, user_info=users, item_info=items)

    if held_out:
        bunch.train_indices, bunch.test_indices = leave_one_out(ratings["user_id"][rows])

    if return_X_y:
        return data, target
    return bunch


def _parse_archive_1m(path: Path) -> dict[str, Any]:
    """Parse the MovieLens 1M zip archive into NumPy arrays."""
    with zipfile.ZipFile(path) as archive:
        raw = _read_dat(archive, "ratings.dat")
        values = np.array([row.split(_SEPARATOR_1M) for row in raw], dtype=np.int64)
        if values.shape[1] != _N_RATING_FIELDS_1M:
            raise ValueError("Malformed ratings.dat row: expected user::item::rating::timestamp.")
        order = np.lexsort((values[:, 1], values[:, 3], values[:, 0]))
        values = values[order]
        ratings = {
            "user_id": values[:, 0],
            "item_id": values[:, 1],
            "rating": values[:, 2],
            "timestamp": values[:, 3],
        }
        users = _parse_users_1m(_read_dat(archive, "users.dat"))
        items, genre_names = _parse_items_1m(_read_dat(archive, "movies.dat"))
    return {"ratings": ratings, "users": users, "items": items, "genre_names": genre_names}


def _read_dat(archive: zipfile.ZipFile, name: str) -> list[str]:
    """Read one ``::``-separated member, dropping the blank line at its end."""
    with archive.open(f"{_ARCHIVE_ROOT_1M}/{name}") as handle:
        text = handle.read().decode("latin-1")
    return [line for line in text.split("\n") if line]


def _parse_users_1m(rows: list[str]) -> dict[str, NDArray[Any]]:
    user_id, gender, age, occupation, zip_code = zip(
        *(row.split(_SEPARATOR_1M) for row in rows), strict=True
    )
    return {
        "user_id": np.array(user_id, dtype=np.int64),
        "gender": np.array(gender),
        "age": np.array(age, dtype=np.int64),
        "occupation": np.array(occupation, dtype=np.int64),
        "zip_code": np.array(zip_code),
    }


def _parse_items_1m(rows: list[str]) -> tuple[dict[str, NDArray[Any]], list[str]]:
    """Parse ``movies.dat``; the year is the parenthesized suffix of the title."""
    records = [row.split(_SEPARATOR_1M, 2) for row in rows]
    if any(len(record) != _N_MOVIE_FIELDS_1M for record in records):
        raise ValueError("Malformed movies.dat row: expected item::title::genres.")
    per_movie = [record[2].split("|") for record in records]
    genre_names = sorted({genre for genres in per_movie for genre in genres})
    index = {genre: position for position, genre in enumerate(genre_names)}
    flags = np.zeros((len(records), len(genre_names)), dtype=bool)
    for row, genres in enumerate(per_movie):
        flags[row, [index[genre] for genre in genres]] = True
    return {
        "item_id": np.array([record[0] for record in records], dtype=np.int64),
        "title": np.array([record[1] for record in records]),
        "year": np.array([_parse_year(record[1]) for record in records], dtype=np.int64),
        "genres": flags,
    }, genre_names


def _parse_year(title: str) -> int:
    """The ``(1995)`` a MovieLens title ends with, or 0 when it carries none."""
    suffix = title.rstrip()[-6:]
    if suffix.startswith("(") and suffix.endswith(")") and suffix[1:-1].isdigit():
        return int(suffix[1:-1])
    return 0

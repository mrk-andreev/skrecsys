import pytest

from skrecsys.datasets import fetch_movielens_100k


@pytest.fixture(scope="session")
def movielens_100k_ua():
    """MovieLens 100K with the official ``ua`` split: 10 held-out ratings per user."""
    return fetch_movielens_100k(subset="ua")

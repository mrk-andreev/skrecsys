"""The Amazon Books loader against the real download: does it still produce the dataset
the sequential-recommendation literature reports on?

The identifier counts below are what the reference preprocessing of
`generative-recommenders <https://github.com/meta-recsys/generative-recommenders>`_
asserts (695,762 items) and what the HSTU paper tabulates (694.9K users, 674.2K items).
A change in the filtering, the encoding or the truncation window moves them, and silently
published numbers would then no longer be comparable with anyone else's.
"""

import numpy as np
import pytest

from skrecsys.datasets import fetch_amazon_books

pytestmark = pytest.mark.benchmark

ITEM_IDENTIFIERS = 695_762
USERS = 694_897
L50_INTERACTIONS = 8_069_177
FIVE_CORE_INTERACTIONS = 10_053_086


@pytest.fixture(scope="module")
def amazon_books():
    return fetch_amazon_books(subset="leave-one-out")


def test_matches_the_reference_preprocessing(amazon_books):
    data = amazon_books.data
    assert len(data) == L50_INTERACTIONS
    assert len(amazon_books.item_info.asin) == ITEM_IDENTIFIERS
    assert len(np.unique(data[:, 0])) == USERS
    # The window the reference pads to; the paper's item count is measured inside it.
    assert len(np.unique(data[:, 1])) == pytest.approx(674_079, abs=1000)


def test_sequences_are_chronological_and_capped(amazon_books):
    data, timestamps = amazon_books.data, amazon_books.timestamps
    users = data[:, 0]
    starts = np.flatnonzero(np.diff(users, prepend=-1) != 0)
    assert np.all(np.diff(users) >= 0), "rows are grouped by user"
    within_user = np.diff(users) == 0
    assert np.all(np.diff(timestamps)[within_user] >= 0), "and chronological inside a user"
    lengths = np.diff(np.append(starts, len(users)))
    # 50 interactions of history plus the held-out one
    assert lengths.max() == 51
    assert lengths.min() >= 5


def test_untruncated_keeps_every_five_core_interaction():
    full = fetch_amazon_books(max_sequence_length=None)
    assert len(full.data) == FIVE_CORE_INTERACTIONS
    assert len(np.unique(full.data[:, 0])) == USERS
    assert set(np.unique(full.target).tolist()) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_leave_one_out_holds_out_the_last_interaction(amazon_books):
    train, test = amazon_books.train_indices, amazon_books.test_indices
    assert len(test) == USERS
    assert len(train) + len(test) == len(amazon_books.data)
    # the held-out row of a user is their latest
    users = amazon_books.data[:, 0]
    assert np.array_equal(np.unique(users[test]), np.unique(users))
    assert np.all(np.diff(users[test]) > 0), "one held-out row per user, in user order"

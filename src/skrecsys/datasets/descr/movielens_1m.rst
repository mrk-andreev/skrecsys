.. _movielens_1m_dataset:

MovieLens 1M dataset
--------------------

**Data Set Characteristics:**

:Number of Ratings: 1,000,209
:Number of Users: 6,040
:Number of Items: 3,706 rated; 3,883 in the movie table, 3,952 identifiers
:Rating Scale: integers 1 to 5
:Density: 4.5%
:Collection Period: April 2000 to February 2003
:Minimum Ratings per User: 20

One million movie ratings collected by the GroupLens Research Project at the University
of Minnesota, from users who joined MovieLens in 2000. Every user in the dataset has
rated at least 20 movies; no filtering of any kind is applied by this loader, and the
identifiers are the ones MovieLens ships rather than renumbered codes, so side
information keyed by movie id joins directly.

``data`` holds ``[user_id, item_id]`` pairs, ``target`` the rating and ``timestamps`` the
rating time in Unix seconds. Rows are sorted by user, timestamp and item, so each user's
block is their viewing history in order, which is what a sequential model reads.

``user_info`` holds the demographics MovieLens ships, in its own coding: ``gender`` is
``"M"`` or ``"F"``, ``age`` is the lower bound of an age band (1 for under 18, then 18,
25, 35, 45, 50 and 56), and ``occupation`` is a code from 0 to 20 whose meanings are
listed in the archive's README. ``item_info`` holds the title, the year parsed out of it
and 18 binary genre flags.

Sequential evaluation
~~~~~~~~~~~~~~~~~~~~~

``subset="leave-one-out"`` holds out the last rating of every user and
``max_sequence_length`` caps how much history precedes it. Together with
``max_sequence_length=200`` this is the ``ml-1m-l200`` dataset that HSTU, SASRec and the
sequential recommenders benchmarked against them report on. The cap counts history: the
held-out rating is kept on top of it, so a scored interaction is predicted from a full
window.

Random interaction splits ignore time. They are invalid when the production decision is
time ordered; leave-one-out is time ordered within each user, though not across users.

**License:** the dataset may be used for research purposes only. It may not be
redistributed without permission from GroupLens, which is why skrecsys downloads it from
https://grouplens.org/datasets/movielens/1m/ instead of bundling it. Publications using
the dataset must cite:

.. rubric:: References

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
  Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
  https://doi.org/10.1145/2827872

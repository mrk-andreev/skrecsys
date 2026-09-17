.. _movielens_100k_dataset:

MovieLens 100K dataset
----------------------

**Data Set Characteristics:**

:Number of Ratings: 100,000
:Number of Users: 943
:Number of Items: 1,682
:Rating Scale: integers 1 to 5
:Density: 6.3%
:Collection Period: September 1997 to April 1998
:Minimum Ratings per User: 20

The MovieLens 100K dataset holds movie ratings collected by the GroupLens Research
Project at the University of Minnesota through the MovieLens web site. Users with fewer
than 20 ratings or incomplete demographic information were removed.

``data`` holds ``[user_id, item_id]`` pairs, ``target`` the rating and ``timestamps`` the
rating time in Unix seconds. Rows are sorted by user, timestamp and item.

``user_info`` holds user demographics (age, gender, occupation, zip code) and ``item_info`` movie
metadata (title, release date, IMDb URL and 19 binary genre flags). A movie may belong to
several genres.

The dataset ships predefined splits, selected with ``subset``:

- ``u1`` to ``u5``: disjoint 80%/20% train/test splits of the ratings; the five test
  sets together form a 5-fold cross-validation.
- ``ua`` and ``ub``: exactly 10 ratings per user in the test set; ``ua.test`` and
  ``ub.test`` are disjoint.

Random interaction splits ignore time. They are invalid when the production decision is
time ordered.

**License:** the dataset may be used for research purposes only. It may not be
redistributed without permission from GroupLens, which is why skrecsys downloads it from
https://grouplens.org/datasets/movielens/100k/ instead of bundling it. Publications using
the dataset must cite:

.. rubric:: References

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
  Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
  https://doi.org/10.1145/2827872

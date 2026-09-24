.. _amazon_books_dataset:

Amazon Books dataset
--------------------

**Data Set Characteristics:**

:Number of Interactions: 8,069,177 with the defaults; 10,053,086 untruncated
:Number of Users: 694,897
:Number of Items: 674,079 rated with the defaults; 695,762 identifiers
:Rating Scale: integers 1 to 5
:Density: 0.002%
:Collection Period: May 1996 to July 2014
:Minimum Interactions per User: 5

Book ratings from the 2014 Amazon product review dump, collected by Julian McAuley's
group at UCSD. The raw file holds 22,507,155 ratings; what this loader returns is the
5-core of that file, cut to the most recent interactions of each user.

``data`` holds ``[user_id, item_id]`` pairs, ``target`` the rating and ``timestamps`` the
rating time in Unix seconds. Rows are sorted by user, timestamp and item. The
identifiers are contiguous integer codes; ``user_info`` and ``item_info`` map them back
to the original reviewer identifiers and product ASINs, which is what side information
such as reviews, prices or categories is keyed by.

Preprocessing
~~~~~~~~~~~~~

The defaults reproduce ``amzn-books-l50``, the dataset that HSTU, SASRec and the
sequential recommenders benchmarked against them report on, so numbers measured here
are comparable with published ones. Three steps, in this order, matching the reference
implementation of `generative-recommenders
<https://github.com/meta-recsys/generative-recommenders>`_:

1. **5-core.** Interactions whose user or whose item occurs fewer than five times in the
   raw file are dropped. The counts come from the raw file and the filter runs once, so
   dropping a row does not then drop the users and items it was propping up; iterating
   this to a true 5-core would leave a different, smaller dataset.
2. **Encoding.** Surviving reviewers and ASINs are numbered from zero in sorted order.
   Users left with fewer than five interactions by step 1 are only dropped afterwards,
   so the identifier space covers 695,762 items while 686,623 of them are rated by a
   surviving user, and 674,079 inside the default window. That off-by-a-few-thousand is
   a property of the reference preprocessing, and reproducing it is what makes item
   counts line up with published ones.
3. **Truncation.** Each user keeps their last ``max_sequence_length`` interactions, plus
   the held-out one under ``subset="leave-one-out"``: the reference implementation pads
   sequences to ``max_sequence_length + 1`` so that a scored interaction is predicted
   from a full history. ``max_sequence_length=None`` keeps every 5-core interaction.

Sequences are long-tailed: a median user has 7 interactions and the longest has 27,508,
which is why the window matters to both quality and cost.

Evaluation
~~~~~~~~~~

``subset="leave-one-out"`` holds out the last interaction of every user, the protocol
the sequential-recommendation literature evaluates this dataset with, and the one whose
numbers are comparable across papers. It is a time-ordered split within each user but
not across users: a training interaction of one user may be later in wall-clock time
than a test interaction of another.

**License:** the dataset is distributed for academic use. Publications using it are
asked to cite:

.. rubric:: References

- Julian McAuley, Christopher Targett, Qinfeng Shi, Anton van den Hengel. 2015.
  Image-based recommendations on styles and substitutes. SIGIR.
  https://doi.org/10.1145/2766462.2767755
- Ruining He, Julian McAuley. 2016. Ups and downs: Modeling the visual evolution of
  fashion trends with one-class collaborative filtering. WWW.
  https://doi.org/10.1145/2872427.2883037
- Jiaqi Zhai et al. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential
  Transducers for Generative Recommendations. ICML. https://arxiv.org/abs/2402.17152

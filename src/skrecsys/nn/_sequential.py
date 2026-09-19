"""Shared machinery for recommenders that read the *order* of the interactions.

A sequential recommender is trained to predict the item a user touches next, so its
input is not the interaction matrix every other estimator here fits but each user's rows
in the order they arrived. What that order is, is the row order of ``X`` within a user:
:meth:`SequentialRecommender.fit` keeps the identifier arrays as they came in, and the
window builder below turns them into the padded ``(n_users, window)`` code matrices the
modules train and score from.
"""

from typing import Any, Self

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import override
from skrecsys.indexing import DenseSpace
from skrecsys.nn._base import ModuleT, TorchRecommender
from skrecsys.utils.validation import check_interactions, encode_ids

__all__ = ["PAD", "SequentialRecommender"]

#: Item code 0 is the padding slot, so a real item ``c`` is stored as ``c + 1``.
PAD = 0


class SequentialRecommender(TorchRecommender[ModuleT]):
    """Base class for next-item recommenders fitted with PyTorch.

    Subclasses implement ``_build_sequence_module``, which receives the windowed
    histories, plus the ``_export`` and scoring hooks of
    :class:`~skrecsys.nn._base.TorchRecommender`.
    """

    #: What ``_export`` materializes here, in place of the ``*_factors_`` the other
    #: torch recommenders use.
    user_embeddings_: NDArray[np.float64]
    item_embeddings_: NDArray[np.float64]

    #: Every interaction ever fitted, as identifiers, in the order they arrived. A
    #: later batch extends these sequences rather than replacing them, and
    #: ``interactions_`` cannot stand in: a CSR row is a set and has thrown the order
    #: away. It costs one identifier pair per stored interaction.
    ordered_history_: "tuple[NDArray[Any], NDArray[Any]]"

    @override
    def _index_space(self) -> DenseSpace:
        # The sequential models name their exported vectors `*_embeddings_` rather than
        # `*_factors_`; both are already materialized per user by `_export`, so a query
        # vector is a lookup and no history is replayed at recommend time.
        return DenseSpace(np.ascontiguousarray(self.item_embeddings_, dtype=np.float64))

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64]:
        return np.ascontiguousarray(self.user_embeddings_[user_indices], dtype=np.float64)

    #: Read by the window builder; declared here as an annotation only, so that sklearn
    #: still reads `get_params` off the subclass's own `__init__` signature.
    max_sequence_length: int

    @override
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit from ordered user-item interactions.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2)
            ``X[:, 0]`` user identifiers, ``X[:, 1]`` item identifiers. **The rows of a
            user are that user's history, in order**: row ``i`` precedes row ``j``
            whenever ``i < j`` and both belong to the same user. Rows of different users
            may be interleaved. Every dataset loader in :mod:`skrecsys.datasets` already
            sorts its rows by user and then by time, so an unshuffled loader output is
            ready to fit; a shuffled one trains the model on nonsense it cannot detect.

        y : array-like of shape (n_interactions,), default=None
            Ignored beyond validation: a sequential model is trained on which item came
            next, not on how it was rated.

        Returns
        -------
        self : object
        """
        users, items, _ = check_interactions(X, y)
        self._ordered = (users, items)
        try:
            fitted = super().fit(X, y)
        finally:
            del self._ordered
        # The windows cannot be rebuilt from `interactions_`: a CSR row is a set, and
        # the order within a user is the whole of what a sequence model reads. So the
        # identifiers are kept as they arrived, for a later batch to extend.
        self.ordered_history_ = (users.copy(), items.copy())
        return fitted

    @override
    def partial_fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Append an ordered batch to the history and continue training on it.

        The batch's rows extend each user's sequence, so the same ordering rule as
        ``fit`` applies within the batch, and the batch as a whole comes after
        everything already seen. Keeping the history is what makes that possible, and it
        costs one identifier pair per stored interaction on top of ``interactions_``.
        """
        users, items, _ = check_interactions(X, y)
        self._batch = (users, items)
        try:
            return super().partial_fit(X, y)
        finally:
            self.__dict__.pop("_batch", None)

    @override
    def _partial_fit(
        self,
        interactions: sp.csr_array,
        *,
        delta: sp.csr_array,
        new_user_indices: NDArray[np.intp],
        new_item_indices: NDArray[np.intp],
        touched_user_indices: NDArray[np.intp],
        touched_item_indices: NDArray[np.intp],
    ) -> None:
        users, items = self._batch
        past_users, past_items = self.ordered_history_
        self.ordered_history_ = (
            np.concatenate([past_users, users]),
            np.concatenate([past_items, items]),
        )
        # The windows are rebuilt from identifiers rather than codes, so a vocabulary
        # that grew -- and renumbered what came before -- needs nothing done to them.
        self._ordered = self.ordered_history_
        try:
            super()._partial_fit(
                interactions,
                delta=delta,
                new_user_indices=new_user_indices,
                new_item_indices=new_item_indices,
                touched_user_indices=touched_user_indices,
                touched_item_indices=touched_item_indices,
            )
        finally:
            del self._ordered

    @override
    def _build_module(
        self,
        interactions: sp.csr_array,
        device: torch.device,
        rng: np.random.RandomState,
        generator: torch.Generator,
    ) -> ModuleT:
        window = int(self.max_sequence_length)
        users, items = self._ordered
        # A stable sort groups each user's rows without disturbing their order, which is
        # the only thing that carries the sequence.
        order = np.argsort(encode_ids(users, self.user_ids_, name="user"), kind="stable")
        user_codes = encode_ids(users[order], self.user_ids_, name="user")
        # +1 leaves code 0 to mean padding.
        item_codes = encode_ids(items[order], self.item_ids_, name="item") + 1

        lengths = np.bincount(user_codes, minlength=self.n_users_)
        # Training reads one position more than scoring does: the extra column is the
        # target the last input position is asked to predict.
        train = self._pad_last(item_codes, lengths, window + 1)
        score = self._pad_last(item_codes, lengths, window)
        module = self._build_sequence_module(
            torch.from_numpy(train),
            torch.from_numpy(score),
            torch.from_numpy(np.minimum(lengths, window).astype(np.int64)),
            generator,
        )
        return module.to(device)

    def _build_sequence_module(
        self,
        train_sequences: torch.Tensor,
        score_sequences: torch.Tensor,
        lengths: torch.Tensor,
        generator: torch.Generator,
    ) -> ModuleT:
        """Build the module to train from the windowed histories.

        ``train_sequences`` is ``(n_users, window + 1)`` and ``score_sequences`` the same
        histories one column shorter; ``lengths`` counts the real positions of each
        scoring row, which is where the state that scores a user sits.
        """
        raise NotImplementedError

    def _pad_last(
        self, item_codes: NDArray[np.int64], lengths: NDArray[np.int64], window: int
    ) -> NDArray[np.int32]:
        """The last ``window`` codes of each user, right-padded into ``(n_users, window)``.

        Right-padding is what the references do, and it is why the modules carry
        ``lengths``: the state that scores a user is the one at its last real position,
        which is not the last column.
        """
        padded = np.zeros((len(lengths), window), dtype=np.int32)
        kept = np.minimum(lengths, window)
        ends = np.cumsum(lengths)
        rows = np.repeat(np.arange(len(lengths)), kept)
        columns = np.concatenate([np.arange(keep) for keep in kept]) if len(rows) else rows
        sources = np.concatenate(
            [np.arange(end - keep, end) for end, keep in zip(ends, kept, strict=True)]
        ).astype(np.intp)
        padded[rows, columns] = item_codes[sources]
        return padded

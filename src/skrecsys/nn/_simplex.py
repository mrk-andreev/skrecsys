"""SimpleX: cosine contrastive learning over a user's interacted items."""

import numbers
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from torch import nn

from skrecsys._typing import override
from skrecsys.nn._base import (
    TorchRecommender,
    TorchRecommenderModule,
    seeded_linear_,
    seeded_normal_,
)

#: Behaviour aggregators, in the order the paper introduces them.
AGGREGATORS = ("mean", "self_attention", "user_attention")


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the unmasked positions of each row; an empty row gets all zeros."""
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
    # Every position of a user with no history is masked, and softmax over nothing but
    # -inf is NaN. Zero weights are the right answer there: the pooled vector is zero.
    return torch.nan_to_num(weights, nan=0.0)


class _MeanAggregator(nn.Module):
    """Average pooling over the history."""

    @override
    def forward(
        self, history: torch.Tensor, mask: torch.Tensor, user: torch.Tensor
    ) -> torch.Tensor:
        weights = mask.to(history.dtype)
        total = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (history * weights.unsqueeze(-1)).sum(dim=1) / total


class _SelfAttentionAggregator(nn.Module):
    """Additive attention scoring each history item from its own embedding."""

    def __init__(self, n_factors: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_factors, n_factors, bias=False)
        self.score = nn.Linear(n_factors, 1, bias=False)

    @override
    def forward(
        self, history: torch.Tensor, mask: torch.Tensor, user: torch.Tensor
    ) -> torch.Tensor:
        scores = self.score(torch.tanh(self.hidden(history))).squeeze(-1)
        return (_masked_softmax(scores, mask).unsqueeze(-1) * history).sum(dim=1)


class _UserAttentionAggregator(nn.Module):
    """Attention scoring each history item by its similarity to the user embedding."""

    @override
    def forward(
        self, history: torch.Tensor, mask: torch.Tensor, user: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.bmm(history, user.unsqueeze(-1)).squeeze(-1)
        return (_masked_softmax(scores, mask).unsqueeze(-1) * history).sum(dim=1)


def _build_aggregator(aggregator: str, n_factors: int) -> nn.Module:
    if aggregator == "mean":
        return _MeanAggregator()
    if aggregator == "self_attention":
        return _SelfAttentionAggregator(n_factors)
    if aggregator == "user_attention":
        return _UserAttentionAggregator()
    raise ValueError(f"aggregator must be one of {AGGREGATORS}, got {aggregator!r}.")


class _SimpleXModule(TorchRecommenderModule):
    """Embeddings, behaviour aggregation and the cosine contrastive loss."""

    #: Registered buffers. ``nn.Module.__getattr__`` erases their type, so they are
    #: declared here to stay tensors for a type checker.
    history: torch.Tensor
    pair_users: torch.Tensor
    pair_items: torch.Tensor

    #: The item table carries a trailing padding row, which a growing catalog pushes
    #: further out; the user table is a plain code space.
    embedding_axes: ClassVar[dict[str, str]] = {
        "user_embedding.weight": "user",
        "item_embedding.weight": "item_pad_last",
    }

    def __init__(
        self,
        estimator: "SimpleX",
        history: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
        n_items: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        n_factors = int(estimator.n_factors)
        self.n_items = n_items
        self.pad_index = n_items
        self.gamma = float(estimator.gamma)
        self.margin = float(estimator.margin)
        self.negative_weight = float(estimator.negative_weight)
        self.n_negatives = int(estimator.n_negatives)

        self.user_embedding = nn.Embedding(history.shape[0], n_factors)
        self.item_embedding = nn.Embedding(n_items + 1, n_factors, padding_idx=self.pad_index)
        self.projection = nn.Linear(n_factors, n_factors)
        self.aggregator = _build_aggregator(estimator.aggregator, n_factors)
        self.dropout = nn.Dropout(float(estimator.dropout))

        std = 1.0 / np.sqrt(n_factors)
        seeded_normal_(self.user_embedding.weight, generator, std)
        seeded_normal_(self.item_embedding.weight, generator, std)
        with torch.no_grad():
            self.item_embedding.weight[self.pad_index].zero_()
        seeded_linear_(self.projection, generator)
        for layer in self.aggregator.modules():
            if isinstance(layer, nn.Linear):
                seeded_linear_(layer, generator)

        # Buffers, not parameters: they move with `.to(device)` but carry no gradient.
        self.register_buffer("history", history)
        self.register_buffer("pair_users", users)
        self.register_buffer("pair_items", items)

    def user_vectors(self, users: torch.Tensor) -> torch.Tensor:
        """Unit-norm fused user vectors of the given user indices."""
        user_embedded = self.user_embedding(users)
        history = self.history[users]
        pooled = self.aggregator(
            self.dropout(self.item_embedding(history)),
            history != self.pad_index,
            user_embedded,
        )
        fused = self.gamma * user_embedded + (1.0 - self.gamma) * self.projection(pooled)
        return torch.nn.functional.normalize(fused, dim=-1)

    def item_vectors(self, items: torch.Tensor) -> torch.Tensor:
        """Unit-norm item vectors of the given item indices."""
        return torch.nn.functional.normalize(self.item_embedding(items), dim=-1)

    @override
    def set_training_subset(
        self, users: torch.Tensor, pair_users: torch.Tensor, pair_items: torch.Tensor
    ) -> None:
        """Train on the batch's interactions rather than on every stored one.

        The examples here are pairs, so narrowing means replacing the pair buffers.
        Everything else the module holds still describes the whole history, which is what
        keeps a user's aggregated behaviour, and the negative sampling, honest.
        """
        del users
        device = self.pair_users.device
        self.pair_users = pair_users.to(device)
        self.pair_items = pair_items.to(device)

    @override
    def iter_batches(
        self, batch_size: int, generator: torch.Generator
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = self.pair_users.device
        order = torch.randperm(len(self.pair_users), generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            # Uniform over the whole catalog, as the paper samples: a draw that happens to
            # be one of the user's own items is kept rather than redrawn.
            negatives = torch.randint(
                self.n_items, (len(rows), self.n_negatives), generator=generator
            )
            rows = rows.to(device)
            yield self.pair_users[rows], self.pair_items[rows], negatives.to(device)

    @override
    def batch_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        users, items, negatives = batch
        vectors = self.user_vectors(users)
        positive = (vectors * self.item_vectors(items)).sum(dim=-1)
        negative = torch.bmm(self.item_vectors(negatives), vectors.unsqueeze(-1)).squeeze(-1)
        loss = (1.0 - positive) + self.negative_weight * torch.relu(negative - self.margin).mean(
            dim=1
        )
        return loss.mean()


class SimpleX(TorchRecommender[_SimpleXModule]):
    """Cosine contrastive collaborative filtering over a user's interacted items [1]_.

    A user is represented by their own embedding fused with an aggregate of the items
    they interacted with,

    ``e~_u = gamma * e_u + (1 - gamma) * W p_u``, where ``p_u = aggregate({e_j : j in N(u)})``,

    and an item by its embedding ``e_i``. The score of a pair is the cosine similarity
    ``cos(e~_u, e_i)``, and training minimizes the cosine contrastive loss (CCL) of one
    observed item against ``n_negatives`` uniformly drawn ones,

    ``(1 - cos(e~_u, e_i)) + (negative_weight / n_negatives) *
    sum_j max(0, cos(e~_u, e_j) - margin)``.

    The loss is what makes the model competitive rather than the architecture: it pushes
    the positive all the way to a cosine of 1 while ignoring any negative already below
    ``margin``, so the gradient concentrates on the negatives that actually rank too high.
    Interaction values are ignored -- only which pairs were observed matters -- which
    makes this an implicit-feedback model like
    :class:`~skrecsys.recommendation.BayesianPersonalizedRanking`.

    ``N(u)`` is the *set* of items of ``u``, not a sequence, so it is exactly the user's
    row of the fitted interaction matrix and no ordering or timestamp is needed. It
    includes the item being predicted, as the reference implementation lets it. A user
    with more than ``history_size`` items is subsampled uniformly once per fit, drawn from
    ``random_state``.

    Training needs PyTorch (``pip install skrecsys[nn]``); a fitted estimator does not.
    ``fit`` exports unit-norm factors to numpy, so the cosine score is a plain dot
    product and ``recommend`` goes through the same top-k kernel as every other
    estimator here.

    Parameters
    ----------
    n_factors : int, default=64
        Dimensionality of the user and item embeddings.
    n_negatives : int, default=100
        Negative items drawn per observed interaction. The paper's main lever: more
        negatives sharpen the ranking at a proportional cost per epoch.
    negative_weight : float, default=10.0
        Weight ``w`` of the negative term of the CCL loss, relative to the positive one.
    margin : float, default=0.9
        Cosine above which a negative starts contributing to the loss. A negative already
        scored below it is treated as settled and produces no gradient.
    gamma : float, default=0.5
        Weight of the user's own embedding in the fusion. ``1.0`` ignores the history and
        reduces the model to plain cosine matrix factorization; ``0.0`` drops the user
        embedding and represents a user only by what they interacted with.
    aggregator : {"mean", "self_attention", "user_attention"}, default="mean"
        How the history is pooled into ``p_u``. ``"mean"`` averages the embeddings;
        ``"self_attention"`` weights them by an additive attention over each item's own
        embedding; ``"user_attention"`` weights them by their dot product with ``e_u``.
    history_size : int, default=100
        Most items kept per user. Longer histories are subsampled uniformly.
    dropout : float, default=0.0
        Dropout applied to the history embeddings during training.
    learning_rate : float, default=1e-4
        Adam step size.
    regularization : float, default=1e-9
        L2 penalty, passed to Adam as ``weight_decay``.
    max_iter : int, default=100
        Number of epochs. Each one visits every observed interaction once.
    batch_size : int, default=512
        Observed interactions per gradient step.
    device : str or torch.device, default="cpu"
        Where to train. ``"auto"`` picks CUDA, then MPS, then CPU; anything else is
        handed to ``torch.device`` unchanged, so ``"cuda:1"``, ``"mps"`` or ``"xpu"``
        all work. The default is CPU because it is the one backend always present, and
        because a fit this size is rarely the bottleneck.
    random_state : int, RandomState instance or None, default=None
        Seed of the initialization, the history subsampling, the epoch shuffling and the
        negative sampling. A seeded fit on one thread is reproducible; the backend's own
        nondeterminism -- reduction order on a GPU, for instance -- is not controlled here.
    n_jobs : int or None, default=None
        Intra-op threads for the fit. ``None`` and ``-1`` leave torch's own default, which
        already uses every core. Any other value caps ``torch.set_num_threads`` for the
        duration of the fit, which is process-global while it lasts.
    learning_rate_schedule : {"constant", "adaptive"}, default="adaptive"
        How the Adam step size moves during the fit. ``"constant"`` holds it at
        ``learning_rate``; ``"adaptive"`` divides it by five each time the training loss
        fails to improve for ``n_iter_no_change`` epochs, down to a floor of 1e-6.
    early_stopping : bool, default=True
        End the fit once the training loss has stalled and the step size can no longer be
        cut. The signal is the training loss rather than a held-out split, so this ends a
        fit that has converged, not one that has begun to overfit.
    tol : float, default=1e-4
        How much an epoch must beat the best loss so far by to count as progress.
    n_iter_no_change : int, default=5
        Consecutive epochs without that improvement before the step size is cut, or --
        once it has bottomed out -- the fit stops.
    index : None, str or VectorIndex, default=None
        Approximate index used by ``recommend``. ``None`` scores every candidate
        exactly; ``"hnsw"`` or a configured :class:`~skrecsys.indexing.HNSW` walks a
        graph instead, which is where a large catalog pays off. See
        :mod:`skrecsys.indexing`.

    Attributes
    ----------
    user_factors_ : ndarray of shape (n_users_, n_factors)
        Unit-norm fused user vectors ``e~_u``.
    item_factors_ : ndarray of shape (n_items_, n_factors)
        Unit-norm item vectors ``e_i``.
    loss_curve_ : ndarray of shape (n_iter_,)
        Mean CCL loss of each epoch.
    n_iter_ : int
        Epochs actually run: ``max_iter`` unless the fit stopped early.
    best_loss_ : float
        The lowest epoch loss reached, or nan when ``max_iter=0``.

    References
    ----------
    .. [1] K. Mao, J. Zhu, J. Wang, Q. Dai, Z. Dong, X. Xiao, and X. He, "SimpleX: A
       Simple and Strong Baseline for Collaborative Filtering", CIKM 2021.
       https://arxiv.org/abs/2109.12613

    Examples
    --------
    >>> from skrecsys.nn import SimpleX
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = SimpleX(n_factors=4, max_iter=20, n_negatives=2, random_state=0).fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(
        self,
        n_factors: int = 64,
        n_negatives: int = 100,
        negative_weight: float = 10.0,
        margin: float = 0.9,
        gamma: float = 0.5,
        aggregator: str = "mean",
        history_size: int = 100,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
        regularization: float = 1e-9,
        max_iter: int = 100,
        batch_size: int = 512,
        *,
        device: str | torch.device = "cpu",
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int | None = None,
        learning_rate_schedule: str = "adaptive",
        early_stopping: bool = True,
        tol: float = 1e-4,
        n_iter_no_change: int = 5,
        index: Any = None,
    ) -> None:
        self.n_factors = n_factors
        self.n_negatives = n_negatives
        self.negative_weight = negative_weight
        self.margin = margin
        self.gamma = gamma
        self.aggregator = aggregator
        self.history_size = history_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.learning_rate_schedule = learning_rate_schedule
        self.early_stopping = early_stopping
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.index = index

    @override
    def _build_module(
        self,
        interactions: sp.csr_array,
        device: torch.device,
        rng: np.random.RandomState,
        generator: torch.Generator,
    ) -> _SimpleXModule:
        n_users, n_items = interactions.shape
        history = torch.from_numpy(self._build_history(interactions, rng))
        users = torch.from_numpy(
            np.repeat(np.arange(n_users, dtype=np.int64), np.diff(interactions.indptr))
        )
        items = torch.from_numpy(interactions.indices.astype(np.int64, copy=False))
        module = _SimpleXModule(self, history, users, items, n_items, generator)
        return module.to(device)

    def _build_history(
        self, interactions: sp.csr_array, rng: np.random.RandomState
    ) -> NDArray[np.int64]:
        """The padded ``(n_users, length)`` history, subsampling users who have too many.

        The sample is drawn by giving every stored interaction a random key and keeping
        the ``length`` smallest of each row, which is a uniform draw without replacement
        and costs one sort of the whole matrix rather than a pass per user. Every
        aggregator is permutation-invariant, so the order the draw leaves behind carries
        no meaning.
        """
        n_users, n_items = interactions.shape
        indptr, indices = interactions.indptr, interactions.indices
        counts = np.diff(indptr)
        length = max(1, min(int(self.history_size), int(counts.max(initial=1))))

        owners = np.repeat(np.arange(n_users), counts)
        order = np.lexsort((rng.random_sample(len(indices)), owners))
        # After a sort by owner the position within a row is the offset from its start,
        # and the rows keep the sizes the CSR already recorded.
        rank = np.arange(len(indices)) - np.repeat(indptr[:-1], counts)
        keep = rank < length

        history = np.full((n_users, length), n_items, dtype=np.int64)
        history[owners[order][keep], rank[keep]] = indices[order][keep]
        return history

    @override
    def _export(self, module: _SimpleXModule) -> None:
        items = module.item_vectors(torch.arange(self.n_items_, device=module.history.device))
        self.item_factors_ = np.ascontiguousarray(items.cpu().numpy(), dtype=np.float64)

        # One chunk per batch, so exporting never costs more memory than training did.
        chunks = [
            module.user_vectors(
                torch.arange(start, min(start + int(self.batch_size), self.n_users_)).to(
                    module.history.device
                )
            )
            .cpu()
            .numpy()
            for start in range(0, self.n_users_, int(self.batch_size))
        ]
        self.user_factors_ = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float64)

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return self.user_factors_[user_indices] @ self.item_factors_[item_indices].T

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return np.einsum(
            "ij,ij->i", self.user_factors_[user_indices], self.item_factors_[item_indices]
        )

    @override
    def _check_params(self) -> int:
        n_threads = self._check_torch_params()
        for name in ("n_negatives", "history_size"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        if not isinstance(self.negative_weight, numbers.Real) or not self.negative_weight >= 0:
            raise ValueError(
                f"negative_weight must be a real number >= 0, got {self.negative_weight!r}."
            )
        for name in ("margin", "gamma"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a real number in [0, 1], got {value!r}.")
        if not isinstance(self.dropout, numbers.Real) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be a real number in [0, 1), got {self.dropout!r}.")
        if self.aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}, got {self.aggregator!r}.")
        return n_threads

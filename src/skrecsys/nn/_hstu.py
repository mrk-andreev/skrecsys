"""HSTU: the Hierarchical Sequential Transduction Unit of "Actions Speak Louder than
Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
(https://arxiv.org/abs/2402.17152).

Unlike every other estimator here, HSTU reads the *order* of the interactions rather
than a bag of them: each user's rows are a sequence, and the model is trained to predict
the item at every position from the ones before it. What that order is, is the row order
of ``X`` within a user -- see :meth:`HSTU.fit`.
"""

import math
import numbers
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from skrecsys._typing import override
from skrecsys.nn._base import TorchRecommenderModule, seeded_dropout, seeded_normal_
from skrecsys.nn._sequential import PAD, SequentialRecommender

__all__ = ["HSTU"]

#: Standard deviation the reference initializes the projection and bias parameters with.
_INIT_STD = 0.02
_LAYER_NORM_EPS = 1e-6
#: Logit given to a sampled negative that turned out to be the positive, so that it
#: leaves the softmax instead of teaching the model to rank the target below itself.
_COLLISION_LOGIT = -5e4


def relative_position_bias(weights: torch.Tensor, length: int) -> torch.Tensor:
    """Expand ``2 * length - 1`` learned weights into a ``(1, length, length)`` bias.

    Entry ``[0, i, j]`` is the weight of the offset ``i - j``, so one parameter is shared
    by every pair of positions the same distance apart. The unfolding is the reference's:
    padding the weights by ``length`` and reshaping lays consecutive diagonals out in
    consecutive rows, which is cheaper than building an index matrix.
    """
    padded = F.pad(weights[: 2 * length - 1], [0, length]).repeat(length)
    rows = padded[..., :-length].reshape(1, length, 3 * length - 2)
    edge = (2 * length - 1) // 2
    return rows[..., edge:-edge]


class _HSTUBlock(nn.Module):
    """One Sequential Transduction Unit.

    A block splits a single projection of its normalized input into four parts --
    ``u``, ``v``, ``q``, ``k`` -- attends with ``q`` and ``k``, gates the attended ``v``
    by ``u``, and adds the result back onto its input. Two things differ from a
    Transformer block and both matter: the attention is pointwise ``SiLU`` divided by the
    window length rather than a softmax, so attention weights do not compete to sum to
    one, and the gate ``u`` replaces the feed-forward network entirely.
    """

    def __init__(
        self,
        n_factors: int,
        n_heads: int,
        head_dim: int,
        dropout: float,
        window: int,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.window = window
        # One matrix produces u, v, q and k, which is one matmul rather than four.
        self.uvqk = nn.Parameter(torch.empty((n_factors, 4 * head_dim * n_heads)))
        self.output = nn.Linear(head_dim * n_heads, n_factors)
        self.position_weights = nn.Parameter(torch.empty(2 * window - 1))

    def reset_parameters(self, generator: torch.Generator) -> None:
        seeded_normal_(self.uvqk, generator, _INIT_STD)
        seeded_normal_(self.position_weights, generator, _INIT_STD)
        # Xavier uniform, the reference's choice for the output projection, drawn here
        # rather than by torch's initializer so that the fit stays seeded.
        bound = math.sqrt(6.0 / (self.output.in_features + self.output.out_features))
        with torch.no_grad():
            weight = torch.empty(self.output.weight.shape)
            weight.uniform_(-bound, bound, generator=generator)
            self.output.weight.copy_(weight)
            self.output.bias.zero_()

    @override
    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        normed = F.layer_norm(x, [self.n_factors], eps=_LAYER_NORM_EPS)
        projected = F.silu(normed @ self.uvqk)
        u, v, q, k = torch.split(projected, 4 * [self.head_dim * self.n_heads], dim=-1)

        heads = (batch, length, self.n_heads, -1)
        attention = torch.einsum("bnhd,bmhd->bhnm", q.view(heads), k.view(heads))
        attention = attention + relative_position_bias(
            self.position_weights, self.window
        ).unsqueeze(1)
        # Pointwise, and normalized by the window rather than by the attention it paid:
        # a position attending to nothing relevant contributes nothing, instead of being
        # forced to spread a unit of weight over its history.
        attention = F.silu(attention) / self.window
        attention = attention * causal_mask
        attended = torch.einsum("bhnm,bmhd->bnhd", attention, v.view(heads))
        attended = attended.reshape(batch, length, self.head_dim * self.n_heads)

        gated = u * F.layer_norm(attended, [self.head_dim * self.n_heads], eps=_LAYER_NORM_EPS)
        dropped = seeded_dropout(gated, self.dropout, generator, training=self.training)
        projected_out = cast("torch.Tensor", self.output(dropped))
        return x + projected_out


class _HSTUModule(TorchRecommenderModule):
    """Item and position embeddings, the stack of blocks, and the sampled-softmax loss."""

    #: Buffers, annotated so that they read as tensors rather than as the
    #: ``Tensor | Module | None`` that ``register_buffer`` is declared to produce.
    train_sequences: torch.Tensor
    score_sequences: torch.Tensor
    lengths: torch.Tensor
    causal_mask: torch.Tensor

    #: Code 0 is the padding slot, so a real item sits one place along and a growing
    #: catalog extends the table at the far end.
    embedding_axes: ClassVar[dict[str, str]] = {"item_embedding.weight": "item_pad_first"}

    def __init__(
        self,
        estimator: "HSTU",
        train_sequences: torch.Tensor,
        score_sequences: torch.Tensor,
        lengths: torch.Tensor,
        n_items: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_factors = int(estimator.n_factors)
        self.window = train_sequences.size(1) - 1
        self.temperature = float(estimator.temperature)
        self.n_negatives = int(estimator.n_negatives)
        self.dropout = float(estimator.dropout)
        self.generator = generator

        self.register_buffer("train_sequences", train_sequences)
        self.register_buffer("score_sequences", score_sequences)
        self.register_buffer("lengths", lengths)
        # Row 0 is the padding slot and is never a target, so it only ever contributes
        # a zero vector to the encoder.
        self.item_embedding = nn.Embedding(n_items + 1, self.n_factors, padding_idx=PAD)
        self.position_embedding = nn.Parameter(torch.empty(self.window, self.n_factors))
        self.blocks = nn.ModuleList(
            _HSTUBlock(
                n_factors=self.n_factors,
                n_heads=int(estimator.n_heads),
                head_dim=estimator._head_dim(),
                dropout=self.dropout,
                window=self.window,
            )
            for _ in range(int(estimator.n_blocks))
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.window, self.window)).view(1, 1, *2 * [self.window]),
        )
        self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        std = 1.0 / math.sqrt(self.n_factors)
        seeded_normal_(self.item_embedding.weight, generator, std)
        with torch.no_grad():
            self.item_embedding.weight[PAD].zero_()
        seeded_normal_(self.position_embedding, generator, std)
        for block in cast("Iterable[_HSTUBlock]", self.blocks):
            block.reset_parameters(generator)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """Run the blocks over padded sequences; return an L2-normalized state per position.

        Positions holding padding are zeroed after every block. In the reference the
        sequences are jagged and padding simply does not exist; zeroing is what keeps a
        dense batch equivalent to that.
        """
        valid = (sequences != PAD).unsqueeze(-1).to(self.item_embedding.weight.dtype)
        embedded = cast("torch.Tensor", self.item_embedding(sequences))
        x = embedded * math.sqrt(self.n_factors)
        x = x + self.position_embedding[: sequences.size(1)].unsqueeze(0)
        x = seeded_dropout(x, self.dropout, self.generator, training=self.training)
        x = x * valid
        for block in cast("Iterable[_HSTUBlock]", self.blocks):
            x = block(x, self.causal_mask, self.generator) * valid
        return self.normalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the last axis, the reference's ``l2_norm`` postprocessor."""
        return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=_LAYER_NORM_EPS)

    def item_vectors(self, codes: torch.Tensor) -> torch.Tensor:
        """Normalized embeddings of ``codes``, which is what the similarity compares."""
        embedded = cast("torch.Tensor", self.item_embedding(codes))
        return self.normalize(embedded)

    def user_vectors(self, rows: torch.Tensor) -> torch.Tensor:
        """The state at each user's last real position, which is what scores items."""
        encoded = self.encode(self.score_sequences[rows].long())
        last = torch.clamp(self.lengths[rows].long() - 1, min=0)
        return encoded[torch.arange(len(rows), device=encoded.device), last]

    @override
    def iter_batches(self, batch_size: int, generator: torch.Generator) -> Iterator[Any]:
        rows = self.training_rows(len(self.train_sequences))
        order = rows[torch.randperm(len(rows), generator=generator)]
        for start in range(0, len(order), batch_size):
            yield (order[start : start + batch_size],)

    @override
    def batch_loss(self, batch: Any) -> torch.Tensor:
        (rows,) = batch
        device = self.item_embedding.weight.device
        sequences = self.train_sequences[rows.to(device)].long()
        inputs, targets = sequences[:, :-1], sequences[:, 1:]

        encoded = self.encode(inputs)
        weights = (targets != PAD).to(encoded.dtype)
        positive = (encoded * self.item_vectors(targets)).sum(-1, keepdim=True)

        shape = (*targets.shape, self.n_negatives)
        sampled = torch.randint(1, self.n_items + 1, shape, generator=self.generator)
        sampled = sampled.to(device)
        negative = torch.einsum("bld,blrd->blr", encoded, self.item_vectors(sampled))
        # A negative that collided with the target would otherwise ask the model to rank
        # the target below itself.
        negative = torch.where(
            sampled == targets.unsqueeze(-1), _COLLISION_LOGIT, negative / self.temperature
        )

        logits = torch.cat([positive / self.temperature, negative], dim=-1)
        losses = -F.log_softmax(logits, dim=-1)[..., 0]
        # A batch of users who each have a single interaction supervises nothing; the
        # clamp keeps that an honest zero rather than a division by zero.
        return (losses * weights).sum() / weights.sum().clamp(min=1.0)


class HSTU(SequentialRecommender[_HSTUModule]):
    """Hierarchical Sequential Transduction Unit, a sequential recommender.

    HSTU encodes a user's history with a stack of transduction blocks and recommends the
    items whose embeddings best match the state at its last position. It is trained
    autoregressively: every position predicts the next item, against ``n_negatives``
    uniformly sampled items under a sampled softmax.

    Unlike the other estimators here, HSTU reads the order of ``X``. See ``fit``.

    Parameters
    ----------
    n_factors : int, default=50
        Width of the item embeddings and of the block states.

    n_blocks : int, default=2
        Number of transduction blocks.

    n_heads : int, default=1
        Attention heads per block.

    head_dim : int or None, default=None
        Width of each head's query, key, value and gate projections. None uses
        ``n_factors``. The reference lets the query/key and value widths differ, but
        every published configuration sets them equal, so this is the one knob.

    max_sequence_length : int, default=200
        Positions the model attends over. Only a user's last this many interactions are
        read, and the cost of a fit grows with its square.

    n_negatives : int, default=128
        Items sampled uniformly per position as negatives of the sampled softmax.

    temperature : float, default=0.05
        Divides the similarities before the softmax. Below one it sharpens the loss; it
        also scales what ``predict`` returns, but not the ranking ``recommend`` produces.

    dropout : float, default=0.2
        Dropout on the embeddings and on each block's gated output.

    learning_rate : float, default=1e-3
        Adam step size.

    regularization : float, default=0.0
        Adam weight decay.

    max_iter : int, default=100
        Training epochs.

    batch_size : int, default=128
        Users per batch.

    device : str or torch.device, default="cpu"
        Where to train. ``"auto"`` picks CUDA, then MPS, then CPU.

    random_state : int, RandomState instance or None, default=None
        Seeds the initialization, the batch order and the negative sampling.

    n_jobs : int or None, default=None
        Cap on torch's intra-op threads during the fit. None leaves torch's own default.

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
    user_ids_ : ndarray of shape (n_users_,)
    item_ids_ : ndarray of shape (n_items_,)
    n_users_ : int
    n_items_ : int
    user_embeddings_ : ndarray of shape (n_users_, n_factors)
        The encoded state of each user's fitted history, L2-normalized.
    item_embeddings_ : ndarray of shape (n_items_, n_factors)
        L2-normalized item embeddings.
    loss_curve_ : ndarray of shape (n_iter_,)
    n_iter_ : int
        Epochs actually run: ``max_iter`` unless the fit stopped early.
    best_loss_ : float
        The lowest epoch loss reached, or nan when ``max_iter=0``.

    Notes
    -----
    The published HSTU adds a bucketed *time* term to the relative attention bias, from
    the gaps between consecutive interactions. ``fit`` receives identifiers only, so this
    implementation carries the positional half of that bias alone; expect a gap against
    numbers published with the time term.

    A fitted HSTU scores from the history it was fitted on: ``recommend`` takes user
    identifiers, not sequences, so it cannot be asked what follows an arbitrary history
    without refitting.

    References
    ----------
    .. [1] J. Zhai et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential
           Transducers for Generative Recommendations", ICML 2024.
           https://arxiv.org/abs/2402.17152

    Examples
    --------
    >>> from skrecsys.nn import HSTU
    >>> X = [["u1", "a"], ["u1", "b"], ["u1", "c"], ["u2", "b"], ["u2", "c"]]
    >>> rec = HSTU(n_factors=8, max_sequence_length=4, max_iter=2, random_state=0).fit(X)
    >>> items, scores = rec.recommend(["u2"], n_recommendations=1)
    >>> items.shape
    (1, 1)
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_blocks: int = 2,
        n_heads: int = 1,
        head_dim: int | None = None,
        max_sequence_length: int = 200,
        n_negatives: int = 128,
        temperature: float = 0.05,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        regularization: float = 0.0,
        max_iter: int = 100,
        batch_size: int = 128,
        device: "str | torch.device" = "cpu",
        random_state: "int | np.random.RandomState | None" = None,
        n_jobs: int | None = None,
        *,
        learning_rate_schedule: str = "adaptive",
        early_stopping: bool = True,
        tol: float = 1e-4,
        n_iter_no_change: int = 5,
        index: Any = None,
    ) -> None:
        self.n_factors = n_factors
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.n_negatives = n_negatives
        self.temperature = temperature
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
    def _build_sequence_module(
        self,
        train_sequences: torch.Tensor,
        score_sequences: torch.Tensor,
        lengths: torch.Tensor,
        generator: torch.Generator,
    ) -> _HSTUModule:
        return _HSTUModule(
            self, train_sequences, score_sequences, lengths, self.n_items_, generator
        )

    @override
    def _export(self, module: _HSTUModule) -> None:
        device = module.item_embedding.weight.device
        codes = torch.arange(1, self.n_items_ + 1, device=device)
        items = module.item_vectors(codes)
        self.item_embeddings_ = np.ascontiguousarray(items.cpu().numpy(), dtype=np.float64)

        # One chunk per batch, so exporting never costs more memory than training did.
        chunks = [
            module.user_vectors(
                torch.arange(start, min(start + int(self.batch_size), self.n_users_), device=device)
            )
            .cpu()
            .numpy()
            for start in range(0, self.n_users_, int(self.batch_size))
        ]
        self.user_embeddings_ = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float64)

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64]:
        # `_score_users` divides by the temperature, and a positive scalar divisor
        # changes no ordering -- but it does change the scores, which `recommend`
        # returns, so it scales into the query rather than being dropped.
        queries = super()._index_queries(user_indices)
        return queries / float(self.temperature)

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        scores = self.user_embeddings_[user_indices] @ self.item_embeddings_[item_indices].T
        return scores / float(self.temperature)

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        scores = np.einsum(
            "ij,ij->i", self.user_embeddings_[user_indices], self.item_embeddings_[item_indices]
        )
        return scores / float(self.temperature)

    def _head_dim(self) -> int:
        return int(self.head_dim) if self.head_dim is not None else int(self.n_factors)

    @override
    def _check_params(self) -> int:
        n_threads = self._check_torch_params()
        for name in ("n_blocks", "n_heads", "max_sequence_length", "n_negatives"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        if self.head_dim is not None and (
            not isinstance(self.head_dim, numbers.Integral)
            or isinstance(self.head_dim, bool)
            or self.head_dim < 1
        ):
            raise ValueError(f"head_dim must be None or an integer >= 1, got {self.head_dim!r}.")
        if not isinstance(self.temperature, numbers.Real) or not self.temperature > 0:
            raise ValueError(f"temperature must be a real number > 0, got {self.temperature!r}.")
        if not isinstance(self.dropout, numbers.Real) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be a real number in [0, 1), got {self.dropout!r}.")
        return n_threads

"""Mamba4Rec: "Mamba4Rec: Towards Efficient Sequential Recommendation with Selective
State Space Models" (https://arxiv.org/abs/2403.03900).

Like :class:`~skrecsys.nn._hstu.HSTU` this reads the *order* of the interactions, but it
replaces attention with a selective state space model: the history is consumed by a
linear recurrence whose decay and input gates are themselves functions of the current
item, so a position mixes its whole past in one pass instead of comparing itself against
every other position. The cost per position is constant rather than linear in the window,
which is the paper's point.

The selective scan here is a plain PyTorch recurrence, not the fused kernel the reference
imports from ``mamba-ssm``; see the Notes of :class:`Mamba4Rec`.
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

__all__ = ["Mamba4Rec"]

#: Standard deviation the reference initializes the embeddings and projections with.
_INIT_STD = 0.02
#: The reference's layer-norm epsilon, which is RecBole's rather than torch's default.
_LAYER_NORM_EPS = 1e-12
#: Hidden width of each layer's feed-forward network, as a multiple of ``n_factors``.
#: The reference's configuration sets exactly this multiple and never varies it.
_FEEDFORWARD_EXPANSION = 4
#: Range the step sizes are initialized to span, and the floor they are clamped at.
#: These are Mamba's own numbers, and what makes an untrained block neither forget its
#: input immediately nor integrate it forever.
_DT_MIN, _DT_MAX, _DT_FLOOR = 1e-3, 1e-1, 1e-4


def causal_depthwise_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Convolve each channel of ``(batch, length, channels)`` with its own short filter.

    Position ``t`` of the output reads positions ``t - k + 1 .. t`` of the input, which
    is what makes the convolution causal; the window is left-padded with zeros so that
    the output keeps the input's length.

    This is what ``nn.Conv1d(channels, channels, k, groups=channels)`` computes, written
    as ``k`` shifted multiply-accumulates instead. The filter is a handful of taps, and a
    grouped convolution that narrow falls off torch's fast CPU paths: the loop below runs
    the same arithmetic an order of magnitude faster, which matters because every fitted
    position pays for it.
    """
    taps = weight.shape[-1]
    padded = F.pad(x, [0, 0, taps - 1, 0])
    out = bias.expand_as(x).clone()
    for tap in range(taps):
        out = out + padded[:, tap : tap + x.shape[1]] * weight[:, tap]
    return out


def selective_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Run the selective state space recurrence and return its output per position.

    The state is ``h_t = exp(delta_t * A) h_{t-1} + delta_t B_t x_t`` and the output is
    ``C_t h_t + D x_t``. ``A`` is diagonal, so every channel carries ``d_state``
    independent scalar states and the whole thing is elementwise; ``delta``, ``B`` and
    ``C`` are produced from the input itself, which is the "selective" part: a position
    can decide to keep its state or to overwrite it with what it just read.

    Shapes are ``(batch, length, channels)`` for ``x`` and ``delta``, ``(channels,
    state)`` for ``a``, ``(batch, length, state)`` for ``b`` and ``c``, ``(channels,)``
    for ``d``.

    The recurrence is sequential by nature and is run as a Python loop over positions,
    the way every unfused PyTorch implementation of Mamba does. It is the expensive part
    of a fit: the loop is ``length`` iterations long and each one keeps a ``(batch,
    channels, state)`` tensor alive for the backward pass.
    """
    decay = torch.exp(delta.unsqueeze(-1) * a)
    update = delta.unsqueeze(-1) * b.unsqueeze(2) * x.unsqueeze(-1)

    state = torch.zeros(x.shape[0], x.shape[2], a.shape[1], dtype=x.dtype, device=x.device)
    outputs = []
    for position in range(x.shape[1]):
        state = decay[:, position] * state + update[:, position]
        outputs.append(torch.einsum("bcs,bs->bc", state, c[:, position]))
    return torch.stack(outputs, dim=1) + x * d


class _MambaBlock(nn.Module):
    """One Mamba block: gated input, causal depthwise convolution, selective scan.

    The input is projected up to ``expand * n_factors`` channels twice over -- once into
    the branch that is scanned and once into a gate -- convolved over a short causal
    window to give the scan local context, scanned, gated, and projected back down.
    """

    def __init__(
        self,
        n_factors: int,
        d_state: int,
        d_conv: int,
        d_inner: int,
        dt_rank: int,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(n_factors, 2 * d_inner, bias=False)
        # One filter per channel, so the convolution mixes positions and not channels;
        # the scan is what mixes over the window. See `causal_depthwise_conv` for why
        # the taps are parameters here rather than an `nn.Conv1d`.
        self.conv_weight = nn.Parameter(torch.empty(d_inner, d_conv))
        self.conv_bias = nn.Parameter(torch.empty(d_inner))
        # The step size, the input gate and the output gate of the scan, all read off
        # the current position. `dt_rank` is a bottleneck on the step size alone.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        # A is stored as the log of its negation: the parameter is unconstrained while
        # the state matrix it produces stays negative, so the recurrence cannot diverge.
        self.a_log = nn.Parameter(torch.empty(d_inner, d_state))
        self.d = nn.Parameter(torch.empty(d_inner))
        self.out_proj = nn.Linear(d_inner, n_factors, bias=False)

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Initialize the projections as RecBole does and the scan as Mamba does.

        The projections take the ``N(0, 0.02)`` that RecBole applies to every linear
        layer it owns. The scan's own parameters do not: ``A`` enumerates the state
        dimensions so that each state decays at its own rate, and the ``dt`` bias is
        drawn so that softplus turns it into a step size spanning :data:`_DT_MIN` to
        :data:`_DT_MAX`. The reference applies its blanket initializer to these too,
        which costs the block the spread of timescales the architecture is built
        around; that much of the reference is not followed here.
        """
        for linear in (self.in_proj, self.x_proj, self.dt_proj, self.out_proj):
            seeded_normal_(linear.weight, generator, _INIT_STD)
        # Torch's own convolution bound, drawn here rather than by its initializer so
        # that the fit stays seeded. A depthwise filter sees `d_conv` inputs.
        bound = 1.0 / math.sqrt(self.d_conv)
        with torch.no_grad():
            for parameter in (self.conv_weight, self.conv_bias):
                draws = torch.empty(parameter.shape, dtype=parameter.dtype)
                draws.uniform_(-bound, bound, generator=generator)
                parameter.copy_(draws)
            self.a_log.copy_(
                torch.arange(1, self.d_state + 1, dtype=self.a_log.dtype)
                .log()
                .expand(self.d_inner, self.d_state)
            )
            self.d.fill_(1.0)
            # Uniform in log space, then inverted through softplus so that the bias the
            # layer holds produces exactly those step sizes.
            draws = torch.empty(self.d_inner, dtype=self.dt_proj.bias.dtype)
            draws.uniform_(math.log(_DT_MIN), math.log(_DT_MAX), generator=generator)
            steps = draws.exp().clamp(min=_DT_FLOOR)
            self.dt_proj.bias.copy_(steps + torch.log(-torch.expm1(-steps)))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scanned, gate = cast("torch.Tensor", self.in_proj(x)).chunk(2, dim=-1)
        scanned = F.silu(causal_depthwise_conv(scanned, self.conv_weight, self.conv_bias))

        projected = cast("torch.Tensor", self.x_proj(scanned))
        delta, b, c = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Softplus keeps the step size positive, so `exp(delta * A)` is a decay in (0, 1).
        delta = F.softplus(cast("torch.Tensor", self.dt_proj(delta)))
        scanned = selective_scan(scanned, delta, -self.a_log.exp(), b, c, self.d)

        return cast("torch.Tensor", self.out_proj(scanned * F.silu(gate)))


class _Mamba4RecLayer(nn.Module):
    """A Mamba block with the reference's normalization, and its feed-forward network.

    Whether a residual connection wraps the block depends on how many layers there are,
    which is the reference's own rule and is kept: a single-layer model normalizes the
    block's output alone, and a deeper one adds the layer's input back first.
    """

    def __init__(
        self,
        n_factors: int,
        d_state: int,
        d_conv: int,
        d_inner: int,
        dt_rank: int,
        feedforward_size: int,
        dropout: float,
        *,
        residual: bool,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.mamba = _MambaBlock(n_factors, d_state, d_conv, d_inner, dt_rank)
        self.norm = nn.LayerNorm(n_factors, eps=_LAYER_NORM_EPS)
        self.feedforward = nn.Linear(n_factors, feedforward_size)
        self.feedforward_out = nn.Linear(feedforward_size, n_factors)
        self.feedforward_norm = nn.LayerNorm(n_factors, eps=_LAYER_NORM_EPS)

    def reset_parameters(self, generator: torch.Generator) -> None:
        self.mamba.reset_parameters(generator)
        for linear in (self.feedforward, self.feedforward_out):
            seeded_normal_(linear.weight, generator, _INIT_STD)
            with torch.no_grad():
                linear.bias.zero_()

    @override
    def forward(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        hidden = self.mamba(x)
        hidden = seeded_dropout(hidden, self.dropout, generator, training=self.training)
        hidden = cast("torch.Tensor", self.norm(hidden + x if self.residual else hidden))

        inner = F.gelu(cast("torch.Tensor", self.feedforward(hidden)))
        inner = seeded_dropout(inner, self.dropout, generator, training=self.training)
        inner = cast("torch.Tensor", self.feedforward_out(inner))
        inner = seeded_dropout(inner, self.dropout, generator, training=self.training)
        return cast("torch.Tensor", self.feedforward_norm(inner + hidden))


class _Mamba4RecModule(TorchRecommenderModule):
    """Item embeddings, the stack of Mamba layers, and the full-catalog softmax loss."""

    #: Buffers, annotated so that they read as tensors rather than as the
    #: ``Tensor | Module | None`` that ``register_buffer`` is declared to produce.
    train_sequences: torch.Tensor
    score_sequences: torch.Tensor
    lengths: torch.Tensor

    #: Code 0 is the padding slot, so a real item sits one place along and a growing
    #: catalog extends the table at the far end.
    embedding_axes: ClassVar[dict[str, str]] = {"item_embedding.weight": "item_pad_first"}

    def __init__(
        self,
        estimator: "Mamba4Rec",
        train_sequences: torch.Tensor,
        score_sequences: torch.Tensor,
        lengths: torch.Tensor,
        n_items: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_factors = int(estimator.n_factors)
        self.dropout = float(estimator.dropout)
        self.generator = generator

        self.register_buffer("train_sequences", train_sequences)
        self.register_buffer("score_sequences", score_sequences)
        self.register_buffer("lengths", lengths)
        # Row 0 is the padding slot and is never a target, so it only ever contributes
        # a zero vector to the encoder.
        self.item_embedding = nn.Embedding(n_items + 1, self.n_factors, padding_idx=PAD)
        self.norm = nn.LayerNorm(self.n_factors, eps=_LAYER_NORM_EPS)
        n_blocks = int(estimator.n_blocks)
        self.layers = nn.ModuleList(
            _Mamba4RecLayer(
                n_factors=self.n_factors,
                d_state=int(estimator.d_state),
                d_conv=int(estimator.d_conv),
                d_inner=estimator._d_inner(),
                dt_rank=estimator._dt_rank(),
                feedforward_size=_FEEDFORWARD_EXPANSION * self.n_factors,
                dropout=self.dropout,
                residual=n_blocks > 1,
            )
            for _ in range(n_blocks)
        )
        self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        seeded_normal_(self.item_embedding.weight, generator, _INIT_STD)
        with torch.no_grad():
            self.item_embedding.weight[PAD].zero_()
        for layer in cast("Iterable[_Mamba4RecLayer]", self.layers):
            layer.reset_parameters(generator)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """Run the layers over padded sequences; return the state at every position.

        There is no positional embedding: a recurrence reads its input in order, so the
        position is already in the state. Padding is zeroed after every layer, which is
        what keeps a dense batch equivalent to the jagged sequences of the reference --
        the convolution would otherwise let a padding column reach the positions after
        it, and a padding position's own state is never read.
        """
        valid = (sequences != PAD).unsqueeze(-1).to(self.item_embedding.weight.dtype)
        embedded = cast("torch.Tensor", self.item_embedding(sequences))
        x = seeded_dropout(embedded, self.dropout, self.generator, training=self.training)
        x = cast("torch.Tensor", self.norm(x)) * valid
        for layer in cast("Iterable[_Mamba4RecLayer]", self.layers):
            x = cast("torch.Tensor", layer(x, self.generator)) * valid
        return x

    def item_scores(self, states: torch.Tensor) -> torch.Tensor:
        """Score every real item against ``states``, which is what the softmax is over.

        The padding row is left out: it is not an item, and a model that can spend
        probability mass on it is one whose loss never quite sums over the catalog.
        """
        return states @ self.item_embedding.weight[1:].T

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

        logits = self.item_scores(self.encode(inputs))
        # `targets - 1` undoes the padding offset, and a padded position is dropped by
        # the ignore index. The mean is taken here rather than by `cross_entropy` so
        # that a batch of users who each have a single interaction -- and so supervise
        # nothing -- is an honest zero rather than a division by zero.
        summed = F.cross_entropy(
            logits.reshape(-1, self.n_items),
            torch.where(targets == PAD, -1, targets - 1).reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )
        return summed / (targets != PAD).sum().clamp(min=1)


class Mamba4Rec(SequentialRecommender[_Mamba4RecModule]):
    """Selective state space sequential recommender.

    Mamba4Rec encodes a user's history with a stack of Mamba blocks and recommends the
    items whose embeddings best match the state at its last position. It is trained
    autoregressively: every position predicts the next item, under a softmax over the
    whole catalog.

    Unlike the non-sequential estimators here, Mamba4Rec reads the order of ``X``. See
    ``fit``.

    Parameters
    ----------
    n_factors : int, default=64
        Width of the item embeddings and of the states between layers.

    n_blocks : int, default=1
        Mamba layers to stack. The reference tunes this to one on MovieLens, and one is
        also what the residual connection depends on: a single-layer model runs the
        block without a skip around it, which is the reference's own rule.

    d_state : int, default=32
        States each channel of a block carries. This is the memory of the recurrence.

    d_conv : int, default=4
        Width of the causal depthwise convolution before the scan.

    expand : int, default=2
        Channel expansion inside a block: it scans ``expand * n_factors`` channels.

    dt_rank : int or None, default=None
        Rank of the projection that produces the scan's step size. None uses
        ``ceil(n_factors / 16)``, Mamba's "auto".

    max_sequence_length : int, default=200
        Positions the model reads. Only a user's last this many interactions are used.
        Unlike an attention model the cost grows linearly rather than quadratically in
        it, which is the architecture's selling point.

    dropout : float, default=0.2
        Dropout on the embeddings, on each block's output and inside its feed-forward
        network.

    learning_rate : float, default=1e-3
        Adam step size.

    regularization : float, default=0.0
        Adam weight decay.

    max_iter : int, default=100
        Training epochs.

    batch_size : int, default=64
        Users per batch. It is the knob that decides a fit's peak memory: the scan keeps
        ``batch_size x max_sequence_length x expand * n_factors x d_state`` values alive
        for the backward pass, and the loss a further ``batch_size x
        max_sequence_length x n_items``.

    device : str or torch.device, default="cpu"
        Where to train. ``"auto"`` picks CUDA, then MPS, then CPU.

    random_state : int, RandomState instance or None, default=None
        Seeds the initialization, the batch order and the dropout masks.

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
        The encoded state of each user's fitted history.
    item_embeddings_ : ndarray of shape (n_items_, n_factors)
        The item embeddings the state is scored against.
    loss_curve_ : ndarray of shape (n_iter_,)
    n_iter_ : int
        Epochs actually run: ``max_iter`` unless the fit stopped early.
    best_loss_ : float
        The lowest epoch loss reached, or nan when ``max_iter=0``.

    Notes
    -----
    The selective scan is a PyTorch recurrence, a Python loop over the window, where the
    reference calls the fused CUDA kernel of ``mamba-ssm``. The arithmetic is the same
    and the results are, up to floating point, the reference's; the constant factor is
    not, and it is what makes a fit here expensive. The efficiency claims of the paper
    are claims about that kernel and cannot be checked against this implementation.

    Training supervises every position of a user's window against the full catalog. The
    reference reaches the same supervision by augmenting each user into one training
    example per prefix; one causal pass over the window is that, without materializing
    the prefixes. What it costs is a logit per position per item, so a catalog of six
    figures needs a smaller ``batch_size`` than the default.

    A fitted Mamba4Rec scores from the history it was fitted on: ``recommend`` takes user
    identifiers, not sequences, so it cannot be asked what follows an arbitrary history
    without refitting.

    References
    ----------
    .. [1] C. Liu, J. Lin, J. Wang, H. Liu, J. Caverlee. "Mamba4Rec: Towards Efficient
           Sequential Recommendation with Selective State Space Models", 2024.
           https://arxiv.org/abs/2403.03900
    .. [2] A. Gu, T. Dao. "Mamba: Linear-Time Sequence Modeling with Selective State
           Spaces", 2023. https://arxiv.org/abs/2312.00752

    Examples
    --------
    >>> from skrecsys.nn import Mamba4Rec
    >>> X = [["u1", "a"], ["u1", "b"], ["u1", "c"], ["u2", "b"], ["u2", "c"]]
    >>> rec = Mamba4Rec(n_factors=8, max_sequence_length=4, max_iter=2, random_state=0)
    >>> items, scores = rec.fit(X).recommend(["u2"], n_recommendations=1)
    >>> items.shape
    (1, 1)
    """

    def __init__(
        self,
        n_factors: int = 64,
        n_blocks: int = 1,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        max_sequence_length: int = 200,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        regularization: float = 0.0,
        max_iter: int = 100,
        batch_size: int = 64,
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
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.max_sequence_length = max_sequence_length
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
    ) -> _Mamba4RecModule:
        return _Mamba4RecModule(
            self, train_sequences, score_sequences, lengths, self.n_items_, generator
        )

    @override
    def _export(self, module: _Mamba4RecModule) -> None:
        items = module.item_embedding.weight[1:]
        self.item_embeddings_ = np.ascontiguousarray(items.cpu().numpy(), dtype=np.float64)

        device = module.item_embedding.weight.device
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
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return self.user_embeddings_[user_indices] @ self.item_embeddings_[item_indices].T

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return np.einsum(
            "ij,ij->i", self.user_embeddings_[user_indices], self.item_embeddings_[item_indices]
        )

    def _d_inner(self) -> int:
        return int(self.expand) * int(self.n_factors)

    def _dt_rank(self) -> int:
        if self.dt_rank is not None:
            return int(self.dt_rank)
        return math.ceil(int(self.n_factors) / 16)

    @override
    def _check_params(self) -> int:
        n_threads = self._check_torch_params()
        for name in ("n_blocks", "d_state", "d_conv", "expand", "max_sequence_length"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        if self.dt_rank is not None and (
            not isinstance(self.dt_rank, numbers.Integral)
            or isinstance(self.dt_rank, bool)
            or self.dt_rank < 1
        ):
            raise ValueError(f"dt_rank must be None or an integer >= 1, got {self.dt_rank!r}.")
        if not isinstance(self.dropout, numbers.Real) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be a real number in [0, 1), got {self.dropout!r}.")
        return n_threads

"""XSimGCL: a LightGCN backbone whose contrastive views are two of its own layers."""

import numbers
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from torch import nn

from skrecsys._typing import override
from skrecsys.nn._base import TorchRecommender, TorchRecommenderModule, seeded_normal_

#: Redraw rounds for a negative that turned out to be one of the user's own items. The
#: reference redraws until the draw is valid, which cannot terminate for a user who has
#: interacted with the whole catalog; a triplet still clashing after this many rounds is
#: dropped from its batch instead, as our own BPR drops one rather than redrawing it.
_RESAMPLE_ROUNDS = 16


def normalized_adjacency(interactions: sp.csr_array) -> torch.Tensor:
    """The symmetrically normalized bipartite adjacency, as a sparse float32 tensor.

    Users occupy the first ``n_users`` rows and items the rest, so one propagation moves
    item information onto users and user information onto items. Only the structure of
    ``interactions`` is used -- XSimGCL is an implicit-feedback model -- so an entry is 1
    however often or however highly the pair was rated, and a node's degree is its number
    of distinct partners.
    """
    n_users, n_items = interactions.shape
    coo = interactions.tocoo()
    rows = np.concatenate([coo.coords[0], coo.coords[1] + n_users])
    cols = np.concatenate([coo.coords[1] + n_users, coo.coords[0]])

    degree = np.bincount(rows, minlength=n_users + n_items).astype(np.float64)
    scale = np.zeros_like(degree)
    # An isolated node keeps a zero scale, which leaves its row of the product at zero
    # rather than dividing by a degree of nothing.
    np.divide(1.0, np.sqrt(degree), out=scale, where=degree > 0)

    indices = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
    values = torch.from_numpy((scale[rows] * scale[cols]).astype(np.float32))
    size = (n_users + n_items, n_users + n_items)
    # Opting in explicitly rather than inheriting torch's warning about the default.
    # The graph is built once per fit, so one pass over nnz is free next to training.
    tensor = torch.sparse_coo_tensor(indices, values, size, check_invariants=True)
    return tensor.coalesce()


def info_nce(view1: torch.Tensor, view2: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE between two views of the same rows, with every other row as a negative."""
    view1 = torch.nn.functional.normalize(view1, dim=1)
    view2 = torch.nn.functional.normalize(view2, dim=1)
    positive = (view1 * view2).sum(dim=-1) / temperature
    total = (view1 @ view2.T / temperature).logsumexp(dim=1)
    return (total - positive).mean()


class _XSimGCLModule(TorchRecommenderModule):
    """Perturbed LightGCN propagation, the BPR objective and the cross-layer contrast."""

    #: Registered buffers. ``nn.Module.__getattr__`` erases their type, so they are
    #: declared here to stay tensors for a type checker.
    adjacency: torch.Tensor
    pair_users: torch.Tensor
    pair_items: torch.Tensor
    stored_pairs: torch.Tensor

    #: Neither table is padded here: both are plain code spaces.
    embedding_axes: ClassVar[dict[str, str]] = {
        "user_embedding.weight": "user",
        "item_embedding.weight": "item",
    }

    def __init__(
        self,
        estimator: "XSimGCL",
        adjacency: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
        n_users: int,
        n_items: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        n_factors = int(estimator.n_factors)
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = int(estimator.n_layers)
        self.contrastive_layer = int(estimator.contrastive_layer)
        self.contrastive_weight = float(estimator.contrastive_weight)
        self.temperature = float(estimator.temperature)
        self.eps = float(estimator.eps)
        self.regularization = float(estimator.regularization)

        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)

        std = 1.0 / np.sqrt(n_factors)
        seeded_normal_(self.user_embedding.weight, generator, std)
        seeded_normal_(self.item_embedding.weight, generator, std)

        # Not a buffer or a parameter: it drives the perturbation and the sampling, and
        # it is gone with the module once the fit ends.
        self.generator = generator

        self.register_buffer("adjacency", adjacency)
        self.register_buffer("pair_users", users)
        self.register_buffer("pair_items", items)
        # One sorted key per observed pair, so a drawn negative can be rejected with a
        # binary search instead of a scan of the user's row.
        self.register_buffer("stored_pairs", torch.sort(users * n_items + items).values)

    def propagate(self, *, perturbed: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the layer-averaged embeddings and the contrastive layer's own.

        The perturbation is the whole of XSimGCL: a uniform noise vector of fixed length
        ``eps``, turned to point the same way as the embedding it is added to, so it moves
        a representation without flipping what it encodes. The two contrastive views are
        then two layers of this one pass rather than two extra forward passes, which is
        what makes the model cheaper than the SimGCL it simplifies.
        """
        embeddings: torch.Tensor = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight]
        )
        contrastive: torch.Tensor = embeddings
        layers: list[torch.Tensor] = []
        for layer in range(1, self.n_layers + 1):
            # `@` dispatches to the same sparse-dense kernel as `torch.sparse.mm`,
            # and unlike it the operator is typed as returning a tensor.
            embeddings = self.adjacency @ embeddings
            if perturbed:
                noise = torch.rand(
                    embeddings.shape, generator=self.generator, dtype=embeddings.dtype
                )
                noise = torch.nn.functional.normalize(noise, dim=-1).to(embeddings.device)
                embeddings = embeddings + torch.sign(embeddings) * noise * self.eps
            layers.append(embeddings)
            if layer == self.contrastive_layer:
                contrastive = embeddings
        # The mean is over the propagated layers only, as the reference takes it; the
        # input embeddings enter it through the first layer rather than on their own.
        return torch.stack(layers, dim=1).mean(dim=1), contrastive

    def _split(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return embeddings[: self.n_users], embeddings[self.n_users :]

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
            rows = order[start : start + batch_size].to(device)
            users, items = self.pair_users[rows], self.pair_items[rows]
            negatives, usable = self._draw_negatives(users, generator)
            if not bool(usable.all()):
                users, items, negatives = users[usable], items[usable], negatives[usable]
            if len(users):
                yield users, items, negatives

    def _draw_negatives(
        self, users: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A uniform item per user, and which of them the user had not interacted with.

        Membership is a binary search into the sorted pair keys, so rejecting a draw costs
        a logarithm of the interaction count rather than a scan of the user's row.
        """
        device = users.device
        negatives = torch.randint(
            self.n_items, (len(users),), generator=generator, device="cpu"
        ).to(device)
        clash = torch.isin(users * self.n_items + negatives, self.stored_pairs)
        for _ in range(_RESAMPLE_ROUNDS):
            if not bool(clash.any()):
                break
            redrawn = torch.randint(
                self.n_items, (int(clash.sum()),), generator=generator, device="cpu"
            )
            negatives = negatives.clone()
            negatives[clash] = redrawn.to(device)
            clash = torch.isin(users * self.n_items + negatives, self.stored_pairs)
        return negatives, ~clash

    @override
    def batch_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        users, items, negatives = batch
        ranking, contrastive = self.propagate(perturbed=True)
        user_view, item_view = self._split(ranking)

        user_emb = user_view[users]
        positive = (user_emb * item_view[items]).sum(dim=1)
        negative = (user_emb * item_view[negatives]).sum(dim=1)
        loss = torch.nn.functional.softplus(negative - positive).mean()

        contrastive_users, contrastive_items = self._split(contrastive)
        unique_users, unique_items = torch.unique(users), torch.unique(items)
        loss = loss + self.contrastive_weight * (
            info_nce(user_view[unique_users], contrastive_users[unique_users], self.temperature)
            + info_nce(item_view[unique_items], contrastive_items[unique_items], self.temperature)
        )

        # The reference penalizes the embeddings the batch touched, scaled by the batch
        # size, rather than decaying every parameter on every step; see `_weight_decay`.
        touched = (user_emb, item_view[items], item_view[negatives])
        penalty = torch.stack([one.norm(p=2) / one.shape[0] for one in touched]).sum()
        return loss + self.regularization * penalty


class XSimGCL(TorchRecommender[_XSimGCLModule]):
    """Graph collaborative filtering with a contrast between two of its own layers [1]_.

    The backbone is LightGCN [2]_: user and item embeddings are propagated over the
    symmetrically normalized bipartite interaction graph for ``n_layers`` steps and
    averaged, and a pair scores as the dot product of the results. On top of that,
    XSimGCL adds a contrastive objective whose two views cost nothing extra.

    Every propagation step during training perturbs its output with a uniform noise
    vector of fixed length ``eps``, aligned with the sign of what it is added to. The
    ranking view is the layer average of that perturbed pass and the contrastive view is
    its ``contrastive_layer``-th layer, so a single forward pass yields both views. This
    is the simplification the paper is named for: SimGCL, which it comes from, runs two
    extra perturbed passes per step to get the same contrast.

    The objective is the BPR ranking loss plus ``contrastive_weight`` times the InfoNCE
    agreement of the two views, over the distinct users and items of the batch. Training
    therefore spreads representations over the hypersphere while fitting the ranking,
    which is what the paper credits for the gain rather than the graph augmentation that
    earlier contrastive recommenders relied on.

    Interaction values are ignored: only which pairs are observed matters, both in the
    graph and in the triplets, which makes this an implicit-feedback model like
    :class:`~skrecsys.recommendation.BayesianPersonalizedRanking`.

    Training needs PyTorch (``pip install skrecsys[nn]``); a fitted estimator does not.
    ``fit`` propagates once more without noise and exports the resulting embeddings to
    numpy, so scoring is the same dot product every factor model here does.

    Parameters
    ----------
    n_factors : int, default=64
        Dimensionality of the user and item embeddings.
    n_layers : int, default=3
        Propagation steps. Each one widens a node's neighbourhood by one hop.
    contrastive_weight : float, default=0.2
        Weight of the InfoNCE term against the ranking loss. The paper tunes it per
        dataset over roughly ``0.005`` to ``0.2``; it is the parameter to move first.
    contrastive_layer : int, default=1
        Which propagated layer supplies the contrastive view. ``0`` contrasts against the
        embeddings before any propagation. Must not exceed ``n_layers``.
    temperature : float, default=0.2
        Temperature of the InfoNCE term. Lower concentrates the gradient on the negatives
        that are hardest to tell from the positive.
    eps : float, default=0.2
        Length of the noise vector added after each propagation step. ``0.0`` trains a
        plain LightGCN with a contrast between two of its layers.
    learning_rate : float, default=1e-3
        Adam step size.
    regularization : float, default=1e-4
        L2 penalty on the embeddings each batch touched, as the reference applies it,
        rather than weight decay over every parameter.
    max_iter : int, default=100
        Number of epochs. Each one visits every observed interaction once.
    batch_size : int, default=2048
        Triplets per gradient step. It also sets how many rows the InfoNCE term contrasts
        against each other, so it is not a pure speed knob.
    device : str or torch.device, default="cpu"
        Where to train. ``"auto"`` picks CUDA, then MPS, then CPU; anything else is handed
        to ``torch.device`` unchanged.
    random_state : int, RandomState instance or None, default=None
        Seed of the initialization, the epoch shuffling, the negative sampling and the
        perturbation. A seeded fit on one thread is reproducible; a backend's own
        nondeterminism, such as reduction order on a GPU, is not controlled here.
    n_jobs : int or None, default=None
        Intra-op threads for the fit. ``None`` and ``-1`` leave torch's own default. Any
        other value caps ``torch.set_num_threads`` for the duration of the fit, which is
        process-global while it lasts.
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
        Layer-averaged user embeddings of a propagation without noise.
    item_factors_ : ndarray of shape (n_items_, n_factors)
        Layer-averaged item embeddings of the same propagation.
    loss_curve_ : ndarray of shape (n_iter_,)
        Mean training loss of each epoch, ranking and contrastive terms together.
    n_iter_ : int
        Epochs actually run: ``max_iter`` unless the fit stopped early.
    best_loss_ : float
        The lowest epoch loss reached, or nan when ``max_iter=0``.

    References
    ----------
    .. [1] J. Yu, X. Xia, T. Chen, L. Cui, N. Q. V. Hung, and H. Yin, "XSimGCL: Towards
       Extremely Simple Graph Contrastive Learning for Recommendation", TKDE 2023.
       https://arxiv.org/abs/2209.02544
    .. [2] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, "LightGCN: Simplifying
       and Powering Graph Convolution Network for Recommendation", SIGIR 2020.
       https://arxiv.org/abs/2002.02126

    Examples
    --------
    >>> from skrecsys.nn import XSimGCL
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = XSimGCL(n_factors=4, n_layers=1, max_iter=20, random_state=0).fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(
        self,
        n_factors: int = 64,
        n_layers: int = 3,
        contrastive_weight: float = 0.2,
        contrastive_layer: int = 1,
        temperature: float = 0.2,
        eps: float = 0.2,
        learning_rate: float = 1e-3,
        regularization: float = 1e-4,
        max_iter: int = 100,
        batch_size: int = 2048,
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
        self.n_layers = n_layers
        self.contrastive_weight = contrastive_weight
        self.contrastive_layer = contrastive_layer
        self.temperature = temperature
        self.eps = eps
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
    def _weight_decay(self) -> float:
        """Zero: the penalty is applied to the batch's own embeddings in ``batch_loss``."""
        return 0.0

    @override
    def _build_module(
        self,
        interactions: sp.csr_array,
        device: torch.device,
        rng: np.random.RandomState,
        generator: torch.Generator,
    ) -> _XSimGCLModule:
        n_users, n_items = interactions.shape
        users = torch.from_numpy(
            np.repeat(np.arange(n_users, dtype=np.int64), np.diff(interactions.indptr))
        )
        items = torch.from_numpy(interactions.indices.astype(np.int64, copy=False))
        module = _XSimGCLModule(
            self,
            normalized_adjacency(interactions),
            users,
            items,
            n_users,
            n_items,
            generator,
        )
        return module.to(device)

    @override
    def _export(self, module: _XSimGCLModule) -> None:
        ranking, _ = module.propagate(perturbed=False)
        users, items = ranking[: self.n_users_], ranking[self.n_users_ :]
        self.user_factors_ = np.ascontiguousarray(users.cpu().numpy(), dtype=np.float64)
        self.item_factors_ = np.ascontiguousarray(items.cpu().numpy(), dtype=np.float64)

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
        if not isinstance(self.n_layers, numbers.Integral) or isinstance(self.n_layers, bool):
            raise ValueError(f"n_layers must be an integer >= 1, got {self.n_layers!r}.")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be an integer >= 1, got {self.n_layers!r}.")
        if (
            not isinstance(self.contrastive_layer, numbers.Integral)
            or isinstance(self.contrastive_layer, bool)
            or not 0 <= self.contrastive_layer <= self.n_layers
        ):
            raise ValueError(
                f"contrastive_layer must be an integer in [0, n_layers], "
                f"got {self.contrastive_layer!r}."
            )
        for name in ("contrastive_weight", "eps"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not value >= 0:
                raise ValueError(f"{name} must be a real number >= 0, got {value!r}.")
        if not isinstance(self.temperature, numbers.Real) or not self.temperature > 0:
            raise ValueError(f"temperature must be a real number > 0, got {self.temperature!r}.")
        return n_threads

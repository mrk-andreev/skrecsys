"""Shared machinery for recommenders fitted with PyTorch.

The estimators here are ordinary :class:`~skrecsys.recommendation._base.BaseRecommender`
subclasses: they take the same ``(n_interactions, 2)`` identifier array, expose the same
``fit`` / ``predict`` / ``recommend``, and differ only in how the parameters behind
``_score_users`` are obtained. Training happens inside ``_fit``, which leaves nothing
torch-typed on the estimator: the arrays that score are copied out by ``_export``, and
the arrays that go on learning -- the parameters and the optimizer's moments -- are
copied out beside them. A fitted model therefore pickles, scores and resumes training
without torch being involved in any of it.
"""

import math
import numbers
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from skrecsys._typing import override
from skrecsys.indexing import DenseSpace
from skrecsys.recommendation._base import BaseRecommender
from skrecsys.recommendation._incremental import IncrementalRecommenderMixin

__all__ = ["ModuleT", "TorchRecommender", "TorchRecommenderModule"]

#: Where a growing vocabulary leaves a parameter whose rows it indexes.
#: ``"user"`` and ``"item"`` are plain code spaces; the two padded kinds carry
#: one extra row, for the padding slot, at the end or at the front.
EMBEDDING_AXES = ("user", "item", "item_pad_last", "item_pad_first")

#: What ``learning_rate_schedule="adaptive"`` divides the step size by when the loss
#: stalls, and the floor it stops dividing at. Both are scikit-learn's values for the
#: same schedule in ``MLPRegressor``: five is aggressive enough that a handful of cuts
#: covers three orders of magnitude, and below the floor an epoch cannot move the
#: parameters enough to matter, so there is nothing left to wait for.
LEARNING_RATE_DECAY = 5.0
MIN_LEARNING_RATE = 1e-6

#: The schedules ``learning_rate_schedule`` accepts.
LEARNING_RATE_SCHEDULES = ("constant", "adaptive")


class TorchRecommenderModule(torch.nn.Module):
    """The trainable half of a :class:`TorchRecommender`.

    The module owns every tensor a fit touches -- parameters, buffers and the training
    pairs themselves -- so the estimator can drop it once the fit is over and keep nothing
    that would need torch to unpickle.
    """

    #: Parameter name to the code space that indexes its rows, one of
    #: :data:`EMBEDDING_AXES`. Only the parameters named here are relabelled and grown
    #: when a vocabulary does; every other parameter keeps its shape and is copied
    #: across unchanged.
    embedding_axes: ClassVar[dict[str, str]] = {}

    #: Set by :meth:`set_training_subset`; ``None`` means every row is in play.
    _train_rows: "torch.Tensor | None" = None

    def iter_batches(self, batch_size: int, generator: torch.Generator) -> Iterator[Any]:
        """Yield one epoch of training batches, reshuffled on every call.

        A batch is a tuple whose first element has one row per training example, which is
        what weights its loss in :attr:`TorchRecommender.loss_curve_`.
        """
        raise NotImplementedError

    def batch_loss(self, batch: Any) -> torch.Tensor:
        """Return the mean loss of one batch of :meth:`iter_batches`."""
        raise NotImplementedError

    def set_training_subset(
        self, users: torch.Tensor, pair_users: torch.Tensor, pair_items: torch.Tensor
    ) -> None:
        """Train the next epochs on one batch rather than on the whole history.

        ``users`` are the users the batch mentions and ``pair_users`` / ``pair_items``
        its interactions, all in the grown code space. A module whose examples are
        interactions replaces its pair buffers; one whose examples are users -- a
        sequence model -- narrows the rows it draws, which is what the default does.
        """
        del pair_users, pair_items
        self._train_rows = users

    def training_rows(self, total: int) -> torch.Tensor:
        """The rows the next epoch draws from: every one, unless a batch narrowed them."""
        rows = self._train_rows
        return torch.arange(total) if rows is None else rows


#: The module a :class:`TorchRecommender` subclass trains, so its hooks see the real type.
ModuleT = TypeVar("ModuleT", bound=TorchRecommenderModule)


class TorchRecommender(IncrementalRecommenderMixin, BaseRecommender, Generic[ModuleT]):
    """Base class for recommenders trained by minibatch gradient descent.

    Subclasses implement ``_build_module`` and ``_export``, plus the ``_score_users`` and
    ``_score_pairs`` hooks of :class:`~skrecsys.recommendation._base.BaseRecommender`.
    The training loop, device selection, seeding and thread control live here.

    Fitted attributes
    -----------------
    loss_curve_ : ndarray of shape (n_iter_,)
        Mean training loss of each epoch, weighted by batch size. It describes the
        parameters as they moved through the epoch rather than the ones the epoch ended
        on, so the curve lags the model by one epoch.
    n_iter_ : int
        Epochs actually run, which is ``max_iter`` unless the fit stopped early.
    best_loss_ : float
        The lowest epoch loss reached, or nan when ``max_iter=0`` left nothing to run.

    Notes
    -----
    ``partial_fit`` resumes the descent from where the last call left it, over the batch
    rather than over the whole history, so a user the batch does not mention is not
    updated and the model drifts towards recent behaviour. The stall counter and the
    adaptive step size start afresh on each call, which is what ``MLPRegressor`` does for
    the same reason: a batch is a new descent, not a continuation of the last schedule.

    What carries across is ``_training_state_``: the trained parameters and Adam's
    moments, held as numpy arrays rather than as a live module. The module itself is
    rebuilt from the grown interaction matrix on every call, because the padded
    histories, normalized adjacency and windowed sequences it holds are functions of that
    matrix and change meaning the moment the catalog grows.
    """

    #: Constructor parameters every subclass takes, declared here because the training
    #: loop reads them. They are annotations only: sklearn builds `get_params` from the
    #: subclass's own `__init__` signature, and a class attribute would shadow it.
    n_factors: int
    max_iter: int
    batch_size: int
    learning_rate: float
    learning_rate_schedule: str
    early_stopping: bool
    tol: float
    n_iter_no_change: int
    regularization: float
    device: "str | torch.device"
    random_state: "int | np.random.RandomState | None"
    n_jobs: int | None

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        device = self._resolve_device()
        rng = check_random_state(self.random_state)
        self._rng = rng
        # A CPU generator drives every draw, so the sampled indices of a seeded fit do not
        # depend on the device they are later copied to, and torch's global RNG is left
        # alone the way `check_random_state` leaves numpy's.
        generator = torch.Generator()
        generator.manual_seed(int(rng.randint(2**31)))

        with self._thread_limit(n_threads):
            module = self._build_module(interactions, device, rng, generator)
            optimizer = self._make_optimizer(module)
            self.loss_curve_ = self._train(module, optimizer, generator)
            self._finish(module, optimizer)

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
        """Rebuild the module around the grown data, carrying the training across.

        The module is rebuilt rather than patched because everything it holds besides its
        parameters -- padded histories, a normalized adjacency, windowed sequences, the
        composite keys one of them encodes pairs with -- is a function of the interaction
        matrix, and half of it changes meaning the moment the catalog grows.
        ``_build_module`` already knows how to derive all of it; what has to survive is
        the learning, which is the parameters and the optimizer's moments -- and that is
        exactly what ``_training_state_`` holds, as numpy, from the last call.
        """
        del new_user_indices, new_item_indices, touched_item_indices
        n_threads = self._check_params()
        device = self._resolve_device()
        rng = self._incremental_rng()
        generator = torch.Generator()
        generator.manual_seed(int(rng.randint(2**31)))

        with self._thread_limit(n_threads):
            learned, moments = self._training_state_
            user_perm, item_perm = self.__dict__.pop("_pending_perm", (None, None))
            module = self._build_module(interactions, device, rng, generator)
            carry_parameters(learned, module, user_perm, item_perm)
            optimizer = self._make_optimizer(module)
            carry_optimizer_state(moments, module, optimizer, user_perm, item_perm)
            rows = np.repeat(np.arange(delta.shape[0]), np.diff(delta.indptr))
            module.set_training_subset(
                torch.from_numpy(np.ascontiguousarray(touched_user_indices, dtype=np.int64)),
                torch.from_numpy(rows.astype(np.int64, copy=False)),
                torch.from_numpy(delta.indices.astype(np.int64, copy=False)),
            )
            # The curve is the whole training history, so a caller can see every batch.
            self.loss_curve_ = np.concatenate(
                [self.loss_curve_, self._train(module, optimizer, generator)]
            )
            self._finish(module, optimizer)

    @override
    def _remap(
        self,
        *,
        user_perm: NDArray[np.intp],
        item_perm: NDArray[np.intp],
        n_users: int,
        n_items: int,
    ) -> None:
        super()._remap(user_perm=user_perm, item_perm=item_perm, n_users=n_users, n_items=n_items)
        # The module is rebuilt from the grown matrix rather than relabelled, so what it
        # needs is not the new arrays but the permutation its embedding rows moved by.
        # `_partial_fit` takes it from here and drops it.
        self._pending_perm = (user_perm, item_perm)

    def _finish(self, module: ModuleT, optimizer: torch.optim.Optimizer) -> None:
        """Record what the epochs reached, and keep what a later batch will need.

        Both halves leave torch behind. ``_export`` copies out the arrays that score;
        ``_training_state_`` copies out the arrays that go on learning. The module and
        the optimizer are then let go, because everything else they hold is derived from
        the interaction matrix and ``_build_module`` rebuilds it on the next call anyway.
        Keeping numpy rather than the module is what makes a fitted model picklable,
        and resumable after it is unpickled, with torch nowhere in the file.
        """
        self.n_iter_ = len(self.loss_curve_)
        self.best_loss_ = float(self.loss_curve_.min()) if self.n_iter_ else float("nan")
        module.eval()
        with torch.no_grad():
            self._export(module)
        self._training_state_ = training_state(module, optimizer)

    def _make_optimizer(self, module: TorchRecommenderModule) -> torch.optim.Optimizer:
        """The optimizer a fit descends with, built here so a later batch can reuse it."""
        return torch.optim.Adam(
            module.parameters(),
            lr=float(self.learning_rate),
            weight_decay=self._weight_decay(),
        )

    def _train(
        self,
        module: TorchRecommenderModule,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> np.ndarray:
        """Run at most ``max_iter`` epochs of Adam and return the loss of each one.

        An epoch counts as an improvement when it beats the best loss so far by at least
        ``tol``; ``n_iter_no_change`` consecutive epochs that do not is what "the loss has
        stopped moving" means here. What happens then is the schedule's business:
        ``learning_rate_schedule="adaptive"`` divides the step size by five and gives the
        fit that many epochs again, and only once the step size has bottomed out does
        ``early_stopping`` end the fit. The two are one mechanism on purpose -- a fit
        that stalls has either not finished descending, which a smaller step fixes, or it
        has, which is exactly when stopping is right -- and it is the arrangement
        scikit-learn's ``MLPRegressor`` uses for the same reason.

        The signal is the *training* loss, so this stops a fit that has converged rather
        than one that has begun to overfit; catching that would need a validation split,
        which would change what ``fit`` is given.
        """
        batch_size = int(self.batch_size)
        tol = float(self.tol)
        patience = int(self.n_iter_no_change)
        adaptive = self.learning_rate_schedule == "adaptive"
        learning_rate = float(self.learning_rate)

        losses: list[float] = []
        best = np.inf
        stalled = 0
        module.train()
        for _ in range(int(self.max_iter)):
            total, seen = 0.0, 0
            for batch in module.iter_batches(batch_size, generator):
                optimizer.zero_grad(set_to_none=True)
                loss = module.batch_loss(batch)
                loss.backward()
                optimizer.step()
                size = len(batch[0])
                total += float(loss.detach()) * size
                seen += size
            epoch_loss = total / seen if seen else 0.0
            losses.append(epoch_loss)

            stalled = 0 if epoch_loss < best - tol else stalled + 1
            best = min(best, epoch_loss)
            if stalled < patience:
                continue
            if adaptive and learning_rate > MIN_LEARNING_RATE:
                learning_rate = max(learning_rate / LEARNING_RATE_DECAY, MIN_LEARNING_RATE)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            elif self.early_stopping:
                break
            # Either the step size just shrank, or a constant schedule that is not allowed
            # to stop has nothing to do about the stall. Both get a fresh count: the first
            # so the smaller step is given its own `n_iter_no_change` epochs to work, the
            # second so the stall is not re-triggered on every remaining epoch.
            stalled = 0
        return np.asarray(losses, dtype=np.float64)

    def _weight_decay(self) -> float:
        """How ``regularization`` reaches the optimizer, as Adam's ``weight_decay``.

        A subclass whose reference penalizes something narrower -- the embeddings a batch
        touched rather than every parameter -- returns 0.0 here and applies its own term
        inside ``batch_loss`` instead.
        """
        return float(self.regularization)

    def _build_module(
        self,
        interactions: sp.csr_array,
        device: torch.device,
        rng: np.random.RandomState,
        generator: torch.Generator,
    ) -> ModuleT:
        """Build the module to train, already on ``device`` and already initialized."""
        raise NotImplementedError

    #: Exported by ``_export``, and what the default index hooks below read. A subclass
    #: that names its exported vectors differently overrides both hooks; see
    #: :class:`~skrecsys.nn._sequential.SequentialRecommender`.
    user_factors_: NDArray[np.float64]
    item_factors_: NDArray[np.float64]

    @override
    def _index_space(self) -> DenseSpace:
        # The neural models score by a plain dot product of exported factors, so their
        # space is the factors themselves -- no bias to fold in, nothing to reconstruct.
        return DenseSpace(np.ascontiguousarray(self.item_factors_, dtype=np.float64))

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64]:
        return np.ascontiguousarray(self.user_factors_[user_indices], dtype=np.float64)

    def _export(self, module: ModuleT) -> None:
        """Copy what scoring needs out of ``module`` into numpy fitted attributes."""
        raise NotImplementedError

    def _resolve_device(self) -> torch.device:
        """The device to train on.

        ``"cpu"`` is the default because it is the only backend every wheel has and the
        only one the test matrix can rely on. ``"auto"`` picks the fastest backend that is
        actually available, and anything else is handed to ``torch.device`` unchanged, so
        ``"cuda:1"``, ``"mps"``, ``"xpu"`` or a device object all work.
        """
        device = self.device
        if isinstance(device, torch.device):
            return device
        if not isinstance(device, str):
            raise ValueError(f"device must be a string or torch.device, got {device!r}.")
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            resolved = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"device {device!r} is not a valid torch device.") from exc
        # An unavailable backend fails deep inside the first allocation otherwise, with a
        # message that says nothing about which parameter caused it.
        try:
            torch.zeros(1, device=resolved)
        except (AssertionError, RuntimeError) as exc:
            raise ValueError(f"device {device!r} is not available on this machine.") from exc
        return resolved

    @contextmanager
    def _thread_limit(self, n_threads: int) -> Iterator[None]:
        """Cap torch's intra-op threads for the fit, then restore the previous cap.

        ``torch.set_num_threads`` is process-global, so the cap is in force for whatever
        else the process is doing meanwhile; ``n_jobs=None`` avoids touching it at all.
        """
        if n_threads <= 0:
            yield
            return
        previous = torch.get_num_threads()
        torch.set_num_threads(n_threads)
        try:
            yield
        finally:
            torch.set_num_threads(previous)

    def _check_torch_params(self) -> int:
        """Validate the parameters every torch estimator has; return the thread cap.

        Zero means "leave torch's own default alone", which is what ``n_jobs=None`` and
        ``n_jobs=-1`` both ask for: torch already spreads across every core.
        """
        for name in ("n_factors", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        if (
            not isinstance(self.max_iter, numbers.Integral)
            or isinstance(self.max_iter, bool)
            or self.max_iter < 0
        ):
            raise ValueError(f"max_iter must be an integer >= 0, got {self.max_iter!r}.")
        if not isinstance(self.learning_rate, numbers.Real) or not self.learning_rate > 0:
            raise ValueError(
                f"learning_rate must be a real number > 0, got {self.learning_rate!r}."
            )
        if not isinstance(self.regularization, numbers.Real) or not self.regularization >= 0:
            raise ValueError(
                f"regularization must be a real number >= 0, got {self.regularization!r}."
            )
        self._check_stopping_params()
        self._resolve_device()
        if self.n_jobs is None or self.n_jobs == -1:
            return 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)

    def _check_stopping_params(self) -> None:
        """Validate the parameters that decide when a fit ends and how the step size moves."""
        if self.learning_rate_schedule not in LEARNING_RATE_SCHEDULES:
            raise ValueError(
                f"learning_rate_schedule must be one of {LEARNING_RATE_SCHEDULES}, "
                f"got {self.learning_rate_schedule!r}."
            )
        if not isinstance(self.early_stopping, bool):
            raise ValueError(f"early_stopping must be a boolean, got {self.early_stopping!r}.")
        if (
            not isinstance(self.tol, numbers.Real)
            or isinstance(self.tol, bool)
            or not math.isfinite(float(self.tol))
            or self.tol < 0
        ):
            raise ValueError(f"tol must be a finite real number >= 0, got {self.tol!r}.")
        if (
            not isinstance(self.n_iter_no_change, numbers.Integral)
            or isinstance(self.n_iter_no_change, bool)
            or self.n_iter_no_change < 1
        ):
            raise ValueError(
                f"n_iter_no_change must be an integer >= 1, got {self.n_iter_no_change!r}."
            )

    def _check_params(self) -> int:
        """Validate every parameter and return the thread cap for the fit."""
        return self._check_torch_params()


def growth_rows(
    axis: str,
    rows: int,
    old_rows: int,
    user_perm: NDArray[np.intp] | None,
    item_perm: NDArray[np.intp] | None,
) -> torch.Tensor:
    """Where each row of an old parameter lands in the grown one.

    Two things move a row. The identifiers are kept sorted, so an identifier that does
    not sort last renumbers the codes handed out before it, and the permutation says
    where each one went. And a padded embedding keeps its padding slot at one end while
    the catalog grows into the other: ``"item_pad_last"`` carries a trailing padding row
    past every new item, and ``"item_pad_first"`` keeps it at the front with every real
    item one place along.
    """
    if axis == "user":
        return _as_perm(user_perm, old_rows)
    if axis == "item":
        return _as_perm(item_perm, old_rows)
    if axis == "item_pad_last":
        items = _as_perm(item_perm, old_rows - 1)
        return torch.cat([items, torch.tensor([rows - 1])])
    if axis == "item_pad_first":
        items = _as_perm(item_perm, old_rows - 1)
        return torch.cat([torch.zeros(1, dtype=items.dtype), items + 1])
    raise ValueError(f"Unknown embedding axis {axis!r}; it must be one of {EMBEDDING_AXES}.")


def _as_perm(perm: NDArray[np.intp] | None, length: int) -> torch.Tensor:
    """``perm`` as a tensor, or the identity when no identifier moved."""
    if perm is None:
        return torch.arange(length)
    return torch.from_numpy(np.ascontiguousarray(perm, dtype=np.int64))


#: The numpy form of everything a later batch needs: the trained parameters by name,
#: and Adam's moments and step count beside them.
TrainingState = tuple[
    dict[str, NDArray[np.float64]],
    dict[str, dict[str, "NDArray[np.float64] | float"]],
]


def training_state(
    module: TorchRecommenderModule, optimizer: torch.optim.Optimizer
) -> TrainingState:
    """Copy the trained parameters and Adam's moments out of torch, as numpy.

    This is what a fitted model keeps in place of the module. Holding it as numpy is not
    a detail: it is what makes the estimator picklable without torch, and therefore what
    makes training resumable after it is unpickled on a machine that has no torch until
    the moment it wants to train again.
    """
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    learned = {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in module.named_parameters()
    }
    moments: dict[str, dict[str, Any]] = {}
    for parameter, state in optimizer.state.items():
        name = names.get(id(parameter))
        if name is None:
            continue
        moments[name] = {
            "step": float(state["step"]),
            "exp_avg": state["exp_avg"].detach().cpu().numpy().copy(),
            "exp_avg_sq": state["exp_avg_sq"].detach().cpu().numpy().copy(),
        }
    return learned, moments


def carry_parameters(
    learned: dict[str, NDArray[np.float64]],
    module: TorchRecommenderModule,
    user_perm: NDArray[np.intp] | None,
    item_perm: NDArray[np.intp] | None,
) -> None:
    """Load what the last call learned into the freshly built ``module``.

    A parameter whose shape is unchanged is loaded outright, which is safe because only
    a new identifier can change a code and a new identifier also changes the row count.
    One that grew keeps the rows it had, relabelled, and leaves the new ones at the
    initialization ``module`` drew for them -- the same distribution a fresh fit uses.
    """
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            stored = learned.get(name)
            if stored is None:
                continue
            source = torch.from_numpy(stored).to(parameter.dtype)
            if source.shape == parameter.shape:
                parameter.copy_(source.to(parameter.device))
                continue
            axis = module.embedding_axes.get(name)
            if axis is None:
                raise ValueError(
                    f"{type(module).__name__} parameter {name!r} changed shape from "
                    f"{tuple(source.shape)} to {tuple(parameter.shape)} without naming "
                    "the code space that indexes it in `embedding_axes`."
                )
            rows = growth_rows(axis, parameter.shape[0], source.shape[0], user_perm, item_perm)
            parameter[rows.to(parameter.device)] = source.to(parameter.device)


def carry_optimizer_state(
    moments: dict[str, dict[str, Any]],
    module: TorchRecommenderModule,
    optimizer: torch.optim.Optimizer,
    user_perm: NDArray[np.intp] | None,
    item_perm: NDArray[np.intp] | None,
) -> None:
    """Load Adam's moments back onto the rebuilt module's parameters.

    Adam keys its state by the parameter object, so a rebuilt module starts with none of
    it and every batch would take its first steps as if from a cold start -- which with
    a zero second moment means steps as large as the step size allows, right where the
    model was already converged. The moments are matched by name, grown the way the
    parameters were, and ``step`` is carried over so the bias correction keeps counting.
    """
    for name, parameter in module.named_parameters():
        state = moments.get(name)
        if state is None:
            continue
        carried: dict[str, Any] = {"step": torch.tensor(float(state["step"]))}
        for moment in ("exp_avg", "exp_avg_sq"):
            source = torch.from_numpy(state[moment]).to(parameter.dtype)
            if source.shape == parameter.shape:
                carried[moment] = source.to(parameter.device)
                continue
            grown = torch.zeros_like(parameter)
            axis = module.embedding_axes[name]
            rows = growth_rows(axis, parameter.shape[0], source.shape[0], user_perm, item_perm)
            grown[rows.to(grown.device)] = source.to(grown.device)
            carried[moment] = grown
        optimizer.state[parameter] = carried


def seeded_normal_(tensor: torch.Tensor, generator: torch.Generator, std: float) -> None:
    """Fill ``tensor`` in place with ``N(0, std)`` drawn from ``generator``.

    Torch's own initializers read the global RNG, which would make a seeded fit depend on
    whatever else the process drew first.
    """
    with torch.no_grad():
        draws = torch.empty(tensor.shape, dtype=tensor.dtype)
        draws.normal_(0.0, std, generator=generator)
        tensor.copy_(draws)


def seeded_linear_(linear: torch.nn.Linear, generator: torch.Generator) -> None:
    """Re-initialize ``linear`` the way torch does, but from ``generator``."""
    bound = 1.0 / np.sqrt(linear.in_features)
    with torch.no_grad():
        weight = torch.empty(linear.weight.shape, dtype=linear.weight.dtype)
        weight.uniform_(-bound, bound, generator=generator)
        linear.weight.copy_(weight)
        # The stubs type `bias` as a Parameter, but a `bias=False` layer really does
        # hold None there, which is how the attention aggregators build theirs.
        if linear.bias is not None:  # ty: ignore[redundant-condition-strict]
            bias = torch.empty(linear.bias.shape, dtype=linear.bias.dtype)
            bias.uniform_(-bound, bound, generator=generator)
            linear.bias.copy_(bias)


def seeded_dropout(
    x: torch.Tensor, p: float, generator: torch.Generator, *, training: bool
) -> torch.Tensor:
    """Inverted dropout drawing its mask from ``generator`` rather than the global RNG.

    ``torch.nn.Dropout`` reads torch's process-global RNG, which would make a seeded fit
    depend on whatever else the process drew first. The mask is drawn on the generator's
    own device and moved, which costs a copy per call on an accelerator and nothing on
    the CPU default.
    """
    if not training or p <= 0.0:
        return x
    keep = torch.empty(x.shape, dtype=x.dtype)
    keep.bernoulli_(1.0 - p, generator=generator)
    return x * keep.to(x.device) / (1.0 - p)

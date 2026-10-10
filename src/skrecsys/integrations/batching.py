"""Micro-batching of concurrent requests for asyncio services.

A service that answers one user per request calls ``recommend`` with a single query each
time, though the model ranks a batch of queries much cheaper per query. This module
coalesces the requests that are in flight together into one call. It needs no extra and
imports no web framework: all it takes is a running asyncio event loop, so it fits FastAPI,
Starlette, aiohttp, Litestar or plain asyncio alike.

- :class:`MicroBatcher` is the generic primitive: any ``async`` function that maps a list of
  keys to a list of results;
- :class:`AsyncRecommender` puts it in front of a fitted recommender's ``recommend``.

The batcher never waits on purpose. With nothing in flight, a request is processed at once,
together with the requests submitted in the same loop iteration. While a batch is being
processed, new requests queue up and go together as the next batch, ``max_batch_size`` at
most, so the busier the model, the larger the batches.

>>> import asyncio
>>> async def double(keys: list[int]) -> list[int]:
...     return [key * 2 for key in keys]
>>> async def main() -> list[int]:
...     batcher = MicroBatcher(double, max_size=100)
...     return await asyncio.gather(*(batcher.submit(key) for key in (1, 2, 3)))
>>> asyncio.run(main())
[2, 4, 6]
"""

import asyncio
import enum
import logging
from collections.abc import Awaitable, Callable, Hashable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys.typing import FittedRecommender

__all__ = ["AsyncRecommender", "MicroBatcher"]

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")
U = TypeVar("U", bound=Hashable)


class _Raise(enum.Enum):
    RAISE = enum.auto()


RAISE = _Raise.RAISE


class MicroBatcher(Generic[K, V]):
    """Process the keys that pile up while the previous batch is being processed, in one call.

    Parameters
    ----------
    process : async callable
        Takes a list of keys and returns one result per key, in order. Every caller gets
        its own result.

    max_size : int
        The most keys in one call to ``process``.

    fallback : object, default=RAISE
        What the callers of a batch get when ``process`` fails or returns the wrong number
        of results. By default they get the exception itself. Anything else is returned to
        every caller of the failed batch, and the failure is logged.
    """

    def __init__(
        self,
        process: Callable[[list[K]], Awaitable[list[V]]],
        *,
        max_size: int,
        fallback: V | _Raise = RAISE,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}.")
        self._process = process
        self._fallback = fallback
        self.max_size = max_size
        self._pending: list[tuple[K, asyncio.Future[V]]] = []
        self._draining = False
        self._closed = False
        self._tasks: set[asyncio.Task[None]] = set()  # the loop keeps only weak references

    async def submit(self, key: K) -> V:
        """Queue ``key`` and wait for its result."""
        if self._closed:
            raise RuntimeError("The batcher is closed.")
        loop = asyncio.get_running_loop()
        future = asyncio.Future[V](loop=loop)
        self._pending.append((key, future))
        if not self._draining:
            self._draining = True
            # lets the callers that are already scheduled join the first batch
            loop.call_soon(self._start)
        return await future

    async def aclose(self) -> None:
        """Refuse new keys, cancel the queued ones and wait for the running batch to end."""
        self._closed = True
        for _, future in self._pending:
            future.cancel()
        self._pending.clear()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _start(self) -> None:
        task = asyncio.get_running_loop().create_task(self._drain())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drain(self) -> None:
        try:
            while self._pending:
                batch = self._pending[: self.max_size]
                del self._pending[: self.max_size]
                await self._run(batch)
        finally:
            self._draining = False

    async def _run(self, batch: list[tuple[K, asyncio.Future[V]]]) -> None:
        try:
            results = await self._process([key for key, _ in batch])
            if len(results) != len(batch):
                raise ValueError(f"Got {len(results)} results for a batch of {len(batch)} keys.")
        except Exception as error:
            fallback = self._fallback
            if isinstance(fallback, _Raise):
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)
                return
            logger.exception("Batch of %d failed", len(batch), exc_info=error)
            results = [fallback] * len(batch)
        for (_, future), result in zip(batch, results, strict=True):
            # a caller that went away has cancelled its future
            if not future.done():
                future.set_result(result)


class AsyncRecommender(Generic[U]):
    """Answer one user per ``await``, ranking the concurrent users in one ``recommend`` call.

    The blocking ``recommend`` runs in an executor, so the event loop stays free while the
    model ranks; its kernels release the GIL, so the two really overlap. Users asked at the
    same time share one call, and an identical user within it is ranked once.

    Parameters
    ----------
    model : fitted recommender
        Anything with ``recommend(X, *, n_recommendations, exclude_interactions)``, such as
        a fitted estimator from :mod:`skrecsys.recommendation`. Replace it with the
        :attr:`model` setter when a new version arrives; a batch in flight finishes on the
        model it started with.

    n_recommendations : int, default=10
        Number of items to return per user.

    max_batch_size : int, default=100
        The most users in one ``recommend`` call.

    executor : concurrent.futures.Executor, default=None
        Where ``recommend`` runs. ``None`` creates a single-worker thread pool, which
        :meth:`aclose` shuts down: the kernels bring their own parallelism, and one batch
        at a time keeps the latency of each predictable. A pool you pass stays yours.

    Notes
    -----
    A failed ``recommend``, such as a ``ValueError`` for an unknown user, is raised to every
    caller of that batch, so serve the fallback of your choice for such users before asking,
    or catch the error and ask again one by one.
    """

    def __init__(
        self,
        model: FittedRecommender,
        *,
        n_recommendations: int = 10,
        max_batch_size: int = 100,
        executor: Executor | None = None,
    ) -> None:
        self.model = model
        self.n_recommendations = n_recommendations
        self._owns_executor = executor is None
        self._executor = ThreadPoolExecutor(max_workers=1) if executor is None else executor
        self._batcher = MicroBatcher[tuple[U, tuple[object, ...]], tuple[NDArray, NDArray]](
            self._recommend_batch, max_size=max_batch_size
        )

    async def recommend(
        self, user: U, *, exclude_items: ArrayLike = ()
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Recommend for one user.

        Parameters
        ----------
        user : hashable
            A user identifier seen during ``fit``.

        exclude_items : array-like of shape (n_items,), default=()
            Items this user must not get, such as the events since the model was fitted;
            see ``exclude_interactions`` of ``recommend``. Several requests for the same
            user in one batch share the union of their ``exclude_items``.

        Returns
        -------
        items : ndarray of shape (n_recommendations,)
            Recommended item identifiers ranked by descending score.

        scores : ndarray of shape (n_recommendations,)
            Scores of the recommended items.
        """
        excluded = tuple(np.asarray(exclude_items).ravel().tolist())
        return await self._batcher.submit((user, excluded))

    async def aclose(self) -> None:
        """Cancel the queued requests, wait for the running batch and release the executor."""
        await self._batcher.aclose()
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    async def _recommend_batch(
        self, keys: list[tuple[U, tuple[object, ...]]]
    ) -> list[tuple[NDArray[np.generic], NDArray[np.floating]]]:
        model = self.model  # read once: a swap lands between batches, never inside one
        users = list(dict.fromkeys(user for user, _ in keys))
        pairs = list(dict.fromkeys((user, item) for user, excluded in keys for item in excluded))
        exclude = _as_pairs(pairs) if pairs else None
        loop = asyncio.get_running_loop()
        items, scores = await loop.run_in_executor(
            self._executor,
            lambda: model.recommend(
                _as_column(users),
                n_recommendations=self.n_recommendations,
                exclude_interactions=exclude,
            ),
        )
        row = {user: position for position, user in enumerate(users)}
        return [(items[row[user]], scores[row[user]]) for user, _ in keys]


def _as_column(values: list[U]) -> NDArray[np.generic]:
    return np.asarray(values)


def _as_pairs(pairs: list[tuple[U, object]]) -> NDArray[np.generic]:
    users = np.asarray([user for user, _ in pairs])
    items = np.asarray([item for _, item in pairs])
    if users.dtype == items.dtype:
        return np.stack([users, items], axis=1)
    result = np.empty((len(pairs), 2), dtype=object)
    result[:, 0] = users
    result[:, 1] = items
    return result

import asyncio
import unittest
from typing import Any
from unittest import mock

import numpy as np
from numpy.typing import ArrayLike

from skrecsys.integrations.batching import AsyncRecommender, MicroBatcher
from skrecsys.recommendation import ItemKNNRecommender
from skrecsys.typing import override
from tests.compose._data import trending_interactions


class MicroBatcherTest(unittest.IsolatedAsyncioTestCase):
    @override
    def setUp(self) -> None:
        self.batches: list[list[int]] = []

    async def double(self, keys: list[int]) -> list[int]:
        self.batches.append(keys)
        return [key * 2 for key in keys]

    async def test_keys_submitted_together_go_in_one_call(self) -> None:
        batcher = MicroBatcher(self.double, max_size=100)

        results = await asyncio.gather(*(batcher.submit(key) for key in (1, 2, 3)))

        self.assertEqual(results, [2, 4, 6])
        self.assertEqual(self.batches, [[1, 2, 3]])

    async def test_idle_batcher_does_not_wait(self) -> None:
        batcher = MicroBatcher(self.double, max_size=100)

        async with asyncio.timeout(1):
            self.assertEqual(await batcher.submit(1), 2)

        self.assertEqual(self.batches, [[1]])

    async def test_keys_arriving_during_a_batch_form_the_next_one(self) -> None:
        release = asyncio.Event()

        async def slow(keys: list[int]) -> list[int]:
            self.batches.append(keys)
            await release.wait()
            return [key * 2 for key in keys]

        batcher = MicroBatcher(slow, max_size=100)
        first = asyncio.create_task(batcher.submit(1))
        await asyncio.sleep(0.01)
        later = [asyncio.create_task(batcher.submit(key)) for key in (2, 3, 4)]
        await asyncio.sleep(0.01)

        release.set()

        self.assertEqual(await first, 2)
        self.assertEqual(await asyncio.gather(*later), [4, 6, 8])
        self.assertEqual(self.batches, [[1], [2, 3, 4]])

    async def test_batch_is_capped_at_max_size(self) -> None:
        batcher = MicroBatcher(self.double, max_size=2)

        results = await asyncio.gather(*(batcher.submit(key) for key in (1, 2, 3)))

        self.assertEqual(results, [2, 4, 6])
        self.assertEqual(self.batches, [[1, 2], [3]])

    async def test_batcher_starts_again_after_going_idle(self) -> None:
        batcher = MicroBatcher(self.double, max_size=100)

        self.assertEqual(await batcher.submit(1), 2)
        self.assertEqual(await batcher.submit(2), 4)

        self.assertEqual(self.batches, [[1], [2]])

    async def test_failed_batch_raises_to_every_caller(self) -> None:
        async def fail(keys: list[int]) -> list[int]:
            raise RuntimeError(f"no answer for {keys}")

        batcher = MicroBatcher(fail, max_size=100)

        results = await asyncio.gather(batcher.submit(1), batcher.submit(2), return_exceptions=True)

        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))

    async def test_failed_batch_answers_with_the_fallback(self) -> None:
        async def fail(keys: list[int]) -> list[int | None]:
            raise RuntimeError(f"no answer for {keys}")

        batcher = MicroBatcher(fail, max_size=100, fallback=None)

        with self.assertLogs("skrecsys.integrations.batching", level="ERROR"):
            results = await asyncio.gather(batcher.submit(1), batcher.submit(2))

        self.assertEqual(results, [None, None])

    async def test_wrong_number_of_results_is_a_failure(self) -> None:
        async def lose_one(keys: list[int]) -> list[int]:
            return [key * 2 for key in keys[1:]]

        batcher = MicroBatcher(lose_one, max_size=100)

        with self.assertRaises(ValueError):
            await batcher.submit(1)

    async def test_caller_that_left_does_not_break_the_batch(self) -> None:
        batcher = MicroBatcher(self.double, max_size=100)
        gone = asyncio.create_task(batcher.submit(1))
        stayed = asyncio.create_task(batcher.submit(2))
        await asyncio.sleep(0)

        gone.cancel()

        self.assertEqual(await stayed, 4)
        self.assertTrue(gone.cancelled())
        self.assertEqual(self.batches, [[1, 2]])

    async def test_closed_batcher_refuses_keys(self) -> None:
        batcher = MicroBatcher(self.double, max_size=100)

        await batcher.aclose()

        with self.assertRaises(RuntimeError):
            await batcher.submit(1)

    async def test_close_cancels_queued_keys(self) -> None:
        release = asyncio.Event()

        async def slow(keys: list[int]) -> list[int]:
            await release.wait()
            return keys

        batcher = MicroBatcher(slow, max_size=1)
        running = asyncio.create_task(batcher.submit(1))
        queued = asyncio.create_task(batcher.submit(2))
        await asyncio.sleep(0.01)

        closing = asyncio.create_task(batcher.aclose())
        await asyncio.sleep(0.01)
        release.set()
        await closing

        self.assertEqual(await running, 1)
        with self.assertRaises(asyncio.CancelledError):
            await queued

    def test_rejects_an_empty_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            MicroBatcher(self.double, max_size=0)


class AsyncRecommenderTest(unittest.IsolatedAsyncioTestCase):
    @override
    def setUp(self) -> None:
        self.model = ItemKNNRecommender().fit(trending_interactions())
        self.calls: list[int] = []
        recommend = self.model.recommend

        def spy(X: ArrayLike, **kwargs: Any) -> Any:
            self.calls.append(len(np.atleast_1d(X)))
            return recommend(X, **kwargs)

        patcher = mock.patch.object(self.model, "recommend", spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_concurrent_users_share_one_call_and_match_a_direct_one(self) -> None:
        users = [0, 1, 2, 1]
        rec = AsyncRecommender(self.model, n_recommendations=5)

        results = await asyncio.gather(*(rec.recommend(user) for user in users))
        await rec.aclose()

        self.assertEqual(self.calls, [3])  # user 1 is ranked once
        items, scores = (
            ItemKNNRecommender().fit(trending_interactions()).recommend(users, n_recommendations=5)
        )
        for position, (got_items, got_scores) in enumerate(results):
            np.testing.assert_array_equal(got_items, items[position])
            np.testing.assert_allclose(got_scores, scores[position])

    async def test_exclude_items_apply_to_their_user_only(self) -> None:
        rec = AsyncRecommender(self.model, n_recommendations=5)
        (plain_items, _), _ = await asyncio.gather(rec.recommend(0), rec.recommend(1))
        banned = int(plain_items[0])

        (items, _), (other_items, _) = await asyncio.gather(
            rec.recommend(0, exclude_items=[banned]), rec.recommend(1)
        )
        await rec.aclose()

        self.assertNotIn(banned, items.tolist())
        self.assertEqual(len(other_items), 5)

    async def test_unknown_user_raises_to_the_caller(self) -> None:
        rec = AsyncRecommender(self.model)

        with self.assertRaises(ValueError):
            await rec.recommend(10_000)
        await rec.aclose()

    async def test_swapped_model_serves_the_next_batch(self) -> None:
        rec = AsyncRecommender(self.model, n_recommendations=5)
        await rec.recommend(0)
        newer = ItemKNNRecommender().fit(trending_interactions(seed=1))

        rec.model = newer
        items, _ = await rec.recommend(0)
        await rec.aclose()

        expected, _ = newer.recommend([0], n_recommendations=5)
        np.testing.assert_array_equal(items, expected[0])


if __name__ == "__main__":
    unittest.main()

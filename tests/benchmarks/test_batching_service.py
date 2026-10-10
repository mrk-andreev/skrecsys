"""The service behind benchmarks/batching.py answers alike with and without batching."""

import asyncio
import json
import unittest
from collections.abc import Awaitable, Callable
from typing import cast

import batching

from skrecsys.typing import override


async def get(port: int, path: str) -> tuple[str, dict[str, object]]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: test\r\n\r\n".encode())
    await writer.drain()
    status = (await reader.readline()).decode().strip()
    length = 0
    while line := (await reader.readline()).strip():
        name, _, value = line.decode().partition(":")
        if name.lower() == "content-length":
            length = int(value)
    body = cast("dict[str, object]", json.loads(await reader.readexactly(length)))
    writer.close()
    return status, body


class ServiceTest(unittest.IsolatedAsyncioTestCase):
    @override
    def setUp(self) -> None:
        self.model = batching.build_model(users=50, items=200, interactions=2000, seed=0)

    async def serve(self, mode: str, work: Callable[[int], Awaitable[None]]) -> None:
        service = batching.Service(self.model, mode, max_batch_size=100)
        server = await asyncio.start_server(service.handle, "127.0.0.1", 0)
        try:
            await work(server.sockets[0].getsockname()[1])
        finally:
            server.close()
            await server.wait_closed()

    async def test_both_modes_answer_the_same_items(self) -> None:
        answers: dict[str, list[int]] = {}

        for mode in batching.MODES:

            async def ask(port: int, mode: str = mode) -> None:
                status, body = await get(port, "/recs/3")
                self.assertEqual(status, "HTTP/1.1 200 OK")
                answers[mode] = cast("list[int]", body["item_ids"])

            await self.serve(mode, ask)

        self.assertEqual(len(answers["direct"]), batching.N_RECOMMENDATIONS)
        self.assertEqual(answers["direct"], answers["batched"])

    async def test_concurrent_requests_share_calls_only_when_batched(self) -> None:
        calls: dict[str, int] = {}

        for mode in batching.MODES:

            async def ask(port: int, mode: str = mode) -> None:
                await asyncio.gather(*(get(port, f"/recs/{user}") for user in range(20)))
                _, stats = await get(port, "/stats")
                calls[mode] = cast("int", stats["calls"])

            await self.serve(mode, ask)

        self.assertEqual(calls["direct"], 20)
        self.assertLess(calls["batched"], 20)

    async def test_unknown_user_is_not_found(self) -> None:
        async def ask(port: int) -> None:
            status, _ = await get(port, "/recs/99999")
            self.assertEqual(status, "HTTP/1.1 404 Not Found")

        await self.serve("batched", ask)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
"""When does micro-batching pay off? Serve one model with and without it, under rising load.

    python benchmarks/batching.py run                      # sweep, print a table
    python benchmarks/batching.py run --rps 500 2000 8000 --duration 10
    python benchmarks/batching.py serve --mode batched     # just the service, to poke at

``run`` starts the service twice, once per mode, and drives each at every ``--rps`` with the
Rust load generator in ``benchmarks/loadgen``:

- ``direct``: every request makes its own ``recommend`` call for one user, in the executor
  that ``AsyncRecommender`` would use, so the two modes differ in batching alone;
- ``batched``: the requests in flight together share one ``recommend`` call through
  :class:`skrecsys.integrations.batching.AsyncRecommender`.

The service is a bare ``asyncio`` HTTP/1.1 server with no framework, so what is measured is
the model and the batcher, not a web stack. The load is open-loop: requests are sent on a
schedule whether or not the service keeps up, and each is charged from the moment it was
due, so a service that falls behind shows it as latency instead of slowing the generator
down. The generator is a separate Rust process; a Python one would share the machine's
cores and the GIL-bound work with the service it measures.

The numbers belong to the model and the machine they were measured on. How long one
``recommend`` call takes decides the load at which batching starts to matter: the direct
service saturates near 1 / (time of one call). Your model takes a different time, so your
break-even load and the size of the gain will differ; change the model with ``--users``,
``--items`` and ``--interactions``, pick ``--rps`` values around your own saturation point,
and re-measure.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np

from skrecsys.integrations.batching import AsyncRecommender
from skrecsys.recommendation import ItemKNNRecommender

LOADGEN = Path(__file__).resolve().parent / "loadgen"
MODES = ("direct", "batched")
USAGE_ERROR = 2  # the exit code of the load generator for a bad setup
N_RECOMMENDATIONS = 10


def build_model(users: int, items: int, interactions: int, seed: int) -> ItemKNNRecommender:
    """A fitted model on synthetic interactions: popular items repeat, as in real logs."""
    rng = np.random.default_rng(seed)
    pairs = np.column_stack(
        [rng.integers(0, users, interactions), rng.zipf(1.3, interactions) % items]
    ).astype(np.int64)
    return ItemKNNRecommender().fit(pairs)


class Service:
    """``GET /recs/{user}`` over one model, with or without batching, and ``GET /stats``."""

    def __init__(self, model: ItemKNNRecommender, mode: str, max_batch_size: int) -> None:
        self.model = model
        self.mode = mode
        self.calls = 0  # recommend calls, which is what batching is meant to cut
        self.requests = 0
        recommend = model.recommend

        def counted(users: Sequence[int], **kwargs: int) -> tuple[np.ndarray, np.ndarray]:
            self.calls += 1  # only the executor's single thread gets here
            return recommend(users, **kwargs)  # ty: ignore[invalid-argument-type]

        model.recommend = counted  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._batched = AsyncRecommender(
            model,
            n_recommendations=N_RECOMMENDATIONS,
            max_batch_size=max_batch_size,
            executor=self._executor,
        )

    async def recommend(self, user: int) -> list[int]:
        self.requests += 1
        if self.mode == "batched":
            items, _ = await self._batched.recommend(user)
        else:
            loop = asyncio.get_running_loop()
            rows, _ = await loop.run_in_executor(
                self._executor,
                lambda: self.model.recommend([user], n_recommendations=N_RECOMMENDATIONS),
            )
            items = rows[0]
        return [int(item) for item in items]

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    return
                while (await reader.readline()).strip():  # the headers; none of them matter
                    pass
                status, body = await self.route(request_line.decode().split())
                head = (
                    f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n"
                )
                writer.write(head.encode() + body)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            return
        finally:
            writer.close()

    async def route(self, request: list[str]) -> tuple[str, bytes]:
        path = request[1] if len(request) > 1 else ""
        if path.startswith("/recs/"):
            try:
                items = await self.recommend(int(path.removeprefix("/recs/")))
            except ValueError:  # an unknown user
                return "404 Not Found", b'{"error": "unknown user"}'
            return "200 OK", json.dumps({"item_ids": items}).encode()
        if path == "/stats":
            return "200 OK", json.dumps({"calls": self.calls, "requests": self.requests}).encode()
        if path == "/healthz":
            return "200 OK", b'{"ok": true}'
        return "404 Not Found", b"{}"


async def serve(args: argparse.Namespace) -> None:
    model = build_model(args.users, args.items, args.interactions, args.seed)
    service = Service(model, args.mode, args.max_batch_size)
    server = await asyncio.start_server(service.handle, "127.0.0.1", args.port, backlog=4096)
    print(f"ready {args.mode} users={len(model.user_ids_)}", flush=True)
    async with server:
        await server.serve_forever()


def fetch(url: str) -> dict[str, int]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return cast("dict[str, int]", json.loads(response.read()))


@contextlib.contextmanager
def running_service(args: argparse.Namespace, mode: str) -> Iterator[tuple[str, int]]:
    """Start the service as a child process; yield its address and the users it knows."""
    command = [
        sys.executable,
        __file__,
        "serve",
        "--mode",
        mode,
        "--port",
        str(args.port),
        *(f"--{name.replace('_', '-')}={getattr(args, name)}" for name in SERVE_OPTIONS),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        if process.stdout is None:
            raise RuntimeError("the service has no output to read")
        ready = process.stdout.readline().split()
        if not ready or ready[0] != "ready":
            raise RuntimeError(f"the {mode} service did not start: {ready}")
        yield f"http://127.0.0.1:{args.port}", int(ready[-1].removeprefix("users="))
    finally:
        process.terminate()
        process.wait(timeout=10)


@dataclasses.dataclass(frozen=True)
class Row:
    """What one mode measured at one offered load."""

    mode: str
    target_rps: float
    achieved_rps: float
    sent: int
    ok: int
    dropped: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    first_error: str | None
    mean_batch: float | None  # requests per ``recommend`` call

    @property
    def failed(self) -> int:
        return self.sent - self.ok + self.dropped


def loadgen(url: str, users: int, rps: float, args: argparse.Namespace) -> dict[str, object]:
    done = subprocess.run(
        [
            str(LOADGEN / "target" / "release" / "loadgen"),
            f"--url={url}/recs/{{user}}",
            f"--users={users}",
            f"--rps={rps}",
            f"--duration={args.duration}",
            f"--warmup={args.warmup}",
            f"--seed={args.seed}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode == USAGE_ERROR:
        raise RuntimeError(done.stderr.strip())
    return cast("dict[str, object]", json.loads(done.stdout))


def measure(args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    for mode in MODES:
        with running_service(args, mode) as (url, users):
            for rps in args.rps:
                before = fetch(f"{url}/stats")
                result = loadgen(url, users, rps, args)
                after = fetch(f"{url}/stats")
                calls = after["calls"] - before["calls"]
                requests = after["requests"] - before["requests"]
                mean_batch = requests / calls if calls else None
                rows.append(Row(mode=mode, mean_batch=mean_batch, **result))  # ty: ignore[invalid-argument-type]
                print(f"{mode:>8} {rps:>7.0f} req/s done", file=sys.stderr, flush=True)
                time.sleep(args.pause)
    return rows


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def table(rows: list[Row]) -> str:
    """One line per mode and offered load, the two modes of a load next to each other."""
    loads = list(dict.fromkeys(row.target_rps for row in rows))
    lines = [
        "| offered req/s | mode | served req/s | failed | p50 ms | p95 ms | p99 ms "
        "| requests per call |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {load:.0f} | {row.mode} | {row.achieved_rps:.0f} | {row.failed} "
        f"| {_fmt(row.p50_ms)} | {_fmt(row.p95_ms)} | {_fmt(row.p99_ms)} "
        f"| {_fmt(row.mean_batch, 1)} |"
        for load in loads
        for row in rows
        if row.target_rps == load
    )
    return "\n".join(lines)


SERVE_OPTIONS = ("users", "items", "interactions", "max_batch_size", "seed")


def add_service_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=20_000)
    parser.add_argument("--interactions", type=int, default=1_000_000)
    parser.add_argument("--max-batch-size", type=int, default=100)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    serving = commands.add_parser("serve", help="run the service")
    serving.add_argument("--mode", choices=MODES, default="batched")
    add_service_options(serving)
    running = commands.add_parser("run", help="sweep the offered load in both modes")
    add_service_options(running)
    running.add_argument(
        "--rps", type=float, nargs="+", default=[500, 2000, 4000, 6000, 8000, 12000, 16000]
    )
    running.add_argument("--duration", type=float, default=10.0)
    running.add_argument("--warmup", type=float, default=2.0)
    running.add_argument("--pause", type=float, default=2.0, help="seconds between loads")
    running.add_argument("--output", type=Path, help="also write the measurements here, as JSON")
    args = parser.parse_args(argv)

    if args.command == "serve":
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(serve(args))
        return 0
    if shutil.which("cargo") is None:
        print("cargo is needed to build the load generator", file=sys.stderr)
        return 2
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(LOADGEN / "Cargo.toml")],
        check=True,
    )
    rows = measure(args)
    print(table(rows))
    print(
        "\nThese numbers are for this model and machine. A different model has a different "
        "time per recommend call, and with it a different load at which batching pays off.",
        file=sys.stderr,
    )
    if args.output:
        args.output.write_text(
            json.dumps([dataclasses.asdict(row) for row in rows], indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

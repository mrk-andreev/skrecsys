#!/usr/bin/env python
"""Build `skrecsys._core` with profile-guided optimization.

    python scripts/pgo.py                  # instrument, train, rebuild, install
    python scripts/pgo.py --keep           # also leave the profile behind to inspect

PGO compiles the crate twice. The first build is instrumented and writes a counter
dump per process; a training run then exercises it, the dumps are merged, and the
second build hands the merged profile back to LLVM as branch and inlining evidence.

This is opt-in and is not part of `uv sync`, `maturin develop` or the release
workflow, because on this project it does not currently pay for itself: an
interleaved A/B on an Apple M4 Pro found the kernels within a few percent either way
(EASE and ranking slightly faster, BM25 and RP3Beta slightly slower) for a 3% smaller
binary and roughly triple the build time. That result is workload- and
micro-architecture-specific, so the script stays here to be re-run: tight numeric
loops with predictable branches give PGO little to work with, but a different CPU or a
training workload closer to your own may not behave the same way.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Instrumented code is several times slower, and the profile only needs to see which
# branches are hot, so the training run is deliberately shorter than a real benchmark.
TRAINING = ["benchmarks/leaderboard.py", "--repeat", "2", "--rank-repeat", "100"]
# Longer command lines are summarized when echoed; only `llvm-profdata merge` reaches it.
MAX_ECHOED_ARGS = 8


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Run `command` from the repository root, echoing it first.

    The merge step takes one argument per dumped profile, so a long tail is summarized
    rather than printed: the count is the informative part.
    """
    shown = (
        command
        if len(command) <= MAX_ECHOED_ARGS
        else [*command[:5], f"... +{len(command) - 5} more"]
    )
    print(f"$ {' '.join(shown)}", flush=True)  # noqa: T201
    return subprocess.run(command, cwd=ROOT, text=True, check=True, **kwargs)  # type: ignore[call-overload]  # noqa: S603


def rustc_llvm_version() -> str:
    """The LLVM major version rustc emits profiles for, for example `22`."""
    probe = [os.environ.get("RUSTC", "rustc"), "-vV"]
    out = subprocess.run(probe, capture_output=True, text=True, check=True).stdout  # noqa: S603
    match = re.search(r"^LLVM version:\s*(\d+)", out, re.MULTILINE)
    if match is None:
        raise SystemExit("could not read rustc's LLVM version from `rustc -vV`.")
    return match.group(1)


def find_llvm_profdata(major: str) -> str:
    """Locate an `llvm-profdata` that can read rustc's raw profiles.

    The raw format is versioned, and a newer tool refuses an older dump outright, so a
    plain `llvm-profdata` on PATH is only usable when its major version matches.
    """
    candidates = [
        *(Path.home() / ".rustup" / "toolchains").glob("*/lib/rustlib/*/bin/llvm-profdata"),
        Path(f"/opt/homebrew/opt/llvm@{major}/bin/llvm-profdata"),
        Path(f"/usr/lib/llvm-{major}/bin/llvm-profdata"),
        Path(f"/usr/bin/llvm-profdata-{major}"),
    ]
    rustup = shutil.which("rustup")
    if rustup is not None:
        sysroot = subprocess.run(  # noqa: S603
            [rustup, "run", "stable", "rustc", "--print", "sysroot"],
            capture_output=True,
            text=True,
            check=False,
        )
        if sysroot.returncode == 0:
            candidates[:0] = Path(sysroot.stdout.strip()).glob("lib/rustlib/*/bin/llvm-profdata")
    on_path = shutil.which("llvm-profdata")
    if on_path is not None:
        candidates.append(Path(on_path))

    for candidate in candidates:
        if not candidate.exists():
            continue
        version = subprocess.run(  # noqa: S603
            [str(candidate), "--version"], capture_output=True, text=True, check=False
        )
        if version.returncode == 0 and f"version {major}" in version.stdout:
            return str(candidate)

    raise SystemExit(
        f"no llvm-profdata for LLVM {major} (rustc's version). Install a matching one, "
        f"for example `rustup component add llvm-tools-preview` or, on Homebrew, "
        f"`brew install llvm@{major}`."
    )


def maturin(profile_flag: str, profile_dir: Path) -> None:
    """Build and install the extension with one extra codegen flag."""
    env = dict(os.environ)
    existing = env.get("RUSTFLAGS", "")
    env["RUSTFLAGS"] = f"{existing} {profile_flag}={profile_dir}".strip()
    run(
        ["uv", "run", "--quiet", "--with", "maturin", "maturin", "develop", "--release"],
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the raw and merged profiles")
    args = parser.parse_args(argv)

    major = rustc_llvm_version()
    profdata = find_llvm_profdata(major)
    print(f"rustc emits LLVM {major} profiles; using {profdata}", flush=True)  # noqa: T201

    workdir = Path(tempfile.mkdtemp(prefix="skrecsys-pgo-"))
    raw, merged = workdir / "raw", workdir / "merged.profdata"
    raw.mkdir()
    try:
        print("\n== 1/3 instrumented build ==", flush=True)  # noqa: T201
        maturin("-Cprofile-generate", raw)

        print("\n== 2/3 training run ==", flush=True)  # noqa: T201
        run(["uv", "run", "--quiet", "python", *TRAINING], stdout=subprocess.DEVNULL)
        dumps = sorted(raw.glob("*.profraw"))
        if not dumps:
            raise SystemExit(
                "the training run produced no .profraw files; the instrumented build "
                "may not have been the one imported."
            )
        print(f"collected {len(dumps)} profile dumps", flush=True)  # noqa: T201
        run([profdata, "merge", "-o", str(merged), *map(str, dumps)])

        print("\n== 3/3 optimized build ==", flush=True)  # noqa: T201
        maturin("-Cprofile-use", merged)
    finally:
        if args.keep:
            print(f"\nprofiles left in {workdir}", flush=True)  # noqa: T201
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(  # noqa: T201
        "\nPGO build installed. It stays until the next `uv sync` or `maturin develop`, "
        "which rebuild without a profile.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

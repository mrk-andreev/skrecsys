"""Score a recommender on a sequential, leave-one-out split.

This is the benchmark the sequential-recommendation literature reports on: each user's
last interaction is held out, the model sees the history before it, and the held-out
item is ranked against the whole catalog. Scores are hit rate and NDCG at several
cutoffs, which is the shape of the tables in the SASRec and HSTU papers, so a row here
can be read next to a published one.

It is a separate report from the leaderboard because it asks a different question. The
leaderboard splits interactions and asks which items a user likes; this asks which item
comes *next*, which is the only question a sequential model such as
:class:`skrecsys.nn.HSTU` or :class:`skrecsys.nn.Mamba4Rec` is trained to answer. The two
tables are not comparable, and mixing them into one would invite exactly that comparison.

Which models run, and under what settings, is ``benchmarks/config/sequential.json``;
``benchmarks/run.py`` runs them. Fitting and timing are the leaderboard's.

The neural models are configured with ``device="auto"`` rather than the estimator
default of ``"cpu"``: the default is what every wheel can rely on, but a benchmark that
has to fit 1M interactions should use whatever accelerator the host has. Expect a fifth
off rather than an order of magnitude. Measured on an M4 Pro, one ml-1m epoch of
Mamba4Rec is 170 s on the CPU against 141 s on MPS, and five HSTU epochs 65 s against
50 s. Neither model is compute-bound at this size: Mamba4Rec's scan is a Python loop over
the 200 positions of the window, so the fit is spent dispatching small kernels, which is
the cost a GPU cannot amortize away.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leaderboard import Evaluation

from skrecsys.metrics import catalog_coverage_at_k, hit_rate_at_k, ndcg_at_k


def device_facts() -> dict[str, str] | None:
    """What ``device="auto"`` resolves to on this host, or ``None`` without torch.

    The neural rows are timed on an accelerator when the host has one, so a result that
    named only the CPU would understate what produced it.
    """
    # Imported by name because the `nn` extra is optional, and absent where ty runs.
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    if torch.cuda.is_available():
        device = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "MPS"
    else:
        device = "CPU"
    return {"device": device, "torch": str(torch.__version__)}


def score(evaluation: Evaluation, cutoffs: Sequence[int]) -> dict[str, str]:
    """Hit rate and NDCG at every cutoff, plus what the top of the list covers.

    Exactly one interaction is held out per user, so the hit rate is also recall, and
    precision is the hit rate over ``k``; reporting all three would be one number three
    times. Catalog coverage is here because a sequential model that has collapsed onto
    the popular head can still look respectable on NDCG alone.
    """
    row: dict[str, str] = {}
    for cutoff in cutoffs:
        row[f"HR@{cutoff}"] = (
            f"{float(hit_rate_at_k(evaluation.y_true, evaluation.y_pred, k=cutoff)):.4f}"
        )
        row[f"NDCG@{cutoff}"] = (
            f"{float(ndcg_at_k(evaluation.y_true, evaluation.y_pred, k=cutoff)):.4f}"
        )
    coverage = catalog_coverage_at_k(
        evaluation.y_true, evaluation.y_pred, k=cutoffs[0], catalog=evaluation.catalog
    )
    row[f"cat cov@{cutoffs[0]}"] = f"{float(coverage):.4f}"
    return row

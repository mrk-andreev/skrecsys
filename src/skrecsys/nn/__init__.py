"""Recommenders trained by gradient descent with PyTorch.

Importing this subpackage requires torch, which is an optional extra::

    pip install skrecsys[nn]

Everything outside :mod:`skrecsys.nn` runs without it. A model fitted here exports its
parameters to numpy, so a *fitted* estimator scores, pickles and unpickles in an
environment that has no torch at all; only ``fit`` needs the extra.
"""

try:
    from skrecsys.nn._hstu import HSTU
    from skrecsys.nn._mamba4rec import Mamba4Rec
    from skrecsys.nn._simplex import SimpleX
    from skrecsys.nn._xsimgcl import XSimGCL
except ModuleNotFoundError as _exc:  # pragma: no cover - covered by the torch-free CI matrix
    if _exc.name != "torch":
        raise
    raise ImportError(
        "skrecsys.nn requires PyTorch. Install it with `pip install skrecsys[nn]`."
    ) from _exc

__all__ = ["HSTU", "Mamba4Rec", "SimpleX", "XSimGCL"]

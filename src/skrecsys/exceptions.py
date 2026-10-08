"""Exceptions raised by skrecsys, beyond the built-in ones it uses for bad arguments."""

__all__ = ["InsufficientDataError"]


class InsufficientDataError(ValueError):
    """The interactions are too few, or too uniform, for an estimator to learn from.

    Raised by ``fit`` when the parameters are valid and the data is the problem: a
    :class:`~skrecsys.compose.Cascade` whose ranker has no held-out interaction to rank
    above the others, say. It is a :class:`ValueError`, which is what such a ``fit`` raised
    before there was a class of its own, so code catching that goes on working; catch this
    one to fall back to a simpler model without hiding a mistake in the arguments, which
    is what ``skip_insufficient`` of a :class:`~skrecsys.compose.Backfill` does.
    """

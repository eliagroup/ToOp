"""Evaluate initial overload thresholds."""

from beartype.typing import Optional


def overload_is_at_or_below_threshold(
    overload_n_0: Optional[float], overload_n_1: Optional[float], threshold: Optional[float]
) -> bool:
    """Return whether both known overload energies are at or below the threshold."""
    return (
        threshold is not None
        and overload_n_0 is not None
        and overload_n_1 is not None
        and overload_n_0 <= threshold
        and overload_n_1 <= threshold
    )

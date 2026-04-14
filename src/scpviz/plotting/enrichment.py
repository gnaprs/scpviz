"""Enrichment plot forwarding to scpviz.enrichment."""

from typing import Any


def plot_enrichment_svg(*args: Any, **kwargs: Any):
    """
    Plot STRING enrichment results as an SVG figure.

    This is a wrapper that redirects to the implementation in `enrichment.py`
    for convenience and discoverability.

    Args:
        *args (Any): Positional arguments passed to `scpviz.enrichment.plot_enrichment_svg`.
        **kwargs (Any): Keyword arguments passed to `scpviz.enrichment.plot_enrichment_svg`.

    Returns:
        svg (SVG): SVG figure object.

    See Also:
        scpviz.enrichment.plot_enrichment_svg
    """
    from scpviz.enrichment import plot_enrichment_svg as actual_plot

    return actual_plot(*args, **kwargs)

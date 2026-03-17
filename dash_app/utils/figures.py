"""Matplotlib to Dash image conversion helpers."""

from __future__ import annotations

import base64
import io
from contextlib import contextmanager
from typing import Generator, Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


@contextmanager
def new_figure(figsize: Tuple[float, float] = (6.0, 4.0)) -> Generator[plt.Figure, None, None]:
    """Create and dispose a matplotlib figure safely."""
    fig = plt.figure(figsize=figsize)
    try:
        yield fig
    finally:
        plt.close(fig)


def fig_to_data_uri(fig: plt.Figure, fmt: str = "png") -> str:
    """Convert matplotlib figure to browser-displayable data URI."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, bbox_inches="tight", dpi=140)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


"""Shared console reporting helpers for differential expression (de / mixed_de)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from scpviz.utils.formatting import format_log_prefix

GroupLabelStyle = Literal["values", "slash", "underscore", "repr"]


def _normalize_group_def(
    group: dict[str, Any] | list[Any] | Any,
) -> dict[str, Any] | list[Any] | Any:
    """Unwrap single-entry filter lists (as used by ``pAnnData.de``)."""
    if isinstance(group, list) and len(group) == 1:
        return group[0]
    return group


def format_de_group_label(
    group: dict[str, Any] | list[Any] | Any,
    *,
    class_type: str | list[str] | None = None,
    style: GroupLabelStyle = "values",
) -> str:
    """Format a single DE group for logs, stats keys, or volcano annotations."""
    group = _normalize_group_def(group)
    if isinstance(group, dict):
        if style == "repr":
            return repr(group)
        sep = "/" if style == "slash" else "_"
        return sep.join(str(v) for v in group.values())
    if isinstance(group, list) and isinstance(class_type, list) and len(group) == len(class_type):
        sep = "/" if style == "slash" else "_"
        return sep.join(str(v) for v in group)
    return str(group)


def format_de_comparison_label(
    group1: dict[str, Any] | list[Any] | Any,
    group2: dict[str, Any] | list[Any] | Any,
    *,
    class_type: str | list[str] | None = None,
    style: GroupLabelStyle = "values",
) -> str:
    """Return ``'{group1} vs {group2}'`` contrast text for stats keys and logs."""
    g1 = format_de_group_label(group1, class_type=class_type, style=style)
    g2 = format_de_group_label(group2, class_type=class_type, style=style)
    return f"{g1} vs {g2}"


def parse_contrast_label(text: str) -> tuple[str, str] | None:
    """Parse ``'{group1} vs {group2}'`` labels used by de/mixed_de stats keys."""
    if " vs " not in text:
        return None
    left, right = text.split(" vs ", 1)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return None
    return left, right


def print_de_threshold_line(
    *,
    correct_fdr: bool,
    threshold: float,
    log2fc_thresh: float,
) -> None:
    """Print significance threshold line shared by de and mixed_de run headers."""
    if correct_fdr:
        print(f"   🔸 Adj p-value threshold: {threshold} | Log2FC threshold: {log2fc_thresh}")
    else:
        print(f"   🔸 P-value threshold: {threshold} | Log2FC threshold: {log2fc_thresh}")


def format_de_column_hint(
    *,
    correct_fdr: bool,
    extra_cols: tuple[str, ...] = (),
) -> str:
    """Column summary for DE result printouts."""
    cols = ["log2fc", "p_value"]
    if correct_fdr:
        cols.append("adj_p_value")
    cols.append("significance")
    cols.extend(extra_cols)
    if extra_cols:
        return ", ".join(cols)
    return ", ".join(cols) + ", etc."


def print_de_run_header(
    *,
    assay: str,
    comparing: str,
    layer_line: str,
    method_line: str,
    correct_fdr: bool,
    threshold: float,
    log2fc_thresh: float,
    group_sizes: str | None = None,
    group_sizes_label: str = "Group sizes",
    extra_lines: list[str] | None = None,
) -> None:
    """Print pre-run USER summary for standard (non-mixed) DE."""
    log_prefix = format_log_prefix("user")
    print(f"{log_prefix} Running differential expression [{assay}]")
    print(f"   🔸 Comparing groups: {comparing}")
    if group_sizes is not None:
        print(f"   🔸 {group_sizes_label}: {group_sizes}")
    for line in extra_lines or []:
        print(line)
    print(f"   🔸 Layer: {layer_line}")
    print(f"   🔸 Method: {method_line}")
    print_de_threshold_line(
        correct_fdr=correct_fdr,
        threshold=threshold,
        log2fc_thresh=log2fc_thresh,
    )


def print_de_result_summary(
    volcano_df: pd.DataFrame,
    *,
    title: str = "DE",
    stats_location: str,
    correct_fdr: bool,
    column_hint: str | None = None,
    contrast_label: str | None = None,
    include_storage_guide: bool = True,
    include_not_comparable: bool = True,
    extra_footer_lines: list[str] | None = None,
) -> None:
    """Print post-run DE summary (de, mixed_de, and optional de_adata callers)."""
    sig_counts = volcano_df["significance"].value_counts().to_dict()
    n_up = int(sig_counts.get("upregulated", 0))
    n_down = int(sig_counts.get("downregulated", 0))
    n_ns = int(sig_counts.get("not significant", 0))
    n_nc = int(sig_counts.get("not comparable", 0))

    header = f"{format_log_prefix('result_only', indent=2)} {title} complete."
    if contrast_label is None:
        print(f"{header} Results stored in:")
    else:
        print(f"{header} Contrast {contrast_label!r}:")

    indent = "       " if contrast_label is None else "         "
    if include_storage_guide:
        print(f"{indent}• {stats_location}")
        hint = column_hint or format_de_column_hint(correct_fdr=correct_fdr)
        print(f"{indent}• Columns: {hint}")

    counts = (
        f"{indent}• Upregulated: {n_up} | Downregulated: {n_down} | "
        f"Not significant: {n_ns}"
    )
    if include_not_comparable:
        counts += f" | Not comparable: {n_nc}"
    print(counts)

    for line in extra_footer_lines or []:
        if line == "":
            print("")
        elif line.startswith("• "):
            print(f"{indent}{line}")
        else:
            print(f"{indent}• {line}")

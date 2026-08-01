"""Publication-style grouped and clustered heatmaps (protein/peptide × sample)."""
from __future__ import annotations

from collections import Counter
from typing import Any, TYPE_CHECKING

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle

from scpviz.utils.data import (
    get_adata,
    get_adata_layer,
    infer_layer_is_log,
    resolve_accessions,
    resolve_input_layer,
    resolve_peptides,
)
from scpviz.utils.formatting import format_log_prefix
from scpviz.utils.mixed_de import expr_looks_non_log
from scpviz.utils.stats import correlation_linkage
from scpviz.plotting.style import get_color

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from scpviz.pAnnData.pAnnData import pAnnData

MISSING_GREY = (0.80, 0.80, 0.80, 1.0)
GAP_WHITE = (1.0, 1.0, 1.0, 0.0)
UNASSIGNED_COLOUR = "#dddddd"
UNASSIGNED_LABEL_COLOUR = "#999999"

# Colorbar + legend column on the left (figure fraction / fixed inches).
# Anchored near the dendrogram / main axes (subplots left≈0.24), not the figure edge.
_CBAR_LEFT = 0.12
_CBAR_WIDTH_IN = 0.14
_CBAR_HEIGHT_IN = 1.35
_CBAR_TOP = 0.82  # figure-fraction top of colorbar (inline path)
_LEGEND_X = 0.12
_LEGEND_PAD_BELOW_CBAR_IN = 0.12

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_sample_order(
    obs: pd.DataFrame,
    samples: list[str],
    classes: list[str],
    sort_by: dict[str, list[str]] | None = None,
) -> list[str]:
    """Multi-key column sort: outer→inner follows ``classes`` order."""
    sort_by = sort_by or {}
    rank_cols: dict[str, pd.Series] = {}
    for cls in classes:
        if cls not in obs.columns:
            raise KeyError(f"classes entry {cls!r} not found in .obs columns.")
        col_vals = obs.loc[samples, cls].astype(str)
        order = sort_by.get(cls)
        if order is None:
            order = list(dict.fromkeys(col_vals.tolist()))
        else:
            order = [str(v) for v in order]
        cat_rank = {v: i for i, v in enumerate(order)}
        mapped = col_vals.map(cat_rank)
        if mapped.isna().any():
            missing = col_vals[mapped.isna()].unique().tolist()
            print(
                f"{format_log_prefix('warn')} Values {missing} in .obs[{cls!r}] "
                f"are not in sort_by[{cls!r}]; appending after listed order."
            )
            next_rank = len(cat_rank)
            for v in missing:
                cat_rank[str(v)] = next_rank
                next_rank += 1
            mapped = col_vals.map(cat_rank)
        rank_cols[cls] = mapped
    rank_df = pd.DataFrame(rank_cols, index=samples)
    return rank_df.sort_values(list(rank_df.columns)).index.tolist()

def _prepare_heatmap_expr(
    adata: ad.AnnData,
    layer: str,
    *,
    auto_log2: bool = True,
    log_pseudocount: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load ``(n_obs, n_vars)`` expression; mirror mixed_de log2 policy."""
    layer_is_log = infer_layer_is_log(layer, adata)
    resolved_layer = resolve_input_layer(adata, layer)
    expr = np.asarray(get_adata_layer(adata, layer), dtype=float)
    meta: dict[str, Any] = {
        "input_layer": layer,
        "resolved_layer": resolved_layer,
        "layer_is_log": layer_is_log,
        "expr_auto_log2": False,
        "log_pseudocount": None,
    }

    if layer_is_log:
        return expr, meta

    looks_non_log = expr_looks_non_log(expr)
    if not looks_non_log:
        finite = expr[np.isfinite(expr)]
        med = float(np.median(finite)) if finite.size else float("nan")
        resolved_note = (
            f" (resolved {layer!r} → {resolved_layer!r})"
            if resolved_layer != layer
            else ""
        )
        print(
            f"{format_log_prefix('info')} Layer {layer!r}{resolved_note} is not "
            f"registered as log-transformed but values look log-like "
            f"(median={med:.2g}). Proceeding. For provenance tracking, run "
            f"`pdata.log_transform(..., base=2)` (sets "
            f"``.uns['current_X_layer']`` when ``set_X=True``) or pass "
            f"`layer={resolved_layer!r}` / ``layer='X_log2'``."
        )
        return expr, meta

    finite = expr[np.isfinite(expr)]
    med = float(np.median(finite)) if finite.size else float("nan")
    if not auto_log2:
        raise ValueError(
            f"Layer {layer!r} does not appear to be log2-transformed (median={med:.2g}). "
            "Heatmaps expect log-scale values before z-scoring. "
            "Normalize on linear scale, log2-transform, then pass that layer — e.g.\n\n"
            f"    pdata.log_transform(layer={layer!r}, base=2)\n"
            f"    pdata.plot_grouped_heatmap(..., layer='X_log2')\n\n"
            "Or set auto_log2=True (default) to apply log2(x + pseudocount) in memory "
            "for this plot only."
        )

    n_negative = int(np.sum(finite < 0)) if finite.size else 0
    if n_negative > 0:
        print(
            f"{format_log_prefix('warn')} {n_negative} value(s) < 0 in layer {layer!r}. "
            f"log2(x + {log_pseudocount}) will be applied."
        )
    print(
        f"{format_log_prefix('warn')} Layer {layer!r} appears non-log (median={med:.2g}). "
        f"Applying log2(x + {log_pseudocount}) in memory for this heatmap only. "
        "To persist: "
        f"`pdata.log_transform(layer={layer!r}, base=2)` then `layer='X_log2'`."
    )
    expr = np.log2(expr + log_pseudocount)
    meta["expr_auto_log2"] = True
    meta["log_pseudocount"] = log_pseudocount
    return expr, meta

def _layer_looks_zscored(adata: ad.AnnData, layer: str, values_T: np.ndarray) -> bool:
    """Prefer provenance; fall back to per-row mean/std heuristic on ND-masked rows."""
    resolved = resolve_input_layer(adata, layer)
    registry = adata.uns.get("layer_provenance", {})
    # Walk chain looking for an explicit scale/zscore op (future-proof).
    seen: set[str] = set()
    cur = resolved
    while cur and cur not in seen:
        seen.add(cur)
        rec = registry.get(cur)
        if not isinstance(rec, dict):
            break
        op = str(rec.get("op", "")).lower()
        if any(tok in op for tok in ("zscore", "z_score", "scale", "standardize")):
            return True
        cur = rec.get("input_layer")

    # Heuristic on up to 200 random rows (proteins × samples matrix).
    n_rows = values_T.shape[0]
    if n_rows == 0:
        return False
    rng = np.random.default_rng(0)
    idx = rng.choice(n_rows, size=min(200, n_rows), replace=False)
    ok = 0
    checked = 0
    for i in idx:
        row = values_T[i]
        finite = row[np.isfinite(row) & (row != 0)]
        if finite.size < 2:
            continue
        checked += 1
        mu = float(np.mean(finite))
        sd = float(np.std(finite, ddof=0))
        if abs(mu) < 0.3 and 0.7 <= sd <= 1.3:
            ok += 1
    if checked == 0:
        return False
    return (ok / checked) > 0.8

def _zscore_rows(values: np.ndarray) -> np.ndarray:
    """Nan-aware per-row z-score. Constant / empty rows become all-NaN."""
    out = np.full_like(values, np.nan, dtype=float)
    for i in range(values.shape[0]):
        row = values[i]
        finite = row[np.isfinite(row)]
        if finite.size < 2:
            continue
        sd = float(np.nanstd(row, ddof=0))
        if sd == 0 or not np.isfinite(sd):
            continue
        mu = float(np.nanmean(row))
        out[i] = (row - mu) / sd
    return out

def _values_from_expr(expr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n_obs, n_vars)`` → ``(n_vars, n_obs)`` with zeros/non-finite as missing."""
    values = np.array(expr.T, dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    values[values == 0] = np.nan
    is_missing = ~np.isfinite(values)
    return values, is_missing

def _suggest_display_scale(
    significance_categories: list[str] | None,
    *,
    from_stats_key: bool,
) -> tuple[str, str]:
    """
    Return ``(scale, reason)`` for ``display_scale='auto'``.

    Rules: any ``"not comparable"`` → ``log``; else upregulated/downregulated
    (or no DE context) → ``zscore``.
    """
    if from_stats_key and significance_categories is not None:
        if "not comparable" in significance_categories:
            return (
                "log",
                "'not comparable' in significance_categories "
                "(z-scores are driven by within-group variation)",
            )
        if any(
            c in significance_categories
            for c in ("upregulated", "downregulated")
        ):
            return (
                "zscore",
                "upregulated/downregulated in significance_categories",
            )
        return ("zscore", "DE selection without 'not comparable'")
    return ("zscore", "default (explicit protein list / no DE stats_key)")

def _resolve_display_scale_arg(
    display_scale: str,
    significance_categories: list[str] | None,
    *,
    from_stats_key: bool,
) -> str:
    """Resolve ``auto`` / validate; print info or override warnings."""
    allowed = {"auto", "zscore", "log", "raw"}
    if display_scale not in allowed:
        raise ValueError(
            f"display_scale must be one of {sorted(allowed)}, got {display_scale!r}"
        )
    suggested, reason = _suggest_display_scale(
        significance_categories, from_stats_key=from_stats_key
    )
    if display_scale == "auto":
        print(
            f"{format_log_prefix('info')} display_scale='{suggested}' "
            f"(auto: {reason})."
        )
        return suggested
    if display_scale != suggested:
        print(
            f"{format_log_prefix('warn')} display_scale={display_scale!r} differs "
            f"from auto suggestion {suggested!r} ({reason}). Using "
            f"{display_scale!r} as requested."
        )
    else:
        print(
            f"{format_log_prefix('info')} display_scale={display_scale!r} "
            f"(matches auto: {reason})."
        )
    return display_scale

def _resolve_cmap_for_scale(
    cmap: str | None,
    display_scale: str,
) -> str:
    """Auto viridis for log/raw; warn if user forces RdBu_r on those scales."""
    if cmap is None:
        return "RdBu_r" if display_scale == "zscore" else "viridis"
    if display_scale in ("log", "raw") and cmap == "RdBu_r":
        print(
            f"{format_log_prefix('warn')} cmap='RdBu_r' with "
            f"display_scale={display_scale!r} is often a poor fit (diverging "
            f"around zero). Consider omitting cmap to use 'viridis', or pass "
            f"another sequential colormap."
        )
    return cmap

def _resolve_display_and_cluster(
    adata: ad.AnnData,
    layer: str,
    *,
    display_scale: str,
    auto_log2: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool, bool]:
    """
    Return display + clustering matrices.

    ``(display, display_missing, cluster_z, cluster_missing, cbar_label,
    symmetric_norm, layer_is_log)``. Clustering matrix is always z-scored
    (after the usual auto-log2 prep), independent of ``display_scale``.
    """
    # Clustering path (unchanged policy)
    expr_z, meta_z = _prepare_heatmap_expr(adata, layer, auto_log2=auto_log2)
    vals_z, miss_z = _values_from_expr(expr_z)
    if _layer_looks_zscored(adata, layer, vals_z):
        cluster_z = vals_z.copy()
    else:
        cluster_z = _zscore_rows(vals_z)
        cluster_z[miss_z] = np.nan

    layer_is_log = bool(meta_z.get("layer_is_log")) or infer_layer_is_log(layer, adata)

    if display_scale == "zscore":
        return cluster_z, miss_z, cluster_z, miss_z, "z-score", True, layer_is_log

    if display_scale == "log":
        # Ensure log space without double-logging (prepare + infer policy)
        expr_log, _meta_log = _prepare_heatmap_expr(adata, layer, auto_log2=True)
        display, miss = _values_from_expr(expr_log)
        return display, miss, cluster_z, miss_z, "log2 abundance", False, True

    # raw: layer values as stored, no in-memory log2
    expr_raw = np.asarray(get_adata_layer(adata, layer), dtype=float)
    display, miss = _values_from_expr(expr_raw)
    layer_is_log = infer_layer_is_log(layer, adata)
    cbar_label = "log2 abundance" if layer_is_log else "abundance"
    return display, miss, cluster_z, miss_z, cbar_label, False, layer_is_log

def _composite_heatmap_rgba(
    values: np.ndarray,
    is_missing: np.ndarray,
    is_gap_row: np.ndarray | None,
    cmap_name: str,
    *,
    symmetric: bool = True,
    vmax: float | None = None,
) -> tuple[np.ndarray, Normalize]:
    finite_mask = np.isfinite(values) & ~is_missing
    if is_gap_row is not None:
        finite_mask = finite_mask & ~is_gap_row[:, None]
    if symmetric:
        if vmax is None:
            if np.any(finite_mask):
                vmax = float(np.nanpercentile(np.abs(values[finite_mask]), 95))
            else:
                vmax = 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        if np.any(finite_mask):
            vmin = float(np.nanpercentile(values[finite_mask], 5))
            vmax_s = float(np.nanpercentile(values[finite_mask], 95))
        else:
            vmin, vmax_s = 0.0, 1.0
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax_s) or vmax_s <= vmin:
            vmax_s = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax_s)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(np.nan_to_num(values, nan=0.0)))
    rgba[is_missing] = MISSING_GREY
    if is_gap_row is not None:
        rgba[is_gap_row, :, :] = GAP_WHITE
    return rgba, norm


_ROW_PX = 10  # content-row pixel height; gap_rows=0.5 → 5 px of white

def _rasterize_with_gaps(
    rgba: np.ndarray,
    row_groups: list[str | None],
    labels: list[str],
    gap_rows: float,
    *,
    row_px: int = _ROW_PX,
) -> tuple[np.ndarray, list[float], list[str], dict[str, tuple[float, float]]]:
    """
    Expand content rows so fractional ``gap_rows`` become partial-height spacers.

    Returns ``(rgba_disp, tick_positions, tick_labels, group_yrange)`` where
    ``group_yrange[g] = (y0, y1)`` are inclusive display-row indices for brackets.
    """
    if gap_rows < 0:
        raise ValueError(f"gap_rows must be >= 0, got {gap_rows!r}")
    gap_px = int(round(float(gap_rows) * row_px))
    n_content, n_cols = rgba.shape[0], rgba.shape[1]
    if n_content != len(row_groups) or n_content != len(labels):
        raise ValueError("rgba, row_groups, and labels length mismatch")

    # Identify group blocks in order
    blocks: list[tuple[str, int, int]] = []  # (gname, start, end) inclusive content idx
    i = 0
    while i < n_content:
        g = row_groups[i]
        if g is None:
            i += 1
            continue
        j = i
        while j + 1 < n_content and row_groups[j + 1] == g:
            j += 1
        blocks.append((g, i, j))
        i = j + 1

    disp_chunks: list[np.ndarray] = []
    tick_pos: list[float] = []
    tick_lab: list[str] = []
    group_yrange: dict[str, tuple[float, float]] = {}
    cursor = 0.0

    for bi, (gname, i0, i1) in enumerate(blocks):
        g_start = cursor
        for i in range(i0, i1 + 1):
            tick_pos.append(cursor + (row_px - 1) / 2.0)
            tick_lab.append(labels[i])
            tile = np.repeat(rgba[i : i + 1], row_px, axis=0)
            disp_chunks.append(tile)
            cursor += row_px
        g_end = cursor - 1
        # If group appears twice (duplicate blocks), extend range
        if gname in group_yrange:
            prev0, prev1 = group_yrange[gname]
            group_yrange[gname] = (min(prev0, g_start), max(prev1, g_end))
        else:
            group_yrange[gname] = (g_start, g_end)
        if bi != len(blocks) - 1 and gap_px > 0:
            gap = np.zeros((gap_px, n_cols, 4), dtype=float)
            gap[:] = GAP_WHITE
            disp_chunks.append(gap)
            cursor += gap_px

    if not disp_chunks:
        return (
            np.zeros((0, n_cols, 4), dtype=float),
            [],
            [],
            {},
        )
    return np.concatenate(disp_chunks, axis=0), tick_pos, tick_lab, group_yrange

def _build_color_map(
    categories: list[Any],
    overrides: dict[Any, str] | None = None,
) -> dict[Any, str]:
    overrides = overrides or {}
    n = len(categories)
    defaults = list(get_color("colors", max(n, 1))) if n else []
    out: dict[Any, str] = {}
    for i, cat in enumerate(categories):
        out[cat] = overrides.get(cat, defaults[i % len(defaults)])
    return out

def _sample_tick_labels(
    obs: pd.DataFrame,
    samples: list[str],
    sample_label_col: str | None = None,
) -> list[str]:
    """Bottom tick labels: ``.obs`` column values, or sample index names if None."""
    if sample_label_col is None:
        return [str(s) for s in samples]
    if sample_label_col not in obs.columns:
        raise KeyError(
            f"sample_label_col {sample_label_col!r} not found in .obs columns. "
            f"Available: {list(obs.columns)}"
        )
    return [str(obs.loc[s, sample_label_col]) for s in samples]

def _render_header_rows(
    fig: "Figure",
    gs_rows: list[Any],
    obs: pd.DataFrame,
    samples: list[str],
    classes: list[str],
    header_colors: dict[str, dict[str, str]] | None,
    text_size: int = 8,
) -> list[tuple[str, list[Rectangle], list[Any]]]:
    """Draw categorical header strips; return legend specs ``(title, handles, labels)``."""
    header_colors = header_colors or {}
    legend_specs: list[tuple[str, list[Rectangle], list[Any]]] = []
    for ax, col in zip(gs_rows, classes):
        vals = [str(obs.loc[s, col]) for s in samples]
        cats = list(dict.fromkeys(vals))
        # Allow overrides keyed by original or stringified category labels
        raw_override = header_colors.get(col) or {}
        override = {str(k): v for k, v in raw_override.items()}
        color_map = _build_color_map(cats, override)
        codes = np.array([[cats.index(v) for v in vals]])
        lcmap = ListedColormap([color_map[c] for c in cats])
        ax.imshow(codes, aspect="auto", cmap=lcmap, vmin=-0.5, vmax=len(cats) - 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([col], fontsize=text_size)
        ax.set_xticks([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        handles = [Rectangle((0, 0), 1, 1, color=color_map[c]) for c in cats]
        legend_specs.append((col, handles, cats))
    return legend_specs


_LEGEND_TEXT_COLOR = "black"

def _style_legend_title(leg) -> None:
    title = leg.get_title()
    if title is not None:
        title.set_color(_LEGEND_TEXT_COLOR)

def _style_colorbar_text(cbar, text_size: int) -> None:
    """Match colorbar label/ticks to legend text color."""
    cbar.ax.yaxis.label.set_color(_LEGEND_TEXT_COLOR)
    cbar.ax.tick_params(labelsize=text_size - 1, colors=_LEGEND_TEXT_COLOR)

def _legend_block_height_in(n_labels: int, text_size: int) -> float:
    """Approximate vertical space (inches) for one titled legend block."""
    scale = text_size / 8.0
    return (0.22 + 0.145 * max(n_labels, 1)) * scale

def _colorbar_axes_rect(
    fig: "Figure",
    *,
    left: float,
    top: float,
    cbar_scale: float = 1.0,
) -> list[float]:
    """Fixed-inch colorbar as ``[left, bottom, width, height]`` in figure fraction."""
    fig_w, fig_h = fig.get_size_inches()
    width = _CBAR_WIDTH_IN / fig_w
    height = min((_CBAR_HEIGHT_IN * cbar_scale) / fig_h, 0.55)
    return [left, top - height, width, height]

def _stack_legends_below_cbar(
    fig: "Figure",
    legend_specs: list[tuple[str, list, list]],
    *,
    cbar_bottom: float,
    legend_x: float,
    text_size: int,
) -> None:
    """Stack legends tightly below the colorbar using inch-based spacing."""
    fig_h = fig.get_size_inches()[1]
    y_cursor = cbar_bottom - (_LEGEND_PAD_BELOW_CBAR_IN / fig_h)
    for title, handles, labels in legend_specs:
        leg = fig.legend(
            handles,
            labels,
            title=title,
            loc="upper left",
            bbox_to_anchor=(legend_x, y_cursor),
            frameon=False,
            fontsize=text_size - 1,
            title_fontsize=text_size,
            labelcolor=_LEGEND_TEXT_COLOR,
        )
        _style_legend_title(leg)
        y_cursor -= _legend_block_height_in(len(labels), text_size) / fig_h

def _render_cbar_and_legends(
    fig: "Figure",
    sm: ScalarMappable,
    legend_specs: list[tuple[str, list, list]],
    text_size: int = 8,
    cbar_label: str = "z-score",
    *,
    cbar_scale: float = 1.0,
    separate_legend: bool = False,
) -> "Figure | None":
    """
    Place colorbar + legends on ``fig``, or on a dedicated second figure.

    When ``separate_legend=True``, neither colorbar nor legends are drawn on
    ``fig``; both go on the returned legend figure (colorbar on top, legends
    stacked below with the same inch-based spacing). Otherwise they are drawn
    on the left of ``fig`` and this returns ``None``.
    """
    if cbar_scale <= 0:
        raise ValueError(f"cbar_scale must be > 0, got {cbar_scale!r}")

    if separate_legend:
        cbar_h_in = _CBAR_HEIGHT_IN * cbar_scale
        blocks_in = sum(
            _legend_block_height_in(len(labs), text_size)
            for _, _, labs in legend_specs
        )
        margin_top_in = 0.25
        margin_bot_in = 0.30
        leg_h = max(
            2.0,
            margin_top_in
            + cbar_h_in
            + _LEGEND_PAD_BELOW_CBAR_IN
            + blocks_in
            + margin_bot_in,
        )
        leg_w = 2.8
        target = plt.figure(figsize=(leg_w, leg_h))
        # Host axes so fig.legend artists display (empty figures skip them)
        host = target.add_axes([0.0, 0.0, 1.0, 1.0])
        host.set_axis_off()
        cbar_left = 0.12
        cbar_top = 1.0 - (margin_top_in / leg_h)
        legend_x = cbar_left
    else:
        target = fig
        cbar_left = _CBAR_LEFT
        cbar_top = _CBAR_TOP
        legend_x = _LEGEND_X

    cbar_rect = _colorbar_axes_rect(
        target, left=cbar_left, top=cbar_top, cbar_scale=cbar_scale
    )
    cbar_ax = target.add_axes(cbar_rect)
    cbar = target.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=text_size)
    _style_colorbar_text(cbar, text_size)
    _stack_legends_below_cbar(
        target,
        legend_specs,
        cbar_bottom=cbar_rect[1],
        legend_x=legend_x,
        text_size=text_size,
    )
    return target if separate_legend else None

def _finish_heatmap_legends(
    fig: "Figure",
    sm: ScalarMappable,
    legend_specs: list[tuple[str, list, list]],
    *,
    text_size: int,
    cbar_label: str,
    cbar_scale: float,
    separate_legend: bool,
    left_with_legends: float,
    left_heatmap_only: float,
    right: float,
    top: float,
    bottom: float = 0.10,
) -> "Figure | tuple[Figure, Figure]":
    """Apply margins, draw cbar/legends, return ``fig`` or ``(fig, legend_fig)``."""
    left = left_heatmap_only if separate_legend else left_with_legends
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    legend_fig = _render_cbar_and_legends(
        fig,
        sm,
        legend_specs,
        text_size=text_size,
        cbar_label=cbar_label,
        cbar_scale=cbar_scale,
        separate_legend=separate_legend,
    )
    if separate_legend:
        assert legend_fig is not None
        return fig, legend_fig
    return fig

def _resolve_feature_list(
    pdata: "pAnnData",
    namelist: list[str],
    on: str,
    *,
    quiet: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Resolve user identifiers to ``.var_names``.

    Returns ``(resolved_accessions_or_peptides, unresolved_names)``.
    Uses ``all_matches=True`` so gene symbols expand to every matching accession.
    """
    on_norm = on.lower()
    if on_norm in ("peptide", "pep"):
        pep_names = set(pdata.pep.var_names.astype(str))  # type: ignore[union-attr]
        resolved = resolve_peptides(
            pdata, namelist, all_matches=True, on_empty="return", quiet=quiet
        ) or []
        unresolved = []
        for name in namelist:
            s = str(name)
            if s in pep_names:
                continue
            one = resolve_peptides(
                pdata, [name], all_matches=True, on_empty="return", quiet=True
            ) or []
            if not one:
                unresolved.append(s)
        return list(resolved), unresolved

    adata = get_adata(pdata, on)
    resolved = resolve_accessions(
        adata, namelist, all_matches=True, on_empty="return", quiet=quiet
    ) or []
    unresolved = []
    for name in namelist:
        one = resolve_accessions(
            adata, [name], all_matches=True, on_empty="return", quiet=True
        ) or []
        if not one:
            unresolved.append(str(name))
    return list(resolved), unresolved

def _gene_for_feature(adata: ad.AnnData, feature_id: str, gene_col: str = "Genes") -> str | None:
    if feature_id not in adata.var_names:
        return None
    if gene_col in adata.var.columns:
        g = adata.var.loc[feature_id, gene_col]
        if pd.notna(g) and str(g).strip():
            return str(g)
    return None

def _display_labels(
    adata: ad.AnnData,
    feature_ids: list[str | None],
    *,
    gene_col: str = "Genes",
    fallback_names: list[str | None] | None = None,
) -> list[str]:
    """Gene labels by default; disambiguate duplicates as ``GENE (accession)``."""
    genes: list[str | None] = []
    for i, fid in enumerate(feature_ids):
        if fid is None:
            fb = fallback_names[i] if fallback_names else None
            genes.append(str(fb) if fb is not None else "")
            continue
        g = _gene_for_feature(adata, fid, gene_col=gene_col)
        genes.append(g)

    counts = Counter(g for g in genes if g)
    labels: list[str] = []
    for fid, g in zip(feature_ids, genes):
        if fid is None:
            labels.append(g or "")
        elif g and counts[g] > 1:
            labels.append(f"{g} ({fid})")
        elif g:
            labels.append(g)
        else:
            labels.append(str(fid))
    return labels

def _validate_classes(obs: pd.DataFrame, classes: list[str]) -> list[str]:
    if classes is None:
        raise ValueError("`classes` is required (list of .obs column names).")
    if isinstance(classes, str):
        classes = [classes]
    if not classes:
        raise ValueError("`classes` must contain at least one .obs column name.")
    missing = [c for c in classes if c not in obs.columns]
    if missing:
        raise KeyError(f"classes not found in .obs: {missing}")
    return list(classes)

def _lookup_stats_df(pdata: "pAnnData", stats_key: str) -> pd.DataFrame:
    if stats_key not in pdata.stats:
        available = list(pdata.stats.keys())
        raise KeyError(
            f"stats_key {stats_key!r} not found in pdata.stats. "
            f"Available keys: {available}"
        )
    stored = pdata.stats[stats_key]
    if isinstance(stored, dict) and "contrasts" in stored:
        raise ValueError(
            f"stats_key {stats_key!r} is a mixed_de collection. "
            "Pass one contrast via proteins= list extracted from "
            f"pdata.stats[stats_key]['contrasts'][<label>], "
            "or store/use a single-contrast DE table (same as plot_volcano)."
        )
    if not isinstance(stored, pd.DataFrame):
        raise TypeError(
            f"stats_key {stats_key!r} did not resolve to a DataFrame "
            f"(got {type(stored).__name__})."
        )
    if "significance" not in stored.columns:
        raise KeyError(
            f"DE table at stats_key {stats_key!r} has no 'significance' column."
        )
    return stored

def _impute_row_median(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Impute NaNs with per-row median for clustering only.

    Returns ``(imputed, keep_mask)`` where ``keep_mask`` is False for rows with
    no finite values or zero variance after imputation.
    """
    imputed = mat.copy()
    n_rows = mat.shape[0]
    keep = np.ones(n_rows, dtype=bool)
    for i in range(n_rows):
        row = imputed[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            keep[i] = False
            continue
        med = float(np.median(finite))
        nan_idx = ~np.isfinite(row)
        row[nan_idx] = med
        if float(np.var(row)) == 0:
            keep[i] = False
    return imputed, keep

# ---------------------------------------------------------------------------
# plot_grouped_heatmap
# ---------------------------------------------------------------------------

def plot_grouped_heatmap(
    pdata: "pAnnData",
    protein_groups: dict[str, list[str]],
    classes: list[str],
    *,
    on: str = "protein",
    sort_by: dict[str, list[str]] | None = None,
    layer: str = "X",
    display_scale: str = "auto",
    group_colors: dict[str, str] | None = None,
    header_colors: dict[str, dict[str, str]] | None = None,
    cmap: str | None = None,
    gap_rows: float = 0.5,
    group_bar_pad: float = 0.25,
    sample_label_col: str | None = None,
    figsize: tuple[float, float] | None = None,
    text_size: int = 8,
    cbar_scale: float = 1.0,
    auto_log2: bool = True,
    gene_col: str = "Genes",
    separate_legend: bool = False,
    **kwargs: Any,
) -> "Figure | tuple[Figure, Figure]":
    """
    Plot a protein/peptide x sample heatmap with curated group blocks and sample headers.

    Proteins are ordered by ``protein_groups`` (dict insertion order), sorted by gene
    symbol within each group, with optional gap rows between groups. Samples are
    ordered by a multi-key sort over ``classes`` (optionally customized via
    ``sort_by``). Missing proteins still occupy a grey row so group block sizes stay
    stable. To subset samples, filter the object beforehand.

    Args:
        pdata (pAnnData): Input object.
        protein_groups (dict): Mapping ``group_name → list of gene/accession ids``.
            A protein listed in two groups is drawn twice. Duplicates *within* one
            group are uniquified.
        classes (list of str): ``.obs`` columns for header strips (top→bottom) and
            sample sort keys (outer→inner). Required.
        on (str): ``"protein"`` / ``"prot"`` or ``"peptide"`` / ``"pep"``.
        sort_by (dict, optional): Per-class category order, e.g.
            ``{"condition": ["Control", "Treated"]}``. Omitted classes use
            first-seen order.
        layer (str): Abundance layer (default ``"X"``).
        display_scale (str): ``"auto"`` (default), ``"zscore"``, ``"log"``, or
            ``"raw"``. Auto uses z-score for curated lists. ``"log"`` / ``"raw"``
            use a sequential colormap by default.
        group_colors (dict, optional): Override colors for group brackets/legend.
            Keys are group names; unspecified groups use package defaults (``get_color('colors', n)``).
        header_colors (dict, optional): Nested ``{class: {category: color}}``
            overrides for header strips. Unspecified categories use package defaults (``get_color('colors', n)``).
        cmap (str, optional): Colormap. ``None`` (default) selects ``RdBu_r`` for
            z-score and ``viridis`` for log/raw.
        gap_rows (float): Vertical spacer between groups in units of one protein
            row (default ``0.5``). Use ``0`` for no gap, ``1`` for a full row, etc.
        group_bar_pad (float): Horizontal gap (in heatmap column units) between the
            right edge of the heatmap and the colored group bars (default 0.25).
        sample_label_col (str, optional): ``.obs`` / ``.summary`` column for bottom
            tick labels (e.g. ``"replicate"``). Default ``None`` uses sample index
            names (``obs_names``).
        figsize (tuple, optional): ``(width, height)``. If None, height scales with
            row count.
        text_size (int): Base font size for ticks, colorbar, and legends (default 8).
            Relative sizes follow ``plot_pairwise_correlation`` (``text_size`` /
            ``text_size - 1`` / ``text_size + 3`` for title).
        cbar_scale (float): Vertical scale factor for the colorbar (default ``1.0``;
            base height ≈ 1.35 in). Category legends stack tightly just below it
            (inch-based spacing, so tall heatmaps do not stretch legend gaps).
            Also applies when ``separate_legend=True``.
        auto_log2 (bool): If True (default), apply in-memory ``log2(x+pseudocount)``
            when the layer looks linear-scale (same policy as ``mixed_de``). Used for
            z-score / log display paths; ignored for ``display_scale="raw"``.
        gene_col (str): ``.var`` column for gene display labels (default ``"Genes"``).
        separate_legend (bool): If True, draw colorbar and legends on a second
            figure (avoids overlap when resizing). Returns ``(fig, legend_fig)``.
            Default False returns ``fig`` only.
        **kwargs: Optional ``title`` (str) drawn as ``fig.suptitle``.

    Returns:
        fig (matplotlib.figure.Figure): The constructed heatmap figure.
        legend_fig (matplotlib.figure.Figure, optional): Returned when
            ``separate_legend=True``.

    Example:
        Grouped pathway heatmap with condition headers:
            ```python
            from scpviz import plotting as scplt

            fig = scplt.plot_grouped_heatmap(
                pdata,
                protein_groups={
                    "Cell cycle": ["CDK1", "CDK2", "PCNA"],
                    "Stress": ["HSPA1A", "HSP90AA1"],
                },
                classes=["treatment", "cellline"],
                sort_by={"treatment": ["DMSO", "Drug"]},
                layer="X",
            )
            ```

        Custom header and group colors (via ``get_color`` or hex):
            ```python
            c = scplt.get_color("colors", 7)
            fig = scplt.plot_grouped_heatmap(
                pdata,
                protein_groups={
                    "Cell cycle": ["CDK1", "PCNA"],
                    "Stress": ["HSPA1A"],
                },
                classes=["condition"],
                sort_by={"condition": ["SWI", "Uninjured"]},
                group_colors={"Cell cycle": c[0], "Stress": c[1]},
                header_colors={
                    "condition": {"SWI": "#FF0000", "Uninjured": "#6EDC00"},
                },
            )
            ```

        Spacing, fonts, display scale, and colorbar height:
            ```python
            fig = scplt.plot_grouped_heatmap(
                pdata,
                protein_groups={
                    "Cell cycle": ["CDK1", "PCNA"],
                    "Stress": ["HSPA1A"],
                },
                classes=["condition"],
                sample_label_col="replicate",
                gap_rows=0.5,
                group_bar_pad=0.15,
                display_scale="zscore",
                text_size=10,
                cbar_scale=0.8,
            )
            ```

        Legends on a separate figure (useful for setting a specific fig size):
            ```python
            fig, legend_fig = scplt.plot_grouped_heatmap(
                pdata,
                protein_groups={
                    "Cell cycle": ["CDK1", "PCNA"],
                    "Stress": ["HSPA1A"],
                },
                classes=["condition"],
                figsize=(4,3),
                separate_legend=True,
                cbar_scale=1.25,
            )
            ```
    """
    if not isinstance(protein_groups, dict) or not protein_groups:
        raise ValueError("`protein_groups` must be a non-empty dict of group → id list.")

    pdata._check_data(on)  # type: ignore[attr-defined]
    adata = get_adata(pdata, on)
    classes = _validate_classes(adata.obs, classes)
    samples = _compute_sample_order(
        adata.obs, list(adata.obs_names.astype(str)), classes, sort_by
    )
    sample_labels = _sample_tick_labels(adata.obs, samples, sample_label_col)

    scale = _resolve_display_scale_arg(
        display_scale, None, from_stats_key=False
    )
    cmap_resolved = _resolve_cmap_for_scale(cmap, scale)
    display_full, miss_full, _z_full, _miss_z, cbar_label, symmetric, _layer_is_log = (
        _resolve_display_and_cluster(
            adata, layer, display_scale=scale, auto_log2=auto_log2
        )
    )
    var_index = list(adata.var_names.astype(str))
    var_to_i = {v: i for i, v in enumerate(var_index)}

    group_names = list(protein_groups.keys())
    gcolors = _build_color_map(group_names, group_colors)

    row_features: list[str | None] = []
    row_groups: list[str | None] = []
    row_fallback: list[str | None] = []

    for gname in group_names:
        members = list(dict.fromkeys(protein_groups[gname]))
        _, unresolved = _resolve_feature_list(pdata, members, on, quiet=True)
        for u in unresolved:
            print(
                f"{format_log_prefix('warn')} protein '{u}' not found in this "
                f"pAnnData — row shown empty"
            )

        entries: list[tuple[str, str | None, str | None]] = []
        seen_in_group: set[str] = set()
        for name in members:
            one, _un = _resolve_feature_list(pdata, [name], on, quiet=True)
            if not one:
                entries.append((str(name).lower(), None, str(name)))
                continue
            for fid in one:
                if fid in seen_in_group:
                    continue
                seen_in_group.add(fid)
                g = _gene_for_feature(adata, fid, gene_col=gene_col) or fid
                entries.append((str(g).lower(), fid, None))

        entries.sort(key=lambda t: t[0])
        for _sk, fid, fb in entries:
            row_features.append(fid)
            row_groups.append(gname)
            row_fallback.append(fb)

    n_content = len(row_features)
    n_samples = len(samples)
    sample_idx = [list(adata.obs_names.astype(str)).index(s) for s in samples]

    values = np.full((n_content, n_samples), np.nan)
    is_missing = np.zeros((n_content, n_samples), dtype=bool)

    for i, fid in enumerate(row_features):
        if fid is None or fid not in var_to_i:
            is_missing[i, :] = True
            print(
                f"{format_log_prefix('info')} Row '{row_fallback[i] or fid}' is missing "
                f"from data (all cells grey)."
            )
            continue
        vi = var_to_i[fid]
        row_vals = display_full[vi, sample_idx]
        row_miss = miss_full[vi, sample_idx]
        values[i, :] = row_vals
        is_missing[i, :] = row_miss
        if bool(np.all(row_miss)):
            print(
                f"{format_log_prefix('info')} Protein '{fid}' has no detected values "
                f"in the plotted samples (all cells grey)."
            )

    labels = _display_labels(
        adata, row_features, gene_col=gene_col, fallback_names=row_fallback
    )

    rgba, norm = _composite_heatmap_rgba(
        values, is_missing, None, cmap_resolved, symmetric=symmetric
    )
    rgba_disp, tick_pos, tick_lab, group_yrange = _rasterize_with_gaps(
        rgba, row_groups, labels, gap_rows
    )
    n_disp = rgba_disp.shape[0]

    n_header = len(classes)
    header_h = 0.35
    main_h = max(8, max(n_disp, 1) * 0.024 * _ROW_PX)
    if figsize is None:
        fig_w, fig_h = 11.5, header_h * n_header + main_h * 0.42 + 1.5
    else:
        fig_w, fig_h = figsize

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_header + 1,
        ncols=1,
        height_ratios=[header_h] * n_header + [main_h],
        hspace=0.06,
    )
    header_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_header)]
    ax_main = fig.add_subplot(gs[n_header, 0])

    legend_specs = _render_header_rows(
        fig, header_axes, adata.obs, samples, classes, header_colors, text_size=text_size
    )

    ax_main.imshow(rgba_disp, aspect="auto", zorder=2)
    ax_main.set_xticks(range(n_samples))
    ax_main.set_xticklabels(sample_labels, fontsize=text_size - 1, rotation=90)
    ax_main.set_yticks(tick_pos)
    ax_main.set_yticklabels(tick_lab, fontsize=text_size - 2)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.tick_params(left=False, bottom=False)

    bar_x0 = n_samples - 0.5 + float(group_bar_pad)
    bar_x1 = bar_x0 + 0.4
    text_x = bar_x1 + 0.55
    for g, (r0, r1) in group_yrange.items():
        ax_main.add_patch(
            Rectangle(
                (bar_x0, r0 - 0.5),
                bar_x1 - bar_x0,
                (r1 - r0) + 1,
                facecolor=gcolors.get(g, "#999999"),
                edgecolor="none",
                clip_on=False,
                zorder=3,
            )
        )
        y_mid = (r0 + r1) / 2
        ax_main.text(
            text_x,
            y_mid,
            g,
            fontsize=text_size - 1,
            va="center",
            ha="center",
            color="#333333",
            rotation=-90,
            rotation_mode="anchor",
            clip_on=False,
            zorder=3,
        )

    sm = ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])

    group_handles = [
        Rectangle((0, 0), 1, 1, color=gcolors[g]) for g in group_names
    ]
    legend_specs = list(legend_specs) + [
        ("protein group", group_handles, group_names)
    ]

    title = kwargs.get("title")
    if title:
        fig.suptitle(title, fontsize=text_size + 3, fontweight="bold", x=0.55)

    return _finish_heatmap_legends(
        fig,
        sm,
        legend_specs,
        text_size=text_size,
        cbar_label=cbar_label,
        cbar_scale=cbar_scale,
        separate_legend=separate_legend,
        left_with_legends=0.24,
        left_heatmap_only=0.06,
        right=0.72,
        top=0.95,
    )

# ---------------------------------------------------------------------------
# plot_clustered_heatmap
# ---------------------------------------------------------------------------

def plot_clustered_heatmap(
    pdata: "pAnnData",
    classes: list[str],
    *,
    proteins: list[str] | None = None,
    stats_key: str | None = None,
    significance_categories: list[str] | str | None = None,
    protein_groups: dict[str, list[str]] | None = None,
    on: str = "protein",
    sort_by: dict[str, list[str]] | None = None,
    layer: str = "X",
    display_scale: str = "auto",
    metric: str = "correlation",
    cor_method: str = "pearson",
    linkage_method: str = "average",
    optimal_ordering: bool = True,
    show_unassigned: bool = True,
    group_colors: dict[str, str] | None = None,
    header_colors: dict[str, dict[str, str]] | None = None,
    label_color: str | None = None,
    sample_label_col: str | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    text_size: int = 8,
    cbar_scale: float = 1.0,
    auto_log2: bool = True,
    gene_col: str = "Genes",
    separate_legend: bool = False,
    **kwargs: Any,
) -> "Figure | tuple[Figure, Figure]":
    """
    Plot a hierarchically clustered protein/peptide × sample heatmap.

    Row order comes from hierarchical clustering (correlation or Euclidean
    distance with average linkage). Optional ``protein_groups`` are shown as a
    colour strip + coloured/bold right-side labels (not spatial blocks). Sample
    columns use the same ``classes`` / ``sort_by`` ordering as
    :func:`plot_grouped_heatmap`.

    Exactly one of ``proteins`` or ``stats_key`` must be provided.

    Args:
        pdata (pAnnData): Input object (needed for ``stats_key`` and peptide resolve).
        classes (list of str): ``.obs`` columns for headers and sample order. Required.
        proteins (list of str, optional): Explicit gene/accession list to cluster.
        stats_key (str, optional): Key in ``pdata.stats`` for a DE / mixed_de
            volcano table (same convention as ``plot_volcano``). Features whose
            ``significance`` is in ``significance_categories`` are plotted.
        significance_categories (list of str or str, optional): Categories kept when
            using ``stats_key``. Default ``["upregulated", "downregulated"]``.
            Pass a bare string or a one-element list for a single category, e.g.
            ``"not comparable"`` or ``["not comparable"]``.
        protein_groups (dict, optional): Optional annotation overlay. Proteins not
            in any group are labelled ``Unassigned`` (grey). If None, all rows are
            Unassigned. Set ``show_unassigned=False`` to drop Unassigned rows.
        on (str): ``"protein"`` or ``"peptide"``.
        sort_by (dict, optional): Per-class category order for sample columns.
        layer (str): Abundance layer (default ``"X"``).
        display_scale (str): ``"auto"`` (default), ``"zscore"``, ``"log"``, or
            ``"raw"``. Auto selects ``log`` when ``"not comparable"`` is among
            ``significance_categories``, otherwise ``zscore``. Clustering always
            uses z-scored values regardless of display scale.
        metric (str): ``"correlation"`` (default) or ``"euclidean"``.
        cor_method (str): ``"pearson"`` or ``"spearman"``; only used when
            ``metric="correlation"``.
        linkage_method (str): Passed to scipy linkage (default ``"average"``).
        optimal_ordering (bool): Improve leaf order for display (default True).
        show_unassigned (bool): If False, drop proteins not in ``protein_groups``.
        group_colors (dict, optional): Override colors for the group strip/legend
            (same pattern as ``plot_grouped_heatmap``).
        header_colors (dict, optional): Nested ``{class: {category: color}}``
            overrides for header strips.
        label_color (str, optional): Fixed color for all gene row labels (e.g.
            ``"black"``). If ``None`` (default), labels use ``protein_groups``
            colors (bold) and Unassigned stays grey.
        sample_label_col (str, optional): ``.obs`` / ``.summary`` column for bottom
            tick labels (e.g. ``"replicate"``). Default ``None`` uses sample index
            names (``obs_names``).
        cmap (str, optional): Colormap. ``None`` (default) selects ``RdBu_r`` for
            z-score and ``viridis`` for log/raw.
        figsize (tuple, optional): Figure size; auto height if None.
        text_size (int): Base font size for ticks, colorbar, and legends (default 8).
            Same convention as ``plot_grouped_heatmap`` / ``plot_pairwise_correlation``.
        cbar_scale (float): Vertical scale factor for the colorbar (default ``1.0``).
            Same behavior as ``plot_grouped_heatmap`` (legends stack below; applies
            to ``separate_legend`` too).
        auto_log2 (bool): Same in-memory log2 policy as ``plot_grouped_heatmap``.
        gene_col (str): Gene label column in ``.var``.
        separate_legend (bool): If True, colorbar and legends are drawn on a
            second figure. Returns ``(fig, legend_fig)``.
        **kwargs: Optional ``title``.

    Returns:
        fig (matplotlib.figure.Figure): The constructed heatmap figure.
        legend_fig (matplotlib.figure.Figure, optional): Returned when
            ``separate_legend=True``.

    Raises:
        ValueError: Invalid ``proteins``/``stats_key`` combination, ``metric``,
            or fewer than 2 clusterable rows after filtering.
        KeyError: ``stats_key`` missing from ``pdata.stats``.

    Example:
        Cluster an explicit protein list:
            ```python
            from scpviz import plotting as scplt

            fig = scplt.plot_clustered_heatmap(
                pdata,
                classes=["treatment", "cellline"],
                proteins=["CDK1", "PCNA", "HSPA1A", "GAPDH"],
                protein_groups={"Cell cycle": ["CDK1", "PCNA"]},
                show_unassigned=True,
            )
            ```

        Cluster DE hits from a stored contrast (default up + down):
            ```python
            fig = scplt.plot_clustered_heatmap(
                pdata,
                classes=["condition"],
                stats_key="SWI vs Uninjured",
                sort_by={"condition": ["SWI", "Uninjured"]},
            )
            ```

        Only upregulated, or only not-comparable features:
            ```python
            fig = scplt.plot_clustered_heatmap(
                pdata,
                classes=["condition"],
                stats_key="SWI vs Uninjured",
                significance_categories=["upregulated"],
            )
            fig = scplt.plot_clustered_heatmap(
                pdata,
                classes=["condition"],
                stats_key="SWI vs Uninjured",
                significance_categories="not comparable",
                label_color="black",
            )
            ```

        Fonts, row-label color, display scale, and colorbar height:
            ```python
            fig = scplt.plot_clustered_heatmap(
                pdata,
                classes=["condition"],
                stats_key="SWI vs Uninjured",
                sample_label_col="replicate",
                label_color="black",
                display_scale="log",
                text_size=10,
                cbar_scale=0.8,
            )
            ```

        Custom colors with a separate legend figure (useful for setting a specific fig size):
            ```python
            c = scplt.get_color("colors", 7)
            fig, legend_fig = scplt.plot_clustered_heatmap(
                pdata,
                classes=["condition"],
                stats_key="SWI vs Uninjured",
                protein_groups={"Cell cycle": ["Tubb2a", "Ncstn"]},
                group_colors={"Cell cycle": c[0]},
                header_colors={
                    "condition": {"SWI": "#FF0000", "Uninjured": "#6EDC00"},
                },
                figsize=(4,3),
                separate_legend=True,
                cbar_scale=1.25,
            )
            ```
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    if (proteins is None) == (stats_key is None):
        raise ValueError("Provide exactly one of `proteins` or `stats_key`.")

    pdata._check_data(on)  # type: ignore[attr-defined]
    adata = get_adata(pdata, on)
    classes = _validate_classes(adata.obs, classes)
    samples = _compute_sample_order(
        adata.obs, list(adata.obs_names.astype(str)), classes, sort_by
    )
    sample_labels = _sample_tick_labels(adata.obs, samples, sample_label_col)
    sample_idx = [list(adata.obs_names.astype(str)).index(s) for s in samples]

    sig_cats: list[str] | None = None
    if stats_key is not None:
        de_df = _lookup_stats_df(pdata, stats_key)
        if significance_categories is None:
            sig_cats = ["upregulated", "downregulated"]
        elif isinstance(significance_categories, str):
            sig_cats = [significance_categories]
        else:
            sig_cats = list(significance_categories)
        if "not comparable" in sig_cats:
            print(
                f"{format_log_prefix('warn')} Plotting 'not comparable' features: "
                f"these typically lack abundance in one group, so row z-scores "
                f"mostly reflect within-group variation (and missing values) "
                f"rather than a cross-group fold change."
            )
        mask = de_df["significance"].astype(str).isin(set(sig_cats))
        proteins = [str(i) for i in de_df.loc[mask].index]
        if not proteins:
            counts = (
                de_df["significance"].astype(str).value_counts().to_dict()
            )
            raise ValueError(
                f"No features with significance in {sig_cats!r} for "
                f"stats_key {stats_key!r}. Available counts: {counts}."
            )

    scale = _resolve_display_scale_arg(
        display_scale,
        sig_cats,
        from_stats_key=stats_key is not None,
    )
    cmap_resolved = _resolve_cmap_for_scale(cmap, scale)

    assert proteins is not None
    resolved, unresolved = _resolve_feature_list(pdata, proteins, on, quiet=False)
    for u in unresolved:
        print(
            f"{format_log_prefix('warn')} protein '{u}' not found in this "
            f"pAnnData — excluded from clustered heatmap"
        )
    # Preserve resolve order but uniquify
    feature_ids = list(dict.fromkeys(resolved))
    if not feature_ids:
        raise ValueError("No resolvable features to plot.")

    # Map feature → group (first wins); expand group member lists via resolve
    feature_to_group: dict[str, str] = {}
    group_names: list[str] = []
    if protein_groups:
        group_names = list(protein_groups.keys())
        for gname, members in protein_groups.items():
            mem_resolved, _ = _resolve_feature_list(
                pdata, list(dict.fromkeys(members)), on, quiet=True
            )
            for fid in mem_resolved:
                if fid not in feature_to_group:
                    feature_to_group[fid] = gname

    if not show_unassigned:
        if not protein_groups:
            raise ValueError(
                "show_unassigned=False requires protein_groups; otherwise every "
                "row is Unassigned and nothing remains to plot."
            )
        kept = [f for f in feature_ids if f in feature_to_group]
        n_drop = len(feature_ids) - len(kept)
        if n_drop:
            print(
                f"{format_log_prefix('info')} Dropping {n_drop} unassigned "
                f"feature(s) (show_unassigned=False)."
            )
        feature_ids = kept
        if not feature_ids:
            raise ValueError("No features remain after dropping unassigned rows.")

    display_full, miss_disp_full, z_full, miss_z_full, cbar_label, symmetric, _layer_is_log = (
        _resolve_display_and_cluster(
            adata, layer, display_scale=scale, auto_log2=auto_log2
        )
    )
    var_index = list(adata.var_names.astype(str))
    var_to_i = {v: i for i, v in enumerate(var_index)}

    # Assemble matrices for selected features × ordered samples
    rows_z = []
    rows_disp = []
    rows_miss_z = []
    rows_miss_disp = []
    valid_features: list[str] = []
    for fid in feature_ids:
        if fid not in var_to_i:
            print(
                f"{format_log_prefix('warn')} protein '{fid}' missing from "
                f".var_names — excluded"
            )
            continue
        vi = var_to_i[fid]
        rows_z.append(z_full[vi, sample_idx])
        rows_disp.append(display_full[vi, sample_idx])
        rows_miss_z.append(miss_z_full[vi, sample_idx])
        rows_miss_disp.append(miss_disp_full[vi, sample_idx])
        valid_features.append(fid)

    if not valid_features:
        raise ValueError("No features present in the abundance matrix.")

    values_z = np.vstack(rows_z)
    values_disp = np.vstack(rows_disp)
    is_missing_z = np.vstack(rows_miss_z)
    is_missing_disp = np.vstack(rows_miss_disp)

    # Clustering uses median-imputed z-scores; display keeps original NaNs
    cluster_mat = values_z.copy()
    cluster_mat[is_missing_z] = np.nan
    imputed, keep_mask = _impute_row_median(cluster_mat)

    if not np.all(keep_mask):
        dropped = [f for f, k in zip(valid_features, keep_mask) if not k]
        for f in dropped:
            print(
                f"{format_log_prefix('warn')} Excluding '{f}' from clustering "
                f"(zero variance or all missing after median imputation)."
            )
        values_z = values_z[keep_mask]
        values_disp = values_disp[keep_mask]
        is_missing_z = is_missing_z[keep_mask]
        is_missing_disp = is_missing_disp[keep_mask]
        imputed = imputed[keep_mask]
        valid_features = [f for f, k in zip(valid_features, keep_mask) if k]

    if len(valid_features) < 2:
        print(
            f"{format_log_prefix('error')} Need at least 2 features for "
            f"hierarchical clustering after filtering; got {len(valid_features)}."
        )
        raise ValueError(
            f"Need at least 2 clusterable features; got {len(valid_features)}."
        )

    if metric not in ("correlation", "euclidean"):
        raise ValueError(
            f"metric must be 'correlation' or 'euclidean', got {metric!r}"
        )
    if metric == "euclidean" and cor_method != "pearson":
        print(
            f"{format_log_prefix('warn')} cor_method={cor_method!r} is ignored "
            f"when metric='euclidean'."
        )

    if metric == "correlation":
        Z, _ = correlation_linkage(
            imputed,
            method=cor_method,
            linkage_method=linkage_method,
            optimal_ordering=optimal_ordering,
        )
    else:
        condensed = pdist(imputed, metric="euclidean")
        Z = linkage(
            condensed, method=linkage_method, optimal_ordering=optimal_ordering
        )

    # Figure layout: headers above heatmap column only
    n_header = len(classes)
    n_rows = len(valid_features)
    n_samples = len(samples)
    header_h = 0.35
    main_h = max(8, n_rows * 0.24)
    if figsize is None:
        fig_w, fig_h = 11.0, header_h * n_header + main_h * 0.42 + 1.5
    else:
        fig_w, fig_h = figsize

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_header + 1,
        ncols=3,
        height_ratios=[header_h] * n_header + [main_h],
        width_ratios=[1.3, 0.18, 10],
        hspace=0.02,
        wspace=0.02,
    )

    # Spacer cells over dendro/strip
    for i in range(n_header):
        for j in range(2):
            ax_blank = fig.add_subplot(gs[i, j])
            ax_blank.axis("off")

    header_axes = [fig.add_subplot(gs[i, 2]) for i in range(n_header)]
    ax_dendro = fig.add_subplot(gs[n_header, 0])
    ax_strip = fig.add_subplot(gs[n_header, 1])
    ax_main = fig.add_subplot(gs[n_header, 2])

    legend_specs = _render_header_rows(
        fig, header_axes, adata.obs, samples, classes, header_colors, text_size=text_size
    )

    dn = dendrogram(
        Z,
        ax=ax_dendro,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#999999",
    )
    ax_dendro.invert_yaxis()
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    leaf_order = dn["leaves"]
    values_disp = values_disp[leaf_order]
    is_missing_disp = is_missing_disp[leaf_order]
    valid_features = [valid_features[i] for i in leaf_order]
    row_groups = [feature_to_group.get(f) for f in valid_features]

    rgba, norm = _composite_heatmap_rgba(
        values_disp, is_missing_disp, None, cmap_resolved, symmetric=symmetric
    )
    ax_main.imshow(rgba, aspect="auto", zorder=2)

    labels = _display_labels(adata, valid_features, gene_col=gene_col)
    ax_main.set_xticks(range(n_samples))
    ax_main.set_xticklabels(sample_labels, fontsize=text_size - 1, rotation=90)
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels(labels, fontsize=text_size - 2)
    ax_main.yaxis.tick_right()

    # Group colors for strip + labels
    if group_names:
        gcolors = _build_color_map(group_names, group_colors)
    else:
        gcolors = {}
    for i, lab in enumerate(ax_main.get_yticklabels()):
        g = row_groups[i]
        if label_color is not None:
            lab.set_color(label_color)
            if g is not None:
                lab.set_fontweight("bold")
        elif g is not None:
            lab.set_color(gcolors.get(g, "#333333"))
            lab.set_fontweight("bold")
        else:
            lab.set_color(UNASSIGNED_LABEL_COLOUR)

    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.tick_params(left=False, right=False, bottom=False)

    # Colour strip
    if group_names:
        strip_codes = np.full((n_rows, 1), np.nan)
        for i, g in enumerate(row_groups):
            if g is not None:
                strip_codes[i, 0] = group_names.index(g)
        strip_cmap = ListedColormap([gcolors[g] for g in group_names])
        strip_cmap.set_bad(UNASSIGNED_COLOUR)
        ax_strip.imshow(
            strip_codes,
            aspect="auto",
            cmap=strip_cmap,
            vmin=-0.5,
            vmax=max(len(group_names) - 0.5, 0.5),
        )
    else:
        strip_codes = np.zeros((n_rows, 1))
        strip_cmap = ListedColormap([UNASSIGNED_COLOUR])
        ax_strip.imshow(strip_codes, aspect="auto", cmap=strip_cmap, vmin=0, vmax=1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    sm = ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])

    # Legends: protein groups then header classes
    group_legend_labels = list(group_names) + ["Unassigned"]
    group_handles = [Rectangle((0, 0), 1, 1, color=gcolors[g]) for g in group_names]
    group_handles.append(Rectangle((0, 0), 1, 1, color=UNASSIGNED_COLOUR))
    stacked = [("protein group", group_handles, group_legend_labels)] + list(
        legend_specs
    )

    title = kwargs.get("title")
    if title:
        fig.suptitle(title, fontsize=text_size + 3, fontweight="bold", x=0.55)

    return _finish_heatmap_legends(
        fig,
        sm,
        stacked,
        text_size=text_size,
        cbar_label=cbar_label,
        cbar_scale=cbar_scale,
        separate_legend=separate_legend,
        left_with_legends=0.24,
        left_heatmap_only=0.06,
        right=0.88,
        top=0.94,
    )

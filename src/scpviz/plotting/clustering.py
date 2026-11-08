"""Pairwise correlation and publication heatmaps / clustermaps.

Includes ``plot_pairwise_correlation``, ``plot_grouped_heatmap``,
``plot_clustered_heatmap``, and ``plot_clustermap``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, TYPE_CHECKING

import copy
import warnings

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from scpviz import utils
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

from .style import _get_cmap, _resolve_subset_mask, get_color

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from scpviz.pAnnData.pAnnData import pAnnData

def _pairwise_corr_subset_cache_key(mask: np.ndarray) -> tuple[int, ...] | None:
    """Match :meth:`pairwise_correlation` ``subset_indices`` (None = all samples)."""
    if mask.shape[0] == 0:
        return None
    if bool(np.all(mask)):
        return None
    return tuple(np.flatnonzero(mask).tolist())

def plot_pairwise_correlation(
    pdata: pAnnData,
    classes: str | list[str],
    on: str = "protein",
    layer: str = "X",
    method: str = "pearson",
    order: list | None = None,
    show_samples: bool = False,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    annotation_cmap: str | dict | list = "default",
    figsize: tuple | None = None,
    text_size: int = 9,
    colorbar_label: str | None = None,
    annot: bool = False,
    annot_fmt: str = ".2f",
    annot_size: int = 7,
    title: str | None = None,
    force: bool = False,
    subset_mask: np.ndarray | pd.Series | list | None = None,
    show_annotation_legend: bool = True,
    legend_anchor_x: float = 0.3,
    show_ticklabels: bool | None = None,
    ticklabels_auto_max_samples: int = 20,
) -> "tuple[Figure, plt.Axes]":
    """  # noqa: D401
    Plot a pairwise protein/peptide abundance correlation heatmap across groups or samples in `.obs`.

    Automatically runs :meth:`~scpviz.pAnnData.pAnnData.pairwise_correlation` if
    results are not already cached (or if ``force=True``). The figure is created
    internally; no ``ax`` argument is needed.

    Cached analysis results are reused when ``classes``, ``method``, ``layer``, and
    ``subset_mask`` (via the same key as ``pairwise_correlation``) match. If
    ``show_samples=True`` but the cache lacks a sample matrix, analysis is rerun with
    ``compute_sample_matrix=True``. Group-level plots may reuse a cache that already
    includes a sample matrix (nothing is stripped). Display ``order`` is applied only
    when drawing and does not require recomputation.

    Args:
        pdata: Input pAnnData object.
        classes: `.obs` column(s) defining groups — passed to ``pairwise_correlation``.
        on: ``"protein"`` or ``"peptide"`` (default ``"protein"``).
        layer: Data layer (default ``"X"``).
        method: ``"pearson"``, ``"spearman"``, or ``"euclidean"``.
        order: Optional row/column order. Must match the matrix being plotted:

            - ``show_samples=False``: group labels — for a single ``classes`` column,
              values like ``"AS"``; for ``classes=[...]``, combined strings exactly as
              produced by :func:`~scpviz.utils.get_samplenames` (e.g. ``"AS, kd"`` with
              the stored comma-space separator).

            - ``show_samples=True``: **observation names** only — i.e. entries of
              ``adata.obs_names`` (however your object labels samples, e.g. PD import
              sample IDs), **not** combined group strings. To order samples by group,
              build a list of those obs names in the desired sequence (e.g. all
              samples of one group, then the next).

            If ``None``, uses storage order (group order from analysis, or sample order
            used when computing the sample matrix).
        show_samples: If False (default), plot the group × group matrix. If True,
            plot the sample × sample matrix (requires ``compute_sample_matrix`` in cache
            or triggers a run that computes it).
        cmap: Matplotlib colormap for the heatmap.
        vmin: Colormap lower limit; correlation methods default to ``-1`` if ``None``.
        vmax: Colormap upper limit; correlation methods default to ``1`` if ``None``.
        annotation_cmap: ``"default"`` (independent palette per obs column), or a
            single ``dict``, ``list``, or matplotlib cmap name shared across annotation bars.
        figsize: ``(width, height)`` in inches; if ``None``, auto-estimated.
        text_size: Base font size for ticks, colorbar, and legends.
        colorbar_label: Override colorbar label.
        annot: If True, write numeric values in each cell.
        annot_fmt: Format string for cell annotations (e.g. ``".2f"``).
        annot_size: Font size for cell annotations.
        title: Optional figure suptitle.
        force: If True, recompute ``pairwise_correlation`` even if cache matches.
        subset_mask: Boolean mask or boolean ``Series`` aligned to ``adata.obs``
            (same semantics as :func:`plot_pca`). All-True is normalized to
            ``None`` for cache parity with full-data analysis.
        show_annotation_legend: If True (default), draw one legend per annotation
            track in a dedicated GridSpec column right of the colorbar (obs column
            names also appear on the left vertical bar axes; top bars stay unlabeled).
        legend_anchor_x: Horizontal anchor for annotation legends inside the legend
            column, in axes coordinates (``0`` = left edge of that column, ``1`` = right).
            Larger values shift legends to the **right**, away from the colorbar, which
            helps if they overlap the colorbar. Typical values to try: about ``0.15`` to
            ``0.45`` (default ``0.3``). Ignored when ``show_annotation_legend`` is False.
        show_ticklabels: When ``show_samples=True``, controls sample names on the
            **x-axis** only (y-axis stays unlabeled to avoid clashing with annotation
            bars). ``None`` (default) shows ticks if ``n_samples <= ticklabels_auto_max_samples``
            and otherwise hides them and prints an info line. ``True`` / ``False`` force
            on or off. Ignored when ``show_samples=False`` (group-level always shows
            x-axis group labels).
        ticklabels_auto_max_samples: When ``show_ticklabels is None`` and
            ``show_samples=True``, sample names are shown only if the sample count is
            at most this value (default ``20``). Must be >= 1.

    Returns:
        ``(fig, ax_heatmap)``.

    Note:
        Heatmap row (y) tick labels are always omitted (symmetric matrix; x-axis labels
        carry sample or group names as applicable).
        ``tight_layout`` may warn on some backends; layout is non-fatal if it fails.

    Raises:
        ValueError: If ``sample_matrix`` is missing when ``show_samples=True``, or if
            ``ticklabels_auto_max_samples`` < 1.

    Example:
        Sample × sample Pearson correlation on a per-protein z-score layer (``X_pw_zscore``):
            ```python
            import matplotlib.pyplot as plt
            import numpy as np
            from scpviz import plotting as scplt
            from scpviz import utils as scu

            adata = scu.get_adata(pdata_norm, "protein")
            X = np.asarray(scu.get_adata_layer(adata, "X"), dtype=float)
            mu = np.nanmean(X, axis=0, keepdims=True)
            sig = np.nanstd(X, axis=0, keepdims=True)
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)
            adata.layers["X_pw_zscore"] = (X - mu) / sig

            fig, ax = scplt.plot_pairwise_correlation(
                pdata_norm,
                classes=["cellline", "condition"],
                method="pearson",
                show_samples=True,
                layer="X_pw_zscore",
                force=True,
            )
            plt.show()
            ```

        ![Plot pairwise correlation](../../assets/plots/plot_pairwise_correlation.png)

        Same approach on single-cell protein data (``classes`` aligned with UMAP, e.g. ``region``):
            ```python
            import matplotlib.pyplot as plt
            import numpy as np
            from scpviz import plotting as scplt
            from scpviz import utils as scu

            adata = scu.get_adata(pdata_sc, "protein")
            X = np.asarray(scu.get_adata_layer(adata, "X"), dtype=float)
            mu = np.nanmean(X, axis=0, keepdims=True)
            sig = np.nanstd(X, axis=0, keepdims=True)
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)
            adata.layers["X_pw_zscore"] = (X - mu) / sig

            fig, ax = scplt.plot_pairwise_correlation(
                pdata_sc,
                classes=["region"],
                method="pearson",
                show_samples=True,
                layer="X_pw_zscore",
                force=True,
            )
            plt.show()
            ```

        ![Plot pairwise correlation (single-cell)](../../assets/plots/plot_pairwise_correlation_sc.png)

        Imports and group-level heatmap (``show_samples=False``, default). Uses cached
        ``pairwise_correlation`` results when parameters match; pass ``force=True`` to
        recompute after changing ``.X`` or normalization:
            ```python
            from scpviz import plotting as scplt

            fig, ax = scplt.plot_pairwise_correlation(pdata, classes="cellline", method="pearson")
            ```

        Sample × sample heatmap (``show_samples=True``). Triggers or reuses analysis with
        ``compute_sample_matrix=True``. Euclidean distances use NaN-aware geometry on raw
        abundance rows; pick a sequential ``cmap`` (e.g. ``viridis``) for distances:
            ```python
            fig, ax = scplt.plot_pairwise_correlation(
                pdata,
                classes=["cellline", "treatment"],
                show_samples=True,
                method="euclidean",
                cmap="viridis",
            )
            ```

        Force sample names on the x-axis when there are many samples (auto-hide uses
        ``ticklabels_auto_max_samples`` when ``show_ticklabels=None``):
            ```python
            fig, ax = scplt.plot_pairwise_correlation(
                pdata,
                classes="cellline",
                show_samples=True,
                show_ticklabels=True,
            )
            ```

        **annotation_cmap** — ``"default"`` (omit or pass explicitly): independent
        categorical palette per ``.obs`` column, built from sorted unique values:
            ```python
            fig, ax = scplt.plot_pairwise_correlation(
                pdata, classes=["cellline", "treatment"], annotation_cmap="default"
            )
            ```

        **annotation_cmap** — ``dict`` mapping stringified ``.obs`` levels to colors; the
        same dict is reused for every annotation column (cover all levels that appear):
            ```python
            ann = {"AS": "#E41A1C", "BE": "#377EB8", "kd": "#4DAF4A", "sc": "#984EA3"}
            fig, ax = scplt.plot_pairwise_correlation(
                pdata, classes=["cellline", "treatment"], annotation_cmap=ann
            )
            ```

        **annotation_cmap** — ``list`` of colors, assigned in sorted-level order **within
        each** obs column (cycles if there are more levels than colors):
            ```python
            fig, ax = scplt.plot_pairwise_correlation(
                pdata, classes="cellline", annotation_cmap=["#FC9744", "#00AEE8", "#9D9D9D"]
            )
            ```

        **annotation_cmap** — matplotlib colormap **name**: evenly spaced colors for each
        column's sorted uniques:
            ```python
            fig, ax = scplt.plot_pairwise_correlation(pdata, classes="cellline", annotation_cmap="tab10")
            ```

        Custom row/column order without recomputing (labels must exist in the matrix).
        For **group** heatmaps, use combined strings when ``classes`` is a list (e.g.
        ``"AS, kd"``):
            ```python
            fig, ax = scplt.plot_pairwise_correlation(
                pdata, classes=["cellline", "treatment"],
                order=["AS, kd", "BE, sc", "AS, sc", "BE, kd"],
            )
            ```

        For **sample** heatmaps, ``order`` must be **observation names** (same strings as
        ``pdata.prot.obs_names``), not ``"AS, kd"`` group tokens — for example reverse
        or subset the index:
            ```python
            names = list(pdata.prot.obs_names)
            fig, ax = scplt.plot_pairwise_correlation(
                pdata,
                classes=["cellline", "treatment"],
                show_samples=True,
                order=list(reversed(names)),
            )
            ```

        Subset of samples (boolean mask or ``Series`` aligned to ``adata.obs_names``) and
        no annotation legends:
            ```python
            mask = pdata.prot.obs["cellline"].eq("AS").to_numpy()
            fig, ax = scplt.plot_pairwise_correlation(
                pdata, classes="treatment", subset_mask=mask, show_annotation_legend=False
            )
            ```

        Small matrices — show numeric values in cells; adjust legend horizontal position if
        it overlaps the colorbar:
            ```python
            fig, ax = scplt.plot_pairwise_correlation(
                pdata, classes="cellline", annot=True, legend_anchor_x=0.45
            )
            ```
    """
    if ticklabels_auto_max_samples < 1:
        raise ValueError(
            f"{utils.format_log_prefix('error')} ticklabels_auto_max_samples must be >= 1, "
            f"got {ticklabels_auto_max_samples}."
        )

    adata = utils.get_adata(pdata, on)
    mask = _resolve_subset_mask(adata, subset_mask)
    subset_indices_key = _pairwise_corr_subset_cache_key(mask)
    subset_for_pc = None if subset_indices_key is None else mask

    prev = adata.uns.get("pairwise_corr")
    if isinstance(prev, dict):
        needs_sample_matrix = show_samples and not prev.get("compute_sample_matrix", False)
    else:
        needs_sample_matrix = bool(show_samples)

    needs_recompute = (
        force
        or not isinstance(prev, dict)
        or prev.get("classes") != classes
        or prev.get("method") != method
        or prev.get("layer") != layer
        or prev.get("subset_indices") != subset_indices_key
        or needs_sample_matrix
    )

    if needs_recompute:
        pdata.pairwise_correlation(
            classes=classes,
            on=on,
            layer=layer,
            method=method,
            order=None,
            compute_sample_matrix=show_samples,
            subset_mask=subset_for_pc,
            force=force,
        )
    else:
        print(
            f"{utils.format_log_prefix('info')} Using cached pairwise_corr results. "
            "Pass force=True to recompute."
        )

    result = adata.uns["pairwise_corr"]
    classes_list = result["classes_list"]
    separator = result["separator"]
    method_used = result["method"]

    if show_samples:
        if result.get("sample_matrix") is None:
            raise ValueError(
                f"{utils.format_log_prefix('error')} sample_matrix is None — "
                "rerun pairwise_correlation with compute_sample_matrix=True or call "
                "plot_pairwise_correlation with show_samples=True (which requests it)."
            )
        matrix_df = result["sample_matrix"].copy()
    else:
        matrix_df = result["group_matrix"].copy()

    _mat_kind = "sample" if show_samples else "group"
    if order is not None:
        if len(order) != len(set(order)):
            raise ValueError(
                f"{utils.format_log_prefix('error')} order contains duplicate {_mat_kind} labels."
            )
        missing = [x for x in order if x not in matrix_df.index]
        if missing:
            extra = ""
            if show_samples:
                extra = (
                    " For sample-level plots, order must list observation names "
                    "(prot/pep `.obs_names`), not combined group labels like 'AS, kd'. "
                    "Use show_samples=False if you want to reorder by group label."
                )
            raise ValueError(
                f"{utils.format_log_prefix('error')} order contains labels not in the "
                f"{_mat_kind} matrix: {missing}.{extra}"
            )
        matrix_df = matrix_df.reindex(index=order, columns=order)
        order_used = list(order)
    else:
        order_used = list(matrix_df.index)

    n_groups = len(order_used)
    n_ann = len(classes_list)

    if show_samples:
        if show_ticklabels is None:
            _show_ticks = n_groups <= ticklabels_auto_max_samples
            if not _show_ticks:
                print(
                    f"{utils.format_log_prefix('info')} {n_groups} samples — tick labels "
                    f"hidden by default (threshold={ticklabels_auto_max_samples}). "
                    "Pass show_ticklabels=True to force them on."
                )
        else:
            _show_ticks = bool(show_ticklabels)
    else:
        _show_ticks = True

    if figsize is None:
        side = max(5.0, n_groups * 0.55)
        ann_width = n_ann * 0.3
        cbar_width = 0.5
        legend_width = 1.5 if show_annotation_legend else 0.0
        fig_w = side + ann_width * 2 + cbar_width + legend_width
        fig_h = side + ann_width * 2
        figsize = (fig_w, fig_h)
        print(
            f"{utils.format_log_prefix('info')} Auto-computed figsize={figsize}. "
            "Pass figsize=(w, h) to override."
        )

    fig = plt.figure(figsize=figsize)
    height_ratios = [0.04] * n_ann + [1.0]
    if show_annotation_legend:
        legend_col_ratio = 0.25
        width_ratios = [0.04] * n_ann + [1.0, 0.04, legend_col_ratio]
        ncols_gs = n_ann + 3
    else:
        width_ratios = [0.04] * n_ann + [1.0, 0.04]
        ncols_gs = n_ann + 2
    gs = GridSpec(
        nrows=n_ann + 1,
        ncols=ncols_gs,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.02,
        wspace=0.05 if show_annotation_legend else 0.02,
    )
    ax_heatmap = fig.add_subplot(gs[n_ann, n_ann])
    ax_cbar = fig.add_subplot(gs[n_ann, n_ann + 1])
    ax_top = [fig.add_subplot(gs[i, n_ann]) for i in range(n_ann)]
    ax_left = [fig.add_subplot(gs[n_ann, i]) for i in range(n_ann)]
    if show_annotation_legend:
        # One axis spanning the full legend column (plan's per-row 0.04-height cells would crush legends)
        ax_leg_col = fig.add_subplot(gs[0 : n_ann + 1, n_ann + 2])
        ax_leg_col.set_axis_off()
    else:
        ax_leg_col = None

    _grey = "#bfbfbf"

    def _ann_colors_for_column(col: str) -> dict:
        unique_vals = sorted(adata.obs[col].astype(str).unique().tolist())
        n_uv = len(unique_vals)
        if annotation_cmap == "default":
            pal = get_color("colors", n=n_uv)
            return {v: pal[i] for i, v in enumerate(unique_vals)}
        if isinstance(annotation_cmap, dict):
            out: dict = {}
            for v in unique_vals:
                if v not in annotation_cmap:
                    warnings.warn(
                        f"annotation_cmap missing key {v!r} for column {col!r}; using grey.",
                        UserWarning,
                        stacklevel=2,
                    )
                    out[v] = _grey
                else:
                    out[v] = annotation_cmap[v]
            return out
        if isinstance(annotation_cmap, list):
            if not annotation_cmap:
                raise ValueError("annotation_cmap list must be non-empty.")
            return {
                v: annotation_cmap[i % len(annotation_cmap)]
                for i, v in enumerate(unique_vals)
            }
        if isinstance(annotation_cmap, str):
            cmap_obj = _get_cmap(annotation_cmap)
            if n_uv == 0:
                return {}
            rgba = cmap_obj(np.linspace(0.0, 1.0, n_uv))
            return {v: rgba[i] for i, v in enumerate(unique_vals)}
        raise TypeError(
            "annotation_cmap must be 'default', dict, non-empty list, or str (cmap name)."
        )

    ann_color_dicts = [_ann_colors_for_column(c) for c in classes_list]

    n_parts = len(classes_list)
    group_parts: list[list[str]] = []
    if not show_samples:
        if separator is not None:
            for combined_label in order_used:
                parts = str(combined_label).split(
                    separator, maxsplit=max(0, n_parts - 1)
                )
                if len(parts) < n_parts:
                    raise ValueError(
                        f"{utils.format_log_prefix('error')} Cannot split combined label "
                        f"{combined_label!r} into {n_parts} parts with separator {separator!r}."
                    )
                group_parts.append(parts)
        else:
            group_parts = [[str(g)] for g in order_used]

    for i, col in enumerate(classes_list):
        if show_samples:
            group_col_labels = [
                str(adata.obs.loc[sample_name, col]) for sample_name in order_used
            ]
        else:
            col_idx = i
            if separator is None:
                group_col_labels = [str(g) for g in order_used]
            else:
                group_col_labels = [row[col_idx] for row in group_parts]

        colors_for_bar = [ann_color_dicts[i][str(lbl)] for lbl in group_col_labels]
        color_row = np.array([mcolors.to_rgba(c) for c in colors_for_bar])[np.newaxis, :, :]
        ax_top[i].imshow(color_row, aspect="auto", interpolation="nearest")
        ax_top[i].set_xticks([])
        ax_top[i].set_yticks([])
        for spine in ax_top[i].spines.values():
            spine.set_visible(False)

        color_col = np.array([mcolors.to_rgba(c) for c in colors_for_bar])[:, np.newaxis, :]
        ax_left[i].imshow(color_col, aspect="auto", interpolation="nearest")
        ax_left[i].set_xticks([0])
        ax_left[i].set_xticklabels([col], fontsize=text_size - 1, rotation=90)
        ax_left[i].xaxis.set_label_position("top")
        ax_left[i].xaxis.tick_top()
        ax_left[i].set_yticks([])
        for spine in ax_left[i].spines.values():
            spine.set_visible(False)

    mat = np.asarray(matrix_df.values, dtype=float)
    if not np.any(np.isfinite(mat)):
        raise ValueError(
            f"{utils.format_log_prefix('error')} Heatmap matrix has no finite values "
            "(often caused by NaNs in sample–sample distances or correlations)."
        )
    if vmin is None:
        vmin = -1.0 if method_used in ("pearson", "spearman") else float(np.nanmin(mat))
    if vmax is None:
        vmax = 1.0 if method_used in ("pearson", "spearman") else float(np.nanmax(mat))

    _cmap_base = _get_cmap(cmap)
    try:
        cmap_obj = _cmap_base.copy()
    except AttributeError:
        cmap_obj = copy.copy(_cmap_base)
    cmap_obj.set_bad(color=(0.82, 0.82, 0.82, 1.0))
    mat_show = np.ma.masked_invalid(mat)

    im = ax_heatmap.imshow(
        mat_show,
        aspect="auto",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if _show_ticks:
        ax_heatmap.set_xticks(range(n_groups))
        ax_heatmap.set_xticklabels(order_used, rotation=90, fontsize=text_size)
    else:
        ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.tick_params(axis="x", which="both", length=0)

    default_cbar_labels = {
        "pearson": "Pearson r",
        "spearman": "Spearman r",
        "euclidean": "Euclidean distance",
    }
    clab = colorbar_label or default_cbar_labels.get(method_used, method_used)
    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label(clab, fontsize=text_size)
    cb.ax.tick_params(labelsize=text_size - 1)

    if annot:
        for row in range(n_groups):
            for col_j in range(n_groups):
                val = mat[row, col_j]
                if not np.isfinite(val):
                    continue
                norm_val = (val - vmin) / (vmax - vmin + 1e-9)
                tcol = "white" if norm_val < 0.5 else "black"
                ax_heatmap.text(
                    col_j,
                    row,
                    format(val, annot_fmt),
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    color=tcol,
                )

    if title:
        fig.suptitle(title, fontsize=text_size + 1, y=1.01)

    if show_annotation_legend and ax_leg_col is not None:
        n_leg = len(classes_list)
        for i, col in enumerate(classes_list):
            handles = [
                mpatches.Patch(color=ann_color_dicts[i][v], label=v)
                for v in sorted(ann_color_dicts[i], key=lambda x: str(x))
            ]
            y_frac = 1.0 - (i + 0.5) / max(n_leg, 1)
            leg = ax_leg_col.legend(
                handles=handles,
                title=col,
                loc="center left",
                bbox_to_anchor=(legend_anchor_x, y_frac),
                bbox_transform=ax_leg_col.transAxes,
                borderaxespad=0.0,
                fontsize=text_size - 1,
                title_fontsize=text_size,
                frameon=False,
            )
            ax_leg_col.add_artist(leg)

    try:
        fig.tight_layout(rect=[0, 0, 1, 0.97] if title else [0, 0, 1, 1])
    except Exception:
        pass
    return fig, ax_heatmap


# ---------------------------------------------------------------------------
# Heatmap helpers and plots
# ---------------------------------------------------------------------------

MISSING_GREY = (0.80, 0.80, 0.80, 1.0)
GAP_WHITE = (1.0, 1.0, 1.0, 0.0)
UNASSIGNED_COLOUR = "#dddddd"
UNASSIGNED_LABEL_COLOUR = "#999999"

# Colorbar + legend column on the left (figure fraction / fixed inches).
# Anchored near the dendrogram / main axes, not the figure edge.
_CBAR_LEFT = 0.02
_CBAR_WIDTH_IN = 0.14
_CBAR_HEIGHT_IN = 1.35
_CBAR_TOP = 0.82  # figure-fraction top of colorbar (inline path)
_LEGEND_X = 0.05
_LEGEND_PAD_BELOW_CBAR_IN = 0.12
_DEFAULT_LEFT_SEPARATE = 0.06  # subplots left when separate_legend=True
_LEGEND_WIDTH_MIN = 0.18
_LEGEND_WIDTH_MAX = 0.55

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


_ROW_PX = 10  # content-row pixel height when row/column gaps are on
_COL_PX = 10  # content-column pixel width when column_spacing > 0 (match _ROW_PX)
# Default gap at row_spacing=True / column_spacing=True: half a content cell
_GROUP_GAP_PX = int(round(0.5 * _ROW_PX))  # 5
# Shared GridSpec spacing between header strips and the heatmap body
_DEFAULT_HEADER_SPACING = 0.06
# Relative GridSpec height for each header strip (vs main heatmap ratio)
_DEFAULT_HEADER_HEIGHT = 0.35
# Grouped-heatmap side bar width in sample-column units (like group_bar_pad)
_DEFAULT_GROUP_BAR_WIDTH = 0.4


def _resolve_spacing_scale(spacing: bool | float, name: str) -> float:
    """``True`` → 1× default gap; ``False``/``0`` → none; float → non-negative scale."""
    if spacing is True:
        return 1.0
    if spacing is False:
        return 0.0
    scale = float(spacing)
    if scale < 0:
        raise ValueError(f"{name} must be >= 0, got {spacing!r}")
    return scale


def _rasterize_with_column_gaps(
    rgba: np.ndarray,
    samples: list[str],
    obs: pd.DataFrame,
    classes: list[str],
    column_spacing: bool | float,
    sample_labels: list[str],
    *,
    col_px: int = _COL_PX,
    gap_px_base: int = _GROUP_GAP_PX,
) -> tuple[np.ndarray, list[float], list[str], list[int | None]]:
    """
    Expand sample columns so ``column_spacing`` inserts white gaps between leaf blocks.

    A leaf block is a run of samples that share the same full ``classes`` tuple
    (after sample ordering). ``column_spacing`` may be ``True`` (1× default),
    ``False``/``0`` (no gaps; 1:1 columns), or a non-negative float scale.

    Returns ``(rgba_disp, tick_positions, tick_labels, col_map)`` where
    ``col_map[j]`` is the sample index into ``samples`` for display column ``j``,
    or ``None`` for a gap column (for header strip alignment).
    """
    scale = _resolve_spacing_scale(column_spacing, "column_spacing")

    n_rows, n_samples = rgba.shape[0], rgba.shape[1]
    if n_samples != len(samples) or n_samples != len(sample_labels):
        raise ValueError("rgba columns, samples, and sample_labels length mismatch")

    if scale == 0 or n_samples == 0:
        col_map: list[int | None] = list(range(n_samples))
        tick_pos = [float(i) for i in range(n_samples)]
        return rgba, tick_pos, list(sample_labels), col_map

    gap_px = int(round(scale * gap_px_base))
    keys = [tuple(str(obs.loc[s, c]) for c in classes) for s in samples]
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n_samples:
        j = i
        while j + 1 < n_samples and keys[j + 1] == keys[i]:
            j += 1
        blocks.append((i, j))
        i = j + 1

    col_map = []
    tick_pos = [0.0] * n_samples
    chunks: list[np.ndarray] = []
    cursor = 0.0
    for bi, (i0, i1) in enumerate(blocks):
        for si in range(i0, i1 + 1):
            tick_pos[si] = cursor + (col_px - 1) / 2.0
            tile = np.repeat(rgba[:, si : si + 1, :], col_px, axis=1)
            chunks.append(tile)
            col_map.extend([si] * col_px)
            cursor += col_px
        if bi != len(blocks) - 1 and gap_px > 0:
            gap = np.zeros((n_rows, gap_px, 4), dtype=float)
            gap[:] = GAP_WHITE
            chunks.append(gap)
            col_map.extend([None] * gap_px)
            cursor += gap_px

    if not chunks:
        return (
            np.zeros((n_rows, 0, 4), dtype=float),
            [],
            [],
            [],
        )
    return np.concatenate(chunks, axis=1), tick_pos, list(sample_labels), col_map


def _rasterize_with_gaps(
    rgba: np.ndarray,
    row_groups: list[str | None],
    labels: list[str],
    row_spacing: bool | float,
    *,
    row_px: int = _ROW_PX,
    gap_px_base: int = _GROUP_GAP_PX,
) -> tuple[np.ndarray, list[float], list[str], dict[str, tuple[float, float]]]:
    """
    Expand content rows so ``row_spacing`` inserts white gaps between protein groups.

    ``row_spacing`` may be ``True`` (1× default half-cell gap), ``False``/``0``
    (no gaps), or a non-negative float scale of that default.

    Returns ``(rgba_disp, tick_positions, tick_labels, group_yrange)`` where
    ``group_yrange[g] = (y0, y1)`` are inclusive display-row indices for brackets.
    """
    scale = _resolve_spacing_scale(row_spacing, "row_spacing")
    gap_px = int(round(scale * gap_px_base))
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

def _normalize_header_colors(
    header_colors: dict[str, dict[str, str]] | dict[str, str] | None,
    classes: list[str],
) -> dict[str, dict[str, str]]:
    """
    Normalize ``header_colors`` to nested ``{class: {category: color}}``.

    Flat ``{category: color}`` is accepted only when ``len(classes) == 1``
    (mapped onto that single class). Nested form is required for multiple
    header rows — one category map per class, not one color per combination.
    """
    if not header_colors:
        return {}
    values = list(header_colors.values())
    if all(isinstance(v, dict) for v in values):
        return header_colors  # type: ignore[return-value]
    if any(isinstance(v, dict) for v in values):
        raise ValueError(
            "header_colors must be either nested {class: {category: color}} "
            "or, when len(classes)==1, flat {category: color}. Mixed forms "
            "are not supported."
        )
    if len(classes) != 1:
        raise ValueError(
            "Flat header_colors={category: color} is only allowed when "
            f"len(classes)==1; got classes={list(classes)!r}. "
            "For multiple classes use "
            "{class_name: {category: color, ...}, ...} — one map per "
            "header row, not one color per class combination."
        )
    return {classes[0]: dict(header_colors)}  # type: ignore[arg-type]

def _render_header_rows(
    fig: "Figure",
    gs_rows: list[Any],
    obs: pd.DataFrame,
    samples: list[str],
    classes: list[str],
    header_colors: dict[str, dict[str, str]] | dict[str, str] | None,
    text_size: int = 8,
    col_map: list[int | None] | None = None,
) -> list[tuple[str, list[Rectangle], list[Any]]]:
    """Draw categorical header strips; return legend specs ``(title, handles, labels)``.

    When ``col_map`` is provided (from ``_rasterize_with_column_gaps``), strips are
    expanded to match the heatmap column layout; ``None`` entries are white gaps.
    """
    header_colors = _normalize_header_colors(header_colors, classes)
    if col_map is None:
        col_map = list(range(len(samples)))
    legend_specs: list[tuple[str, list[Rectangle], list[Any]]] = []
    for ax, col in zip(gs_rows, classes):
        vals = [str(obs.loc[s, col]) for s in samples]
        cats = list(dict.fromkeys(vals))
        raw_override = header_colors.get(col) or {}
        override = {str(k): v for k, v in raw_override.items()}
        color_map = _build_color_map(cats, override)
        disp: list[float] = []
        for idx in col_map:
            if idx is None:
                disp.append(np.nan)
            else:
                disp.append(float(cats.index(vals[idx])))
        codes = np.array([disp], dtype=float)
        lcmap = ListedColormap([color_map[c] for c in cats])
        lcmap.set_bad(GAP_WHITE)
        ax.imshow(
            codes,
            aspect="auto",
            cmap=lcmap,
            vmin=-0.5,
            vmax=len(cats) - 0.5,
            interpolation="nearest",
        )
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

def _legend_block_height_in(n_labels: int, text_size: int) -> float:
    """Approximate vertical space (inches) for one titled legend block."""
    scale = text_size / 8.0
    return (0.22 + 0.145 * max(n_labels, 1)) * scale

def _estimate_legend_width_frac(
    fig_w: float,
    legend_specs: list[tuple[str, list, list]],
    text_size: int,
    cbar_label: str,
    *,
    left_tick_labels: list[str] | None = None,
) -> float:
    """
    Estimate figure-fraction left margin for colorbar + legends.

    Uses longest legend/cbar string and ``text_size``. When ``left_tick_labels``
    is set (grouped heatmaps keep gene names on the left), extra space is
    reserved so those labels do not collide with the legend column.
    """
    strings = [str(cbar_label)]
    for title, _, labels in legend_specs:
        strings.append(str(title))
        strings.extend(str(lab) for lab in labels)
    max_chars = max((len(s) for s in strings), default=8)
    # Slightly generous em width for default sans fonts
    char_in = 0.62 * (text_size / 72.0)
    # Vertical colorbar: bar + tick numbers (rotated axis label adds little width)
    cbar_col_in = (_CBAR_LEFT * fig_w) + _CBAR_WIDTH_IN + 0.60
    # Legends are anchored at ``_LEGEND_X`` (figure fraction)
    legend_row_in = (_LEGEND_X * fig_w) + 0.40 + max_chars * char_in
    content_in = max(cbar_col_in, legend_row_in) + 0.12
    if left_tick_labels:
        max_tick = max((len(str(s)) for s in left_tick_labels), default=0)
        # Gene labels sit just left of the axes spine
        content_in += max_tick * char_in + 0.20
    frac = content_in / max(float(fig_w), 1e-6)
    return float(np.clip(frac, _LEGEND_WIDTH_MIN, _LEGEND_WIDTH_MAX))

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
        title_artist = leg.get_title()
        if title_artist is not None:
            title_artist.set_color(_LEGEND_TEXT_COLOR)
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
    cbar.ax.yaxis.label.set_color(_LEGEND_TEXT_COLOR)
    cbar.ax.tick_params(labelsize=text_size - 1, colors=_LEGEND_TEXT_COLOR)
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
    legend_width: float | None,
    right: float,
    top: float,
    bottom: float = 0.10,
    left_tick_labels: list[str] | None = None,
) -> "Figure | tuple[Figure, Figure]":
    """Apply margins, draw cbar/legends, return ``fig`` or ``(fig, legend_fig)``."""
    if separate_legend:
        left = _DEFAULT_LEFT_SEPARATE
    elif legend_width is None:
        left = _estimate_legend_width_frac(
            fig.get_size_inches()[0],
            legend_specs,
            text_size=text_size,
            cbar_label=cbar_label,
            left_tick_labels=left_tick_labels,
        )
    else:
        if legend_width <= 0 or legend_width >= 1:
            raise ValueError(
                f"legend_width must be in (0, 1), got {legend_width!r}"
            )
        left = float(legend_width)
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

def _impute_row_min(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Impute NaNs with per-row minimum for clustering only (protein-min fill).

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
        row_min = float(np.min(finite))
        nan_idx = ~np.isfinite(row)
        row[nan_idx] = row_min
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
    header_colors: dict[str, dict[str, str]] | dict[str, str] | None = None,
    cmap: str | None = None,
    row_spacing: bool | float = True,
    column_spacing: bool | float = True,
    header_spacing: float = _DEFAULT_HEADER_SPACING,
    header_height: float = _DEFAULT_HEADER_HEIGHT,
    group_bar_pad: float = 0.25,
    group_bar_width: float = _DEFAULT_GROUP_BAR_WIDTH,
    sample_label_col: str | None = None,
    figsize: tuple[float, float] | None = None,
    text_size: int = 8,
    cbar_scale: float = 1.0,
    legend_width: float | None = None,
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
        header_colors (dict, optional): Colors for header strips. Nested
            ``{class: {category: color}}`` for one or more classes. When
            ``len(classes)==1``, a flat ``{category: color}`` is also accepted.
            Each class gets its own header row colored by that column's levels
            (not one color per multi-class combination). Unspecified categories
            use package defaults (``get_color('colors', n)``).
        cmap (str, optional): Colormap. ``None`` (default) selects ``RdBu_r`` for
            z-score and ``viridis`` for log/raw.
        row_spacing (bool or float): Vertical white gaps between protein groups.
            ``True`` (default) uses a half-cell gap; ``False``/``0`` = none; a
            float scales that default (``0.5`` = half, ``2`` = double). Same
            scale semantics as ``column_spacing``.
        column_spacing (bool or float): Horizontal white gaps between sample leaf
            blocks (runs that share the same full ``classes`` combination).
            ``True`` (default) uses the same half-cell thickness as
            ``row_spacing=True``; ``False``/``0`` = none; a float scales that
            default (``0.5`` = half, ``2`` = double).
        header_spacing (float): Vertical GridSpec space between header strips and
            the heatmap (default ``0.06``). Also spaces stacked header rows.
        header_height (float): Relative GridSpec height for each header strip
            (default ``0.35``). Larger values make thicker bands relative to the
            heatmap body (same units as the auto ``main`` height ratio).
        group_bar_pad (float): Horizontal gap (in heatmap column units) between the
            right edge of the heatmap and the colored group bars (default 0.25).
        group_bar_width (float): Width of the colored group bars in sample-column
            units (default ``0.4``). Scaled with column expansion like
            ``group_bar_pad``.
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
        legend_width (float, optional): Figure-fraction left margin for the
            colorbar + legends. ``None`` (default) auto-sizes from legend text
            length, ``text_size``, and (for grouped heatmaps) left-side gene
            labels. Pass a value (e.g. ``0.36``) to override. Ignored when
            ``separate_legend=True``.
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
        Grouped pathway heatmap on bulk PD data (``cellline`` x ``condition`` headers):
            ```python
            from scpviz import plotting as scplt

            fig = scplt.plot_grouped_heatmap(
                pdata_norm,
                protein_groups={
                    "Cell cycle": ["CDK1", "CDK2", "PCNA"],
                    "Housekeeping": ["GAPDH", "TUBB", "ACTB"],
                    "Stress": ["HSP90AA1", "UBE4B"],
                },
                classes=["cellline", "condition"],
                sort_by={"condition": ["sc", "kd"], "cellline": ["AS", "BE"]},
                layer="X",
            )
            ```

        ![Plot grouped heatmap](../../assets/plots/plot_grouped_heatmap.png)

        Custom header and group colors (via ``get_color`` or hex).
        Single class may use a flat category map; multiple classes use one
        nested map per header row (not per combination):
            ```python
            c = scplt.get_color("colors", 7)
            # Single class — flat {category: color} is fine
            fig = scplt.plot_grouped_heatmap(
                pdata_norm,
                protein_groups={"Cell cycle": ["CDK1", "PCNA"]},
                classes=["condition"],
                group_colors={"Cell cycle": c[0]},
                header_colors={"sc": "#55A868", "kd": "#C44E52"},
            )

            # Multiple classes — nested {class: {category: color}}
            fig = scplt.plot_grouped_heatmap(
                pdata_norm,
                protein_groups={
                    "Cell cycle": ["CDK1", "PCNA"],
                    "Housekeeping": ["GAPDH", "ACTB"],
                },
                classes=["cellline", "condition"],
                sort_by={"condition": ["sc", "kd"]},
                group_colors={"Cell cycle": c[0], "Housekeeping": c[1]},
                header_colors={
                    "cellline": {"AS": "#4C72B0", "BE": "#DD8452"},
                    "condition": {"sc": "#55A868", "kd": "#C44E52"},
                },
            )
            ```

        Row and column spacing (``row_spacing`` between protein groups;
        ``column_spacing`` between sample leaf blocks; ``header_spacing``
        between header strips and the heatmap):
            ```python
            fig = scplt.plot_grouped_heatmap(
                pdata_norm,
                protein_groups={
                    "Cell cycle": ["CDK1", "CDK2", "PCNA"],
                    "Housekeeping": ["GAPDH", "TUBB", "ACTB"],
                    "Stress": ["HSP90AA1", "UBE4B"],
                },
                classes=["cellline", "condition"],
                sort_by={"condition": ["sc", "kd"], "cellline": ["AS", "BE"]},
                row_spacing=0.75,
                column_spacing=0.5,
                header_spacing=0.08,
            )
            ```

        ![Plot grouped heatmap spacing](../../assets/plots/plot_grouped_heatmap_spacing.png)

        Fonts, display scale, and colorbar height:
            ```python
            fig = scplt.plot_grouped_heatmap(
                pdata_norm,
                protein_groups={
                    "Cell cycle": ["CDK1", "PCNA"],
                    "Housekeeping": ["GAPDH", "ACTB"],
                },
                classes=["cellline", "condition"],
                sample_label_col="Sample",
                group_bar_pad=0.15,
                display_scale="zscore",
                text_size=10,
                cbar_scale=0.8,
                legend_width=0.36,
            )
            ```

        Legends on a separate figure (useful when fixing a compact heatmap size):
            ```python
            fig, legend_fig = scplt.plot_grouped_heatmap(
                pdata_norm,
                protein_groups={
                    "Cell cycle": ["CDK1", "PCNA"],
                    "Housekeeping": ["GAPDH", "ACTB"],
                },
                classes=["cellline", "condition"],
                figsize=(4, 3),
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
        rgba, row_groups, labels, row_spacing
    )
    rgba_disp, xtick_pos, xtick_lab, col_map = _rasterize_with_column_gaps(
        rgba_disp,
        samples,
        adata.obs,
        classes,
        column_spacing,
        sample_labels,
    )
    n_disp = rgba_disp.shape[0]
    n_disp_cols = rgba_disp.shape[1]

    n_header = len(classes)
    header_h = float(header_height)
    if header_h <= 0:
        raise ValueError(f"header_height must be > 0, got {header_height!r}")
    bar_w_user = float(group_bar_width)
    if bar_w_user <= 0:
        raise ValueError(f"group_bar_width must be > 0, got {group_bar_width!r}")
    main_h = max(8, max(n_disp, 1) * 0.024 * _ROW_PX)
    if figsize is None:
        fig_w, fig_h = 11.5, header_h * n_header + main_h * 0.42 + 1.5
    else:
        fig_w, fig_h = figsize

    if float(header_spacing) < 0:
        raise ValueError(f"header_spacing must be >= 0, got {header_spacing!r}")

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_header + 1,
        ncols=1,
        height_ratios=[header_h] * n_header + [main_h],
        hspace=float(header_spacing),
    )
    header_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_header)]
    ax_main = fig.add_subplot(gs[n_header, 0])

    legend_specs = _render_header_rows(
        fig,
        header_axes,
        adata.obs,
        samples,
        classes,
        header_colors,
        text_size=text_size,
        col_map=col_map,
    )

    ax_main.imshow(rgba_disp, aspect="auto", interpolation="nearest", zorder=2)
    ax_main.set_xticks(xtick_pos)
    ax_main.set_xticklabels(xtick_lab, fontsize=text_size - 1, rotation=90)
    ax_main.set_yticks(tick_pos)
    ax_main.set_yticklabels(tick_lab, fontsize=text_size - 2)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.tick_params(left=False, bottom=False)

    # group_bar_pad / bar width are in sample-column units; scale when columns are expanded
    x_unit = float(_COL_PX) if n_disp_cols != len(samples) else 1.0
    bar_pad = float(group_bar_pad) * x_unit
    bar_w = bar_w_user * x_unit
    bar_x0 = n_disp_cols - 0.5 + bar_pad
    bar_x1 = bar_x0 + bar_w
    text_x = bar_x1 + 0.55 * x_unit
    ax_main.set_xlim(-0.5, text_x + 0.8 * x_unit)
    # Keep header strips aligned with the heatmap matrix (not the group bars/labels)
    for ax_h in header_axes:
        ax_h.set_xlim(ax_main.get_xlim())
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
        fig.suptitle(title, fontsize=text_size + 3, x=0.55)

    return _finish_heatmap_legends(
        fig,
        sm,
        legend_specs,
        text_size=text_size,
        cbar_label=cbar_label,
        cbar_scale=cbar_scale,
        separate_legend=separate_legend,
        legend_width=legend_width,
        right=0.72,
        top=0.95,
        left_tick_labels=list(tick_lab),
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
    header_colors: dict[str, dict[str, str]] | dict[str, str] | None = None,
    label_color: str | None = None,
    sample_label_col: str | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    text_size: int = 8,
    cbar_scale: float = 1.0,
    legend_width: float | None = None,
    column_spacing: bool | float = True,
    header_spacing: float = _DEFAULT_HEADER_SPACING,
    header_height: float = _DEFAULT_HEADER_HEIGHT,
    dendrogram_linewidth: float | None = None,
    auto_log2: bool = True,
    gene_col: str = "Genes",
    separate_legend: bool = False,
    **kwargs: Any,
) -> "Figure | tuple[Figure, Figure]":
    """
    Plot a hierarchically clustered protein/peptide x sample heatmap.

    Row order comes from hierarchical clustering (correlation or Euclidean
    distance with average linkage). Optional ``protein_groups`` are shown as a
    colour strip + coloured right-side labels (not spatial blocks). Sample
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
        header_colors (dict, optional): Colors for header strips. Nested
            ``{class: {category: color}}``, or flat ``{category: color}`` when
            ``len(classes)==1``. Each class is a separate header row (not one
            color per multi-class combination). Same rules as
            ``plot_grouped_heatmap``.
        label_color (str, optional): Fixed color for all gene row labels (e.g.
            ``"black"``). If ``None`` (default), labels use ``protein_groups``
            colors and Unassigned stays grey.
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
        legend_width (float, optional): Figure-fraction left margin for colorbar +
            legends. ``None`` (default) auto-sizes from legend text; pass a float
            to override. Ignored when ``separate_legend=True``.
        column_spacing (bool or float): Horizontal white gaps between sample leaf
            blocks (same full ``classes`` combination). ``True`` (default) uses a
            half-cell gap; ``False``/``0`` = none; float scales the default. Same
            semantics as ``plot_grouped_heatmap``.
        header_spacing (float): Vertical GridSpec space between header strips and
            the heatmap (default ``0.06``). Same semantics as ``plot_grouped_heatmap``.
        header_height (float): Relative GridSpec height for each header strip
            (default ``0.35``). Same semantics as ``plot_grouped_heatmap``.
        dendrogram_linewidth (float, optional): Line width for the hierarchical
            cluster tree. ``None`` (default) keeps matplotlib's default.
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
        Cluster an explicit protein list (optional group strip for curated subsets):
            ```python
            from scpviz import plotting as scplt

            fig = scplt.plot_clustered_heatmap(
                pdata_norm,
                classes=["cellline", "condition"],
                proteins=[
                    "CDK1", "CDK2", "PCNA",
                    "GAPDH", "TUBB", "ACTB",
                    "HSP90AA1", "ENO1", "PGK1",
                ],
                protein_groups={
                    "Cell cycle": ["CDK1", "CDK2", "PCNA"],
                    "Housekeeping": ["GAPDH", "TUBB", "ACTB"],
                },
                sort_by={"condition": ["sc", "kd"], "cellline": ["AS", "BE"]},
                show_unassigned=True,
            )
            ```

        ![Plot clustered heatmap](../../assets/plots/plot_clustered_heatmap.png)

        Cluster DE hits from a stored contrast. Large tables are hard to read in
        docs/figures — here we keep the top 40 by ``|significance_score|``:
            ```python
            de = pdata_norm.stats["BE_kd vs BE_sc"]
            hits = de[de["significance"].isin(["upregulated", "downregulated"])]
            top40 = (
                hits.assign(_s=hits["significance_score"].abs())
                .sort_values("_s", ascending=False)
                .head(40)
            )
            pdata_norm.stats["BE_kd vs BE_sc (top 40)"] = top40

            fig = scplt.plot_clustered_heatmap(
                pdata_norm,
                classes=["cellline", "condition"],
                stats_key="BE_kd vs BE_sc (top 40)",
                sort_by={"condition": ["sc", "kd"], "cellline": ["AS", "BE"]},
            )
            ```

        ![Plot clustered heatmap (DE hits)](../../assets/plots/plot_clustered_heatmap_de.png)

        Column and header spacing (``column_spacing`` between sample leaf blocks;
        ``header_spacing`` between header strips and the heatmap):
            ```python
            fig = scplt.plot_clustered_heatmap(
                pdata_norm,
                classes=["cellline", "condition"],
                proteins=[
                    "CDK1", "CDK2", "PCNA",
                    "GAPDH", "TUBB", "ACTB",
                    "HSP90AA1", "ENO1", "PGK1",
                ],
                protein_groups={
                    "Cell cycle": ["CDK1", "CDK2", "PCNA"],
                    "Housekeeping": ["GAPDH", "TUBB", "ACTB"],
                },
                sort_by={"condition": ["sc", "kd"], "cellline": ["AS", "BE"]},
                column_spacing=0.5,
                header_spacing=0.08,
                show_unassigned=True,
            )
            ```

        ![Plot clustered heatmap spacing](../../assets/plots/plot_clustered_heatmap_spacing.png)

        Only upregulated features, or black row labels on the full contrast key:
            ```python
            fig = scplt.plot_clustered_heatmap(
                pdata_norm,
                classes=["cellline", "condition"],
                stats_key="BE_kd vs BE_sc",
                significance_categories=["upregulated"],
            )
            fig = scplt.plot_clustered_heatmap(
                pdata_norm,
                classes=["cellline", "condition"],
                stats_key="BE_kd vs BE_sc",
                label_color="black",
                display_scale="log",
                text_size=10,
                cbar_scale=0.8,
                legend_width=0.36,
            )
            ```

        Custom colors with a separate legend figure.
        Single class may use flat ``{category: color}``; multiple classes use
        one nested map per header row:
            ```python
            c = scplt.get_color("colors", 7)
            # Single class — flat map
            fig = scplt.plot_clustered_heatmap(
                pdata_norm,
                classes=["condition"],
                proteins=["CDK1", "PCNA", "GAPDH"],
                header_colors={"sc": "#55A868", "kd": "#C44E52"},
            )

            # Multiple classes — nested maps (one strip per class)
            fig, legend_fig = scplt.plot_clustered_heatmap(
                pdata_norm,
                classes=["cellline", "condition"],
                proteins=["CDK1", "PCNA", "GAPDH", "ACTB", "HSP90AA1"],
                protein_groups={"Cell cycle": ["CDK1", "PCNA"]},
                group_colors={"Cell cycle": c[0]},
                header_colors={
                    "cellline": {"AS": "#4C72B0", "BE": "#DD8452"},
                    "condition": {"sc": "#55A868", "kd": "#C44E52"},
                },
                figsize=(4, 3),
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

    # Clustering uses protein-min–imputed z-scores; display keeps original NaNs
    cluster_mat = values_z.copy()
    cluster_mat[is_missing_z] = np.nan
    imputed, keep_mask = _impute_row_min(cluster_mat)

    if not np.all(keep_mask):
        dropped = [f for f, k in zip(valid_features, keep_mask) if not k]
        for f in dropped:
            print(
                f"{format_log_prefix('warn')} Excluding '{f}' from clustering "
                f"(zero variance or all missing after protein-min imputation)."
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
    header_h = float(header_height)
    if header_h <= 0:
        raise ValueError(f"header_height must be > 0, got {header_height!r}")
    main_h = max(8, n_rows * 0.24)
    if figsize is None:
        fig_w, fig_h = 11.0, header_h * n_header + main_h * 0.42 + 1.5
    else:
        fig_w, fig_h = figsize

    if float(header_spacing) < 0:
        raise ValueError(f"header_spacing must be >= 0, got {header_spacing!r}")

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_header + 1,
        ncols=3,
        height_ratios=[header_h] * n_header + [main_h],
        width_ratios=[1.3, 0.18, 10],
        hspace=float(header_spacing),
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

    dn = dendrogram(
        Z,
        ax=ax_dendro,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#999999",
    )
    if dendrogram_linewidth is not None:
        for line in ax_dendro.get_lines():
            line.set_linewidth(dendrogram_linewidth)
        for coll in ax_dendro.collections:
            coll.set_linewidth(dendrogram_linewidth)
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
    rgba, xtick_pos, xtick_lab, col_map = _rasterize_with_column_gaps(
        rgba,
        samples,
        adata.obs,
        classes,
        column_spacing,
        sample_labels,
    )
    legend_specs = _render_header_rows(
        fig,
        header_axes,
        adata.obs,
        samples,
        classes,
        header_colors,
        text_size=text_size,
        col_map=col_map,
    )
    ax_main.imshow(rgba, aspect="auto", interpolation="nearest", zorder=2)

    labels = _display_labels(adata, valid_features, gene_col=gene_col)
    ax_main.set_xticks(xtick_pos)
    ax_main.set_xticklabels(xtick_lab, fontsize=text_size - 1, rotation=90)
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
        elif g is not None:
            lab.set_color(gcolors.get(g, "#333333"))
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
            interpolation="nearest",
        )
    else:
        strip_codes = np.zeros((n_rows, 1))
        strip_cmap = ListedColormap([UNASSIGNED_COLOUR])
        ax_strip.imshow(
            strip_codes,
            aspect="auto",
            cmap=strip_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
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
        fig.suptitle(title, fontsize=text_size + 3, x=0.55)

    return _finish_heatmap_legends(
        fig,
        sm,
        stacked,
        text_size=text_size,
        cbar_label=cbar_label,
        cbar_scale=cbar_scale,
        separate_legend=separate_legend,
        legend_width=legend_width,
        right=0.88,
        top=0.94,
    )


# ---------------------------------------------------------------------------
# plot_clustermap (overview: row + column dendrograms)
# ---------------------------------------------------------------------------

def plot_clustermap(
    pdata: "pAnnData",
    classes: list[str],
    *,
    namelist: list[str] | None = None,
    on: str = "protein",
    layer: str = "X",
    display_scale: str = "auto",
    metric: str = "correlation",
    cor_method: str = "pearson",
    linkage_method: str = "average",
    optimal_ordering: bool = True,
    header_colors: dict[str, dict[str, str]] | dict[str, str] | None = None,
    sample_label_col: str | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    text_size: int = 8,
    cbar_scale: float = 1.0,
    legend_width: float | None = None,
    header_spacing: float = _DEFAULT_HEADER_SPACING,
    header_height: float = _DEFAULT_HEADER_HEIGHT,
    dendrogram_linewidth: float | None = None,
    show_row_labels: bool = False,
    auto_log2: bool = True,
    gene_col: str = "Genes",
    separate_legend: bool = False,
    **kwargs: Any,
) -> "Figure | tuple[Figure, Figure]":
    """
    Plot an overview clustered heatmap with row and column dendrograms.

    Both proteins (rows) and samples (columns) are hierarchically clustered;
    sample order follows the column dendrogram leaf order (not ``classes`` /
    ``sort_by``). Header strips still annotate samples by ``classes``. Intended
    as a general overview; prefer :func:`plot_clustered_heatmap` for curated
    protein lists / DE hits with metadata-ordered samples, or
    :func:`plot_grouped_heatmap` for curated group blocks.

    Missing cells stay grey in the display. Clustering uses protein-min
    imputation on the z-scored matrix only (same fill for row and column
    linkage).

    Args:
        pdata (pAnnData): Input object.
        classes (list of str): ``.obs`` columns for header strips. Required.
        namelist (list of str, optional): Gene names or accessions to plot.
            ``None`` (default) uses all features at ``on``.
        on (str): ``"protein"`` / ``"prot"`` or ``"peptide"`` / ``"pep"``.
        layer (str): Abundance layer (default ``"X"``).
        display_scale (str): ``"auto"`` (default), ``"zscore"``, ``"log"``, or
            ``"raw"``. Auto selects ``zscore`` for overview plots. Clustering
            always uses z-scored values regardless of display scale.
        metric (str): ``"correlation"`` (default) or ``"euclidean"``.
        cor_method (str): ``"pearson"`` or ``"spearman"``; only used when
            ``metric="correlation"``.
        linkage_method (str): Passed to scipy linkage (default ``"average"``).
        optimal_ordering (bool): Improve leaf order for display (default True).
        header_colors (dict, optional): Colors for header strips. Nested
            ``{class: {category: color}}``, or flat ``{category: color}`` when
            ``len(classes)==1``. Same rules as ``plot_grouped_heatmap``.
        sample_label_col (str, optional): ``.obs`` column for bottom tick labels.
            Default ``None`` uses sample index names (``obs_names``).
        cmap (str, optional): Colormap. ``None`` selects ``RdBu_r`` for z-score
            and ``viridis`` for log/raw.
        figsize (tuple, optional): Figure size; auto height if None.
        text_size (int): Base font size for ticks, colorbar, and legends
            (default 8).
        cbar_scale (float): Vertical scale factor for the colorbar (default
            ``1.0``).
        legend_width (float, optional): Figure-fraction left margin for colorbar
            + legends. ``None`` auto-sizes. Ignored when
            ``separate_legend=True``.
        header_spacing (float): Vertical GridSpec space between header strips
            and the heatmap (default ``0.06``).
        header_height (float): Relative GridSpec height for each header strip
            (default ``0.35``).
        dendrogram_linewidth (float, optional): Line width for both dendrogram
            trees. ``None`` keeps matplotlib's default.
        show_row_labels (bool): Show gene/feature labels on the right
            (default False; overview matrices are usually too tall).
        auto_log2 (bool): In-memory log2 when the layer looks linear.
        gene_col (str): Gene label column in ``.var``.
        separate_legend (bool): If True, return ``(fig, legend_fig)`` with
            colorbar and legends on a second figure.
        **kwargs: Optional ``title``.

    Returns:
        fig (matplotlib.figure.Figure): The constructed heatmap figure.
        legend_fig (matplotlib.figure.Figure, optional): Returned when
            ``separate_legend=True``.

    Example:
        Overview clustermap on all proteins:
            ```python
            from scpviz import plotting as scplt

            fig = scplt.plot_clustermap(
                pdata_norm,
                classes=["cellline", "condition"],
                layer="X",
                figsize=(8, 10),
            )
            ```

        ![Plot clustermap](../../assets/plots/plot_clustermap.png)

        Subset via ``namelist`` (genes or accessions) and custom header colors:
            ```python
            fig = scplt.plot_clustermap(
                pdata_norm,
                classes=["cellline", "condition"],
                namelist=["CDK1", "CDK2", "PCNA", "GAPDH", "ACTB"],
                header_colors={
                    "cellline": {"AS": "#4C72B0", "BE": "#DD8452"},
                    "condition": {"sc": "#55A868", "kd": "#C44E52"},
                },
                show_row_labels=True,
                separate_legend=True,
            )
            ```
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    pdata._check_data(on)  # type: ignore[attr-defined]
    adata = get_adata(pdata, on)
    classes = _validate_classes(adata.obs, classes)

    # Sample list starts in obs order; final order comes from column dendrogram
    samples_all = list(adata.obs_names.astype(str))
    sample_idx_all = list(range(len(samples_all)))

    scale = _resolve_display_scale_arg(
        display_scale, None, from_stats_key=False
    )
    cmap_resolved = _resolve_cmap_for_scale(cmap, scale)

    if namelist is not None:
        resolved, unresolved = _resolve_feature_list(
            pdata, namelist, on, quiet=False
        )
        for u in unresolved:
            print(
                f"{format_log_prefix('warn')} protein '{u}' not found in this "
                f"pAnnData — excluded from clustermap"
            )
        feature_ids = list(dict.fromkeys(resolved))
        if not feature_ids:
            raise ValueError("No resolvable features in namelist to plot.")
    else:
        feature_ids = list(adata.var_names.astype(str))

    display_full, miss_disp_full, z_full, miss_z_full, cbar_label, symmetric, _layer_is_log = (
        _resolve_display_and_cluster(
            adata, layer, display_scale=scale, auto_log2=auto_log2
        )
    )
    var_index = list(adata.var_names.astype(str))
    var_to_i = {v: i for i, v in enumerate(var_index)}

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
        rows_z.append(z_full[vi, sample_idx_all])
        rows_disp.append(display_full[vi, sample_idx_all])
        rows_miss_z.append(miss_z_full[vi, sample_idx_all])
        rows_miss_disp.append(miss_disp_full[vi, sample_idx_all])
        valid_features.append(fid)

    if not valid_features:
        raise ValueError("No features present in the abundance matrix.")

    values_z = np.vstack(rows_z)
    values_disp = np.vstack(rows_disp)
    is_missing_z = np.vstack(rows_miss_z)
    is_missing_disp = np.vstack(rows_miss_disp)

    # Protein-min impute on z-scores for both row and column linkage
    cluster_mat = values_z.copy()
    cluster_mat[is_missing_z] = np.nan
    imputed, keep_mask = _impute_row_min(cluster_mat)

    if not np.all(keep_mask):
        dropped = [f for f, k in zip(valid_features, keep_mask) if not k]
        for f in dropped:
            print(
                f"{format_log_prefix('warn')} Excluding '{f}' from clustering "
                f"(zero variance or all missing after protein-min imputation)."
            )
        values_z = values_z[keep_mask]
        values_disp = values_disp[keep_mask]
        is_missing_z = is_missing_z[keep_mask]
        is_missing_disp = is_missing_disp[keep_mask]
        imputed = imputed[keep_mask]
        valid_features = [f for f, k in zip(valid_features, keep_mask) if k]

    if len(valid_features) < 2:
        raise ValueError(
            f"Need at least 2 clusterable features; got {len(valid_features)}."
        )
    if imputed.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 samples for column clustering; got {imputed.shape[1]}."
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

    def _link(mat: np.ndarray) -> np.ndarray:
        if metric == "correlation":
            Z, _ = correlation_linkage(
                mat,
                method=cor_method,
                linkage_method=linkage_method,
                optimal_ordering=optimal_ordering,
            )
            return Z
        condensed = pdist(mat, metric="euclidean")
        return linkage(
            condensed, method=linkage_method, optimal_ordering=optimal_ordering
        )

    Z_row = _link(imputed)
    Z_col = _link(imputed.T)

    n_header = len(classes)
    n_rows = len(valid_features)
    n_samples = len(samples_all)
    header_h = float(header_height)
    if header_h <= 0:
        raise ValueError(f"header_height must be > 0, got {header_height!r}")
    dendro_h = max(1.2, header_h * 2.5)
    main_h = max(8, n_rows * 0.08)
    if figsize is None:
        fig_w, fig_h = 10.0, dendro_h + header_h * n_header + main_h * 0.42 + 1.5
    else:
        fig_w, fig_h = figsize

    if float(header_spacing) < 0:
        raise ValueError(f"header_spacing must be >= 0, got {header_spacing!r}")

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_header + 2,
        ncols=2,
        height_ratios=[dendro_h] + [header_h] * n_header + [main_h],
        width_ratios=[1.3, 10],
        hspace=float(header_spacing),
        wspace=0.02,
    )

    ax_blank_tl = fig.add_subplot(gs[0, 0])
    ax_blank_tl.axis("off")
    ax_col_dendro = fig.add_subplot(gs[0, 1])

    for i in range(n_header):
        ax_blank = fig.add_subplot(gs[i + 1, 0])
        ax_blank.axis("off")
    header_axes = [fig.add_subplot(gs[i + 1, 1]) for i in range(n_header)]

    ax_row_dendro = fig.add_subplot(gs[n_header + 1, 0])
    ax_main = fig.add_subplot(gs[n_header + 1, 1])

    dn_col = dendrogram(
        Z_col,
        ax=ax_col_dendro,
        orientation="top",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#999999",
    )
    dn_row = dendrogram(
        Z_row,
        ax=ax_row_dendro,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#999999",
    )
    if dendrogram_linewidth is not None:
        for ax_d in (ax_col_dendro, ax_row_dendro):
            for line in ax_d.get_lines():
                line.set_linewidth(dendrogram_linewidth)
            for coll in ax_d.collections:
                coll.set_linewidth(dendrogram_linewidth)

    ax_row_dendro.invert_yaxis()
    for ax_d in (ax_col_dendro, ax_row_dendro):
        for spine in ax_d.spines.values():
            spine.set_visible(False)
        ax_d.set_xticks([])
        ax_d.set_yticks([])

    row_leaves = dn_row["leaves"]
    col_leaves = dn_col["leaves"]

    values_disp = values_disp[np.ix_(row_leaves, col_leaves)]
    is_missing_disp = is_missing_disp[np.ix_(row_leaves, col_leaves)]
    valid_features = [valid_features[i] for i in row_leaves]
    samples = [samples_all[i] for i in col_leaves]
    sample_labels = _sample_tick_labels(adata.obs, samples, sample_label_col)

    rgba, norm = _composite_heatmap_rgba(
        values_disp, is_missing_disp, None, cmap_resolved, symmetric=symmetric
    )
    legend_specs = _render_header_rows(
        fig,
        header_axes,
        adata.obs,
        samples,
        classes,
        header_colors,
        text_size=text_size,
        col_map=None,
    )
    ax_main.imshow(rgba, aspect="auto", interpolation="nearest", zorder=2)

    ax_main.set_xticks(range(n_samples))
    ax_main.set_xticklabels(sample_labels, fontsize=text_size - 1, rotation=90)
    if show_row_labels:
        labels = _display_labels(adata, valid_features, gene_col=gene_col)
        ax_main.set_yticks(range(n_rows))
        ax_main.set_yticklabels(labels, fontsize=text_size - 2)
        ax_main.yaxis.tick_right()
    else:
        ax_main.set_yticks([])

    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.tick_params(left=False, right=False, bottom=False)

    # Keep header strips aligned with heatmap columns
    for ax_h in header_axes:
        ax_h.set_xlim(ax_main.get_xlim())

    sm = ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])

    title = kwargs.get("title")
    if title:
        fig.suptitle(title, fontsize=text_size + 3, x=0.55)

    # Store clustering results for downstream helpers (e.g. get_clusters)
    on_key = "prot" if on.lower() in ("protein", "prot") else "pep"
    cluster_key = f"{on_key}_{layer}_clustermap"
    pdata.stats[cluster_key] = {  # type: ignore[index]
        "row_order": list(valid_features),
        "row_indices": list(row_leaves),
        "col_order": list(samples),
        "col_indices": list(col_leaves),
        "row_linkage": Z_row,
        "col_linkage": Z_col,
        "namelist_used": namelist if namelist is not None else "all_proteins",
        "metric": metric,
        "cor_method": cor_method,
        "linkage_method": linkage_method,
    }

    return _finish_heatmap_legends(
        fig,
        sm,
        list(legend_specs),
        text_size=text_size,
        cbar_label=cbar_label,
        cbar_scale=cbar_scale,
        separate_legend=separate_legend,
        legend_width=legend_width,
        right=0.88 if show_row_labels else 0.95,
        top=0.92,
    )

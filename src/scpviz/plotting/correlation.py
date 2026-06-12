"""Pairwise correlation and clustermap plots."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import copy
import warnings

import matplotlib.cm as cm
import matplotlib.collections as clt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.gridspec import GridSpec

from scpviz import utils

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

def plot_clustermap(
    ax: "plt.Axes",
    pdata: pAnnData,
    on: str = "prot",
    classes: str | list[str] | None = None,
    layer: str = "X",
    x_label: str = "accession",
    namelist: list[str] | None = None,
    lut: dict | None = None,
    log2: bool = True,
    cmap: str = "coolwarm",
    figsize: tuple[float, float] = (6, 10),
    force: bool = False,
    impute: str | None = None,
    order: dict | None = None,
    **kwargs: Any,
) -> Any:
    """
    Plot a clustered heatmap of proteins or peptides by samples.

    This function creates a hierarchical clustered heatmap (features × samples)
    with optional column annotations from sample-level metadata. Supports
    custom annotation colors, log2 transformation, and missing value imputation.

    Args:
        ax (matplotlib.axes.Axes): Unused; included for API compatibility.
        pdata (pAnnData): Input pAnnData object.
        on (str): Data level to plot, either `"prot"` or `"pep"`. Default is `"prot"`.
        classes (str or list of str, optional): One or more `.obs` columns to
            annotate samples in the heatmap.
        layer (str): Data layer to use. Defaults to `"X"`.
        x_label (str): Row label mode, either `"accession"` or `"gene"`. Used
            for mapping `namelist`.
        namelist (list of str, optional): Subset of accessions or gene names to plot.
            If None, all rows are included.
        lut (dict, optional): Nested dictionary of `{class_name: {label: color}}`
            controlling annotation bar colors. Missing entries fall back to
            default palettes. See the note 'lut example' below.
        log2 (bool): Whether to log2-transform the abundance matrix. Default is True.
        cmap (str): Colormap for heatmap. Default is `"coolwarm"`.
        figsize (tuple): Figure size in inches. Default is `(6, 10)`.
        force (bool): If True, imputes missing values instead of dropping rows
            with NaNs.
        impute (str, optional): Imputation strategy used when `force=True`.
            
            - `"row_min"`: fill NaNs with minimum value of that protein row.
            - `"global_min"`: fill NaNs with global minimum value of the matrix.

        order (dict, optional): Custom order for categorical annotations.
            Example: `{"condition": ["kd", "sc"], "cellline": ["AS", "BE"]}`.
        **kwargs (Any): Additional keyword arguments passed to `seaborn.clustermap`.

            Common options include:
            
            - `z_score (int)`: Normalize rows (0, features) or columns (1, samples).
            - `standard_scale (int)`: Scale rows or columns to unit variance.
            - `center (float)`: Value to center colormap on (e.g. 0 with `z_score`).
            - `col_cluster (bool)`: Cluster columns (samples). Default is False.
            - `row_cluster (bool)`: Cluster rows (features). Default is True.
            - `linewidth (float)`: Grid line width between cells.
            - `xticklabels` / `yticklabels` (bool): Show axis tick labels.
            - `colors_ratio (tuple)`: Proportion of space allocated to annotation bars.

    Returns:
        g (seaborn.matrix.ClusterGrid): The seaborn clustermap object.

    !!! note
        Function is currently under development, may not produce publication quality graphs yet.
        User discretion for formatting plots is encouraged.
        
    !!! note "lut example"
        Example of a custom lookup table for annotation colors:
            ```python
            lut = {
                "cellline": {
                    "AS": "#e41a1c",
                    "BE": "#377eb8"
                },
                "condition": {
                    "kd": "#4daf4a",
                    "sc": "#984ea3"
               }
            }
            ```

    Example:
        Clustered heatmap with sample annotations:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(1, 1))
            g = scplt.plot_clustermap(
                ax,
                pdata_norm,
                on="prot",
                classes=["cellline", "condition"],
                force=True,
                impute="row_min",
                z_score=0,
                center=0,
                linewidth=0,
                figsize=(10, 6),
            )
            plt.show()
            ```

        ![Plot clustermap](../../assets/plots/plot_clustermap.png)

        Provide a custom LUT for annotation colors:
            ```python
            import seaborn as sns

            paired = sns.color_palette("Paired", 6)

            lut = {
                "timepoint": {
                    "1mo": paired[1],
                    "3mo": paired[3],
                    "6mo": paired[5],
                },
                "aggregate": {
                    "aggN": "#4d4d4d",
                    "aggY": "#bdbdbd",
                },
            }

            fig, ax = plt.subplots(figsize=(6, 4))
            scplt.plot_clustermap(
                ax,
                pdata,
                classes=["timepoint", "aggregate"],
                force=True,
                impute="zero",
                z_score=0,
                center=0,
                lut=lut,
            )
            ```
    """
    # --- Step 1: Extract data ---
    if on not in ("prot", "pep"):
        raise ValueError(f"`on` must be 'prot' or 'pep', got '{on}'")
    
    if namelist is not None:
        df_abund = utils.get_abundance(
        pdata, namelist=namelist, layer=layer, on=on,
        classes=classes, log=log2, x_label=x_label)
        
        pivot_col = "log2_abundance" if log2 else "abundance"
        row_index = "gene" if x_label == "gene" else "accession"
        df = df_abund.pivot(index=row_index, columns="cell", values=pivot_col)
    
    else:
        adata = pdata.prot if on == 'prot' else pdata.pep
        X = adata.layers[layer] if layer in adata.layers else adata.X
        data = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        df = pd.DataFrame(data.T, index=adata.var_names, columns=adata.obs_names)
        if log2:
            with np.errstate(divide='ignore', invalid='ignore'):
                df = np.log2(df)
                df[df == -np.inf] = np.nan

    # --- Handle missing values ---
    nan_rows = df.index[df.isna().any(axis=1)].tolist()
    if nan_rows:
        if not force:
            print(f"Warning: {len(nan_rows)} proteins contain missing values and will be excluded: {nan_rows}")
            print("To include them, rerun with force=True and impute='row_min' or 'global_min'.")
            df = df.drop(index=nan_rows)
        else:
            print(f"{len(nan_rows)} proteins contain missing values: {nan_rows}.\nImputing using strategy: '{impute}'")
            if impute == "row_min":
                global_min = df.min().min()
                df = df.apply(lambda row: row.fillna(row.min() if not np.isnan(row.min()) else global_min), axis=1)
            elif impute == "global_min":
                df = df.fillna(df.min().min())
            else:
                raise ValueError("`impute` must be either 'row_min' or 'global_min' when force=True.")

    # --- Step 2: Column annotations ---
    col_colors = None
    legend_handles, legend_labels = [], []

    if classes is not None:
        if isinstance(classes, str):
            sample_labels = utils.get_samplenames(adata, classes)
            annotations = pd.DataFrame({classes: sample_labels}, index=adata.obs_names)
        else:
            sample_labels = utils.get_samplenames(adata, classes)
            split_labels = [[part.strip() for part in s.split(",")] for s in sample_labels]
            annotations = pd.DataFrame(split_labels, index=adata.obs_names, columns=classes)

        # Optional: apply custom category order from `order` dict
        if order is not None and isinstance(order, dict):
            for col in classes:
                if col in annotations.columns and col in order:
                    cat_type = pd.api.types.CategoricalDtype(order[col], ordered=True)
                    annotations[col] = annotations[col].astype(cat_type)
            unused_keys = set(order) - set(classes)
            if unused_keys:
                print(f"⚠️ Unused keys in `order`: {unused_keys} (not present in `classes`)")

        # Sort columns (samples) by class hierarchy
        sort_order = annotations.sort_values(by=classes).index
        df = df[sort_order]
        annotations = annotations.loc[sort_order]

        if lut is None:
            lut = {}

        full_lut = {}
        for col in annotations.columns:
            unique_vals = sorted(annotations[col].dropna().unique())
            user_colors = lut.get(col, {})
            missing_vals = [v for v in unique_vals if v not in user_colors]
            fallback_palette = sns.color_palette(n_colors=len(missing_vals))
            fallback_colors = dict(zip(missing_vals, fallback_palette))
            full_lut[col] = {**user_colors, **fallback_colors}

            unmatched = set(user_colors) - set(unique_vals)
            if unmatched:
                print(f"Warning: The following labels in `lut['{col}']` are not found in the data: {sorted(unmatched)}")

        col_colors = annotations.apply(lambda col: col.map(full_lut[col.name]))

        # Legend handles
        for col in annotations.columns:
            legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="none", label=col))  # header
            for label, color in full_lut[col].items():
                legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="black", label=label))
            legend_labels.extend([col] + list(full_lut[col].keys()))

    # --- Step 3: Clustermap defaults (user-overridable) ---
    col_cluster = kwargs.pop("col_cluster", False)
    row_cluster = kwargs.pop("row_cluster", True)
    linewidth = kwargs.pop("linewidth", 0)
    yticklabels = kwargs.pop("yticklabels", False)
    xticklabels = kwargs.pop("xticklabels", False)
    colors_ratio = kwargs.pop("colors_ratio", (0.03, 0.02))
    if kwargs.get("z_score", None) == 0:
        zero_var_rows = df.var(axis=1) == 0
        if zero_var_rows.any():
            dropped = df.index[zero_var_rows].tolist()
            print(f"⚠️ {len(dropped)} proteins have zero variance and will be dropped due to z_score=0: {dropped}")
            df = df.drop(index=dropped)

    # --- Step 4: Plot clustermap ---
    try:
        g = sns.clustermap(df,
                        cmap=cmap,
                        col_cluster=col_cluster,
                        row_cluster=row_cluster,
                        col_colors=col_colors,
                        figsize=figsize,
                        xticklabels=xticklabels,
                        yticklabels=yticklabels,
                        linewidth=linewidth,
                        colors_ratio=colors_ratio,
                        **kwargs)
    except Exception as e:
        print(f"Error occurred while creating clustermap: {e}")
        return df

    # --- Step 5: Column annotation legend ---
    if classes is not None:
        g.ax_col_dendrogram.legend(legend_handles, legend_labels,
                                   title=None,
                                   bbox_to_anchor=(0.5, 1.15),
                                   loc="upper center",
                                   ncol=len(classes),
                                   handletextpad=0.5,
                                   columnspacing=1.5,
                                   frameon=False)
        
    # --- Step 6: Row label remapping ---
    if x_label == "gene" and xticklabels:
        _ , prot_map = pdata.get_gene_maps(on='protein' if on == 'prot' else 'peptide')
        row_labels = [prot_map.get(row, row) for row in g.data2d.index]
        g.ax_heatmap.set_yticklabels(row_labels, rotation=0)

    # --- Step 8: Store clustering results ---
    cluster_key  = f"{on}_{layer}_clustermap"
    row_order = list(g.data2d.index)
    row_indices = g.dendrogram_row.reordered_ind

    pdata.stats[cluster_key]  = {
        "row_order": row_order,
        "row_indices": row_indices,
        "row_labels": x_label,   # 'accession' or 'gene'
        "namelist_used": namelist if namelist is not None else "all_proteins",
        "col_order": list(g.data2d.columns),
        "col_indices": g.dendrogram_col.reordered_ind if g.dendrogram_col else None,
        "row_linkage": g.dendrogram_row.linkage,  # <--- NEW
        "col_linkage": g.dendrogram_col.linkage if g.dendrogram_col else None,
    }

    return g

# def plot_heatmap(ax, pdata, classes=None, layer="X", cmap=cm.get_cmap('seismic'), norm_values=[4,5.5,7], linewidth=.5, annotate=True, square=False, cbar_kws = {'label': 'Abundance (AU)'}):
#     """
#     Plot annotated heatmap of protein abundance data.

#     Parameters:
#     ax (matplotlib.axes.Axes): The axes on which to plot the heatmap.
#     pdata (scpviz.pdata): The input pdata object.
#     classes (str or list of str, optional): Class column(s) to group samples. If None, all samples are included.
#     cmap (matplotlib.colors.Colormap): The colormap to use for the heatmap.
#     norm_values (list): The low, mid, and high values used to set colorbar scale. Can be assymetric.
#     linewidth (float): Plot linewidth.
#     annotate (bool): Annotate each heatmap entry with numerical value. True by default.
#     square (bool): Make heatmap square. False by default.
#     cbar_kws (dict): Pass-through keyword arguments for the colorbar. See `matplotlib.figure.Figure.colorbar()` for more information.

#     Returns:
#     ax (matplotlib.axes.Axes): The axes with the plotted heatmap.
#     """
#     # get the abundance data for the specified classes
    
#     if cmap is None:
#         cmap = plt.get_cmap("seismic")
#     if cbar_kws is None:
#         cbar_kws = {'label': 'Abundance (log$_2$ AU)'}

#     # get data
#     adata = pdata.prot
#     data = adata.layers[layer] if layer != "X" else adata.X
#     data = data.toarray() if sparse.issparse(data) else data.copy()


#     # log-transform the data
#     abundance_data_log10 = np.log10(data + 1)

#     mid_norm = mcolors.TwoSlopeNorm(vmin=norm_values[0], vcenter=norm_values[1], vmax=norm_values[2])
#     ax = sns.heatmap(abundance_data_log10, yticklabels=True, square=square, annot=annotate, linewidth=linewidth, cmap=cmap, norm=mid_norm, cbar_kws=cbar_kws)

#     return ax

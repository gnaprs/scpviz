"""Volcano plots and volcano annotation helpers."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import copy
import warnings

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from scpviz import utils
from scpviz.utils.formatting import format_log_prefix

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData

def _resolve_p_col_from_df(p_col: str | None, df: pd.DataFrame) -> tuple[str, bool]:
    """Return ``(p_col, p_col_explicit)``. ``None`` prefers ``adj_p_value`` when present."""
    if p_col is None:
        if "adj_p_value" in df.columns:
            return ("adj_p_value", False)
        return ("p_value", False)
    if p_col not in ("p_value", "adj_p_value"):
        raise ValueError("p_col must be 'p_value' or 'adj_p_value'.")
    return (p_col, True)

def _guard_adj_p_col(df: pd.DataFrame, p_col: str) -> None:
    if p_col == "adj_p_value" and "adj_p_value" not in df.columns:
        raise ValueError(
            "p_col='adj_p_value' requested but 'adj_p_value' column is missing. "
            "Run de() with correct_fdr=True first, or pass p_col='p_value'."
        )
    log10_col = f"-log10({p_col})"
    if log10_col not in df.columns:
        raise ValueError(
            f"p_col={p_col!r} requested but {log10_col!r} column is missing in the "
            "differential expression results."
        )

def _warn_volcano_de_inconsistency(
    df: pd.DataFrame,
    *,
    de_data_provided: bool,
    correct_fdr: bool,
    p_col: str,
    p_col_explicit: bool,
) -> None:
    if correct_fdr and "adj_p_value" not in df.columns:
        suffix = (
            "using supplied de_data unchanged."
            if de_data_provided
            else "FDR correction may not have been applied."
        )
        print(
            f"{format_log_prefix('warn')} correct_fdr=True but DE results have no "
            f"'adj_p_value' column; {suffix}"
        )
    if (
        correct_fdr
        and "adj_p_value" in df.columns
        and p_col == "p_value"
        and p_col_explicit
    ):
        print(
            f"{format_log_prefix('warn')} correct_fdr=True and DE results include "
            "adjusted p-values, but p_col='p_value'; significance labels reflect "
            "FDR while the y-axis shows raw p-values. Omit p_col or pass "
            "p_col='adj_p_value' for a consistent volcano."
        )

def plot_volcano(ax: "plt.Axes", pdata: pAnnData | None = None, values: Any = None, method: str = 'ttest', fold_change_mode: str = 'mean', label: Any = 5,
                 label_type='Gene', color=None, alpha=0.5, threshold=0.05, log2fc=1, linewidth=0.5,
                 p_col: str | None = None, correct_fdr: bool = False, equal_var: bool = True,
                 pval: float | None = None,
                 fontsize=8, no_marks=False, classes=None, de_data=None, return_df=False,
                 group_annot=True, group_annot_kwargs=None, group1_kwargs=None, group2_kwargs=None, up_kwargs=None, down_kwargs=None, **kwargs: Any) -> Any:
    """
    Plot a volcano plot of differential expression results.

    This function calculates differential expression (DE) between two groups
    and visualizes results as a volcano plot. Alternatively, it can use
    pre-computed DE results (e.g. from `pdata.de()`).

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData, optional): Input pAnnData object. Required if `de_data`
            is not provided.
        values (list or dict, optional): Values to compare between groups.
            
            - Legacy list format: `["group1", "group2"]`
            
            - Dictionary format: list of dicts specifying multiple conditions,
              e.g. `[{"cellline": "HCT116", "treatment": "DMSO"},
                     {"cellline": "HCT116", "treatment": "DrugX"}]`.

        method (str): Statistical test method. Default is `"ttest"`. Options are `"ttest"`, `"mannwhitneyu"`, `"wilcoxon"`.
        fold_change_mode (str): Method for computing fold change.
            
            - `"mean"`: log2(mean(group1) / mean(group2))
            - `"pairwise_median"`: median of all pairwise log2 ratios.
            - "pep_pairwise_median": median of peptide-level pairwise log2 ratios, aggregated per protein

        label (int, list, or None): Features to highlight.
            
            - If int: label top and bottom *n* features.
            - If list of str: label only the specified features.
            - If list of two ints: `[top, bottom]` to label asymmetric counts.
            - If None: no labels plotted.

        label_type (str): Label content type. Currently `"Gene"` is recommended.
        color (dict, optional): Dictionary mapping significance categories
            to colors. Defaults to grey/red/blue.
        alpha (float): Point transparency. Default is 0.5.
        threshold (float): Significance cutoff for threshold lines and for DE when
            computed inline. Matches ``de()``: raw ``p_value`` when ``correct_fdr=False``,
            ``adj_p_value`` when ``correct_fdr=True``. Default is 0.05.
        pval (float, optional): Deprecated alias for ``threshold``.
        log2fc (float): Log2 fold change threshold for significance. Default is 1.
        p_col (str or None): Column for the volcano y-axis: ``'p_value'`` or ``'adj_p_value'``.
            Default ``None`` auto-selects ``'adj_p_value'`` when ``correct_fdr=True``, otherwise
            ``'p_value'``. Pass explicitly to override.
        correct_fdr (bool): Passed to ``pdata.de()`` when DE is computed inline, adds multiple testing correction.
            When True, ``p_col`` defaults to ``'adj_p_value'`` unless overridden. Ignored when
            ``de_data`` is provided except for ``p_col`` auto-selection and consistency warnings.
        equal_var (bool): Passed to ``pdata.de()`` for Student vs Welch t-test.
        linewidth (float): Line width for threshold lines. Default is 0.5.
        fontsize (int): Font size for feature labels. Default is 8.
        no_marks (bool): If True, suppress coloring of significant points and
            plot all points in grey. Default is False.
        classes (str, optional): Sample class column to use for group comparison.
        de_data (pandas.DataFrame, optional): Pre-computed DE results. Must contain
            `"log2fc"`, `"p_value"`, and `"significance"` columns.
        return_df (bool): If True, return both the axis and the DataFrame used
            for plotting. Default is False.
        group_annot (bool): If True, annotate group names and differential
            expression counts (n) at the top of the plot. If False, suppress
            all group-related annotations. Default is True.
        group_annot_kwargs (dict, optional): Global configuration for group
            annotations. Supported keys include:

            - `"pos"`: Dictionary controlling annotation positions in axes
              fraction coordinates. Expected keys are `"group1_xy"`,
              `"group2_xy"`, `"up_xy"`, and `"down_xy"`, each mapping to
              an `(x, y)` tuple.
            
            - `"bbox"`: Dictionary of bounding box properties passed to
              `matplotlib.text.Annotation`, or `None` to disable the bounding
              box for group labels.

        group1_kwargs (dict, optional): Keyword arguments passed to
            `ax.annotate()` for the first group label (right-aligned by
            default). Can be used to override font size, weight, alignment,
            or other text properties.
        group2_kwargs (dict, optional): Keyword arguments passed to
            `ax.annotate()` for the second group label (left-aligned by
            default). Can be used to override font size, weight, alignment,
            or other text properties.
        up_kwargs (dict, optional): Keyword arguments passed to
            `ax.annotate()` for the upregulated feature count (`n=...`).
            Useful for adjusting font size, color, or vertical spacing
            independently of other annotations.
        down_kwargs (dict, optional): Keyword arguments passed to
            `ax.annotate()` for the downregulated feature count (`n=...`).
            Useful for adjusting font size, color, or vertical spacing
            independently of other annotations.
        **kwargs (Any): Additional keyword arguments passed to `matplotlib.pyplot.scatter`.

    Returns:
        ax (matplotlib.axes.Axes): Axis with the volcano plot if `return_df=False`.
        tuple (matplotlib.axes.Axes, pandas.DataFrame): Returned if `return_df=True`.

    Usage Tips:
        mark_volcano: Highlight specific features on an existing volcano plot.  
        - For selective highlighting, set `no_marks=True` to render all points
          in grey, then call `mark_volcano()` to add specific features of interest.

        add_volcano_legend: Add standard legend handles for volcano plots.
        - Use the helper function `add_volcano_legend(ax)` to add standard
          significance legend handles.

    Example:
        Dictionary-style input:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            ax, df = scplt.plot_volcano(ax, pdata_norm, values=values, return_df=True)
            plt.show()
            ```

        ![Plot volcano](../../assets/plots/plot_volcano.png)

        Legacy input:
            ```python
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            colors = sns.color_palette("Paired")[4:6]
            color_dict = dict(zip(["downregulated", "upregulated"], colors))
            ax, df = scplt.plot_volcano(
                ax,
                pdata_norm,
                classes="condition",
                values=["kd", "sc"],
                color=color_dict,
                return_df=True,
            )
            scplt.add_volcano_legend(ax)
            plt.show()
            ```
        To tweak styling:

            Move positions up/down and tweak styling:
            ```python
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            scplt.plot_volcano(
                ax, pdata_norm, values=values,
                group_annot_kwargs={"pos": {"group1_xy": (0.98, 1.10), "group2_xy": (0.02, 1.10)}},
                up_kwargs={"fontsize": 9},
                down_kwargs={"fontsize": 9},
            )
            ```
            Remove the bbox but keep text:
            ```python
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            scplt.plot_volcano(
                ax, pdata_norm, values=values,
                group_annot_kwargs={"bbox": None},
            )
            ```
            Turn off all text:
            ```python
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            scplt.plot_volcano(ax, pdata_norm, values=values, group_annot=False)
            ```

        Multiple testing correction (Benjamini-Hochberg FDR):
            Proteomics DE tests thousands of proteins at once; without correction,
            many raw p-values will appear significant by chance. Set ``correct_fdr=True``
            to apply Benjamini-Hochberg FDR adjustment: significance labels and the
            dashed horizontal threshold use adjusted p-values, and the y-axis defaults
            to ``-log10(adj_p_value)`` (override with ``p_col`` if needed):

            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            ax, df = scplt.plot_volcano(
                ax, pdata_norm, values=values,
                correct_fdr=True, threshold=0.05, return_df=True,
            )
            scplt.add_volcano_legend(ax)
            plt.show()
            ```

            With pre-computed DE results from ``pdata.de(correct_fdr=True)``:

            ```python
            de_df = pdata_norm.de(values=values, correct_fdr=True, threshold=0.05)
            scplt.plot_volcano(ax, de_data=de_df, correct_fdr=True)
            ```

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    import matplotlib.patheffects as PathEffects

    if de_data is None and pdata is None:
        raise ValueError("Either de_data or pdata must be provided.")

    if pval is not None:
        print(
            f"{format_log_prefix('warn')} `pval` is deprecated in plot_volcano(); "
            f"use `threshold` instead (applied pval={pval})."
        )
        if pval != threshold:
            print(
                f"{format_log_prefix('warn')} Both `threshold`={threshold} and "
                f"`pval={pval}` were passed to plot_volcano(); using `pval`."
            )
        threshold = pval

    if p_col is None:
        p_col = "adj_p_value" if correct_fdr else "p_value"
        p_col_explicit = False
    else:
        if p_col not in ("p_value", "adj_p_value"):
            raise ValueError("p_col must be 'p_value' or 'adj_p_value'.")
        p_col_explicit = True

    if de_data is not None:
        volcano_df = de_data.copy()
    else:
        if values is None:
          raise ValueError("If pdata is provided, values must also be provided.")
        if isinstance(values, list) and isinstance(values[0], dict):
          volcano_df = pdata.de(
              values=values, method=method, threshold=threshold, log2fc=log2fc,
              fold_change_mode=fold_change_mode, correct_fdr=correct_fdr, equal_var=equal_var,
          )
        else:
            volcano_df = pdata.de(
                class_type=classes, values=values, method=method, threshold=threshold,
                log2fc=log2fc, fold_change_mode=fold_change_mode, correct_fdr=correct_fdr,
                equal_var=equal_var,
            )

    _warn_volcano_de_inconsistency(
        volcano_df,
        de_data_provided=de_data is not None,
        correct_fdr=correct_fdr,
        p_col=p_col,
        p_col_explicit=p_col_explicit,
    )
    _guard_adj_p_col(volcano_df, p_col)

    df = volcano_df.copy()
    log10_col = f"-log10({p_col})"
    volcano_df = volcano_df.dropna(subset=[p_col]).copy()
    volcano_df = volcano_df[volcano_df["significance"] != "not comparable"]

    default_color = {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}
    if color:
        default_color.update(color)
    elif no_marks:
        default_color = {k: 'grey' for k in default_color}

    scatter_kwargs = dict(s=20, edgecolors='none')
    scatter_kwargs.update(kwargs)
    colors = volcano_df['significance'].astype(str).map(default_color)

    ax.scatter(volcano_df['log2fc'], volcano_df[log10_col],
               c=colors, alpha=alpha, **scatter_kwargs)

    ax.axhline(-np.log10(threshold), color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(-log2fc, color='black', linestyle='--', linewidth=linewidth)

    ax.set_xlabel('$log_{2}$ fold change')
    ylabel = '-$log_{10}$ adjusted p value' if p_col == 'adj_p_value' else '-$log_{10}$ p value'
    ax.set_ylabel(ylabel)

    log2fc_clean = volcano_df['log2fc'].replace([np.inf, -np.inf], np.nan).dropna()
    if log2fc_clean.empty:
        max_abs_log2fc = 1  # default range if nothing valid
    else:
        max_abs_log2fc = log2fc_clean.abs().max() + 0.5
    ax.set_xlim(-max_abs_log2fc, max_abs_log2fc)


    if not no_marks and label not in [None, 0, [0, 0]]:
        if isinstance(label, int):
            upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
            downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)
            label_df = pd.concat([upregulated.head(label), downregulated.head(label)])
        elif isinstance(label, list):
            if len(label) == 2 and all(isinstance(i, int) for i in label):
                upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
                downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)
                label_df = pd.concat([upregulated.head(label[0]), downregulated.head(label[1])])
            else:
                label_lower = [str(l).lower() for l in label]
                label_df = volcano_df[
                volcano_df.index.str.lower().isin(label_lower) |
                volcano_df['Genes'].str.lower().isin(label_lower)
            ]

        else:
            raise ValueError("label must be int or list")

        texts = []
        for i in range(len(label_df)):
            gene = label_df.iloc[i].get('Genes', label_df.index[i])
            txt = plt.text(label_df.iloc[i]['log2fc'],
                           label_df.iloc[i][log10_col],
                           s=gene,
                           fontsize=fontsize,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.6))
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            texts.append(txt)
        
        adjust_text(texts, expand=(2, 2), arrowprops=dict(arrowstyle='->', color='k', zorder=5))

    # Add group names and DE counts to plot
    def format_group(values_entry, classes):
        if isinstance(values_entry, dict):
            return "/".join(str(v) for v in values_entry.values())
        elif isinstance(values_entry, list) and isinstance(classes, list) and len(values_entry) == len(classes):
            return "/".join(str(v) for v in values_entry)
        return str(values_entry)

    group1 = group2 = ""
    if isinstance(values, list) and len(values) == 2:
        group1 = format_group(values[0], classes)
        group2 = format_group(values[1], classes)

    up_count = (volcano_df['significance'] == 'upregulated').sum()
    down_count = (volcano_df['significance'] == 'downregulated').sum()

    # --- Group annotations (configurable) ---
    if group_annot:
        def _merge(base, extra):
            out = dict(base)
            if extra:
                out.update(extra)
            return out

        group_annot_kwargs = group_annot_kwargs or {}
        group1_kwargs = group1_kwargs or {}
        group2_kwargs = group2_kwargs or {}
        up_kwargs = up_kwargs or {}
        down_kwargs = down_kwargs or {}

        # Defaults (can be overridden via *_kwargs)
        bbox_style = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black")

        base_text = dict(xycoords="axes fraction", fontsize=fontsize, annotation_clip=False,        )
        base_group = dict(weight="bold", bbox=bbox_style, va="bottom")
        base_count = dict(va="bottom")

        # Default positions (can be overridden globally or per-item)
        default_pos = dict(group1_xy=(0.98, 1.07), up_xy=(0.98, 1.015),  group2_xy=(0.02, 1.07), down_xy=(0.02, 1.015),)
        pos = _merge(default_pos, group_annot_kwargs.get("pos"))

        # Allow overriding bbox (or disabling it by bbox=None)
        bbox_override = group_annot_kwargs.get("bbox", bbox_style)
        if bbox_override is None:
            base_group = dict(base_group)
            base_group.pop("bbox", None)
        else:
            base_group = dict(base_group, bbox=bbox_override)

        # Group labels
        ax.annotate(group1, xy=pos["group1_xy"],  ha="right", **_merge(_merge(base_text, base_group), group1_kwargs),
        )
        ax.annotate(group2, xy=pos["group2_xy"], ha="left", **_merge(_merge(base_text, base_group), group2_kwargs),
        )

        # Counts
        ax.annotate(f"n={up_count}", xy=pos["up_xy"], ha="right",
            **_merge(_merge(_merge(base_text, base_count), {"color": default_color.get("upregulated", "red")}), up_kwargs))
        ax.annotate(f"n={down_count}", xy=pos["down_xy"], ha="left",
            **_merge(_merge(_merge(base_text, base_count), {"color": default_color.get("downregulated", "blue")}), down_kwargs))
        
    if return_df:
        return ax, df
    else:
        return ax

def plot_volcano_adata(ax: "plt.Axes", adata: Any = None, values: Any = None, class_type: Any = None, de_data: Any = None,
    gene_col=None, method='ttest', fold_change_mode='mean', layer='X', label=5, fontsize=8,
    alpha=0.5, color=None, linewidth=0.5, threshold=0.05, log2fc=1.0, p_col: str | None = None,
    correct_fdr: bool = False, equal_var: bool = True, pval: float | None = None,
    no_marks=False,
    return_df=False, **kwargs
) -> Any:
    """
    Volcano plot for AnnData with the *same API behavior* as pdata.plot_volcano.

    Required:
        - Either ``de_data`` OR (``adata`` and ``values``). For legacy-style ``values`` (group labels or list-of-lists), also pass ``class_type`` as documented in :func:`scpviz.utils.stats.de_adata`.
        
    Supports:
        - Dictionary-style values: [{"cellline":"HCT116","tx":"DMSO"}, {...}]
        - Legacy-style values: ["A","B"]
        - Legacy multi-col values: [["HCT116","DMSO"], ["HCT116","DrugX"]]

    Produces: identical volcano to pAnnData version.

    Args:
        correct_fdr (bool): If True, apply Benjamini-Hochberg FDR correction for
            multiple testing when DE is computed inline (via ``de_adata``). Significance
            and the y-axis use adjusted p-values by default. See :func:`plot_volcano`.
        p_col (str or None): ``'p_value'`` or ``'adj_p_value'``. Default ``None`` auto-selects
            from ``correct_fdr``. Other arguments mirror :func:`plot_volcano` / ``de_adata``.
        pval (float, optional): Deprecated alias for ``threshold``.

    Example:
        After DE on ``adata`` with the same comparison as :func:`plot_volcano`, the figure matches :func:`plot_volcano` (same PNG):

            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            ax, df = scplt.plot_volcano_adata(
                ax, pdata_norm.prot, values=values, return_df=True
            )
            plt.show()
            ```

        ![Plot volcano (same style as plot_volcano_adata)](../../assets/plots/plot_volcano.png)

        Multiple testing correction (Benjamini-Hochberg FDR):
            Same as :func:`plot_volcano` — use when testing many features and you want
            FDR-controlled significance rather than raw p-values:

            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            ax, df = scplt.plot_volcano_adata(
                ax, pdata_norm.prot, values=values,
                correct_fdr=True, threshold=0.05, return_df=True,
            )
            plt.show()
            ```

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    import matplotlib.patheffects as PathEffects

    if pval is not None:
        print(
            f"{format_log_prefix('warn')} `pval` is deprecated in plot_volcano_adata(); "
            f"use `threshold` instead (applied pval={pval})."
        )
        if pval != threshold:
            print(
                f"{format_log_prefix('warn')} Both `threshold`={threshold} and "
                f"`pval={pval}` were passed to plot_volcano_adata(); using `pval`."
            )
        threshold = pval

    if p_col is None:
        p_col = "adj_p_value" if correct_fdr else "p_value"
        p_col_explicit = False
    else:
        if p_col not in ("p_value", "adj_p_value"):
            raise ValueError("p_col must be 'p_value' or 'adj_p_value'.")
        p_col_explicit = True

    if de_data is not None:
        df = de_data.copy()
        # For labels: user must supply labels manually if needed
        group1_label = df.attrs.get("group1_label", None)
        group2_label = df.attrs.get("group2_label", None)

    else:
        if adata is None or values is None:
            raise ValueError("When de_data is not provided, must supply adata and values.")

        df = utils.de_adata(
            adata=adata, values=values, class_type=class_type,
            method=method, fold_change_mode=fold_change_mode, layer=layer,
            threshold=threshold, log2fc=log2fc, correct_fdr=correct_fdr,
            equal_var=equal_var, gene_col=gene_col,
        )

        def format_group(val, class_type):
            if isinstance(val, dict):
                return "/".join(str(v) for v in val.values())
            elif isinstance(val, list) and isinstance(class_type, list):
                return "/".join(str(v) for v in val)
            else:
                return str(val)

        group1_label = format_group(values[0], class_type)
        group2_label = format_group(values[1], class_type)

    _warn_volcano_de_inconsistency(
        df,
        de_data_provided=de_data is not None,
        correct_fdr=correct_fdr,
        p_col=p_col,
        p_col_explicit=p_col_explicit,
    )
    _guard_adj_p_col(df, p_col)

    # volcano plotting
    log10_col = f"-log10({p_col})"
    volcano_df = df.dropna(subset=[p_col]).copy()
    volcano_df = volcano_df[volcano_df["significance"] != "not comparable"]

    default_color = {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}
    if color:
        default_color.update(color)
    elif no_marks:
        default_color = {k: 'grey' for k in default_color}

    scatter_kwargs = dict(s=20, edgecolors='none')
    scatter_kwargs.update(kwargs)

    colors = volcano_df['significance'].astype(str).map(default_color)

    ax.scatter(
        volcano_df['log2fc'],
        volcano_df[log10_col],
        c=colors, alpha=alpha,
        **scatter_kwargs
    )

    # threshold lines
    ax.axhline(-np.log10(threshold), color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(-log2fc, color='black', linestyle='--', linewidth=linewidth)

    ax.set_xlabel('$log_{2}$ fold change')
    ylabel = '-$log_{10}$ adjusted p value' if p_col == 'adj_p_value' else '-$log_{10}$ p value'
    ax.set_ylabel(ylabel)

    # symmetric x-limits
    log2fc_clean = pd.to_numeric(volcano_df['log2fc'], errors='coerce').dropna()
    max_abs = log2fc_clean.abs().max() + 0.5 if not log2fc_clean.empty else 1
    ax.set_xlim(-max_abs, max_abs)

    if not no_marks and label not in [None, 0, [0, 0]]:
        if isinstance(label, int):
            up = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values(
                'significance_score', ascending=False
            )
            down = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values(
                'significance_score', ascending=True
            )
            label_df = pd.concat([up.head(label), down.head(label)])

        elif isinstance(label, list):
            if len(label) == 2 and all(isinstance(i, int) for i in label):
                up = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values(
                    'significance_score', ascending=False
                )
                down = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values(
                    'significance_score', ascending=True
                )
                label_df = pd.concat([up.head(label[0]), down.head(label[1])])

            else:
                ll = [str(v).lower() for v in label]
                label_df = volcano_df[
                    volcano_df.index.str.lower().isin(ll) |
                    volcano_df.get("Genes", pd.Series("", index=volcano_df.index)).str.lower().isin(ll)
                ]

        else:
            raise ValueError("label must be int or list")

        # plot labels
        texts = []
        for idx, row in label_df.iterrows():
            text_val = row.get('Genes', idx)
            txt = ax.text(
                row['log2fc'], row[log10_col],
                s=text_val,
                fontsize=fontsize,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.6)
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            texts.append(txt)

        adjust_text(texts, expand=(2, 2),
                    arrowprops=dict(arrowstyle='->', color='k', zorder=5))

    bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black')

    if group1_label:
        ax.annotate(group1_label, xy=(0.98, 1.07), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=fontsize,
                    weight='bold', bbox=bbox_style)

    if group2_label:
        ax.annotate(group2_label, xy=(0.02, 1.07), xycoords='axes fraction',
                    ha='left', va='bottom', fontsize=fontsize,
                    weight='bold', bbox=bbox_style)

    up_count = (volcano_df['significance'] == 'upregulated').sum()
    down_count = (volcano_df['significance'] == 'downregulated').sum()

    ax.annotate(f'n={up_count}', xy=(0.98, 1.015), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=fontsize,
                color=default_color['upregulated'])

    ax.annotate(f'n={down_count}', xy=(0.02, 1.015), xycoords='axes fraction',
                ha='left', va='bottom', fontsize=fontsize,
                color=default_color['downregulated'])

    return (ax, df) if return_df else ax

def add_volcano_legend(ax: "plt.Axes", colors: dict[str, str] | None = None) -> None:
    """
    Add a standard legend for volcano plots.

    This function appends a legend to a volcano plot axis, showing handles for
    upregulated, downregulated, and non-significant features. Colors can be
    customized, but default to grey, red, and blue.

    Args:
        ax (matplotlib.axes.Axes): Axis object to which the legend will be added.

        colors (dict, optional): Custom colors for significance categories.
            Keys must include `"upregulated"`, `"downregulated"`, and
            `"not significant"`. Defaults to:
            
            ```python
            {
                "not significant": "grey",
                "upregulated": "red",
                "downregulated": "blue"
            }
            ```

    Example:
        Add legend handles for significance categories:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(3, 2))
            scplt.add_volcano_legend(ax)
            plt.show()
            ```

        ![Add volcano legend](../../assets/plots/add_volcano_legend.png)

    Returns:
        None
    """
    from matplotlib.lines import Line2D
    import numpy as np

    default_colors = {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}
    if colors is None:
        colors = default_colors.copy()
    else:
        default_colors.update(colors)
        colors = default_colors

    handles = [
        Line2D([0], [0], marker='o', color='w', label='Up', markerfacecolor=colors['upregulated'], markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Down', markerfacecolor=colors['downregulated'], markersize=6),
        Line2D([0], [0], marker='o', color='w', label='NS', markerfacecolor=colors['not significant'], markersize=6)
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=7)

def mark_volcano(ax: "plt.Axes", volcano_df: pd.DataFrame, label: Any, label_color: str = "black", text_color: str | None = None, label_type: str = 'Gene', s: float = 10, alpha: float = 1, show_names: bool = True, fontsize: int = 8, p_col: str | None = None, return_texts: bool = False) -> Any:
    """
    Mark a volcano plot with specific proteins or genes.

    This function highlights selected features on an existing volcano plot,
    optionally labeling them with names.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        volcano_df (pandas.DataFrame): DataFrame returned by `plot_volcano()` or
            `pdata.de()`, containing differential expression results.
        label (list): Features to highlight. Can also be a nested list, with
            separate lists of features for different cases.
        label_color (str or list, optional): Marker color(s). Defaults to `"black"`.
            If a list is provided, each case receives a different color.
        text_color (str, optional): Text color. Defaults to the same as label_color if not explicitly provided.
        label_type (str): Type of label to display. Default is `"Gene"`.
        s (float): Marker size. Default is 10.
        alpha (float): Marker transparency. Default is 1.
        show_names (bool): Whether to show labels for the selected features.
            Default is True.
        fontsize (int): Font size for labels. Default is 8.
        p_col (str or None): Column for y-positions: ``'p_value'`` or ``'adj_p_value'``.
            Default ``None`` uses ``'adj_p_value'`` when that column is present (e.g. after
            ``de(correct_fdr=True)``), otherwise ``'p_value'``. Pass explicitly to override.
        return_texts (bool): Whether to return the list of created text artists.
            This is useful when labeling multiple groups and performing a single
            global `adjust_text()` call at the end.

    Returns:
        ax (matplotlib.axes.Axes): Axis with the highlighted volcano plot.

    Example:
        Highlight specific features on a volcano plot:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            ax, volcano_df = scplt.plot_volcano(
                ax, pdata_norm, values=values, return_df=True, label=[0, 0]
            )
            scplt.mark_volcano(ax, volcano_df, label=["GAPDH", "TUBB", "ACTB"])
            plt.show()
            ```

        ![Mark volcano](../../assets/plots/mark_volcano.png)

    Note:
        This function works especially well in combination with
        `plot_volcano(..., no_marks=True)` to render all points in grey,
        followed by `mark_volcano()` to selectively highlight features of interest.
    """
    p_col, _ = _resolve_p_col_from_df(p_col, volcano_df)
    _guard_adj_p_col(volcano_df, p_col)
    log10_col = f"-log10({p_col})"

    if return_texts and not show_names:
        print(f"{utils.format_log_prefix('warn_only')} "
            "return_texts=True but show_names=False; no text labels will be returned.")

    if not isinstance(label[0], list):
        label = [label]
        label_color = [label_color] if isinstance(label_color, str) else label_color

    if "Genes" in volcano_df.columns:
        gene_col = volcano_df["Genes"].astype(str)
    else:
        # fallback: use the index as feature names
        gene_col = volcano_df.index.astype(str)

    all_texts = []
    for i, label_group in enumerate(label):
        color = label_color[i % len(label_color)] if isinstance(label_color, list) else label_color
        txt_color = text_color if text_color is not None else color

        # Match by index or 'Genes' column
        match_mask = (
            volcano_df.index.isin(label_group) |
            gene_col.isin(label_group)
        )
        match_df = volcano_df[match_mask]

        ax.scatter(match_df['log2fc'], match_df[log10_col],
                   c=color, s=s, alpha=alpha, edgecolors='none')

        if show_names:
            texts = []
            for idx, row in match_df.iterrows():
                if label_type == "Gene" and "Genes" in volcano_df.columns:
                    text = row.get("Genes", idx)
                else:
                    text = idx

                txt = ax.text(row['log2fc'], row[log10_col],
                              s=text,
                              fontsize=fontsize,
                              color=txt_color ,
                              bbox=dict(facecolor='white', edgecolor=txt_color , boxstyle='round'))
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
                texts.append(txt)
                all_texts.append(txt)

            if not return_texts:
                adjust_text(texts, expand=(2, 2),
                            arrowprops=dict(arrowstyle='->', color=txt_color , zorder=5))

    if return_texts:
        return ax, all_texts
    return ax

def mark_volcano_by_significance(
    ax: "plt.Axes",
    volcano_df: pd.DataFrame,
    label: Any,
    color: Any = None,
    text_color: str | None = None,
    label_type: str = "Gene",
    s: float = 10,
    alpha: float = 1,
    show_names: bool = True,
    fontsize: int = 8,
    p_col: str | None = None,
    return_texts: bool = False,
) -> Any:
    """
    Mark a volcano plot with specific proteins or genes, colored by significance.

    This function highlights selected features on an existing volcano plot,
    using the `significance` column in `volcano_df` to determine colors
    (e.g. "upregulated", "downregulated", "not significant").

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        volcano_df (pandas.DataFrame): DataFrame returned by `plot_volcano()` or
            `pdata.de()`, containing differential expression results and a
            `significance` column with values such as:
            "upregulated", "downregulated", "not significant".
        label (list): Features to highlight. Can also be a nested list, with
            separate lists of features for different cases. All features are
            colored according to their `significance`, not by group.
        color (dict, optional): Mapping from significance category to color.
            Defaults to:
                {
                    "not significant": "grey",
                    "upregulated": "red",
                    "downregulated": "blue",
                }
            You can override any of these by passing a dict with the same keys.
        text_color (str, optional): Text color. Default is None, which makes each label follow its corresponding marker color.
            - If str: all labels use the same text color.
            - If dict: mapping from significance category to text color
              (e.g. "upregulated", "downregulated", "not significant").
              Categories not found in the dict fall back to the `"not significant"`
              text color (or black if not provided).

        label_type (str): Type of label to display. Default is `"Gene"`.
        s (float): Marker size. Default is 10.
        alpha (float): Marker transparency. Default is 1.
        show_names (bool): Whether to show labels for the selected features.
            Default is True.
        fontsize (int): Font size for labels. Default is 8.
        p_col (str or None): Column for y-positions: ``'p_value'`` or ``'adj_p_value'``.
            Default ``None`` uses ``'adj_p_value'`` when that column is present (e.g. after
            ``de(correct_fdr=True)``), otherwise ``'p_value'``. Pass explicitly to override.
        return_texts (bool): Whether to return the list of created text artists.
            This is useful when labeling multiple groups and performing a single
            global `adjust_text()` call at the end.

    Returns:
        matplotlib.axes.Axes: Axis with highlighted points if `return_texts=False`.
        tuple (matplotlib.axes.Axes, list): Returned if `return_texts=True`,
            where the list contains the text artists for further adjustment.

    Example:
        Highlight specific features on a volcano plot using significance colors;
        ``label`` is required. This example marks the top up- and down-regulated
        features by ``significance_score``:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            values = [
                {"cellline": "BE", "condition": "kd"},
                {"cellline": "BE", "condition": "sc"},
            ]
            ax, volcano_df = scplt.plot_volcano(
                ax, pdata_norm, values=values, return_df=True, label=[0, 0]
            )
            up_ids = (
                volcano_df[volcano_df["significance"] == "upregulated"]
                .sort_values("significance_score", ascending=False)
                .head(5)
                .index.tolist()
            )
            down_ids = (
                volcano_df[volcano_df["significance"] == "downregulated"]
                .sort_values("significance_score", ascending=True)
                .head(5)
                .index.tolist()
            )
            scplt.mark_volcano_by_significance(ax, volcano_df, label=up_ids + down_ids)
            plt.show()
            ```

        ![Mark volcano by significance](../../assets/plots/mark_volcano_by_significance.png)

    Note:
        This function is designed to work seamlessly with
        `plot_volcano(..., no_marks=True)` for workflows where you first plot
        all points in grey and then selectively highlight features of interest.
    """

    default_color = {
        "not significant": "grey",
        "upregulated": "red",
        "downregulated": "blue",
    }
    if color:
        default_color.update(color)

    if "significance" not in volcano_df.columns:
        raise ValueError(
            "volcano_df must contain a 'significance' column to use "
            "`mark_volcano_by_significance`."
        )

    p_col, _ = _resolve_p_col_from_df(p_col, volcano_df)
    _guard_adj_p_col(volcano_df, p_col)
    log10_col = f"-log10({p_col})"

    if return_texts and not show_names:
        print(f"{utils.format_log_prefix('warn_only')} "
            "return_texts=True but show_names=False; no text labels will be returned.")

    if not isinstance(label[0], list):
        label = [label]

    # Decide what we match on for names
    if "Genes" in volcano_df.columns:
        gene_col = volcano_df["Genes"].astype(str)
    else:
        gene_col = volcano_df.index.astype(str)

    all_texts = []
    for label_group in label:
        # Match by index or 'Genes' column
        match_mask = (
            volcano_df.index.isin(label_group) |
            gene_col.isin(label_group)
        )
        match_df = volcano_df[match_mask].copy()

        if match_df.empty:
            continue

        sig_series = match_df["significance"].astype(str)
        point_colors = sig_series.map(default_color).fillna(default_color["not significant"])

        ax.scatter(
            match_df["log2fc"],
            match_df[log10_col],
            c=point_colors,
            s=s,
            alpha=alpha,
            edgecolors="none",
        )

        if show_names:
            texts = []

            # Resolve text colors
            if text_color is None:
                text_colors = point_colors  # follow marker color (per-point)
            elif isinstance(text_color, dict):
                tc = sig_series.map(text_color)
                fallback = text_color.get("not significant", "black")
                text_colors = tc.fillna(fallback)
            else:
                text_colors = [text_color] * len(match_df)  # single str

            for (idx, row), c, tc in zip(match_df.iterrows(), point_colors, text_colors):
                if label_type == "Gene" and "Genes" in volcano_df.columns:
                    text = row.get("Genes", idx)
                else:
                    text = idx

                txt = ax.text(
                    row["log2fc"],
                    row[log10_col],
                    s=text,
                    fontsize=fontsize,
                    color=tc,
                    bbox=dict(facecolor="white", edgecolor=tc, boxstyle="round"),
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=3, foreground="w")]
                )
                texts.append(txt)
                all_texts.append(txt)

            if not return_texts:
                adjust_text(
                    texts,
                    expand=(2, 2),
                    arrowprops=dict(arrowstyle="->", color="black", zorder=5),
                )

    if return_texts:
        return ax, all_texts
    return ax

def volcano_adjust_and_outline_texts(
    texts: list[Any],
    expand: tuple[float, float] = (2, 2),
    arrowprops: dict[str, Any] = dict(arrowstyle="->", color="k", lw=0.8),
    linewidth: float = 3,
    outline_color: str = "w",
) -> Any:
    """
    Adjust text labels for volcano plots and apply a white outline for readability.

    This function runs `adjust_text()` on a list of text artists while temporarily
    removing their path effects to ensure stable label placement. A white outline
    is re-applied after adjustment to improve legibility on dense volcano plots
    or scatter backgrounds.

    Args:
        texts (list): List of `matplotlib.text.Text` objects, typically returned
            from `mark_volcano_by_significance(..., return_texts=True)`.
        expand (tuple): Expansion parameters passed to `adjust_text()`.
            Default is `(2, 2)`.
        arrowprops (dict or None): Arrow properties passed to `adjust_text()`.
            Set to `None` to disable arrow drawing. Default draws black arrows.
        linewidth (float): Line width of the outline applied after adjustment.
            Default is 3.
        outline_color (str): Color of the outline stroke. Default is `"w"`.

    Returns:
        list: The same list of text objects (modified in place).

    Example:
        Adjust and outline labels for multiple marked volcano groups:

            ```python
            ax, volcano_df = scplt.plot_volcano(
                ax, pdata_6mo_snpc_norm, values=case_values,
                return_df=True, no_marks=True
            )

            rps_dict={'downregulated': '#5166FF'}
            rpl_dict={'downregulated': '#1F2CCF'}

            # in this case, two sets of texts from mark_volcano or mark_volcano_by_significance (return_texts=True)
            texts = []
            ax, t = scplt.mark_volcano(
                ax, volcano_df, label=rpl_top5, label_color='#1F2CCF',return_texts=True
            )
            texts.extend(t)

            ax, t = scplt.mark_volcano_by_significance(
                ax, volcano_df, label=rps_top5, color=rps_dict, return_texts=True
            )
            texts.extend(t)

            # and for others, use show_names=False to not show any names/arrows
            scplt.mark_volcano_by_significance(
                ax, volcano_df, label=rpl_others, color=rpl_dict, show_names=False
            )
            scplt.mark_volcano_by_significance(
                ax, volcano_df, label=rps_others, color=rps_dict, show_names=False
            )

            volcano_adjust_and_outline_texts(texts, expand=(2, 2))
            ```

        ![Volcano adjust and outline texts](../../assets/plots/volcano_adjust_and_outline_texts.png)

    Note:
        This function is designed to be used after collecting all labels from
        multiple `mark_volcano_by_significance(..., return_texts=True)` calls.
        Running `adjust_text()` once globally produces cleaner layouts than
        multiple per-group adjustments.
    """

    from adjustText import adjust_text
    import matplotlib.patheffects as PathEffects

    orig_effects = []
    for txt in texts:
        orig_effects.append(txt.get_path_effects())
        txt.set_path_effects([])

    # adjustText
    adjust_kwargs = {"expand": expand}
    if arrowprops is not None:
        adjust_kwargs["arrowprops"] = arrowprops

    adjust_text(texts, **adjust_kwargs)

    for txt in texts:
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=linewidth, foreground=outline_color)
        ])

    return texts


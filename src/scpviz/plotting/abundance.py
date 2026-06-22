"""Abundance, distribution, CV, rank, raincloud plots."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

from scpviz import utils
from scpviz.utils.formatting import format_log_prefix

from .style import get_color, plot_significance

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData

def _plotting_pkg_utils():
    """`scpviz.utils` as exposed on `scpviz.plotting` (tests may patch ``scplt.utils``)."""
    import scpviz.plotting as _pkg

    return _pkg.utils

def plot_cv(
    ax: "plt.Axes",
    pdata: pAnnData,
    classes: str | list[str] | None = None,
    layer: str = "X",
    on: str = "protein",
    order: list[str] | None = None,
    palette: Any = None,
    return_df: bool = False,
    extra_cols: list[str] = ["Accession", "Genes"],
    show_n: bool = False,
    annotate: str | dict[str, str] | None = None,
    n_kwargs: dict[str, Any] | None = None,
    annotate_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Plot coefficient of variation (CV) distributions as violins.

    This function computes CV values across proteins or peptides, grouped by
    sample-level classes, and visualizes their distribution. CV is stored as a
    ratio in ``pdata.var``; the plot and ``CV_pct`` column use percent
    (ratio × 100).

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object containing protein or peptide data.
        classes (str or list of str, optional): One or more `.obs` columns to use
            for grouping samples in the plot. If None, no grouping is applied.
        layer (str): Data layer to use for CV calculation. Default is `'X'`.
        on (str): Data level to compute CV on, either `'protein'` or `'peptide'`.
        order (list, optional): Custom order of classes for plotting.
            If None, defaults to alphabetical order.
        palette (dict or list, optional): Custom color palette for class groups.
            If None, defaults to `scviz` package color palette.
        return_df (bool): If True, returns the underlying DataFrame used for plotting.
        extra_cols (list): Additional columns to include in returned dataframe.
        show_n (bool): If True, annotate each violin with the total sample count
            in that group (``n={count}``), placed below the x-axis tick labels.
        annotate (str or dict, optional): Per-violin annotations along the top of
            the plotting area (just above the upper axis spine).
            - ``"median"`` or ``"mean"``: summary stat of ``CV_pct`` on two lines
              (e.g. ``median`` / ``12.3%``).
            - ``dict``: custom label per class key; keys not present are skipped.
        n_kwargs (dict, optional): Styling for ``show_n`` labels. Recognized keys
            include Matplotlib text options plus ``offset`` (distance below the
            x-axis in axes coordinates; default ``0.12``).
        annotate_kwargs (dict, optional): Styling for ``annotate`` labels. Recognized
            keys include Matplotlib text options plus ``offset`` (y position in axes
            coordinates above the top spine; default ``1.03``).
        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes): The axis with the plotted CV distribution.
        cv_df (pandas.DataFrame): Optional, returned if `return_df=True`. Columns
            include ``CV`` (ratio) and ``CV_pct`` (percent).

    Example:
        Basic CV violins grouped by cell line and condition:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(3, 3))
            scplt.plot_cv(ax, pdata, classes=["cellline", "condition"])
            plt.show()
            ```

        ![Plot cv](../../assets/plots/plot_cv.png)

        Sample counts below each violin and median CV above the plot:
            ```python
            fig, ax = plt.subplots(figsize=(3, 3))
            scplt.plot_cv(
                ax, pdata, classes=["cellline", "condition"],
                show_n=True,
                annotate="median",
                annotate_kwargs={"fontsize": 7},
            )
            plt.show()
            ```

        ![Plot cv annotate](../../assets/plots/plot_cv_annotate.png)

        Custom per-group labels:
            ```python
            fig, ax = plt.subplots(figsize=(3, 3))
            scplt.plot_cv(
                ax, pdata, classes=["cellline", "condition"],
                annotate={"AS_kd": "replicate set A"},
            )
            plt.show()
            ```

        ![Plot cv custom annotate](../../assets/plots/plot_cv_custom_annotate.png)

        Export the underlying table (``CV`` ratio and ``CV_pct`` percent columns):
            ```python
            cv_df = scplt.plot_cv(
                None, pdata, classes=["cellline", "condition"], return_df=True
            )
            ```
    """
    if annotate is not None and annotate not in ("median", "mean") and not isinstance(annotate, dict):
        raise ValueError("annotate must be 'median', 'mean', a dict keyed by class, or None.")

    # Compute CVs for the selected layer
    pdata.cv(classes=classes, on=on, layer=layer)
    adata = utils.get_adata(pdata, on)
    classes_list = utils.get_classlist(adata, classes=classes, order=order)

    ex_cols = [col for col in extra_cols if col in adata.var.columns]

    cv_data = []
    for class_value in classes_list:
        cv_col = f"CV: {class_value}"
        if cv_col in adata.var.columns:
            cv_values = adata.var[cv_col].values
            row = {"Class": class_value, "CV": cv_values}
            for col in ex_cols:
                row[col] = adata.var[col].values
            cv_data.append(pd.DataFrame(row))

    if not cv_data:
        print(f"{utils.format_log_prefix('warn')} No valid CV subsets found — skipping plot.")
        return ax if ax is not None else None

    cv_df = pd.concat(cv_data, ignore_index=True)
    cv_df["CV_pct"] = cv_df["CV"] * 100

    if return_df:
        return cv_df

    if palette is None:
        palette = get_color("palette")

    # Ensure consistent class ordering
    if order is not None:
        cat_type = pd.api.types.CategoricalDtype(order, ordered=True)
        cv_df["Class"] = cv_df["Class"].astype(cat_type)
    else:
        cv_df["Class"] = pd.Categorical(
            cv_df["Class"],
            categories=sorted(cv_df["Class"].unique()),
            ordered=True,
        )

    plot_df = cv_df.copy()
    plot_df.loc[~np.isfinite(plot_df["CV_pct"]), "CV_pct"] = np.nan

    violin_kwargs = dict(inner="box", linewidth=1, cut=0, alpha=0.6, density_norm="width")
    violin_kwargs.update(kwargs)

    sns.violinplot(x="Class", y="CV_pct", data=plot_df, ax=ax, palette=palette, **violin_kwargs)

    ax.set_xlabel("")
    ax.set_ylabel("CV (%)")

    if show_n or annotate is not None:
        n_defaults = dict(fontsize=7, color="black", ha="center", va="top", zorder=10, offset=0.12)
        if n_kwargs is not None:
            n_defaults.update(n_kwargs)

        annotate_defaults = dict(fontsize=7, color="black", ha="center", va="bottom", zorder=10, offset=1.03)
        if annotate_kwargs is not None:
            annotate_defaults.update(annotate_kwargs)

        categories = list(cv_df["Class"].cat.categories)
        for x_center, class_value in enumerate(categories):
            sub = plot_df.loc[plot_df["Class"] == class_value, "CV_pct"]
            sub_finite = sub[np.isfinite(sub)]

            annotate_text = None
            if annotate == "median":
                val = np.nanmedian(sub_finite.to_numpy()) if len(sub_finite) else np.nan
                if np.isfinite(val):
                    annotate_text = f"median\n{val:.1f}%"
            elif annotate == "mean":
                val = np.nanmean(sub_finite.to_numpy()) if len(sub_finite) else np.nan
                if np.isfinite(val):
                    annotate_text = f"mean\n{val:.1f}%"
            elif isinstance(annotate, dict) and class_value in annotate:
                annotate_text = annotate[class_value]

            if annotate_text is not None:
                ax.text(
                    x_center,
                    annotate_defaults["offset"],
                    annotate_text,
                    transform=ax.get_xaxis_transform(),
                    fontsize=annotate_defaults["fontsize"],
                    color=annotate_defaults["color"],
                    ha=annotate_defaults["ha"],
                    va=annotate_defaults["va"],
                    zorder=annotate_defaults["zorder"],
                    clip_on=False,
                )

            if show_n:
                filtered = utils.resolve_class_filter(adata, classes, class_value)
                ax.text(
                    x_center,
                    -n_defaults["offset"],
                    f"n={filtered.n_obs}",
                    transform=ax.get_xaxis_transform(),
                    fontsize=n_defaults["fontsize"],
                    color=n_defaults["color"],
                    ha=n_defaults["ha"],
                    va=n_defaults["va"],
                    zorder=n_defaults["zorder"],
                    clip_on=False,
                )

    return ax

def plot_abundance_housekeeping(ax: "plt.Axes", pdata: pAnnData, classes: str | list[str] | None = None, loading_control: str = "all", **kwargs: Any) -> Any:
    """
    Plot abundance of housekeeping proteins.

    This function visualizes the abundance of canonical housekeeping proteins
    as loading controls, grouped by sample-level metadata if specified.
    Different sets of proteins are supported depending on the chosen loading
    control type.

    Args:
        ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes): Axis or list of axes to plot on.
            If `loading_control='all'`, must provide a list of 3 axes.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str, optional): One or more `.obs` columns to use for grouping samples.
        loading_control (str): Type of housekeeping controls to plot. Options:

            - `'whole cell'`: GAPDH, TBCD (β-tubulin), ACTB (β-actin), VCL (vinculin), TBP (TATA-binding protein)
            
            - `'nuclear'`: COX (cytochrome c oxidase), LMNB1 (lamin B1), PCNA (proliferating cell nuclear antigen), HDAC1 (histone deacetylase 1)
            
            - `'mitochondrial'`: VDAC1 (voltage-dependent anion channel 1)
            
            - `'all'`: plots all three categories across separate subplots.

        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes):
            Axis or list of axes with the plotted protein abundances.
    Note:
        This function assumes that the specified housekeeping proteins are annotated in `.prot.var['Genes']`. Missing proteins will be skipped during plotting and may result in empty or partially filled plots.
            
    !!! example
        Plot housekeeping protein abundance for whole cell controls:
            ```python
            from scpviz import plotting as scplt
            fig, ax = plt.subplots(figsize=(6,4))
            scplt.plot_abundance_housekeeping(ax, pdata, loading_control='whole cell', classes='condition')
            ```

        ![Plot abundance housekeeping](../../assets/plots/plot_abundance_housekeeping.png)
    """

    loading_controls = {
        'whole cell': ['GAPDH', 'TBCD', 'ACTB', 'VCL', 'TBP'],
        'nuclear': ['COX', 'LMNB1', 'PCNA', 'HDAC1'],
        'mitochondrial': ['VDAC1'],
        'all': ['GAPDH', 'TBCD', 'ACTB', 'VCL', 'TBP', 'COX', 'LMNB1', 'PCNA', 'HDAC1', 'VDAC1']
    }

    # Check validity
    if loading_control not in loading_controls:
        raise ValueError(f"❌ Invalid loading control type: {loading_control}")

    # Plot all categories as subplots
    if loading_control == 'all':
        # Create 1x3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        groups = ['whole cell', 'nuclear', 'mitochondrial']
        for ax_sub, group in zip(axes, groups):
            palette = get_color('colors', n=len(loading_controls[group]))
            plot_abundance(ax_sub, pdata, namelist=loading_controls[group], classes=classes, layer='X', palette=palette, **kwargs)
            ax_sub.set_title(group.title())
        fig.suptitle("Housekeeping Protein Abundance", fontsize=14)
        return fig, axes
    else:
        palette = get_color('colors', n=len(loading_controls[loading_control]))
        plot_abundance(ax, pdata, namelist=loading_controls[loading_control], classes=classes, layer='X', palette=palette, **kwargs)
        ax.set_title(loading_control.title())

def plot_abundance(ax: "plt.Axes | None", pdata: pAnnData, namelist: list[str] | None = None, layer: str = "X", on: str = "protein",
                   classes=None, return_df=False, order=None, palette=None,
                   log=False, facet=None, height=4, aspect=0.5,
                   plot_points=True, x_label='gene', kind='auto', **kwargs: Any):
    """
    Plot abundance of proteins or peptides across samples.

    This function visualizes expression values for selected proteins or peptides
    using violin + box + strip plots, or bar plots when the number of replicates
    per group is small. Supports grouping, faceting, and custom ordering.

    **Important default behavior:**
    - Abundances are **not log-transformed** by default (`log=False`)
    - The plotted abundance values remain **raw**
    - The **y-axis is transformed to log10 scale**, so the plot displays
      log10(abundance) even when raw abundances are used.    

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on. Ignored if `facet` is used.
        pdata (pAnnData): Input pAnnData object.
        namelist (list of str, optional): List of accessions or gene names to plot.
            If None, all available features are considered.
        layer (str): Data layer to use for abundance values. Default is `'X'`.
        on (str): Data level to plot, either `'protein'` or `'peptide'`.
        classes (str or list of str, optional): `.obs` column(s) to use for grouping
            samples. Determines coloring and grouping structure.
        return_df (bool): If True, returns the DataFrame of replicate and summary values.
        order (dict or list, optional): Custom order of classes. For dictionary input,
            keys are class names and values are the ordered categories.  
            Example: `order = {"condition": ["sc", "kd"]}`.
        palette (list or dict, optional): Color palette mapping groups to colors.
        log (bool): If True, apply log2 transformation to abundance values. Default is False (raw values used; y-axis log10-scaled instead).
        facet (str, optional): `.obs` column to facet by, creating multiple subplots.
        height (float): Height of each facet plot. Default is 4.
        aspect (float): Aspect ratio of each facet plot. Default is 0.5.
        plot_points (bool): Whether to overlay stripplot of individual samples.
        x_label (str): Label for the x-axis, either `'gene'` or `'accession'`.
        kind (str): Type of plot. Options:

            - `'auto'`: Default; uses barplot if groups have ≤ 3 samples, otherwise violin.
            - `'violin'`: Always use violin + box + strip.
            - `'bar'`: Always use barplot.

        **kwargs (Any): Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes or seaborn.FacetGrid): The axis or facet grid containing the plot.
        df (pandas.DataFrame, optional): Returned if `return_df=True`.

    !!! example
        Plot abundance of selected marker proteins grouped by cell line and condition:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            scplt.plot_abundance(
                ax, pdata, namelist=["GAPDH", "TUBB", "ACTB"], classes=["cellline", "condition"]
            )
            plt.show()
            ```

        ![Plot abundance](../../assets/plots/plot_abundance.png)
    """

    # Get abundance DataFrame
    df = utils.get_abundance(
        pdata, namelist=namelist, layer=layer, on=on,
        classes=classes, log=log, x_label=x_label
    )

    # custom class ordering
    if classes is not None and order is not None:
        unused = set(order) - (set([classes]) if isinstance(classes, str) else set(classes))
        if unused:
            print(f"⚠️ Unused keys in `order`: {unused} (not in `classes`)")
            
        if isinstance(classes, str):
            if classes in order:
                cat_type = pd.api.types.CategoricalDtype(order[classes], ordered=True)
                df['class'] = df['class'].astype(cat_type)
        else:
            for cls in classes:
                if cls in order and cls in df.columns:
                    cat_type = pd.api.types.CategoricalDtype(order[cls], ordered=True)
                    df[cls] = df[cls].astype(cat_type)

    # sort the dataframe so group order is preserved in plotting
    if classes is not None:
        sort_cols = ['x_label_name']
        if isinstance(classes, str):
            sort_cols.append('class')
        else:
            sort_cols.extend(classes)
        df = df.sort_values(by=sort_cols)

    # Facet handling
    df['facet'] = df[facet] if facet else 'all'

    if facet and classes and facet == classes:
        raise ValueError("`facet` and `classes` must be different.")

    if return_df:
        return df

    if palette is None:
        palette = get_color('palette')

    x_col = 'x_label_name'
    y_col = 'log2_abundance' if log else 'abundance'
    df = df.dropna(subset=[y_col])

    if kind == 'auto':
        sample_counts = df.groupby([x_col, 'class', 'facet'], observed=False).size()
        min_count = sample_counts.min() if len(sample_counts) else np.inf
        kind = 'bar' if min_count <= 3 else 'violin'

    def _plot_bar(df):
        bar_kwargs = dict(
            ci='sd',
            capsize=0.2,
            errwidth=1.5,
            palette=palette
        )
        bar_kwargs.update(kwargs)
        if facet and df['facet'].nunique() > 1:
            plot_df = df[[x_col, y_col, 'class', 'facet']]
            g = sns.FacetGrid(plot_df, col='facet', height=height, aspect=aspect, sharey=True, dropna=False, legend_out=log)
            g.map_dataframe(sns.barplot, x=x_col, y=y_col, hue='class', **bar_kwargs)
            g.set_axis_labels("Gene" if x_label == 'gene' else "Accession", "log2(Abundance)" if log else "Abundance")
            g.set_titles("{col_name}")
            g.add_legend(title='Class', frameon=True)
            if not log:
                for ax_ in g.axes.flatten():
                    ax_.set_yscale("log")
            return g
        else:
            if ax is None:
                fig, _ax = plt.subplots(figsize=(6, 4))
            else:
                _ax = ax

            sns.barplot(data=df, x=x_col, y=y_col, hue='class', ax=_ax, **bar_kwargs)
            _ax.set_yscale("log") if not log else None

            # deduplicate legend
            handles, labels = _ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            _ax.legend(by_label.values(), by_label.keys(), title='Class', frameon=True)
            _ax.set_ylabel("log2(Abundance)" if log else "Abundance")
            _ax.set_xlabel("Gene" if x_label == 'gene' else "Accession")

            return _ax

    def _plot_violin(df):
        violin_kwargs = dict(inner="box", linewidth=1, cut=0, alpha=0.5, density_norm="width")
        violin_kwargs.update(kwargs)
        if facet and df['facet'].nunique() > 1:
            plot_df = df[[x_col, y_col, 'class', 'facet']]
            g = sns.FacetGrid(plot_df, col='facet', height=height, aspect=aspect, sharey=True, dropna=False, legend_out=log)
            g.map_dataframe(sns.violinplot, x=x_col, y=y_col, hue='class', palette=palette, **violin_kwargs)
            if plot_points:
                def _strip(data, color, **kwargs_inner):
                    sns.stripplot(data=data, x=x_col, y=y_col, hue='class', dodge=True, jitter=True,
                                  color='black', size=3, alpha=0.5, legend=False, **kwargs_inner)
                g.map_dataframe(_strip)
            g.set_axis_labels("Gene" if x_label == 'gene' else "Accession", "log2(Abundance)" if log else "Abundance")
            g.set_titles("{col_name}")
            g.add_legend(title='Class', frameon=True)
            if not log:
                for ax_ in g.axes.flatten():
                    ax_.set_yscale("log")
            return g
        else:
            if ax is None:
                fig, _ax = plt.subplots(figsize=(6, 4))
            else:
                _ax = ax
            sns.violinplot(data=df, x=x_col, y=y_col, hue='class', palette=palette, ax=_ax, **violin_kwargs)
            if plot_points:
                sns.stripplot(data=df, x=x_col, y=y_col, hue='class', dodge=True, jitter=True,
                              color='black', size=3, alpha=0.5, legend=False, ax=_ax)
            handles, labels = _ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            _ax.legend(by_label.values(), by_label.keys(), title='Class', frameon=True)
            _ax.set_ylabel("log2(Abundance)" if log else "Abundance")
            _ax.set_xlabel("Gene" if x_label == 'gene' else "Accession")
            _ax.set_yscale("log") if not log else None
            return _ax

    return _plot_bar(df) if kind == 'bar' else _plot_violin(df)


_SIG_KW_DEFAULTS = {
    "sig_test": "ttest",
    "sig_equal_var": True,
    "spacing_frac": 0.08,
    "h_frac": 0.03,
    "base_offset_frac": 0.05,
}

_ND_KW_DEFAULTS = {
    "nd_label": "ND",
    "color": "#888888",
    "fontsize": 7,
    "y_axes_offset": 0.06,
    "y_log10_offset": 0.3,
    "zorder": 10,
}

_SIG_KW_LAYOUT_KEYS = frozenset(
    {"sig_test", "sig_equal_var", "spacing_frac", "h_frac", "base_offset_frac"}
)


def _resolve_sig_group_label(
    spec: Any,
    group_col: str,
    classes_original: str | list[str] | tuple[str, ...] | None,
) -> str:
    """Map volcano/de-style group spec to the plotted ``class`` label string."""
    if isinstance(classes_original, (list, tuple)) or group_col == "class":
        if isinstance(spec, dict):
            return "_".join(str(v) for v in spec.values())
        if isinstance(spec, (list, tuple)):
            if isinstance(classes_original, (list, tuple)) and len(spec) != len(classes_original):
                raise ValueError(
                    f"Group spec {spec!r} length must match `classes` {list(classes_original)!r}."
                )
            return "_".join(str(v) for v in spec)
        if isinstance(spec, str):
            return spec
        raise TypeError(
            f"Group spec must be dict, list, or str for composite classes; got {type(spec).__name__}."
        )

    if isinstance(spec, dict):
        if group_col in spec:
            return str(spec[group_col])
        if len(spec) == 1:
            return str(next(iter(spec.values())))
        raise ValueError(
            f"Group dict {spec!r} must include column {group_col!r} when `classes` is a single column."
        )
    return str(spec)


def annotate_abundance_boxgrid_significance(
    panel_info: list[dict[str, Any]],
    sig_pairs: list[tuple[Any, Any]] | bool,
    *,
    classes: str,
    classes_original: str | list[str] | tuple[str, ...] | None,
    sig_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Add pairwise significance brackets to ``plot_abundance_boxgrid`` panels.

    Called internally by :func:`plot_abundance_boxgrid` when ``sig_pairs`` is set,
    or directly on ``panel_info`` returned from a prior plotting pass.

    Args:
        panel_info: Per-gene dicts with keys ``gene``, ``ax``, ``sub``, ``unique_classes``,
            ``x_centers``, and ``nd_groups``.
        sig_pairs: ``True`` for an automatic two-group comparison on each panel, or a list
            of ``(group1, group2)`` specs in volcano / ``de()`` format (dict, list, or str).
        classes: Column in each ``sub`` DataFrame used for grouping (often ``"class"``).
        classes_original: Original ``classes`` argument passed to boxgrid (for label resolution).
        sig_kwargs: Optional overrides merged onto defaults. Known keys:

            - ``sig_test``: ``"ttest"``, ``"mannwhitneyu"``, or ``"wilcoxon"`` (default ``"ttest"``).
            - ``sig_equal_var``: Student vs Welch t-test when ``sig_test="ttest"`` (default ``True``).
            - ``spacing_frac``, ``h_frac``, ``base_offset_frac``: bracket layout (axis fractions).

            Remaining keys are forwarded to :func:`plot_significance` (e.g. ``col``, ``fontsize``).
            See :func:`plot_significance` for full drawing options.

    Returns:
        stats_df: One row per gene × comparison with test results and draw status.
    """
    merged_sig = dict(_SIG_KW_DEFAULTS)
    if sig_kwargs:
        merged_sig.update(sig_kwargs)
    layout = {k: merged_sig.pop(k) for k in _SIG_KW_LAYOUT_KEYS if k in merged_sig}
    plot_sig_kwargs = merged_sig

    method = layout.get("sig_test", "ttest")
    equal_var = layout.get("sig_equal_var", True)
    spacing_frac = layout.get("spacing_frac", 0.08)
    h_frac = layout.get("h_frac", 0.03)
    base_offset_frac = layout.get("base_offset_frac", 0.05)

    if method not in {"ttest", "mannwhitneyu", "wilcoxon"}:
        raise ValueError(f"Unsupported sig_test {method!r}.")

    wilcoxon_warned = False
    rows: list[dict[str, Any]] = []

    for panel in panel_info:
        gene = panel["gene"]
        ax = panel["ax"]
        sub = panel["sub"]
        unique_classes = panel["unique_classes"]
        x_centers = panel["x_centers"]
        nd_groups = panel["nd_groups"]
        class_to_x = dict(zip(unique_classes, x_centers))

        if sig_pairs is True:
            if len(unique_classes) != 2:
                raise ValueError(
                    f"sig_pairs=True requires exactly two groups on panel {gene!r}; "
                    f"found {len(unique_classes)}: {unique_classes!r}."
                )
            pair_list = [(unique_classes[0], unique_classes[1])]
        else:
            pair_list = sig_pairs

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax > ymin else 1.0
        h = h_frac * y_range

        bracket_level = 0
        for g1_spec, g2_spec in pair_list:
            label1 = _resolve_sig_group_label(g1_spec, classes, classes_original)
            label2 = _resolve_sig_group_label(g2_spec, classes, classes_original)

            base_row = {
                "gene": gene,
                "group1": label1,
                "group2": label2,
                "group1_spec": g1_spec,
                "group2_spec": g2_spec,
                "method": method,
            }

            if label1 in nd_groups or label2 in nd_groups:
                nd_names = [lbl for lbl in (label1, label2) if lbl in nd_groups]
                print(
                    f"{format_log_prefix('warn')} {gene}: skipping comparison "
                    f"{label1!r} vs {label2!r} — ND group(s): {nd_names!r}."
                )
                rows.append(
                    {
                        **base_row,
                        "n1": np.nan,
                        "n2": np.nan,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "label": "",
                        "status": "skipped_nd",
                        "reason": f"ND group(s): {nd_names}",
                    }
                )
                continue

            x1 = class_to_x.get(label1)
            x2 = class_to_x.get(label2)
            if x1 is None or x2 is None or np.isnan(x1) or np.isnan(x2):
                missing = [lbl for lbl, x in ((label1, x1), (label2, x2)) if x is None or (x is not None and np.isnan(x))]
                print(
                    f"{format_log_prefix('warn')} {gene}: skipping comparison "
                    f"{label1!r} vs {label2!r} — group(s) not on axis: {missing!r}."
                )
                rows.append(
                    {
                        **base_row,
                        "n1": np.nan,
                        "n2": np.nan,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "label": "",
                        "status": "skipped_pair",
                        "reason": f"Missing x position for {missing!r}",
                    }
                )
                continue

            x1_vals = sub.loc[sub[classes] == label1, "abundance"].to_numpy(dtype=float)
            x1_vals = x1_vals[np.isfinite(x1_vals) & (x1_vals > 0)]
            x2_vals = sub.loc[sub[classes] == label2, "abundance"].to_numpy(dtype=float)
            x2_vals = x2_vals[np.isfinite(x2_vals) & (x2_vals > 0)]
            n1, n2 = int(x1_vals.size), int(x2_vals.size)

            if n1 < 2 or n2 < 2:
                print(
                    f"{format_log_prefix('warn')} {gene}: skipping comparison "
                    f"{label1!r} vs {label2!r} — need ≥2 valid replicates per group "
                    f"(n1={n1}, n2={n2})."
                )
                rows.append(
                    {
                        **base_row,
                        "n1": n1,
                        "n2": n2,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "label": "",
                        "status": "skipped_n",
                        "reason": f"n1={n1}, n2={n2}",
                    }
                )
                continue

            if method == "wilcoxon" and not wilcoxon_warned:
                warnings.warn(
                    "wilcoxon is a paired test; ensure samples are matched before interpreting "
                    "boxgrid significance brackets.",
                    UserWarning,
                    stacklevel=2,
                )
                wilcoxon_warned = True

            try:
                if method == "ttest":
                    res = ttest_ind(x1_vals, x2_vals, equal_var=equal_var, nan_policy="omit")
                elif method == "mannwhitneyu":
                    res = mannwhitneyu(x1_vals, x2_vals, alternative="two-sided")
                else:
                    res = wilcoxon(x1_vals, x2_vals)
                statistic = float(res.statistic)
                p_value = float(res.pvalue)
            except Exception as exc:
                print(
                    f"{format_log_prefix('warn')} {gene}: test failed for "
                    f"{label1!r} vs {label2!r}: {exc}"
                )
                rows.append(
                    {
                        **base_row,
                        "n1": n1,
                        "n2": n2,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "label": "",
                        "status": "skipped_n",
                        "reason": str(exc),
                    }
                )
                continue

            anchor_vals = []
            for lbl in (label1, label2):
                dat = sub.loc[sub[classes] == lbl, "plot_abundance"].to_numpy(dtype=float)
                if dat.size:
                    anchor_vals.append(float(np.nanmax(dat)))
                    anchor_vals.append(float(np.nanpercentile(dat, 75)))
            data_top = max(anchor_vals) if anchor_vals else ymin
            y = data_top + base_offset_frac * y_range + bracket_level * spacing_frac * y_range

            plot_significance(ax, y, h, x1=float(x1), x2=float(x2), pval=p_value, **plot_sig_kwargs)
            label = "n.s." if p_value > 0.05 else "*" * int(np.floor(-np.log10(p_value)))
            rows.append(
                {
                    **base_row,
                    "n1": n1,
                    "n2": n2,
                    "statistic": statistic,
                    "p_value": p_value,
                    "label": label,
                    "status": "ok",
                    "reason": "",
                }
            )
            bracket_level += 1

        if bracket_level > 0:
            cur_ymin, cur_ymax = ax.get_ylim()
            ax.set_ylim(cur_ymin, cur_ymax + spacing_frac * y_range)

    return pd.DataFrame(rows)


def plot_abundance_boxgrid(pdata: pAnnData, namelist: list[str] | None = None, ax: Any = None, layer: str = "X", on: str = "protein", classes: str | list[str] | None = None, return_df: bool = False,
    order=None, plot_type="box", log_scale=False, figsize=(2,2), palette=None, y_min=None, y_max=None, label_x=True, show_n=False,
    global_legend=True, box_kwargs=None, hline_kwargs=None, bar_kwargs=None, bar_error='sd', violin_kwargs=None, text_kwargs=None, strip_kwargs=None,
    sig_pairs: list[tuple[Any, Any]] | bool | None = None, sig_kwargs: dict[str, Any] | None = None, nd_kwargs: dict[str, Any] | None = None):
    """
    Plot abundance values in a one-row panel of boxplots, mean-lines, bars, or violins.

    This function generates a clean horizontal panel, with one subplot per gene,
    using ``plot_type`` to select boxplots (default), mean-lines, bar plots, or
    violin plots. If ``log_scale=True``, abundance values are visualized in
    log10 units (with zero or negative values clipped to 0 before transformation).
    The layout is optimized for compact manuscript figure panels and supports
    custom global legends, count annotations, and flexible formatting via keyword
    dictionaries.

    Args:
        pdata (pAnnData): Input pAnnData object.
        namelist (list of str, optional): List of accessions or gene names to plot.
            If None, all available features are considered.
        ax (matplotlib.axes.Axes): Axis to plot on. Generates a new axis if None.
        layer (str): Data layer to use for abundance values. Default is `'X'`.
        on (str): Data level to plot, either `'protein'` or `'peptide'`.
        return_df (bool): If True, returns the DataFrame of replicate and summary values.
        order (list of str): Ordered list to plot by. If None, plots by given dataframe order.
        classes (str): Column in `.obs` to use for grouping samples (default: None).
        plot_type (str): Type of plot, select from one of {"box", "line", "bar", "violin"}.
            Defaults to "box".
        log_scale (bool): If True, plot log10-transformed abundances on a linear axis.
            If False (default), plot raw abundance values on a linear axis.
        figsize (tuple): Figure size as (width, height) in inches.
        palette (dict or list, optional): Color palette for grouping categories.
            Defaults to ``scplt.get_color("colors", n_classes)``.
        y_min (float or None): Lower y-axis limit in plotting units. If ``log_scale=True``,
            this is in log10 units (e.g., 2 → 10²). If ``log_scale=False``, this is in
            raw abundance units. If None, inferred.
        y_max (float or None): Upper y-axis limit in plotting units. If ``log_scale=True``,
            this is in log10 units (e.g., 6 → 10⁶). If ``log_scale=False``, this is in
            raw abundance units. If None, inferred.
        label_x (bool): Whether to display x tick labels inside each subplot.
        show_n (bool): Whether to annotate each subplot with sample counts.
        global_legend (bool): Whether to display a single global legend.
        box_kwargs (dict, optional): Additional arguments passed to ``sns.boxplot``
            (used when ``plot_type="box"``).
        hline_kwargs (dict, optional): Styling for mean segments when ``plot_type="line"``.
            Recognized keys include Matplotlib ``hlines`` options plus ``half_width``
            (float, default 0.15): half the segment length in x-axis units; use a
            smaller value when dodged groups would otherwise overlap.
        bar_kwargs (dict, optional): Passed to ``Axes.bar`` when ``plot_type="bar"``
            (e.g. ``width`` in x-axis units; default here is 0.3—decrease when many
            hue levels overlap on one gene tick).
        bar_error (str, optional): Error bar for bar plot. Select from one of
            {"sd", "sem", None, <callable>}, where callable takes a 1D array and returns
            a scalar error. Defaults to "sd".
        violin_kwargs (dict, optional): Additional arguments passed to ``sns.violinplot``
            (used when ``plot_type="violin"``).
        text_kwargs (dict, optional): Keyword arguments for count labels
            (e.g., fontsize, offset).
        strip_kwargs (dict, optional): Keyword arguments for strip (raw points),
            e.g. ``{"darken_factor": 0.65}``.
        sig_pairs (list, bool, or None): Pairwise comparisons for significance brackets.
            ``None`` (default) disables testing. ``True`` auto-compares the two hue groups
            when exactly two are present. Otherwise pass a list of ``(group1, group2)`` specs
            in the same dict/list/str format as :func:`plot_volcano` / ``de()`` values.
        sig_kwargs (dict, optional): Significance options merged onto defaults
            ``{"sig_test": "ttest", "sig_equal_var": True}``. Layout keys
            ``spacing_frac``, ``h_frac``, and ``base_offset_frac`` are consumed locally;
            remaining keys (e.g. ``col``, ``fontsize``, ``h``) are passed to
            :func:`plot_significance`.
        nd_kwargs (dict, optional): Not-detected annotation options merged onto defaults
            ``{"nd_label": "ND", "color": "#888888", "fontsize": 7, "y_axes_offset": 0.06,
            "y_log10_offset": 0.3}``. On linear scales, ``y_axes_offset`` is the vertical
            offset in axes coordinates (blended transform). On log-scale panels,
            ``y_log10_offset`` is added above the axis minimum in log10 data units.
            Shown when a group has no valid (non-zero) abundances for a gene; plots are unchanged.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        axes (list of matplotlib.axes.Axes): One axis per gene.
        df (pandas.DataFrame, optional): Returned if ``return_df=True``.
        stats_df (pandas.DataFrame, optional): Returned if ``return_df=True`` and
            ``sig_pairs`` is set; one row per gene × comparison.

    !!! note
        Default customizations for keyword dictionaries:

        Boxplot styling (used when ``plot_type="box"``):
        ```python
        box_kwargs = {
            "showcaps": False,
            "whiskerprops": {"visible": False},
            "showfliers": False,
            "boxprops": {"alpha": 0.6, "linewidth": 1},
            "linewidth": 1,
            "dodge": True,
        }
        ```

        Mean-line styling (used when ``plot_type="line"``):
        ```python
        hline_kwargs = {
            "color": "k",
            "linewidth": 2.0,
            "zorder": 5,
            "half_width": 0.15,
        }
        ```
        ``half_width`` is in x-axis units; lower it when several classes are dodged
        and mean segments would cross.

        Bar styling (used when ``plot_type="bar"``):
        ```python
        bar_kwargs = {
            "alpha": 0.8,
            "edgecolor": "black",
            "linewidth": 0.6,
            "width": 0.3,
            "capsize": 2,
            "zorder": 3,
        }
        ```
        ``width`` is passed to ``Axes.bar`` (x-axis units); use a smaller value when
        bars from neighboring hue levels overlap.

        Violin styling (used when ``plot_type="violin"``):
        ```python
        violin_kwargs = {
            "inner": "quartile",
            "dodge": True,
            "zorder": 5,
        }
        ```

        Strip styling (raw points; used for all plot types):
        ```python
        strip_kwargs = {
            "jitter": True,
            "alpha": 0.4,
            "size": 3,
            "zorder": 7,
            "darken_factor": 0.65,
        }
        ```

        Text annotation styling (used when ``show_n=True``):
        ```python
        text_kwargs = {
            "fontsize": 7,
            "color": "black",
            "ha": "center",
            "va": "bottom",
            "zorder": 10,
            "offset": 0.1,
        }
        ```

    !!! example
        Basic usage (grouped boxplots):
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            plot_type="box",
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        ![Plot abundance boxgrid](../../assets/plots/plot_abundance_boxgrid.png)

        Bar plots with error bars:
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            plot_type="bar",
            bar_error="sd",  # "sd", "sem", None, or callable
            bar_kwargs={"width": 0.14},  # narrower bars when many groups dodge
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        ![Plot abundance boxgrid bar](../../assets/plots/plot_abundance_boxgrid_bar.png)

        Mean-lines with count annotations:
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            plot_type="line",
            show_n=True,
            hline_kwargs={"half_width": 0.08},  # shorter segments when groups dodge
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        ![Plot abundance boxgrid line](../../assets/plots/plot_abundance_boxgrid_line.png)

        Violin plots (distribution-focused):
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            plot_type="violin",
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        ![Plot abundance boxgrid violin](../../assets/plots/plot_abundance_boxgrid_violin.png)

        Customizing appearance (palette, order, and styling):
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            plot_type="box",
            box_kwargs={"boxprops": {"alpha": 0.45}, "linewidth": 1.2},
            strip_kwargs={"size": 4, "alpha": 0.6},
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        ![Plot abundance boxgrid custom](../../assets/plots/plot_abundance_boxgrid_custom.png)

        Return the plotting DataFrame for downstream checks:
        ```python
        fig, axes, df = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            plot_type="box",
            return_df=True,
        )

        display(df.head())
        plt.show()
        ```

        ![Plot abundance boxgrid](../../assets/plots/plot_abundance_boxgrid.png)

        Significance brackets (explicit pairs, volcano-style dicts):
        ```python
        fig, axes, df, stats = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            sig_pairs=[
                ({"cellline": "BE", "condition": "sc"}, {"cellline": "BE", "condition": "kd"}),
                ({"cellline": "AS", "condition": "sc"}, {"cellline": "AS", "condition": "kd"}),
            ],
            sig_kwargs={"fontsize": 8},
            return_df=True,
        )
        plt.show()
        ```

        ![Plot abundance boxgrid significance](../../assets/plots/plot_abundance_boxgrid_significance.png)

        Multiple comparisons with a shared group (same group may appear in more than one pair):
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH", "TUBB", "ACTB"],
            classes=["cellline", "condition"],
            sig_pairs=[
                ({"cellline": "BE", "condition": "sc"}, {"cellline": "BE", "condition": "kd"}),
                ({"cellline": "BE", "condition": "kd"}, {"cellline": "AS", "condition": "kd"}),
            ],
            sig_kwargs={"fontsize": 8},
        )
        plt.show()
        ```

        ![Plot abundance boxgrid significance multi](../../assets/plots/plot_abundance_boxgrid_significance_multi.png)

        Two hue groups only — auto comparison:
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["GAPDH"],
            classes="treatment",
            sig_pairs=True,
        )
        plt.show()
        ```
    """
    from matplotlib.colors import to_rgba
    from matplotlib.transforms import blended_transform_factory

    if classes is None:
        df = pdata.get_abundance(
            namelist=namelist,
            on=on,
            layer=layer,
        )
    else:
        df = pdata.get_abundance(
            namelist=namelist,
            classes=classes,
            on=on,
            layer=layer,
        )

    df = df.copy()

    if sig_pairs is not None and classes is None:
        raise ValueError("`sig_pairs` requires sample grouping; pass `classes`.")

    merged_nd = dict(_ND_KW_DEFAULTS)
    if nd_kwargs:
        merged_nd.update(nd_kwargs)
    nd_label = merged_nd.pop("nd_label")

    # --- normalize classes (list/tuple -> df["class"]) ---
    classes_original = classes
    classes_label = classes  # keep original for legend title
    if isinstance(classes, (list, tuple)):
        if "class" not in df.columns:
            raise ValueError(
                "classes was a list/tuple, but get_abundance did not return a 'class' column."
            )
        classes = "class"
        classes_label = ", ".join(list(classes_label))
    elif isinstance(classes, str):
        if classes not in df.columns:
            raise ValueError(f"Column '{classes}' not found in abundance DataFrame.")
    elif classes is not None:
        raise TypeError("classes must be None, a string, or a list/tuple of strings.")

    # --- abundance transform ---
    if log_scale: # Create log10-transformed abundance, preserving zeros as 0
        df["plot_abundance"] = np.nan
        pos = df["abundance"] > 0
        df.loc[pos, "plot_abundance"] = np.log10(df.loc[pos, "abundance"])
        df.loc[~pos, "plot_abundance"] = 0.0
    else:
        df["plot_abundance"] = np.nan
        pos = df["abundance"] > 0
        df.loc[pos, "plot_abundance"] = df.loc[pos, "abundance"]
        df.loc[~pos, "plot_abundance"] = 0.0

    # Get gene list
    genes = df["gene"].unique()
    n = len(genes)

    # Determine unique_classes
    if classes is not None:
        unique_classes = list(df[classes].unique())  # DO NOT sort
    else:
        unique_classes = [None]  # placeholder for no grouping

    # Determine palette
    if classes is not None:
        n_classes = df[classes].nunique()
        if palette is None:
            palette = get_color("colors", n_classes)
    else:
        # no classes → everything is one group, no hue
        n_classes = 1
        if palette is None:
            palette = get_color("colors", 1)  # or any default single color

    # ---------- plot defaults ----------
    # setup kwargs defaults
    boxplot_defaults = dict(showcaps=False, whiskerprops={"visible": False}, showfliers=False, boxprops=dict(alpha=0.6, linewidth=1), linewidth=1, dodge=True)
    if box_kwargs is not None:
        boxplot_defaults.update(box_kwargs)
    if classes is None:
        boxplot_defaults["dodge"] = False

    hline_defaults = dict(color="k", linewidth=2.0, zorder=5, half_width=0.15)
    if hline_kwargs is not None:
        hline_defaults.update(hline_kwargs)

    bar_defaults = dict(alpha=0.8, edgecolor="black", linewidth=0.6, width=0.3, capsize=2, zorder=3)
    if bar_kwargs is not None:
        bar_defaults.update(bar_kwargs)

    violin_defaults = dict(inner="quartile", dodge = True, zorder=5)
    if violin_kwargs is not None:
        violin_defaults.update(violin_kwargs)
    if classes is None:
        violin_defaults["dodge"] = False

    text_defaults = dict(fontsize=7, color="black", ha="center", va="bottom", zorder=10,
        offset=0.1,             # vertical offset from anchor
    )
    if text_kwargs is not None:
        text_defaults.update(text_kwargs)

    strip_defaults = dict(x="gene", y="plot_abundance", jitter=True, alpha=0.4, size=3, legend=False, ax=ax, zorder=7, darken_factor=0.65)
    if plot_type in ("bar","violin"):
        strip_defaults["alpha"] = 0.6
    if strip_kwargs is not None:
        strip_defaults.update(strip_kwargs)

    def _get_err(vals, mode):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return np.nan
        if mode is None:
            return 0.0
        if callable(mode):
            return float(mode(vals))
        if mode == "sd":
            return float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        if mode == "sem":
            return float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        raise ValueError("bar_error must be 'sd', 'sem', None, or a callable")

    def _darken_color(color, factor=0.7):
        """
        Darken an RGB/hex color by multiplying RGB channels.
        factor < 1 darkens, factor > 1 lightens.
        """
        r, g, b, a = to_rgba(color)
        return (r * factor, g * factor, b * factor, a)

    # Create subplots
    fig_width = figsize[0]
    fig_height = figsize[1]

    if ax is None:
        fig, axes = plt.subplots(1, n, figsize=(fig_width * n, fig_height), sharey=True)
        if n == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]  # treat external ax as a single-panel layout

    panel_info: list[dict[str, Any]] = []

    for ax, gene in zip(axes, genes):
        sub = df[df["gene"] == gene]

        if classes is not None:
            if order is not None:
                # Use user-specified hue order, but only keep those present in this subset
                # unique_classes = [c for c in order if c in sub[classes].unique()]
                unique_classes = order
            else:
                unique_classes = list(sub[classes].unique())
        else:
            unique_classes = [None]     

        # Stripplot (raw points) on plot_abundance
        before_n = len(ax.collections)

        # Make per-panel strip kwargs (avoid cross-panel mutation)
        strip_kws = dict(strip_defaults)
        strip_kws["data"] = sub
        strip_kws["ax"] = ax 

        # pull darken_factor without deleting it from the shared defaults
        darken_factor = strip_kws.pop("darken_factor", 1)
        # Clear any prior hue/color/palette that may be present
        strip_kws.pop("hue", None)
        strip_kws.pop("hue_order", None)
        strip_kws.pop("palette", None)
        strip_kws.pop("color", None)

        if classes is None:
            # no hue, everything in one group
            strip_kws["color"] = "black"
            strip_kws["dodge"] = False
        else:
            strip_kws["hue"] = classes
            strip_kws["dodge"] = True
            strip_kws["hue_order"] = unique_classes
            if plot_type == "box":
                strip_kws["palette"] = ["black"] * len(unique_classes)  # keep hue/dodge, but all dots black
            else:
                if isinstance(palette, dict):
                    strip_kws["palette"] = {k: _darken_color(v, factor=darken_factor) for k, v in palette.items()}
                else:
                    strip_kws["palette"] = [_darken_color(c, factor=darken_factor) for c in palette]

        sns.stripplot(**strip_kws)

        after_n = len(ax.collections)
        strip_collections = ax.collections[before_n:after_n]

        x_centers = []
        if classes is not None:
            # one collection per hue level when dodge=True
            for coll in strip_collections:
                offs = coll.get_offsets()
                x_centers.append(np.nanmean(offs[:, 0]) if offs.size > 0 else np.nan)
        else:
            # ungrouped: there should be one collection
            if len(strip_collections) > 0:
                offs = strip_collections[0].get_offsets()
                x_centers = [np.nanmean(offs[:, 0]) if offs.size > 0 else np.nan]
            else:
                x_centers = []

        if plot_type == "box":
            # boxplot on plot abundance
            if classes is None:
                sns.boxplot(
                    data=sub, x="gene", y="plot_abundance",
                    color=palette[0], ax=ax, **boxplot_defaults,
                )
            else:
                sns.boxplot(
                    data=sub, x="gene", y="plot_abundance",
                    hue=classes, hue_order=unique_classes, palette=palette,
                    ax=ax, **boxplot_defaults,
                )

        elif plot_type == "line":
            if classes is None:
                # one mean over all non-zero abundances
                sub_pos = sub[sub["abundance"] > 0]
                mean_val = sub_pos["plot_abundance"].mean()
                x_center = x_centers[0]
                half_width = hline_defaults["half_width"]
                ax.hlines(
                    y=mean_val,
                    xmin=x_center - half_width, xmax=x_center + half_width,
                    color=hline_defaults["color"], linewidth=hline_defaults["linewidth"], zorder=hline_defaults["zorder"],
                )
            else:
                # compute means excluding zeros
                sub_pos = sub[sub["abundance"] > 0]
                group_means = (
                    sub_pos.groupby(classes)["plot_abundance"]
                    .mean()
                    .reindex(unique_classes)
                )

                for cls, x_center in zip(unique_classes, x_centers):
                    mean_val = group_means.loc[cls]
                    if np.isnan(mean_val):
                        continue
                    half_width = hline_defaults["half_width"]
                    ax.hlines(
                        y=mean_val,
                        xmin=x_center - half_width, xmax=x_center + half_width,
                        color=hline_defaults["color"], linewidth=hline_defaults["linewidth"], zorder=hline_defaults["zorder"],
                    )
        elif plot_type == "violin":
            if classes is None:
                sns.violinplot(
                    data=sub, x="gene", y="plot_abundance",
                    color=palette[0], ax=ax, **violin_defaults
                )
            else:
                sns.violinplot(
                    data=sub, x="gene", y="plot_abundance",
                    hue=classes, hue_order=unique_classes, palette=palette,
                    ax=ax, **violin_defaults
                )

        elif plot_type == "bar":
            if classes is None:
                sub_pos = sub # include 0s in calculation?
                vals = sub_pos["plot_abundance"].to_numpy()
                mean_val = np.nanmean(vals)
                err = _get_err(vals, bar_error)
                x_center = x_centers[0] if len(x_centers) else 0.0
                ax.bar(
                    [x_center], [mean_val],
                    color=palette[0], **bar_defaults
                )

                if bar_error is not None:
                    ax.errorbar([x_center],[mean_val],yerr=[err], fmt="none",ecolor='k', zorder=10, capsize=2)
            else:
                sub_pos = sub # include 0s in calculation?
                grp = sub_pos.groupby(classes)["plot_abundance"]
                means = grp.mean().reindex(unique_classes)
                errs = grp.apply(lambda v: _get_err(v.to_numpy(), bar_error)).reindex(unique_classes)

                colors = [palette[c] for c in unique_classes] if isinstance(palette, dict) else palette
                ax.bar(
                    x_centers, means.to_numpy(),
                    color=colors, **bar_defaults
                )

                if bar_error is not None:
                    ax.errorbar(x_centers, means.to_numpy(), yerr=errs.to_numpy(), fmt="none", ecolor='k',zorder=10, capsize=2)

        else:
            raise ValueError("plot_type must be one of: 'box', 'line', 'bar', 'violin'")

        # n = x annotation
        if show_n and classes is not None:
            # Count only non-zero abundances
            n_nonzero = (
                (sub["abundance"] > 0)
                .groupby(sub[classes])
                .sum()
                .reindex(unique_classes)
            )

            for cls, x_center in zip(unique_classes, x_centers):
                # choose y position
                if plot_type != "line":
                    # Q3 for this class
                    dat = sub.loc[sub[classes] == cls, "plot_abundance"]
                    anchor = np.nanpercentile(dat, 75)
                else:
                    # use mean line position
                    anchor = group_means.loc[cls]

                y_anchor = anchor + text_defaults["offset"]
                ax.text(
                    x_center,
                    y_anchor,
                    f"n={int(n_nonzero.loc[cls])}",
                    fontsize=text_defaults["fontsize"],
                    color=text_defaults["color"],
                    ha=text_defaults["ha"],
                    va=text_defaults["va"],
                    zorder=text_defaults["zorder"],
                )

        # Axis formatting (linear axis, log10 units)
        if (y_min is not None) or (y_max is not None):
            cur_ymin, cur_ymax = ax.get_ylim()
            ymin = y_min if y_min is not None else cur_ymin
            ymax = y_max if y_max is not None else cur_ymax
            ax.set_ylim(ymin, ymax)

        if log_scale:
            ymin, ymax = ax.get_ylim()
            ticks = np.arange(min(int(np.floor(ymin)), 0), int(np.ceil(ymax)) + 1)
            ax.set_yticks(ticks)
            ylabel = "log10(Abundance)"
        else:
            ylabel = "Abundance"

        if len(x_centers) == 0:
            # No dodge positions were created (e.g., only one class had data)
            # → Do NOT set xticks or xticklabels
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            if label_x:
                if classes is not None:
                    ax.set_xticks(x_centers)
                    ax.set_xticklabels(unique_classes, rotation=45, ha="right")
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.tick_params(axis="x", bottom=False)

        ax.set_xlabel("") 
        ax.set_ylabel(ylabel if ax == axes[0] else "")

        # Remove subplot legends
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        ax.set_title(gene, fontsize=10)

        nd_groups: set[str] = set()
        if classes is not None:
            for cls in unique_classes:
                cls_vals = sub.loc[sub[classes] == cls, "abundance"].to_numpy(dtype=float)
                n_valid = int(np.sum(np.isfinite(cls_vals) & (cls_vals > 0)))
                if n_valid == 0:
                    nd_groups.add(cls)
                    print(
                        f"{format_log_prefix('warn')} {gene}: group {cls!r} has no "
                        f"detectable abundance ({nd_label!r})."
                    )

            if nd_groups:
                nd_text_kwargs = dict(
                    color=merged_nd.get("color", "#888888"),
                    fontsize=merged_nd.get("fontsize", 7),
                    ha="center",
                    va="bottom",
                    zorder=merged_nd.get("zorder", 10),
                    clip_on=False,
                )
                if log_scale:
                    nd_ymin, nd_ymax = ax.get_ylim()
                    nd_range = nd_ymax - nd_ymin if nd_ymax > nd_ymin else 1.0
                    y_log10_offset = merged_nd.get("y_log10_offset", 0.3)
                    y_nd = nd_ymin + max(float(y_log10_offset), 0.05 * nd_range)
                    for cls, x_center in zip(unique_classes, x_centers):
                        if cls in nd_groups:
                            ax.text(x_center, y_nd, nd_label, **nd_text_kwargs)
                else:
                    nd_trans = blended_transform_factory(ax.transData, ax.transAxes)
                    y_axes_offset = merged_nd.get("y_axes_offset", 0.06)
                    for cls, x_center in zip(unique_classes, x_centers):
                        if cls in nd_groups:
                            ax.text(
                                x_center,
                                y_axes_offset,
                                nd_label,
                                transform=nd_trans,
                                **nd_text_kwargs,
                            )

        panel_info.append(
            {
                "gene": gene,
                "ax": ax,
                "sub": sub,
                "unique_classes": unique_classes,
                "x_centers": x_centers,
                "nd_groups": nd_groups,
            }
        )

    # global legend
    if global_legend and classes is not None:
        # Build custom legend handles from palette
        legend_classes = unique_classes

        if isinstance(palette, dict):
            colors = [palette[c] for c in legend_classes]
        else:
            # palette is a list in class order
            colors = palette

        handles = [
            plt.Line2D([0], [0], color=colors[i], lw=3, label=legend_classes[i])
            for i in range(len(legend_classes))
        ]

        fig.legend(
            handles,
            legend_classes,
            title=classes_label,
            frameon=True,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
        )

    plt.tight_layout()

    stats_df = None
    if sig_pairs is not None:
        stats_df = annotate_abundance_boxgrid_significance(
            panel_info,
            sig_pairs,
            classes=classes,
            classes_original=classes_original,
            sig_kwargs=sig_kwargs,
        )

    if return_df and sig_pairs is not None:
        return fig, axes, df, stats_df
    if return_df:
        return fig, axes, df
    return fig, axes
    
def plot_rankquant(ax: "plt.Axes", pdata: pAnnData, classes: str | list[str] | None = None, layer: str = "X", on: str = "protein", cmap: Any = ["Blues"], color: Any = ["blue"], order: Any = None, s: float = 20, alpha: float = 0.2, calpha: float = 1, exp_alpha: float = 70, debug: bool = False) -> Any:
    """
    Plot rank abundance distributions across samples or groups.

    This function visualizes rank abundance of proteins or peptides, optionally
    grouped by sample-level classes. Distributions are drawn as scatter plots
    with adjustable opacity and color schemes. Mean, standard deviation, and
    rank statistics are written to `.var` for downstream annotation.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str, optional): One or more `.obs` columns to
            group samples. If None, samples are combined into identifier classes.
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level to plot, either `"protein"` or `"peptide"`. Default is `"protein"`.
        cmap (str or list of str): Colormap(s) used for scatter distributions.
            Default is `["Blues"]`.
        color (list of str): List of colors used for scatter distributions.
            Defaults to `["blue"]`.
        order (list of str, optional): Custom order of class categories. If None,
            categories appear in data order.
        s (float): Marker size. Default is 20.
        alpha (float): Marker transparency for distributions. Default is 0.2.
        calpha (float): Marker transparency for class means. Default is 1.
        exp_alpha (float): Exponent for scaling probability density values by
            average abundance. Default is 70.
        debug (bool): If True, print debug information during computation.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the rank abundance plot.
    
    Example:
        Plot rank abundance grouped by cell line and condition:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            scplt.plot_rankquant(ax, pdata, classes=["cellline", "condition"])
            plt.show()
            ```

        ![Plot rankquant](../../assets/plots/plot_rankquant.png)

        Plot rank abundance on single-cell protein data (use the same ``classes`` you use for UMAP, e.g. ``region``):
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            scplt.plot_rankquant(ax, pdata_sc, classes=["region"])
            plt.show()
            ```

        ![Plot rankquant (single-cell)](../../assets/plots/plot_rankquant_sc.png)

    See Also:
        mark_rankquant: Highlight specific proteins or genes on a rank abundance plot.            
            
    """
    # all the plot_dfs should now be stored in pdata.var
    pdata.rank(classes, on, layer)

    adata = utils.get_adata(pdata, on)
    classes_list = utils.get_classlist(adata, classes = classes, order = order)

    # Ensure colormap and color list match number of classes
    cmap = cmap if cmap and len(cmap) == len(classes_list) else get_color('cmap', n=len(classes_list))
    color = color if color and len(color) == len(classes_list) else get_color('colors', n=len(classes_list))

    for j, class_value in enumerate(classes_list):
        if classes is None or isinstance(classes, (str, list)):
            values = class_value.split('_') if classes is not str else class_value
            rank_data = utils.filter(adata, classes, values, debug=False)

        plot_df = rank_data.to_df().transpose()
        plot_df['Average: '+class_value] = np.nanmean(rank_data.X.toarray(), axis=0)
        plot_df['Stdev: '+class_value] = np.nanstd(rank_data.X.toarray(), axis=0)
        plot_df.sort_values(by=['Average: '+class_value], ascending=False, inplace=True)
        plot_df['Rank: '+class_value] = np.where(plot_df['Average: '+class_value].isna(), np.nan, np.arange(1, len(plot_df) + 1))

        sorted_indices = plot_df.index
        plot_df = plot_df.loc[adata.var.index]
        adata.var['Average: ' + class_value] = plot_df['Average: ' + class_value]
        adata.var['Stdev: ' + class_value] = plot_df['Stdev: ' + class_value]
        adata.var['Rank: ' + class_value] = plot_df['Rank: ' + class_value]
        plot_df = plot_df.reindex(sorted_indices)

        # if taking from pdata.var, can continue from here
        # problem is that we need rank_data, the data consisting of samples from this class to make
        # stats df should have 3 column, average stdev and rank
        # plot_df should only have the abundance 
        stats_df = plot_df.filter(regex = 'Average: |Stdev: |Rank: ', axis=1)
        plot_df = plot_df.drop(stats_df.columns, axis=1)
        print(stats_df.shape) if debug else None
        print(plot_df.shape) if debug else None

        nsample = plot_df.shape[1]
        nprot = plot_df.shape[0]

        # Abundance matrix: shape (nprot, nsample)
        X_matrix = plot_df.values  # shape: (nprot, nsample)
        ranks = stats_df['Rank: ' + class_value].values  # shape: (nprot,)
        mu = np.log10(np.clip(stats_df['Average: ' + class_value].values, 1e-6, None))
        std = np.log10(np.clip(stats_df['Stdev: ' + class_value].values, 1e-6, None))
        # Flatten abundance data (X) and repeat ranks (Y)
        X = X_matrix.flatten(order='F')  # Fortran order stacks column-wise, matching your loop
        Y = np.tile(ranks, nsample)
        # Compute Z-values
        logX = np.log10(np.clip(X, 1e-6, None))
        z = ((logX - np.tile(mu, nsample)) / np.tile(std, nsample)) ** 2
        Z = np.exp(-z * exp_alpha)
        # Remove NaNs
        mask = ~np.isnan(X)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        print(f'nsample: {nsample}, nprot: {np.max(Y)}') if debug else None

        ax.scatter(Y, X, c=Z, marker='.',cmap=cmap[j], s=s,alpha=alpha)
        ax.scatter(stats_df['Rank: '+class_value], 
                   stats_df['Average: '+class_value], 
                   marker='.', 
                   color=color[j], 
                   alpha=calpha,
                   label=class_value)
        ax.set_yscale('log')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Abundance')

    # format the argument string classes to be first letter capitalized
    legend_title = (
        "/".join(cls.capitalize() for cls in classes)
        if isinstance(classes, list)
        else classes.capitalize() if isinstance(classes, str)
        else None)

    ax.legend(title=legend_title, loc='best', frameon=True, fontsize='small')
    return ax

def mark_rankquant(plot: "plt.Axes", pdata: pAnnData, mark_df: pd.DataFrame, class_values: list[str], layer: str = "X", on: str = "protein", color: str = "red", s: float = 10, alpha: float = 1, show_label: bool = True, label_type: str = "accession") -> Any:
    """
    Highlight specific features on a rank abundance plot.

    This function marks selected proteins or peptides on an existing rank
    abundance plot, optionally adding labels. It uses statistics stored in
    `.var` during `plot_rankquant()`.

    Args:
        plot (matplotlib.axes.Axes): Axis containing the rank abundance plot.
        pdata (pAnnData): Input pAnnData object.
        mark_df (pandas.DataFrame): Features to highlight.
            
            - DataFrame: Must include an `"accession"` column, and optionally
              `"gene_primary"` if `label_type="gene"`.  
              A typical way to generate this is using
              `scutils.get_upset_query()`, e.g.:
              ```python
              size_upset = scutils.get_upset_contents(pdata_filter, classes="size")
              prot_sc_df = scutils.get_upset_query(
                  size_upset, present=["sc"], absent=["5k", "10k", "20k"],
                  fetch_uniprot=False, pdata=pdata_filter,
              )
              ```
    
        class_values (list of str): Class values to highlight (must match those
            used in `plot_rankquant`).
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level, either `"protein"` or `"peptide"`. Default is `"protein"`.
        color (str): Marker color. Default is `"red"`.
        s (float): Marker size. Default is 10.
        alpha (float): Marker transparency. Default is 1.
        show_label (bool): Whether to display labels for highlighted features.
            Default is True.
        label_type (str): Label type. Options:
            - `"accession"`: show accession IDs.
            - `"gene"`: map to gene names using `"Gene Names"` in `mark_df`.

    Returns:
        ax (matplotlib.axes.Axes): Axis with highlighted features.

    !!! tip 
    
        Works best when paired with `plot_rankquant()`, which stores `Average`,
        `Stdev`, and `Rank` statistics in `.var`. Call `plot_rankquant()` first
        to generate these values, then use `mark_rankquant()` to overlay
        highlights.

    Example:
        Overlay markers after a bulk rank-quant plot:
            ```python
            import matplotlib.pyplot as plt
            import pandas as pd
            from scpviz import plotting as scplt
            from scpviz import utils as scu

            classes_2 = ["cellline", "condition"]
            class_list = scu.get_classlist(pdata.prot, classes_2)
            acc = list(pdata.prot.var_names[:3])
            mark_df = pd.DataFrame({"accession": acc})
            if "Genes" in pdata.prot.var.columns:
                mark_df["gene_primary"] = pdata.prot.var.loc[acc, "Genes"].astype(str).values

            fig, ax = plt.subplots(figsize=(4, 4))
            scplt.plot_rankquant(ax, pdata, classes=classes_2)
            scplt.mark_rankquant(
                ax,
                pdata,
                mark_df=mark_df,
                class_values=class_list[: min(4, len(class_list))],
                color="black",
                label_type="gene",
            )
            plt.show()
            ```

        ![Mark rankquant](../../assets/plots/mark_rankquant.png)

    See Also:
        plot_rankquant: Generate rank abundance plots with statistics stored in `.var`.
        get_upset_query: Create a DataFrame of proteins based on set intersections (obs membership).
    """
    adata = utils.get_adata(pdata, on)
    
    # get entry label
    id_precedence = [
            "accession",    # new default with new Uniprot API
            "Entry",        # legacy uniprot API?
            "id",
            "Accession",
            "Protein IDs",
            ]

    id_col = next((c for c in id_precedence if c in mark_df.columns), None)
    if id_col is None:
        raise ValueError(
            f"mark_df is missing an accession/ID column. "
            f"Tried: {id_precedence}. Columns are: {list(mark_df.columns)}"
        )

    names = mark_df[id_col].astype(str).tolist()
    
    # get gene label if needed
    gene_precedence = [
            "gene_primary",   # NEW default
            "Gene Names",
            "Genes",
            "gene_names",
            "Gene",
        ]

    gene_col = next((c for c in gene_precedence if c in mark_df.columns), None)

    # TEST: check if names are in the data
    pdata._check_rankcol(on, class_values)

    for j, class_value in enumerate(class_values):
        print('Class: ', class_value)
        
        for i, txt in enumerate(names):
            try:
                avg = adata.var[f"Average: {class_value}"].loc[txt]
                rank = adata.var[f"Rank: {class_value}"].loc[txt]
            except Exception as e:
                print(f"Name {txt} not found in {on}.var. Check {on} name for spelling errors and whether it is in data.")
                continue

            label_txt = txt
            if show_label:
                if label_type == 'accession':
                    pass
                elif label_type == 'gene':
                    if gene_col and txt in mark_df[id_col].values:
                        match = mark_df.loc[mark_df[id_col] == txt, gene_col]
                        if not match.empty:
                            label_txt = str(match.values[0])

                plot.annotate(label_txt, (rank, avg), xytext=(rank+10,avg*1.1), fontsize=8)
            plot.scatter(rank, avg, marker='o', color=color, s=s, alpha=alpha)
    return plot

def plot_abundance_2D(ax: "plt.Axes", data: pd.DataFrame, cases: list[list[str]], genes: str | list[str] = "all", cmap: str = "Blues", color: list[str] = ["blue"], s: float = 20, alpha: list[float] = [0.2, 1], calpha: float = 1) -> "plt.Axes":
    """
    Plot a 2D abundance scatter between two case groups.

    This helper computes mean abundance per feature for each case group (from columns matching
    ``"Abundance: "`` + case tokens), then plots a log-log scatter of case1 vs case2. If ``genes``
    is a list, only those genes are highlighted (matched against ``data["Gene Symbol"]``).

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        data (pandas.DataFrame): Long-ish feature table containing abundance columns and a
            ``"Gene Symbol"`` column used for labeling/highlighting.
        cases (list[list[str]]): Exactly two case definitions. Each case is a list of tokens
            used to match abundance columns (joined by underscores).
        genes (str or list[str]): Either ``"all"`` (default) to plot all genes, or a list of gene
            symbols to highlight.
        cmap (str): Colormap name used for the background scatter.
        color (list[str]): Colors for highlights/background points (legacy behavior).
        s (float): Scatter marker size.
        alpha (list[float]): Alpha for background scatter and highlight points.
        calpha (float): Legacy parameter (currently unused).

    Returns:
        matplotlib.axes.Axes: Axis containing the plot.

    Note:
        This function assumes the input table uses scpviz-style abundance column naming. It is
        retained for backwards compatibility and ad-hoc exploratory plots.
    """

    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]
        append_string = '_'.join(vars[1:])

        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]

        # average abundance of proteins across these columns, ignoring NaN values
        data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
        data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

        print(append_string)

    case1_name_string = '_'.join(cases[0][:])
    case2_name_string = '_'.join(cases[1][:])
    
    # find the number for the average column  of the 2 cases
    case1_col = data.columns.get_loc('Average: '+case1_name_string)
    case2_col = data.columns.get_loc('Average: '+case2_name_string)

    # ignore rows where the 2 cases are NaN or 0
    data = data.copy()
    data = data[data.iloc[:,case1_col].notnull()]
    data = data[data.iloc[:,case2_col].notnull()]
    data = data[data.iloc[:,case1_col] != 0]
    data = data[data.iloc[:,case2_col] != 0]

    X = data.iloc[:,case1_col].values
    Y = data.iloc[:,case2_col].values

    # make 2D scatter plot of case1 abundance vs case2 abundance
    ax.scatter(X, Y, marker='.',cmap=cmap, s=s,alpha=alpha[0])
    # set both axis to log
    ax.set_xscale('log')
    ax.set_yscale('log')

    if isinstance(genes, list):
        print('highlighting genes')
        # genes is a list of gene names, so let's extract those that match the accession column
        for i in range(len(genes)):
            # if gene is in data['Gene Symbol'], extract the abundance values for that gene
            if genes[i] in data['Gene Symbol'].values:
                X_highlight = data[data['Gene Symbol']==genes[i]].iloc[:,case1_col].values[0]
                Y_highlight = data[data['Gene Symbol']==genes[i]].iloc[:,case2_col].values[0]
                ax.scatter(X_highlight,Y_highlight,marker='.',color=color[0],s=s,alpha=alpha[1])
                # add gene name to plot
                ax.annotate(genes[i], (X_highlight,Y_highlight), xytext=(X_highlight+10,Y_highlight*1.1), fontsize=10)

    else:
        # plot all genes
        for i, txt in enumerate(data['Gene Symbol']):
            # ax.annotate(txt, (X[i],Y[i]), xytext=(X[i]+10,Y[i]*1.1), fontsize=8)
            ax.scatter(X[i],Y[i],marker='o',color=color[0],s=s,alpha=alpha[1])

    # get min and max of both axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # add a 1:1 line, make line hash dotted with alpha = 0.3
    ax.plot([1e-1,1e7],[1e-1,1e7], ls='--', color='grey', alpha=0.3)

    # set x and y limits to be the same
    minval = min(xmin, ymin)
    maxval = max(xmax, ymax)

    ax.set_xlim([minval, maxval])
    ax.set_ylim([minval, maxval])

    return ax

def plot_raincloud(ax: "plt.Axes", pdata: pAnnData, classes: str | list[str] | None = None, layer: str = "X", on: str = "protein", order: Any = None, color: list[str] = ["blue"], boxcolor: str = "black", linewidth: float = 0.5, debug: bool = False) -> Any:
    """
    Plot raincloud distributions of protein or peptide abundances.

    This function generates a raincloud plot (violin + boxplot + scatter)
    to visualize abundance distributions across groups. Summary statistics
    (average, standard deviation, rank) are written into `.var` for downstream
    use with `mark_raincloud()`.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str, optional): One or more `.obs` columns to
            group samples. If None, all samples are combined.
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level, either `"protein"` or `"peptide"`. Default is `"protein"`.
        order (list of str, optional): Custom order of class categories. If None,
            categories appear in data order.
        color (list of str): Colors for each class distribution. Default is `["blue"]`.
        boxcolor (str): Color for boxplot outlines. Default is `"black"`.
        linewidth (float): Line width for box/whisker elements. Default is 0.5.
        debug (bool): If True, return both axis and computed data arrays.

    Returns:
        ax (matplotlib.axes.Axes): If `debug=False`: axis with raincloud plot.

        tuple (matplotlib.axes.Axes, list of np.ndarray): If `debug=True`: `(axis, data_X)` where `data_X` are the transformed abundance distributions per group.

    Note:
        Statistics (`Average`, `Stdev`, `Rank`) are stored in `.var` and can be
        used with `mark_raincloud()` to highlight specific features.

    Example:
        Plot raincloud distributions by cell line and condition (one color per combined class):
            ```python
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt
            from scpviz import utils as scu

            classes_2 = ["cellline", "condition"]
            rain_colors = [cm.tab10(i % 10) for i in range(len(scu.get_classlist(pdata.prot, classes_2)))]

            fig, ax = plt.subplots(figsize=(5, 4))
            scplt.plot_raincloud(ax, pdata, classes=classes_2, color=rain_colors)
            plt.show()
            ```

        ![Plot raincloud](../../assets/plots/plot_raincloud.png)

        Same pattern on single-cell protein data after ``directlfq`` (``classes`` aligned with UMAP, e.g. ``region``):
            ```python
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt
            from scpviz import utils as scu

            classes_sc = ["region"]
            rain_colors = [cm.tab10(i % 10) for i in range(len(scu.get_classlist(pdata_sc.prot, classes_sc)))]

            fig, ax = plt.subplots(figsize=(5, 4))
            scplt.plot_raincloud(ax, pdata_sc, classes=classes_sc, color=rain_colors)
            plt.show()
            ```

        ![Plot raincloud (single-cell)](../../assets/plots/plot_raincloud_sc.png)

    See Also:
        mark_raincloud: Highlight specific features on a raincloud plot.  
        plot_rankquant: Alternative distribution visualization using rank abundance.
    """
    u = _plotting_pkg_utils()
    adata = u.get_adata(pdata, on)

    classes_list = u.get_classlist(adata, classes=classes, order=order)
    data_X = []

    for j, class_value in enumerate(classes_list):
        rank_data = u.resolve_class_filter(adata, classes, class_value, debug=True)

        plot_df = rank_data.to_df().transpose()
        plot_df['Average: '+class_value] = np.nanmean(rank_data.X.toarray(), axis=0)
        plot_df['Stdev: '+class_value] = np.nanstd(rank_data.X.toarray(), axis=0)
        plot_df.sort_values(by=['Average: '+class_value], ascending=False, inplace=True)
        plot_df['Rank: '+class_value] = np.where(plot_df['Average: '+class_value].isna(), np.nan, np.arange(1, len(plot_df) + 1))

        sorted_indices = plot_df.index

        plot_df = plot_df.loc[adata.var.index]
        adata.var['Average: ' + class_value] = plot_df['Average: ' + class_value]
        adata.var['Stdev: ' + class_value] = plot_df['Stdev: ' + class_value]
        adata.var['Rank: ' + class_value] = plot_df['Rank: ' + class_value]
        plot_df = plot_df.reindex(sorted_indices)

        stats_df = plot_df.filter(regex = 'Average: |Stdev: |Rank: ', axis=1)
        plot_df = plot_df.drop(stats_df.columns, axis=1)

        nsample = plot_df.shape[1]
        nprot = plot_df.shape[0]

        # merge all abundance columns into one column
        X = np.zeros((nsample*nprot))
        for i in range(nsample):
            X[i*nprot:(i+1)*nprot] = plot_df.iloc[:, i].values

        X = X[~np.isnan(X)] # remove NaN values
        X = X[X != 0] # remove 0 values
        X = np.log10(X)

        data_X.append(X)
    
    print('data_X shape: ', len(data_X)) if debug else None

    # boxplot
    bp = ax.boxplot(data_X, positions=np.arange(1,len(classes_list)+1)-0.06, widths=0.1, patch_artist = True,
                    flierprops=dict(marker='o', alpha=0.2, markersize=2, markerfacecolor=boxcolor, markeredgecolor=boxcolor),
                    whiskerprops=dict(color=boxcolor, linestyle='-', linewidth=linewidth),
                    medianprops=dict(color=boxcolor, linewidth=linewidth),
                    boxprops=dict(facecolor='none', color=boxcolor, linewidth=linewidth),
                    capprops=dict(color=boxcolor, linewidth=linewidth))

    # Violinplot
    vp = ax.violinplot(data_X, points=500, vert=True, positions=np.arange(1,len(classes_list)+1)+0.06,
                showmeans=False, showextrema=False, showmedians=False)

    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 1])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1.06, idx+2.06)
        # Change to the desired color
        b.set_color(color[idx])
    # Scatterplot data
    for idx in range(len(data_X)):
        features = data_X[idx]
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=.1, high=.18, size=len(idxs))
        y = out
        ax.scatter(y, features, s=2., c=color[idx], alpha=0.5)

    if debug:
        return ax, data_X
    else:
        return ax

def mark_raincloud(plot: "plt.Axes", pdata: pAnnData, mark_df: pd.DataFrame, class_values: list[str], layer: str = "X", on: str = "protein", lowest_index: int = 0, color: str = "red", s: float = 10, alpha: float = 1) -> Any:
    """
    Highlight specific features on a raincloud plot.

    This function marks selected proteins or peptides on an existing
    raincloud plot, using summary statistics written to `.var` during
    `plot_raincloud()`.

    Args:
        plot (matplotlib.axes.Axes): Axis containing the raincloud plot.
        pdata (pAnnData): Input pAnnData object.
        mark_df (pandas.DataFrame): DataFrame containing entries to highlight.
            Must include an `"Entry"` column.
        class_values (list of str): Class values to highlight (must match those
            used in `plot_raincloud`).
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level, either `"protein"` or `"peptide"`. Default is `"protein"`.
        lowest_index (int): Offset for horizontal positioning. Default is 0.
        color (str): Marker color. Default is `"red"`.
        s (float): Marker size. Default is 10.
        alpha (float): Marker transparency. Default is 1.

    Returns:
        ax (matplotlib.axes.Axes): Axis with highlighted features.

    !!! tip 
    
        Works best when paired with `plot_raincloud()`, which computes and
        stores the required statistics in `.var`.

    Example:
        Highlight proteins on a raincloud after ``plot_raincloud`` (same grouping and colors as that plot):
            ```python
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
            import pandas as pd
            from scpviz import plotting as scplt
            from scpviz import utils as scu

            classes_2 = ["cellline", "condition"]
            class_list = scu.get_classlist(pdata.prot, classes_2)
            rain_colors = [cm.tab10(i % 10) for i in range(len(class_list))]

            var = pdata.prot.var
            want = ["GAPDH", "TUBB", "ACTB"]
            if "Genes" not in var.columns:
                acc = list(var.index[:3])
            else:
                m = var["Genes"].astype(str).isin(want)
                acc = list(var.index[m][:3])
                if len(acc) < 3:
                    acc = list(var.index[:3])
            sub = var.loc[acc].copy().reset_index()
            id_col = "index" if "index" in sub.columns else sub.columns[0]
            mark_df = sub.rename(columns={id_col: "accession"})
            if "Genes" in mark_df.columns:
                mark_df = mark_df.rename(columns={"Genes": "gene_primary"})
            mark_df = mark_df[[c for c in ("accession", "gene_primary") if c in mark_df.columns]]

            fig, ax = plt.subplots(figsize=(5, 4))
            scplt.plot_raincloud(ax, pdata, classes=classes_2, color=rain_colors)
            scplt.mark_raincloud(
                ax,
                pdata,
                mark_df=mark_df,
                class_values=class_list[: min(4, len(class_list))],
                color="black",
            )
            plt.show()
            ```

        ![Mark raincloud](../../assets/plots/mark_raincloud.png)

    See Also:
        plot_raincloud: Generate raincloud plots with distributions per group.  
        plot_rankquant: Alternative distribution visualization using rank abundance.
    """
    adata = _plotting_pkg_utils().get_adata(pdata, on)
    # get entry label
    id_precedence = [
            "accession",    # new default with new Uniprot API
            "Entry",        # legacy uniprot API?
            "id",
            "Accession",
            "Protein IDs",
            ]

    id_col = next((c for c in id_precedence if c in mark_df.columns), None)
    if id_col is None:
        raise ValueError(
            f"mark_df is missing an accession/ID column. "
            f"Tried: {id_precedence}. Columns are: {list(mark_df.columns)}"
        )

    names = mark_df[id_col].astype(str).tolist()
    
    # TEST: check if names are in the data
    pdata._check_rankcol(on, class_values)

    for j, class_value in enumerate(class_values):
        print('Class: ', class_value)

        for i, txt in enumerate(names):
            try:
                y = np.log10(adata.var['Average: '+class_value].loc[txt])
                x = lowest_index + j + .14 + 0.8
            except Exception as e:
                print(f"Name {txt} not found in {on}.var. Check {on} name for spelling errors and whether it is in data.")
                continue
            plot.scatter(x,y,marker='o',color=color,s=s, alpha=alpha)

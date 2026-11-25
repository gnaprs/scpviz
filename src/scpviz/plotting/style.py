"""Shared plotting utilities: colors, legends, summary, significance bars."""
from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING, overload

import warnings

import anndata as ad
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from scpviz.utils.formatting import format_log_prefix

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData


def _get_cmap(cmap: str | mcolors.Colormap) -> mcolors.Colormap:
    """Resolve a matplotlib colormap by name or instance (compatible with mpl >= 3.9)."""
    return mpl.colormaps.get_cmap(cmap)


@overload
def get_color(resource_type: Literal["colors"], n: int) -> list[str]: ...

@overload
def get_color(resource_type: Literal["cmap"], n: int | None = None) -> mcolors.LinearSegmentedColormap | list[mcolors.LinearSegmentedColormap]: ...

@overload
def get_color(resource_type: Literal["palette"], n: None = None) -> list[tuple[float, float, float]]: ...

@overload
def get_color(resource_type: Literal["show"], n: None = None) -> None: ...

def get_color(resource_type: str, n: int | None = None) -> Any:
    """
    Generate a list of colors, a colormap, or a palette from package defaults.

    Args:
        resource_type (str): The type of resource to generate. Options are:
            - 'colors': Return a list of hex color codes.
            - 'cmap': Return a matplotlib colormap.
            - 'palette': Return a seaborn palette.
            - 'show': Display all 7 base colors.

        n (int, optional): The number of colors or colormaps to generate.
            Required for 'colors' and 'cmap'. Colors will repeat if n > 7.

    Returns:
        colors (list of str): If ``resource_type='colors'``, a list of hex color strings. Repeats colors if n > 7.
        cmap (matplotlib.colors.LinearSegmentedColormap): If ``resource_type='cmap'``.
        palette (seaborn.color_palette): If ``resource_type='palette'``.
        None: If ``resource_type='show'``, displays the available colors.

    !!! info "Default Colors"

        The following base colors are used (hex codes):

            ['#FC9744', '#00AEE8', '#9D9D9D', '#6EDC00', '#F4D03F', '#FF0000', '#A454C7']        

        <div style="display:flex;gap:0.5em;">
            <div style="width:1.5em;height:1.5em;background:#FC9744;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#00AEE8;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#9D9D9D;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#6EDC00;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#F4D03F;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#FF0000;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#A454C7;border:1px solid #000"></div>
        </div>
            
    Example:
        Get list of 5 colors:
            ```python
            colors = get_color('colors', 5)
            ```

        <div style="display:flex;gap:0.5em;">
            <div style="width:1.5em;height:1.5em;background:#FC9744;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#00AEE8;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#9D9D9D;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#6EDC00;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#F4D03F;border:1px solid #000"></div>
        </div>

        Get default cmap:
            ```python
            cmap = get_color('cmap', 2)
            ```
        <div style="width:150px;height:20px;background:linear-gradient(to right, white, #FC9744);border:1px solid #000"></div>
        <div style="width:150px;height:20px;background:linear-gradient(to right, white, #00AEE8);border:1px solid #000"></div>
                    
        Get default palette:
            ```python
            palette = get_color('palette')
            ```

        <div style="display:flex;gap:0.3em;">
            <div style="width:1.2em;height:1.2em;background:#FC9744;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#00AEE8;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#9D9D9D;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#6EDC00;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#F4D03F;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#FF0000;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#A454C7;border:1px solid #000"></div>
        </div>
    """

    # --- 
    # Create a list of colors
    base_colors = ['#FC9744', '#00AEE8', '#9D9D9D', '#6EDC00', '#F4D03F', '#FF0000', '#A454C7']
    # ---

    if resource_type == 'colors':
        if n is None:
            raise ValueError("Parameter 'n' must be specified when resource_type is 'colors'")
        if n > len(base_colors):
            warnings.warn(f"Requested {n} colors, but only {len(base_colors)} available. Reusing from the start.")
        return [base_colors[i % len(base_colors)] for i in range(n)]
    
    elif resource_type == 'cmap':
        if n is None:
            n = 1  # Default to generating one colormap from the first base color
        if n > len(base_colors):
            warnings.warn(f"Requested {n} colormaps, but only {len(base_colors)} base colors. Reusing from the start.")
        cmaps = []
        for i in range(n):
            color = base_colors[i % len(base_colors)]
            cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{i}', ['white', color])
            cmaps.append(cmap)
        return cmaps if n > 1 else cmaps[0]
    
    elif resource_type == 'palette':
        return sns.color_palette(base_colors)
    
    elif resource_type == 'show':
        # Show palette and colormaps
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 1]})
        
        # Format labels as "n: #HEX"
        hex_labels = [f"{i}: {mcolors.to_hex(color)}" for i, color in enumerate(base_colors)]

        # --- Palette ---
        for i, color in enumerate(base_colors):
            axs[0].bar(i, 1, color=color)
        axs[0].set_title("Base Colors (Colors and Palette)")
        axs[0].set_xticks(range(len(base_colors)))
        axs[0].set_xticklabels(hex_labels, rotation=45, ha='right')
        axs[0].set_yticks([])

        # --- Colormaps ---
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        n_colors = len(base_colors)

        for i, color in enumerate(base_colors):
            cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{i}', ['white', color])
            axs[1].imshow(
                gradient,
                aspect='auto',
                cmap=cmap,
                extent=(i, i + 1, 0, 1)
            )

        axs[1].set_title("Colormaps")
        axs[1].set_xlim(0, n_colors)
        axs[1].set_xticks(np.arange(n_colors) + 0.5)
        axs[1].set_xticklabels(hex_labels, rotation=45, ha='right')
        axs[1].set_yticks([])

        plt.tight_layout()
        plt.show()
        return None

    else:
        raise ValueError("Invalid resource_type. Options are 'colors', 'cmap', and 'palette'")

def _apply_text_style(txt: Any, color: str, text_style: str) -> None:
    """Style a mark label: ``"outline"`` (white stroke) or ``"box"`` (rounded white box + stroke)."""
    if text_style not in ("outline", "box"):
        raise ValueError(
            f"{format_log_prefix('error')} text_style must be 'outline' or 'box', got {text_style!r}."
        )
    if text_style == "box":
        txt.set_bbox(dict(facecolor="white", edgecolor=color, boxstyle="round"))
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])


def _update_mark_legend(ax: "plt.Axes", category: str, color: str, loc: str) -> None:
    """Add/replace a ``category`` entry in a single marks legend on ``ax``.

    Repeated calls on the same axis merge into one legend. Any pre-existing
    legend (e.g. from ``plot_rankquant``) is kept as a separate artist.
    """
    reg = getattr(ax, "_scpviz_mark_legend", None)
    if reg is None:
        reg = ax._scpviz_mark_legend = {"entries": {}, "legend": None}
    reg["entries"][category] = color
    if reg["legend"] is not None and reg["legend"].axes is ax:
        reg["legend"].remove()
    elif ax.get_legend() is not None:
        ax.add_artist(ax.get_legend())
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", color=c, label=k)
        for k, c in reg["entries"].items()
    ]
    reg["legend"] = ax.legend(handles=handles, loc=loc, frameon=True, fontsize="small")


def adjust_and_outline_texts(
    texts: list[Any],
    expand: tuple[float, float] = (2, 2),
    arrowprops: dict[str, Any] | None = dict(arrowstyle="->", color="k", lw=0.8),
    linewidth: float = 3,
    outline_color: str = "w",
) -> Any:
    """
    Adjust text labels to reduce overlap and apply a white outline for readability.

    Runs `adjust_text()` on a list of text artists while temporarily removing
    their path effects to ensure stable label placement. A white outline is
    re-applied after adjustment to improve legibility on dense plots. Works with
    labels from any `mark_*` function called with `return_texts=True`
    (e.g. `mark_volcano`, `mark_volcano_by_significance`, `mark_rankquant`).

    Args:
        texts (list): List of `matplotlib.text.Text` objects, typically returned
            from a `mark_*` function with `return_texts=True`.
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

            scplt.adjust_and_outline_texts(texts, expand=(2, 2))
            ```

        ![Adjust and outline texts](../../assets/plots/volcano_adjust_and_outline_texts.png)

    Note:
        Collect labels from several `mark_*(..., return_texts=True)` calls and
        run this once; a single global adjustment gives cleaner layouts than
        per-group adjustments. `volcano_adjust_and_outline_texts` is a
        deprecated alias of this function.
    """
    from adjustText import adjust_text

    orig_effects = []
    for txt in texts:
        orig_effects.append(txt.get_path_effects())
        txt.set_path_effects([])

    adjust_kwargs = {"expand": expand}
    if arrowprops is not None:
        adjust_kwargs["arrowprops"] = arrowprops

    adjust_text(texts, **adjust_kwargs)

    for txt in texts:
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=linewidth, foreground=outline_color)
        ])

    return texts


def shift_legend(
    ax: "plt.Axes",
    anchor_pos: tuple[float, float] = (1.05, 1),
    loc: str = "center left",
) -> None:
    """
    Reposition all legends on an axis.

    Moves every Matplotlib legend on the axis to a custom anchor point
    (outside or inside the axis) without modifying contents. When multiple
    legends are present, they are stacked vertically from the anchor point
    downward with a small gap between them.

    Args:
        ax (matplotlib.axes.Axes): Axis containing the legend(s).
        anchor_pos (tuple of float, optional): (x, y) anchor position for the
            first (or only) legend in axis coordinates. Default is `(1.05, 0.5)`,
            placing the legend just outside the right edge.
        loc (str, optional): Legend location relative to the anchor. Default is
            `'center left'`.

    Returns:
        None

    Example:
        Move a single legend outside the right edge:
            ```python
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax = scplt.plot_pca(pdata, classes='treatment')
                    scplt.shift_legend(ax)
            ```

        Stack multiple legends when color, edge, and shape are all mapped:
            ```python
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax = scplt.plot_pca(pdata, color='treatment', edge_color='cellline',
                                        marker_shape='batch')
                    scplt.shift_legend(ax, anchor_pos=(1.05, 1.0))
            ```
    """
    # ax.get_legend() returns only the last; collect all via ax children
    ax_legends = [a for a in ax.get_children()
                  if isinstance(a, plt.matplotlib.legend.Legend)]
    if not ax_legends:
        # Legends hosted on a sibling axis (e.g. plot_pairwise_correlation's
        # legend column): move them, anchored in `ax` coordinates.
        ax_legends = [a for other in ax.get_figure().axes if other is not ax
                      for a in other.get_children()
                      if isinstance(a, plt.matplotlib.legend.Legend)]

    # The last ax.legend() is listed both as ax.legend_ and as an artist
    ax_legends = list(dict.fromkeys(ax_legends))

    if not ax_legends:
        return

    if len(ax_legends) == 1:
        leg = ax_legends[0]
        leg.set_clip_on(False)
        leg.set_bbox_to_anchor(anchor_pos, transform=ax.transAxes)
        leg.set_loc(loc)
        return

    # Multiple legends: stack vertically from anchor_pos downward
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_height_px = ax.get_window_extent(renderer).height

    x, y_cursor = anchor_pos
    for leg in ax_legends:
        leg.set_clip_on(False)
        leg_height_px = leg.get_window_extent(renderer).height
        leg_height_ax = leg_height_px / ax_height_px
        leg.set_bbox_to_anchor((x, y_cursor), transform=ax.transAxes)
        leg.set_loc("upper left")
        y_cursor -= leg_height_ax + 0.02

def plot_significance(
    ax: "plt.Axes",
    y: float,
    h: float,
    x1: float = 0,
    x2: float = 1,
    col: str = "k",
    pval: float | str = "n.s.",
    fontsize: int = 12,
) -> None:
    """
    Plot significance bars on a matplotlib axis.

    This function draws horizontal significance bars (e.g., for statistical annotations)
    between two x-positions with a label indicating the p-value or significance level.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot the significance bars.
        y (float): Vertical coordinate of the top of the bars.
        h (float): Height of the vertical ticks extending downward from `y`.
        x1 (float): X-coordinate of the first bar endpoint.
        x2 (float): X-coordinate of the second bar endpoint.
        col (str): Color of the bars.
        pval (float or str): P-value or significance label.
            
            - If a float, it is compared against thresholds (e.g., 0.05, 0.01) to assign
              significance markers (`*`, `**`, `***`).
            
            - If a string, it is directly rendered as the label.
        
        fontsize (int): Font size of the significance text.

    Returns:
        None

    Example:
        Minimal bar comparison with a significance bracket:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(2, 3))
            ax.bar([0, 1], [10, 15])
            scplt.plot_significance(ax, 16.0, 1.0, x1=0, x2=1, pval="*")
            plt.show()
            ```

        ![Plot significance](../../assets/plots/plot_significance.png)

        Annotate a swarm + bar plot with a t-test p-value:
            ```python
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import ttest_ind
            
            fig, ax = plt.subplots(figsize=(1.74, 2.13))
            sns.swarmplot(data=summary_df, x="treatment", y="protein_count", ax=ax, color="k")
            sns.barplot(
                data=summary_df,
                x="treatment",
                y="protein_count",
                ax=ax,
                errorbar="ci",
                alpha=1,
                palette=color_dict,
            )

            control = summary_df[summary_df["treatment"] == "Control"]["protein_count"]
            treated = summary_df[summary_df["treatment"] == "Treated"]["protein_count"]

            scplt.plot_significance(
                ax,
                y=2630,
                h=30,
                pval=ttest_ind(control, treated).pvalue,
                fontsize=8,
            )
            ```
    """

    # check variable type of pval
    sig = 'n.s.'
    if isinstance(pval, float):
        if pval > 0.05:
            sig = 'n.s.'
        else:
            sig = '*' * int(np.floor(-np.log10(pval)))
    else:
        sig = pval

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col, fontsize=fontsize)
    
def plot_summary(
    ax: "plt.Axes",
    pdata: "pAnnData",
    value: str = "protein_count",
    classes: str | list[str] | None = None,
    plot_mean: bool = True,
    **kwargs: Any,
) -> "plt.Axes | list[plt.Axes]":
    """
    Plot summary statistics of sample metadata.

    This function visualizes values from `pdata.summary` (e.g., protein count,
    peptide count, abundance) as bar plots, optionally grouped by sample-level classes.
    It supports both per-sample visualization and mean values across groups.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object with `.summary` metadata table.
        value (str): Column in `pdata.summary` to plot. Default is `'protein_count'`.
        classes (str or list of str, optional): Sample-level classes to group by.
            - If None: plot per-sample values directly.
            
            - If str: group by the specified column, aggregating with mean if `plot_mean=True`.
            
            - If list: when multiple classes are provided, combinations of class values
              are used for grouping and subplots are created per unique value of `classes[0]`.

        plot_mean (bool): Whether to plot mean ± standard deviation by class.
            If True, `classes` must be provided. Default is True.
        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes): The axis (or 
        list of axes if subplots are created) with the plotted summary.

    Raises:
        ValueError: If `plot_mean=True` but `classes` is not specified.
        ValueError: If `classes` is invalid (not None, str, or non-empty list).

    Example:
        Sample count summary by cell line and condition:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(5, 3))
            scplt.plot_summary(ax, pdata, classes=["cellline", "condition"])
            plt.show()
            ```

        ![Plot summary](../../assets/plots/plot_summary.png)
    """

    if pdata.summary is None:
        pdata._update_summary()

    summary_data = pdata.summary.copy()

    if plot_mean:
        if classes is None:
            raise ValueError("Classes must be specified when plot_mean is True.")
        elif isinstance(classes, str):
            sns.barplot(x=classes, y=value, hue=classes, data=summary_data, errorbar='sd', ax=ax, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif isinstance(classes, list) and len(classes) > 0:
            if len(classes) == 1:
                sns.catplot(x=classes[0], y=value, data=summary_data, hue=classes[0], kind='bar', ax=ax, **kwargs)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            elif len(classes) >= 2:
                summary_data['combined_classes'] = summary_data[classes[1:]].astype(str).agg('-'.join, axis=1)

                unique_values = summary_data[classes[0]].unique()
                num_unique_values = len(unique_values)

                fig, ax = plt.subplots(nrows=num_unique_values, figsize=(10, 5 * num_unique_values))

                if num_unique_values == 1:
                    ax = [ax]

                for ax_sub, unique_value in zip(ax, unique_values):
                    subset_data = summary_data[summary_data[classes[0]] == unique_value]
                    sns.barplot(x='combined_classes', y=value, data=subset_data, hue='combined_classes', ax=ax_sub, **kwargs)
                    ax_sub.set_title(f"{classes[0]}: {unique_value}")
                    ax_sub.set_xticklabels(ax_sub.get_xticklabels(), rotation=45, ha='right')
    else:
        if classes is None:
            sns.barplot(x=summary_data.index, y=value, data=summary_data, ax=ax, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif isinstance(classes, str):
            sns.barplot(x=summary_data.index, y=value, hue=classes, data=summary_data, ax=ax, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif isinstance(classes, list) and len(classes) > 0:
            if len(classes) == 1:
                sns.barplot(x=summary_data.index, y=value, hue=classes[0], data=summary_data, ax=ax, **kwargs)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            elif len(classes) >= 2:
                summary_data['combined_classes'] = summary_data[classes[1:]].astype(str).agg('-'.join, axis=1)
                # Create a subplot for each unique value in classes[0]
                unique_values = summary_data[classes[0]].unique()
                num_unique_values = len(unique_values)
                
                fig, ax = plt.subplots(nrows=num_unique_values, figsize=(10, 5 * num_unique_values))
                
                if num_unique_values == 1:
                    ax = [ax]  # Ensure axes is iterable
                
                for ax_sub, unique_value in zip(ax, unique_values):
                    subset_data = summary_data[summary_data[classes[0]] == unique_value]
                    sns.barplot(x=subset_data.index, y=value, hue='combined_classes', data=subset_data, ax=ax_sub, **kwargs)
                    ax_sub.set_title(f"{classes[0]}: {unique_value}")
                    ax_sub.set_xticklabels(ax_sub.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()            
        else:
            raise ValueError("Invalid 'classes' parameter. It should be None, a string, or a non-empty list.")

    plt.tight_layout()

    return ax

def _resolve_subset_mask(
    adata: ad.AnnData, subset_mask: np.ndarray | pd.Series | list | None
) -> np.ndarray:
    n = adata.n_obs
    if subset_mask is None:
        return np.ones(n, dtype=bool)

    # pandas Series: align by index if possible
    if isinstance(subset_mask, pd.Series):
        if subset_mask.dtype != bool:
            raise TypeError("subset_mask Series must be boolean.")
        if subset_mask.index.equals(adata.obs.index):
            m = subset_mask.to_numpy()
        else:
            # align by index labels (missing -> False)
            m = subset_mask.reindex(adata.obs.index, fill_value=False).to_numpy()
        if m.shape[0] != n:
            raise ValueError("subset_mask has wrong length after alignment.")
        return m.astype(bool)

    # numpy / list
    m = np.asarray(subset_mask)
    if m.dtype != bool:
        raise TypeError("subset_mask must be a boolean array/Series aligned to adata.obs.index.")
    if m.shape[0] != n:
        raise ValueError(f"subset_mask length {m.shape[0]} does not match n_obs {n}.")
    return m.astype(bool)

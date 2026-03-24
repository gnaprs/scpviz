"""
This module provides a collection of plotting utilities for visualizing
protein and peptide abundance data, quality control metrics, and results of
statistical analyses. Functions are organized into categories based on their
purpose, with paired "plot" and "mark" functions where applicable.

Functions are written to work seamlessly with the `pAnnData` object structure and metadata conventions in scpviz.

## Convenience Plotting Wrappers
    get_color: Generate a list of colors, a colormap, or a palette from package defaults.
    shift_legend: Reposition an axis legend outside the plot while maintaining figure size.

## Distribution and Abundance Plots

Functions:
    plot_abundance: Violin/box/strip plots of protein or peptide abundance.
    plot_abundance_housekeeping: Plot abundance of housekeeping proteins.
    plot_rankquant: Rank abundance scatter distributions across groups.
    mark_rankquant: Highlight specific features on a rank abundance plot.
    plot_raincloud: Raincloud plot (violin + box + scatter) of distributions.
    mark_raincloud: Highlight specific features on a raincloud plot.

## Multivariate Dimension Reduction

Functions:
    plot_pca: Principal Component Analysis (PCA) scatter plot.
    plot_pca_scree: Scree plot of PCA variance explained.
    plot_umap: UMAP projection for nonlinear dimensionality reduction.
    resolve_plot_colors: Helper function for resolving PCA/UMAP colors.

## Clustering and Heatmaps

Functions:
    plot_clustermap: Clustered heatmap of proteins/peptides × samples.

## Differential Expression and Volcano Plots

Functions:
    plot_volcano: Volcano plot of differential expression results.
    plot_volcano_adata: Same as above, but for AnnData objects.
    mark_volcano: Highlight specific features on a volcano plot with a specific color.
    mark_volcano_significance: Similar to above, but colored by significance.
    volcano_adjust_and_outline_texts: Adjust text labels for volcano plots after multiple mark_volcanos.
    add_volcano_legend: Add standard legend handles for volcano plots.

## Enrichment Plots

Functions:
    plot_enrichment_svg: Plot STRING enrichment results (forwarded from `enrichment.py`).

## Set Operations

Functions:
    plot_venn: Venn diagrams for 2 to 3 sets.
    plot_upset: UpSet diagrams for >3 sets.

## Summaries and Quality Control

Functions:
    plot_summary: Bar plots summarizing sample-level metadata (e.g., protein counts).
    plot_significance: Add significance bars to plots.
    plot_cv: Boxplots of coefficient of variation (CV) across groups.

## Notes and Tips
!!! tip
    * Most functions accept a `matplotlib.axes.Axes` as the first argument for flexible subplot integration. `ax` can be defined as such:

    ```python
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4)) # configure size as needed
    ```
    
    * "Mark" functions are designed to be used immediately after their paired "plot" functions to highlight features of interest.

---

"""

import re
from matplotlib import markers
import numpy as np
import pandas as pd
import umap.umap_ as umap
import scanpy as sc

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.collections as clt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
import matplotlib.lines as mlines

import upsetplot
from adjustText import adjust_text
import warnings

from scpviz import utils

sns.set_theme(context='paper', style='ticks')

def get_color(resource_type: str, n=None):
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

def shift_legend(ax, anchor_pos=(1.05, 0.5), loc='center left'):
    """
    Reposition an axis legend.

    This helper moves an existing Matplotlib legend to a custom anchor point
    (outside or inside the axis) without modifying its contents.

    Args:
        ax (matplotlib.axes.Axes): Axis containing the legend.
        anchor_pos (tuple of float, optional): (x, y) anchor position for the legend
            in axis coordinates. Default is `(1.05, 0.5)`, placing the legend just
            outside the right edge.
        loc (str, optional): Legend location relative to the anchor. Default is
            `'center left'`.

    Returns:
        None

    Example:
        ```python
        fig, ax = plt.subplots(figsize=(3, 3))
        ax, = scplt.plot_pca(ax, pdata, classes='treatment')
        scplt.shift_legend(ax)
        ```
    """    
    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor(anchor_pos)
        leg.set_loc(loc)

def plot_significance(ax, y, h, x1=0, x2=1, col='k', pval='n.s.', fontsize=12):
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

def plot_cv(ax, pdata, classes=None, layer='X', on='protein', order=None, palette=None, return_df=False, **kwargs):
    """
    Generate a box-and-whisker plot for the coefficient of variation (CV).

    This function computes CV values across proteins or peptides, grouped by
    sample-level classes, and visualizes their distribution as a box plot.

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
        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes): The axis with the plotted CV distribution.
        cv_df (pandas.DataFrame): Optional, returned if `return_df=True`.

    Example:
        Retrieve CV values and customize the violin plot:
            ```python
            import matplotlib.pyplot as plt
            import seaborn as sns

            classes = "size"
            fig, ax = plt.subplots(figsize=(2.795, 3))
            cv_df = scplt.plot_cv(ax, pdata, classes=classes, return_df=True)

            cv_df = cv_df.reset_index()
            sns.violinplot(
                data=cv_df,
                y="Class",
                x="CV",
                orient="h",
                order=order,
                palette=colors,
                linewidth=1,
                inner="quartile",
                saturation=1,
                ax=ax,
            )
            ```
    """
    # Compute CVs for the selected layer
    pdata.cv(classes = classes, on = on, layer = layer)
    adata = utils.get_adata(pdata, on)    
    classes_list = utils.get_classlist(adata, classes = classes, order = order)
    
    cv_data = []
    for class_value in classes_list:
        cv_col = f'CV: {class_value}'
        if cv_col in adata.var.columns:
            cv_values = adata.var[cv_col].values
            cv_data.append(pd.DataFrame({'Class': class_value, 'CV': cv_values}))

    if not cv_data:
        print(f"{utils.format_log_prefix('warn')} No valid CV subsets found — skipping plot.")
        return ax if ax is not None else None
    
    cv_df = pd.concat(cv_data, ignore_index=True)

    # return cv_df for user to plot themselves
    if return_df:
        return cv_df
    
    if palette is None:
        palette = get_color('palette')

    # Ensure consistent class ordering
    if order is not None:
        cat_type = pd.api.types.CategoricalDtype(order, ordered=True)
        cv_df['Class'] = cv_df['Class'].astype(cat_type)
    else:
        cv_df['Class'] = pd.Categorical(cv_df['Class'],
                                        categories=sorted(cv_df['Class'].unique()),
                                        ordered=True)    

    violin_kwargs = dict(inner="box", linewidth=1, cut=0, alpha=0.6, density_norm="width")
    violin_kwargs.update(kwargs)

    sns.violinplot(x='Class', y='CV', data=cv_df, ax=ax, palette=palette, **violin_kwargs)
    
    plt.title('Coefficient of Variation (CV) by Class')
    plt.xlabel('Class')
    plt.ylabel('CV')
    
    return ax

def plot_summary(ax, pdata, value='protein_count', classes=None, plot_mean=True, **kwargs):
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
        Quick QC summary without mean bars:
            ```python
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            scplt.plot_summary(ax, pdata, classes=["amount"], plot_mean=False)
            ```
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

def plot_abundance_housekeeping(ax, pdata, classes=None, loading_control='all', **kwargs):
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

def plot_abundance(ax, pdata, namelist=None, layer='X', on='protein',
                   classes=None, return_df=False, order=None, palette=None,
                   log=False, facet=None, height=4, aspect=0.5,
                   plot_points=True, x_label='gene', kind='auto', **kwargs):
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

        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes or seaborn.FacetGrid): The axis or facet grid containing the plot.
        df (pandas.DataFrame, optional): Returned if `return_df=True`.

    !!! example
        Plot abundance of two selected proteins:
            ```python
            from scpviz import plotting as scplt
            scplt.plot_abundance(ax, pdata, namelist=['Slc12a2','Septin6'])
            ```

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

    if kind == 'auto':
        sample_counts = df.groupby([x_col, 'class', 'facet']).size()
        kind = 'bar' if sample_counts.min() <= 3 else 'violin'

    def _plot_bar(df):
        bar_kwargs = dict(
            ci='sd',
            capsize=0.2,
            errwidth=1.5,
            palette=palette
        )
        bar_kwargs.update(kwargs)
        if facet and df['facet'].nunique() > 1:
            g = sns.FacetGrid(df, col='facet', height=height, aspect=aspect, sharey=True)
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

            # deduplicate legend
            handles, labels = _ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            _ax.legend(by_label.values(), by_label.keys(), title='Class', frameon=True)
            _ax.set_yscale("log") if not log else None
            _ax.set_ylabel("log2(Abundance)" if log else "Abundance")
            _ax.set_xlabel("Gene" if x_label == 'gene' else "Accession")

            return _ax

    def _plot_violin(df):
        violin_kwargs = dict(inner="box", linewidth=1, cut=0, alpha=0.5, density_norm="width")
        violin_kwargs.update(kwargs)
        if facet and df['facet'].nunique() > 1:
            g = sns.FacetGrid(df, col='facet', height=height, aspect=aspect, sharey=True)
            g.map_dataframe(sns.violinplot, x=x_col, y=y_col, hue='class', palette=palette, **violin_kwargs)
            if plot_points:
                def _strip(data, color, **kwargs_inner):
                    sns.stripplot(data=data, x=x_col, y=y_col, hue='class', dodge=True, jitter=True,
                                  color='black', size=3, alpha=0.5, legend=False, **kwargs_inner)
                g.map_dataframe(_strip)
            g.set_axis_labels("Gene" if x_label == 'gene' else "Accession", "log2(Abundance)" if log else "Abundance")
            if not log:
                for ax_ in g.axes.flatten():
                    ax_.set_yscale("log")
            g.set_titles("{col_name}")
            g.add_legend(title='Class', frameon=True)
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

def plot_abundance_boxgrid(pdata, namelist=None, ax=None, layer='X', on='protein', classes=None, return_df=False,
    order=None, plot_type="box", log_scale=False, figsize=(2,2), palette=None, y_min=None, y_max=None, label_x=True, show_n=False,
    global_legend=True, box_kwargs=None, hline_kwargs=None, bar_kwargs=None, bar_error='sd', violin_kwargs=None, text_kwargs=None, strip_kwargs=None):
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
        hline_kwargs (dict, optional): Keyword arguments for mean-lines
            (used when ``plot_type="line"``).
        bar_kwargs (dict, optional): Additional arguments passed to bar plotting
            (used when ``plot_type="bar"``).
        bar_error (str, optional): Error bar for bar plot. Select from one of
            {"sd", "sem", None, <callable>}, where callable takes a 1D array and returns
            a scalar error. Defaults to "sd".
        violin_kwargs (dict, optional): Additional arguments passed to ``sns.violinplot``
            (used when ``plot_type="violin"``).
        text_kwargs (dict, optional): Keyword arguments for count labels
            (e.g., fontsize, offset).
        strip_kwargs (dict, optional): Keyword arguments for strip (raw points),
            e.g. ``{"darken_factor": 0.65}``.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        axes (list of matplotlib.axes.Axes): One axis per gene.
        df (pandas.DataFrame, optional): Returned if `return_df=True`.

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
        Note: ``half_width`` sets the half-length of the mean line.

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
            namelist=["Gapdh", "Vcp", "Ahnak"],
            classes="condition",
            plot_type="box",
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        Bar plots with error bars:
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["Gapdh", "Vcp", "Ahnak"],
            classes="condition",
            plot_type="bar",
            bar_error="sd",  # "sd", "sem", None, or callable
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        Mean-lines with count annotations:
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["Gapdh", "Vcp", "Ahnak"],
            classes="condition",
            plot_type="line",
            show_n=True,
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        Violin plots (distribution-focused):
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["Gapdh", "Vcp", "Ahnak"],
            classes="condition",
            plot_type="violin",
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        Customizing appearance (palette, order, and styling):
        ```python
        palette = {"Control": "#4C72B0", "Treatment": "#DD8452"}

        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["Gapdh", "Vcp", "Ahnak"],
            classes="condition",
            order=["Control", "Treatment"],
            palette=palette,
            plot_type="box",
            box_kwargs={"boxprops": {"alpha": 0.45}, "linewidth": 1.2},
            strip_kwargs={"size": 4, "alpha": 0.6},
            y_min=2,
            y_max=10,
            log_scale=True,
            figsize=(2, 2.5),
        )
        plt.show()
        ```

        Return the plotting DataFrame for downstream checks:
        ```python
        fig, axes, df = pdata.plot_abundance_boxgrid(
            namelist=["Gapdh", "Vcp"],
            classes="condition",
            plot_type="box",
            return_df=True,
        )

        display(df.head())
        plt.show()
        ```
    """
    from matplotlib.colors import to_rgba

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

    # --- normalize classes (list/tuple -> df["class"]) ---
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

    if return_df:
        return fig,axes,df
    else:
        return fig, axes

def plot_pca(ax, pdata, color=None, edge_color=None, marker_shape=None, classes=None, 
             layer="X", on='protein', cmap='default', edge_cmap="default", shape_cmap="default", edge_lw=0.8,
             s=20, alpha=.8, plot_pc=[1, 2], pca_params=None, subset_mask=None,
             force=False, basis='X_pca', text_size=9, show_labels=False, label_column=None,
             add_ellipses=False, ellipse_group=None, ellipse_cmap='default', ellipse_kwargs=None, 
             return_fit=False, **kwargs):
    """
    Plot principal component analysis (PCA) of protein or peptide abundance.

    Computes (or reuses) PCA coordinates and plots samples in 2D or 3D, with
    flexible styling via face color (`color`), edge color (`edge_color`), marker
    shapes (`marker_shape`), labels, and optional confidence ellipses.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on. Must be 3D if plotting 3 PCs.
        pdata (pAnnData): Input pAnnData object with `.prot`, `.pep`, and `.summary`.

        color (str or list of str or None): Face coloring for points.

            - None: grey face color for all points.
            - str: an `.obs` key (categorical or continuous) OR a gene/protein identifier
              (continuous abundance coloring).
            - list of str: combine multiple `.obs` keys into a single categorical label
              (e.g., `["cellline", "treatment"]`).

        edge_color (str or list of str or None): Edge coloring for points (categorical only).

            - None: no edge coloring (edges disabled).
            - str: an `.obs` key (categorical).
            - list of str: combine multiple `.obs` keys into a single categorical label.

        marker_shape (str or list of str or None): Marker shapes for points (categorical only).

            - None: use a single marker (`"o"`).
            - str: an `.obs` key (categorical).
            - list of str: combine multiple `.obs` keys into a single categorical label.

        classes (str or list of str or None): Deprecated alias for `color`.

            - If `classes` is provided and `color` is None, `classes` is used as `color`.
            - If both are provided, `color` is used and `classes` is ignored.

        layer (str): Data layer to use (default: `"X"`).
        on (str): Data level to plot, either `"protein"` or `"peptide"` (default: `"protein"`).

        cmap (str, list, or dict): Palette/colormap for face coloring (`color`).

            - `"default"`: uses internal `get_color()` scheme for categorical coloring and
              defaults to a standard continuous colormap for abundance coloring.
            - list: list of colors assigned to class labels in sorted order (categorical).
            - dict: `{label: color}` mapping (categorical).
            - str / colormap: continuous colormap name/object (abundance).

        edge_cmap (str, list, or dict): Palette for edge coloring (`edge_color`, categorical only).

            - `"default"`: internal categorical palette via `get_color()`.
            - list: colors assigned to sorted class labels.
            - dict: `{label: color}` mapping.

        shape_cmap (str, list, or dict): Marker mapping for `marker_shape` (categorical only).

            - `"default"`: cycles markers in this order:
              `["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]`
            - list: markers assigned to sorted class labels.
            - dict: `{label: marker}` mapping.

        edge_lw (float): Edge linewidth when `edge_color` is used (default: 0.8).
        s (float): Marker size (default: 20).
        alpha (float): Marker opacity (default: 0.8).

        plot_pc (list of int): Principal components to plot, e.g. `[1, 2]` or `[1, 2, 3]`.
        pca_params (dict, optional): Additional parameters for the PCA computation.
        subset_mask (array-like or pandas.Series, optional): Boolean mask to subset samples.
            If a Series is provided, it will be aligned to `adata.obs.index`.
        force (bool): If True, recompute PCA even if cached.
        basis (str): PCA basis in `adata.obsm` (default: `"X_pca"`). Alternative bases
            (e.g., `"X_pca_harmony"`) may be available after running Harmony or other methods.

        text_size (int): Font size for axis labels and legends (default: 9).
        show_labels (bool or list): Whether to label points.

            - False: no labels.
            - True: label all samples.
            - list: label only specified samples.

        label_column (str, optional): Column in `pdata.summary` to use for labels when
            `show_labels=True`. If not provided, sample names are used.

        add_ellipses (bool): If True, overlay confidence ellipses per group (2D only).
        ellipse_group (str or list of str, optional): Explicit `.obs` key(s) to group ellipses.
            If None, grouping is chosen by priority:

            1. categorical `color`
            2. `edge_color`
            3. `marker_shape`
            4. otherwise raises ValueError

        ellipse_cmap (str, list, or dict): Ellipse color mapping.

            - `"default"`: if grouping uses categorical `color` or `edge_color`, ellipses reuse
              those colors; if grouping uses `marker_shape`, ellipses use `get_color()`.
            - list: colors assigned to sorted group labels.
            - dict: `{label: color}` mapping.
            - str: matplotlib colormap name (used to generate a palette across groups).

        ellipse_kwargs (dict, optional): Extra keyword arguments passed to the ellipse patch
            (e.g., `{"alpha": 0.12, "lw": 1.5}`).

        return_fit (bool): If True, also return the fitted PCA object.
        **kwargs: Extra keyword arguments passed to `ax.scatter()`.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the PCA scatter plot.
        pca (sklearn.decomposition.PCA): The fitted PCA object (only if `return_fit=True`).

    Raises:
        AssertionError: If 3 PCs are requested and `ax` is not 3D.
        ValueError: If `edge_color` is continuous (use `color=` for abundance instead).
        ValueError: If `marker_shape` is not a categorical `.obs` key.
        ValueError: If `add_ellipses=True` but no categorical grouping is available.

    Note:
        - `edge_color` and `marker_shape` are categorical only.
        - If `color` is continuous (abundance), a colorbar is shown automatically.
        - Use `classes=` only for backwards compatibility; prefer `color=`.
        - PCA results are cached in `pdata.uns["pca"]` and reused across plotting calls.
        - To force recalculation (e.g., after filtering or normalization), set `force=True`.

    Example:
        Basic usage in grey:
            ```python
            plot_pca(ax, pdata)
            ```

        Face color by a categorical `.obs` key:
            ```python
            plot_pca(ax, pdata, color="treatment")
            ```

        Combine multiple `.obs` keys into one categorical label:
            ```python
            plot_pca(ax, pdata, color=["cellline", "treatment"])
            ```

        Face color by gene/protein abundance (continuous) with a matplotlib colormap:
            ```python
            plot_pca(ax, pdata, color="UBE4B", cmap="plasma")
            ```

        Face color and edge color by different categorical keys with a custom palette:
            ```python
            edge_palette = {"A": "#3627E0", "B": "#F61B0F"}
            plot_pca(ax, pdata, color="condition", edge_color="group", edge_cmap=edge_palette, edge_lw=1.5)
            ```

        Marker shapes by a categorical key:
            ```python
            shape_map = {"WT": "o", "MUT": "s"}
            plot_pca(ax, pdata, color="treatment", marker_shape="genotype", shape_cmap=shape_map)
            ```

        Add ellipses (auto-grouping by categorical `color`):
            ```python
            plot_pca(ax, pdata, color="treatment", add_ellipses=True)
            ```

        Add ellipses grouped explicitly (and force ellipse colors):
            ```python
            ellipse_colors = {"WT": "#000000", "MUT": "#377EB8"}
            plot_pca(
                ax, pdata,
                color="UBE4B", cmap="viridis",
                marker_shape="genotype",
                add_ellipses=True,
                ellipse_group="genotype",
                ellipse_cmap=ellipse_colors,
                ellipse_kwargs={"alpha": 0.10, "lw": 1.5},
            )
            ```

        Label all samples (using a custom label column if present):
            ```python
            plot_pca(ax, pdata, color="treatment", show_labels=True, label_column="short_name")
            ```
    """
    
    # Validate PCA dimensions
    assert isinstance(plot_pc, list) and len(plot_pc) in [2, 3], "plot_pc must be a list of 2 or 3 PCs."
    if len(plot_pc) == 3:
        assert ax.name == '3d', "3 PCs requested — ax must be a 3D projection"

    pc_x, pc_y = plot_pc[0] - 1, plot_pc[1] - 1
    pc_z = plot_pc[2] - 1 if len(plot_pc) == 3 else None

    # check deprecated classes argument
    if classes is not None and color is None:
        print(f"{utils.format_log_prefix('warn')} `classes` is deprecated; use `color=` instead.")
        color = classes
    elif classes is not None and color is not None:
        print(f"{utils.format_log_prefix('warn')} Both `classes` and `color` were provided; using `color` and ignoring `classes`.")

    adata = utils.get_adata(pdata, on)

    default_pca_params = {'n_comps': min(len(adata.obs_names), len(adata.var_names)) - 1, 'random_state': 42}
    user_params = dict(pca_params or {})

    # accept n_components OR n_comps
    if "n_components" in user_params and "n_comps" not in user_params:
        user_params["n_comps"] = user_params.pop("n_components")
    else:
        user_params.pop("n_components", None) 
    pca_param = {**default_pca_params, **user_params}

    if basis != "X_pca":
        # User-specified alternative basis (e.g. Harmony, ICA)
        if basis not in adata.obsm:
            raise KeyError(f"{utils.format_log_prefix('error',2)} Custom PCA basis '{basis}' not found in adata.obsm.")
    else:
        # Standard PCA case
        if "X_pca" not in adata.obsm or force:
            print(f"{utils.format_log_prefix('info')} Computing PCA (force={force})...")
            pdata.pca(on=on, layer=layer, **pca_param)
        else:
            print(f"{utils.format_log_prefix('info')} Using existing PCA embedding.")

    # --- Select PCA basis for plotting ---
    X_pca = adata.obsm[basis] if basis in adata.obsm else adata.obsm["X_pca"]
    pca = adata.uns["pca"]

    # subset if requested
    mask = _resolve_subset_mask(adata, subset_mask)
    obs_names_plot = adata.obs_names[mask]
    pc_idx = [pc_x, pc_y] if len(plot_pc) == 2 else [pc_x, pc_y, pc_z]

    # build PCA-specific axis labels with variance %
    var = pca["variance_ratio"]
    dim_labels = [
        f"PC{pc_x+1} ({var[pc_x]*100:.2f}%)",
        f"PC{pc_y+1} ({var[pc_y]*100:.2f}%)",
    ]
    if len(pc_idx) == 3:
        dim_labels.append(f"PC{pc_z+1} ({var[pc_z]*100:.2f}%)")

    # label series for show_labels
    if label_column and label_column in pdata.summary.columns:
        label_series = pdata.summary.loc[obs_names_plot, label_column]
    else:
        label_series = obs_names_plot

    ax = _plot_embedding_scatter(ax=ax, adata=adata, Xt=X_pca, mask=mask, obs_names_plot=obs_names_plot,
        color=color, edge_color=edge_color, marker_shape=marker_shape, layer=layer,
        cmap=cmap, edge_cmap=edge_cmap, shape_cmap=shape_cmap, edge_lw=edge_lw, s=s, alpha=alpha, text_size=text_size,
        axis_prefix="PC", dim_labels=dim_labels, pc_idx=pc_idx, 
        show_labels=show_labels, label_series=label_series, add_ellipses=add_ellipses, ellipse_kwargs=ellipse_kwargs, ellipse_group=ellipse_group, ellipse_cmap=ellipse_cmap,
        plot_confidence_ellipse=_plot_confidence_ellipse,
        **kwargs,
    )

    if return_fit:
        return ax, pca
    else:
        return ax

def resolve_plot_colors(adata, classes, cmap, layer="X"):
    """
    Resolve colors for PCA or abundance plots.

    This helper function determines how samples should be colored in plotting
    functions based on categorical or continuous class values. It returns mapped
    color values, a colormap (if applicable), and legend handles.

    Args:
        adata (anndata.AnnData): AnnData object (protein or peptide level).
        classes (str): Class used for coloring. Can be:
            
            - An `.obs` column name (categorical or continuous).
            - A gene or protein identifier, in which case coloring is based
              on abundance values from the specified `layer`.

        cmap (str, list, or matplotlib colormap): Colormap to use.
            
            - `"default"`: uses `get_color()` scheme.
            - list of colors: categorical mapping.
            - colormap name or object: continuous mapping.

        layer (str): Data layer to extract abundance values from when `classes`
            is a gene/protein. Default is `"X"`.

    Returns:      
        color_mapped (array-like): Values mapped to colors for plotting.
        cmap_resolved (matplotlib colormap or None): Colormap object for continuous coloring; None if categorical.
        legend_elements (list or None): Legend handles for categorical coloring; None if continuous.

    """
    legend_elements = None

    # Case 1: No coloring, all grey
    if classes is None:
        color_mapped = ['grey'] * len(adata)
        legend_elements = [mpatches.Patch(color='grey', label='All samples')]
        return color_mapped, None, legend_elements

    # Case 2: Single categorical column from obs
    elif isinstance(classes, str) and classes in adata.obs.columns:
        y = utils.get_samplenames(adata, classes)
        class_labels = sorted(set(y))
        if cmap == 'default':
            palette = get_color('colors', n=len(class_labels))
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, list):
            color_dict = {c: cmap[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, dict):
            color_dict = cmap
        else:
            cmap_obj = cm.get_cmap(cmap)
            palette = [mcolors.to_hex(cmap_obj(i / max(len(class_labels) - 1, 1))) for i in range(len(class_labels))]
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        color_mapped = [color_dict[val] for val in y]
        legend_elements = [mpatches.Patch(color=color_dict[c], label=c) for c in class_labels]
        return color_mapped, None, legend_elements

    # Case 3: Multiple categorical columns from obs (combined class)
    elif isinstance(classes, list) and all(c in adata.obs.columns for c in classes):
        y = utils.get_samplenames(adata, classes)
        class_labels = sorted(set(y))
        if cmap == 'default':
            palette = get_color('colors', n=len(class_labels))
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, list):
            color_dict = {c: cmap[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, dict):
            color_dict = cmap
        else:
            cmap_obj = cm.get_cmap(cmap)
            palette = [mcolors.to_hex(cmap_obj(i / max(len(class_labels) - 1, 1))) for i in range(len(class_labels))]
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        color_mapped = [color_dict[val] for val in y]
        legend_elements = [mpatches.Patch(color=color_dict[c], label=c) for c in class_labels]
        return color_mapped, None, legend_elements

    # Case 4: Continuous coloring by protein abundance (accession)
    elif isinstance(classes, str) and classes in adata.var_names:
        X = adata.layers[layer] if layer in adata.layers else adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        idx = list(adata.var_names).index(classes)
        color_mapped = X[:, idx]
        if cmap == 'default':
            cmap = 'viridis'
        cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

        # Add default colorbar handling for abundance-based coloring
        norm = mcolors.Normalize(vmin=color_mapped.min(), vmax=color_mapped.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # required for colorbar

        return color_mapped, cmap, None

    # Case 5: Gene name (mapped to accession)
    elif isinstance(classes, str):
        if "Genes" in adata.var.columns:
            gene_map = adata.var["Genes"].to_dict()
            match = [acc for acc, gene in gene_map.items() if gene == classes]
            if match:
                return resolve_plot_colors(adata, match[0], cmap, layer)
        raise ValueError("Invalid classes input. Must be None, a protein in var_names, or an obs column/list.")

    else:
        raise ValueError("Invalid input. List input supports only classes ([class1, class2]), and string supports classes or protein accession or gene name. ")

def resolve_marker_shapes(adata, marker_shape, shape_cmap="default"):
    """
    Resolve marker shapes for categorical sample groupings.

    Args:
        adata (anndata.AnnData): AnnData object.
        marker_shape (str, list of str, or None): `.obs` column(s) used to assign markers.
            - None: return None (use a single marker).
            - str: categorical `.obs` key.
            - list: combine multiple `.obs` columns into a single categorical label.
        shape_cmap (str, list, or dict): Marker assignment.
            - "default": uses an internal default marker list.
            - list: markers assigned to sorted class labels.
            - dict: {label: marker} mapping.

    Returns:
        markers (list[str] or None): Marker per observation (len = n_obs), or None.
        shape_legend (list[Line2D] or None): Legend handles for marker shapes.
        shape_map (dict or None): Mapping {label: marker}.
    """
    if marker_shape is None:
        return None, None, None

    # only allow categorical `.obs`
    if isinstance(marker_shape, str) and marker_shape in adata.obs.columns:
        labels = utils.get_samplenames(adata, marker_shape)
    elif isinstance(marker_shape, list) and all(c in adata.obs.columns for c in marker_shape):
        labels = utils.get_samplenames(adata, marker_shape)
    else:
        raise ValueError("marker_shape must be an `.obs` categorical key (str) or list of keys.")

    class_labels = sorted(set(labels))

    if shape_cmap == "default":
        marker_list = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]
        shape_map = {c: marker_list[i % len(marker_list)] for i, c in enumerate(class_labels)}
        if len(class_labels) > len(marker_list):
            print(f"{utils.format_log_prefix('warn')} marker_shape has {len(class_labels)} levels; cycling markers.")
    elif isinstance(shape_cmap, list):
        shape_map = {c: shape_cmap[i % len(shape_cmap)] for i, c in enumerate(class_labels)}
    elif isinstance(shape_cmap, dict):
        shape_map = dict(shape_cmap)
    else:
        raise ValueError("shape_cmap must be 'default', a list of markers, or a dict mapping labels to markers.")

    markers = [shape_map[v] for v in labels]

    shape_legend = [
        mlines.Line2D(
            [], [], linestyle="none",
            marker=shape_map[c],
            color="black",  # neutral legend handle
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label=str(c),
        )
        for c in class_labels
    ]

    return markers, shape_legend, shape_map

def _build_scatter_kwargs(
    *,
    base_kwargs,
    color_key,
    face_plot,
    face_all,
    face_cmap_resolved,
    edge_key,
    edge_plot,
    edge_all,
    edge_lw,
    n
):
    """
    Build kwargs for a single ax.scatter call (already subset-aligned).

    Returns:
        dict: kwargs for ax.scatter
    """
    kw = dict(base_kwargs)

    # Face
    if color_key is None:
        kw.setdefault("c", ["grey"] * n)
        kw.pop("cmap", None)
    else:
        kw["c"] = face_plot
        kw["cmap"] = face_cmap_resolved

    # Edge
    if edge_key is None:
        kw["edgecolors"] = "none"
    else:
        kw["edgecolors"] = edge_plot
        kw["linewidths"] = edge_lw

    return kw

def _slice_scatter_kwargs(scatter_kwargs, m):
    """
    Slice array-like scatter kwargs for a subset mask m (boolean array).

    Returns:
        dict: kwargs safe to pass to ax.scatter for that group.
    """
    kw = dict(scatter_kwargs)

    # Slice face colors if array-like
    if "c" in kw and isinstance(kw["c"], (list, np.ndarray)):
        c = np.asarray(kw["c"])
        if c.shape[0] == m.shape[0]:
            kw["c"] = c[m]

    # Slice edgecolors safely:
    # - edgecolors can be "none" / None / a single color string / an array of colors
    if "edgecolors" in kw:
        ec = kw["edgecolors"]

        # scalar "off" cases
        if ec is None or (isinstance(ec, str) and ec == "none"):
            return kw

        # scalar single color string (e.g. "black") -> do not slice
        if isinstance(ec, str):
            return kw

        # array-like -> slice if aligned
        ec_arr = np.asarray(ec)
        if ec_arr.shape[0] == m.shape[0]:
            kw["edgecolors"] = ec_arr[m]

    return kw

def _resolve_subset_mask(adata, subset_mask):
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

def _add_continuous_colorbar(ax, color_mapped, cmap_resolved, label, text_size=10):
    """Attach a colorbar if using continuous coloring."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np

    if cmap_resolved is None or color_mapped is None:
        return

    vals = np.asarray(color_mapped)
    if vals.size == 0:
        return
    if np.issubdtype(vals.dtype, np.number) is False:
        return
    
    norm = mcolors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_resolved)
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label(label, fontsize=text_size)

def _legend_title_from_key(key):
    if key is None:
        return None
    if isinstance(key, list):
        return "/".join(str(c).capitalize() for c in key)
    return str(key).capitalize()

def _plot_confidence_ellipse(x, y, ax, n_std=2.4477, facecolor='none', edgecolor='black', alpha=0.2, **kwargs):
    from matplotlib.patches import Ellipse
    
    if x.size <= 2:
        return
    cov = np.cov(x, y)
    if np.linalg.matrix_rank(cov) < 2:
        return
    mean_x, mean_y = np.mean(x), np.mean(y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Only set defaults if user didn't provide them
    kwargs.setdefault("lw", 1.5)
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                        facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs)
    ax.add_patch(ellipse)

def _plot_embedding_scatter(*, ax, adata, Xt, mask, obs_names_plot,
    color=None, edge_color=None, marker_shape=None, layer="X",
    cmap="default", edge_cmap="default", shape_cmap="default",
    edge_lw=0.8, s=20, alpha=0.8, text_size=10,
    # embedding metadata
    axis_prefix="UMAP",              # "UMAP" or "PC"
    dim_labels=None,                 # list[str] length = n_dim plotted
    pc_idx=None,                     # list[int] of columns in Xt to plot, e.g. [0,1] or [0,1,2]
    # optional “1D embedding” support
    y_1d=None,                       # np.ndarray length n_plot, if 1D: y positions
    # optional extras
    show_labels=False,
    label_series=None,               # pd.Series indexed by obs_names_plot, or list-like aligned to obs_names_plot
    add_ellipses=False, ellipse_kwargs=None, ellipse_group=None, ellipse_cmap="default", 
    plot_confidence_ellipse=None,    # function(x, y, ax, **kwargs)
    return_parts=False,
    **kwargs,
):
    """
    Shared scatter rendering for UMAP/PCA-like embeddings.

    Notes:
      - color can be categorical or continuous (abundance), via resolve_plot_colors
      - edge_color is categorical only
      - marker_shape is categorical only (marker splitting)
    """
    # resolve colors & marker
    face_mapped, face_cmap_resolved, face_legend = resolve_plot_colors(
        adata, color, cmap, layer=layer
    )

    edge_mapped = None
    edge_legend = None
    if edge_color is not None:
        edge_mapped, edge_cmap_resolved, edge_legend = resolve_plot_colors(
            adata, edge_color, edge_cmap, layer=layer
        )
        if edge_cmap_resolved is not None:
            raise ValueError(
                "edge_color does not support continuous (abundance) coloring. "
                "Use `color=` for abundance-based coloring instead."
            )

    markers_all, shape_legend, _ = resolve_marker_shapes(
        adata, marker_shape, shape_cmap=shape_cmap
    )

    # plot
    Xt_plot = Xt[mask]
    n_plot = Xt_plot.shape[0]

    face_plot = None if face_mapped is None else np.asarray(face_mapped)[mask]
    edge_plot = None if edge_mapped is None else np.asarray(edge_mapped)[mask]
    markers_plot = None if markers_all is None else np.asarray(markers_all)[mask]

    base_kwargs = dict(s=s, alpha=alpha, **kwargs)

    scatter_kwargs = _build_scatter_kwargs(
        base_kwargs=base_kwargs,
        color_key=color,
        face_plot=face_plot,
        face_all=face_mapped,
        face_cmap_resolved=face_cmap_resolved,
        edge_key=edge_color,
        edge_plot=edge_plot,
        edge_all=edge_mapped,
        edge_lw=edge_lw,
        n=n_plot,
    )

    # choose plotted dimensions
    if pc_idx is None:
        pc_idx = list(range(min(2, Xt_plot.shape[1])))

    n_dim = len(pc_idx)
    if n_dim not in (1, 2, 3):
        raise ValueError("pc_idx must specify 1, 2, or 3 dimensions.")

    # draw scatter (with marker splitting if requested)
    def _scatter_call(m, marker=None):
        kw = _slice_scatter_kwargs(scatter_kwargs, m)
        if n_dim == 1:
            x = Xt_plot[m, pc_idx[0]]
            y = y_1d[m] if y_1d is not None else np.arange(n_plot)[m]
            ax.scatter(x, y, marker=marker or "o", **kw)
        elif n_dim == 2:
            ax.scatter(
                Xt_plot[m, pc_idx[0]],
                Xt_plot[m, pc_idx[1]],
                marker=marker or "o",
                **kw,
            )
        else:
            ax.scatter(
                Xt_plot[m, pc_idx[0]],
                Xt_plot[m, pc_idx[1]],
                Xt_plot[m, pc_idx[2]],
                marker=marker or "o",
                **kw,
            )

    if markers_plot is None:
        _scatter_call(np.ones(n_plot, dtype=bool), marker=None)
    else:
        for mk in np.unique(markers_plot):
            m = (markers_plot == mk)
            _scatter_call(m, marker=mk)

    # axes labels
    if dim_labels is None:
        if axis_prefix.upper() == "PC":
            dim_labels = [f"PC{i+1}" for i in pc_idx]
        else:
            dim_labels = [f"{axis_prefix} {i+1}" for i in range(n_dim)]

    ax.set_xlabel(dim_labels[0], fontsize=text_size)
    if n_dim >= 2:
        ax.set_ylabel(dim_labels[1], fontsize=text_size)
    if n_dim == 3:
        ax.set_zlabel(dim_labels[2], fontsize=text_size)

    # colorbar for continuous face coloring
    _add_continuous_colorbar(
        ax,
        np.asarray(face_mapped) if face_mapped is not None else None,
        face_cmap_resolved,
        label=_legend_title_from_key(color) or "Abundance",
        text_size=text_size,
    )

    # ellipses (2D only)
    def _is_obs_key(key):
        if key is None:
            return False
        keys = key if isinstance(key, list) else [key]
        return all((k in adata.obs.columns) for k in keys)

    def _is_color_categorical(face_cmap_resolved, color_key):
        # categorical iff key is obs-based and resolve_plot_colors returned cmap_resolved=None
        return (color_key is not None) and _is_obs_key(color_key) and (face_cmap_resolved is None)

    def _resolve_group_labels(group_key):
        # group_key guaranteed categorical obs key
        return utils.get_samplenames(adata, group_key)

    def _resolve_ellipse_color_map(group_labels, *, group_source, ellipse_cmap):
        """
        group_source: one of {"color", "edge_color", "marker_shape"} based on where group_key came from
        ellipse_cmap: "default" | list | dict
        Returns: dict {label: hexcolor}
        """
        class_labels = sorted(set(group_labels))

        # B(1) explicit ellipse_cmap wins
        if isinstance(ellipse_cmap, dict):
            return dict(ellipse_cmap)
        if isinstance(ellipse_cmap, list):
            return {c: ellipse_cmap[i % len(ellipse_cmap)] for i, c in enumerate(class_labels)}
        if ellipse_cmap != "default":
            # treat non-default string as a matplotlib cmap name
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            cmap_obj = cm.get_cmap(ellipse_cmap)
            pal = [mcolors.to_hex(cmap_obj(i / max(len(class_labels) - 1, 1))) for i in range(len(class_labels))]
            return {c: pal[i] for i, c in enumerate(class_labels)}

        # B(2) default behavior depends on grouping source
        if group_source == "color":
            # color is categorical => face_mapped already is concrete colors per obs
            # build dict from first occurrence
            m = {}
            for lab, colr in zip(group_labels, face_mapped):
                if lab not in m:
                    m[lab] = colr
            return m

        if group_source == "edge_color":
            m = {}
            for lab, colr in zip(group_labels, edge_mapped):
                if lab not in m:
                    m[lab] = colr
            return m

        # group_source == "marker_shape": choose a default palette
        pal = get_color("colors", n=len(class_labels))
        return {c: pal[i] for i, c in enumerate(class_labels)}

    if add_ellipses:
        if n_dim != 2:
            print(f"{utils.format_log_prefix('warn')} add_ellipses=True is only supported for 2D embeddings.")
        else:
            if plot_confidence_ellipse is None:
                raise ValueError("plot_confidence_ellipse must be provided when add_ellipses=True.")

            # A) choose grouping key (priority)
            group_source = None
            group_key = ellipse_group

            if group_key is not None:
                if not _is_obs_key(group_key):
                    raise ValueError("ellipse_group must be a categorical `.obs` key (str or list of str).")
                group_source = "ellipse_group"  # treated as explicit; colors handled by ellipse_cmap/default logic
            else:
                if _is_color_categorical(face_cmap_resolved, color):
                    group_key = color
                    group_source = "color"
                elif edge_color is not None and _is_obs_key(edge_color):
                    group_key = edge_color
                    group_source = "edge_color"
                elif marker_shape is not None and _is_obs_key(marker_shape):
                    group_key = marker_shape
                    group_source = "marker_shape"
                else:
                    raise ValueError(
                        "add_ellipses=True requires a categorical grouping. Provide `ellipse_group`, or set "
                        "`color` to a categorical `.obs` key, or provide `edge_color` / `marker_shape` as categorical."
                    )

            # labels per obs (full + plot subset)
            y_all = _resolve_group_labels(group_key)
            y_plot = np.asarray(y_all)[mask]

            df_coords = pd.DataFrame(
                Xt_plot[:, [pc_idx[0], pc_idx[1]]],
                columns=["D1", "D2"],
                index=obs_names_plot,
            )
            df_coords["class"] = y_plot

            # B) resolve ellipse colors
            # if group_key was user-explicit ellipse_group, treat it like "marker_shape" for default palette behavior
            color_source = group_source
            if color_source == "ellipse_group":
                # if ellipse_group equals color or edge_color, reuse those when default
                if group_key == color and _is_color_categorical(face_cmap_resolved, color):
                    color_source = "color"
                elif group_key == edge_color:
                    color_source = "edge_color"
                else:
                    color_source = "marker_shape"

            ellipse_color_map = _resolve_ellipse_color_map(
                y_all,  # use full set to define levels consistently
                group_source=color_source,
                ellipse_cmap=ellipse_cmap,
            )

            e_kwargs = ellipse_kwargs.copy() if ellipse_kwargs else {}

            for cls in df_coords["class"].unique():
                sub = df_coords[df_coords["class"] == cls]
                if sub.shape[0] < 3:
                    continue

                c0 = ellipse_color_map.get(cls, "black")
                k = e_kwargs.copy()
                k.setdefault("facecolor", c0)
                k.setdefault("edgecolor", c0)

                plot_confidence_ellipse(sub["D1"].values, sub["D2"].values, ax=ax, **k)

    # --- labels ---
    if show_labels:
        show_set = set(show_labels) if isinstance(show_labels, list) else set(obs_names_plot)
        if label_series is None:
            label_series = obs_names_plot
        # normalize label_series to something indexable by obs name
        if hasattr(label_series, "loc"):
            lab = label_series
        else:
            lab = pd.Series(list(label_series), index=obs_names_plot)

        for i, sample in enumerate(obs_names_plot):
            if sample not in show_set:
                continue
            label = lab.loc[sample]
            if n_dim == 1:
                x = Xt_plot[i, pc_idx[0]]
                y = (y_1d[i] if y_1d is not None else i)
                ax.text(x, y, str(label), fontsize=8)
            elif n_dim == 2:
                ax.text(Xt_plot[i, pc_idx[0]], Xt_plot[i, pc_idx[1]], str(label),
                        fontsize=8, ha="right", va="bottom")
            else:
                ax.text(Xt_plot[i, pc_idx[0]], Xt_plot[i, pc_idx[1]], Xt_plot[i, pc_idx[2]],
                        str(label), fontsize=8)

    # --- legends (keep separate) ---
    if face_legend:
        leg1 = ax.legend(
            handles=face_legend,
            title=_legend_title_from_key(color),
            loc="best",
            fontsize=text_size,
            frameon=False,
        )
        ax.add_artist(leg1)

    if edge_legend:
        leg2 = ax.legend(
            handles=edge_legend,
            title=_legend_title_from_key(edge_color),
            loc="upper right",
            fontsize=text_size,
            frameon=False,
        )
        ax.add_artist(leg2)

    if shape_legend:
        leg3 = ax.legend(
            handles=shape_legend,
            title=_legend_title_from_key(marker_shape),
            loc="lower right",
            fontsize=text_size,
            frameon=False,
        )
        ax.add_artist(leg3)

    if return_parts:
        return {
            "face_mapped": face_mapped,
            "face_cmap_resolved": face_cmap_resolved,
            "edge_mapped": edge_mapped,
            "markers_all": markers_all,
        }

    return ax

# NOTE: STRING enrichment plots live in enrichment.py, not here.
# This function is re-documented here for discoverability.
def plot_enrichment_svg(*args, **kwargs):
    """
    Plot STRING enrichment results as an SVG figure.

    This is a wrapper that redirects to the implementation in `enrichment.py`
    for convenience and discoverability.

    Args:
        *args: Positional arguments passed to `scpviz.enrichment.plot_enrichment_svg`.
        **kwargs: Keyword arguments passed to `scpviz.enrichment.plot_enrichment_svg`.

    Returns:
        svg (SVG): SVG figure object.

    See Also:
        scpviz.enrichment.plot_enrichment_svg
    """
    from .enrichment import plot_enrichment_svg as actual_plot
    return actual_plot(*args, **kwargs)

def plot_umap(ax, pdata, color=None, edge_color=None, marker_shape=None, classes = None, 
              layer = "X", on = 'protein', cmap='default', edge_cmap="default", shape_cmap="default", show_labels=False, label_column=None,
              s=20, alpha=.8, umap_params={}, text_size = 10, edge_lw=0.8, 
              add_ellipses=False, ellipse_group=None, ellipse_cmap='default', ellipse_kwargs=None, 
              force = False, return_fit=False, subset_mask=None, **kwargs):
    """
    Plot UMAP projection of protein or peptide abundance data.

    Computes (or reuses) a UMAP embedding and visualizes samples in 1D/2D/3D, with
    flexible styling via face color (`color`), edge color (`edge_color`), marker
    shapes (`marker_shape`), labels, and optional confidence ellipses.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on. Must be 3D if `n_components=3`.
        pdata (scpviz.pAnnData): The pAnnData object containing `.prot`, `.pep`, and `.summary`.

        color (str or list of str or None): Face coloring for points.

            - None: grey face color for all points.
            - str: an `.obs` key (categorical or continuous) OR a gene/protein identifier
              (continuous abundance coloring).
            - list of str: combine multiple `.obs` keys into a single categorical label
              (e.g., `["cellline", "treatment"]`).

        edge_color (str or list of str or None): Edge coloring for points (categorical only).

            - None: no edge coloring (edges disabled).
            - str: an `.obs` key (categorical).
            - list of str: combine multiple `.obs` keys into a single categorical label.

        marker_shape (str or list of str or None): Marker shapes for points (categorical only).

            - None: use a single marker (`"o"`).
            - str: an `.obs` key (categorical).
            - list of str: combine multiple `.obs` keys into a single categorical label.

        classes (str or list of str or None): Deprecated alias for `color`.

            - If `classes` is provided and `color` is None, `classes` is used as `color`.
            - If both are provided, `color` is used and `classes` is ignored.

        layer (str): Data layer to use for UMAP input (default: `"X"`).
        on (str): Whether to use `"protein"` or `"peptide"` data (default: `"protein"`).

        cmap (str, list, or dict): Palette/colormap for face coloring (`color`).

            - `"default"`: internal categorical palette via `get_color()`; for continuous
              abundance coloring, uses a standard continuous colormap.
            - list: colors assigned to sorted class labels (categorical).
            - dict: `{label: color}` mapping (categorical).
            - str / colormap: continuous colormap name/object (abundance).

        edge_cmap (str, list, or dict): Palette for edge coloring (`edge_color`, categorical only).

            - `"default"`: internal categorical palette via `get_color()`.
            - list: colors assigned to sorted class labels.
            - dict: `{label: color}` mapping.

        shape_cmap (str, list, or dict): Marker mapping for `marker_shape` (categorical only).

            - `"default"`: cycles markers in this order:
              `["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]`
            - list: markers assigned to sorted class labels.
            - dict: `{label: marker}` mapping.

        show_labels (bool or list): Whether to label points.

            - False: no labels.
            - True: label all samples.
            - list: label only specified samples.

        label_column (str, optional): Column in `pdata.summary` to use for labels when
            `show_labels=True`. If not provided, sample names are used.

        s (float): Marker size (default: 20).
        alpha (float): Marker opacity (default: 0.8).

        umap_params (dict, optional): Parameters for UMAP computation. Common keys:

            - `n_components` (default: 2)
            - `n_neighbors`
            - `min_dist`
            - `metric`
            - `spread`
            - `random_state` (default: 42)
            - `n_pcs` (neighbors step)

        subset_mask (array-like or pandas.Series, optional): Boolean mask to subset samples.
            If a Series is provided, it will be aligned to `adata.obs.index`.

        text_size (int): Font size for axis labels and legends (default: 10).
        edge_lw (float): Edge linewidth when `edge_color` is used (default: 0.8).

        add_ellipses (bool): If True, overlay confidence ellipses per group (2D only).
        ellipse_group (str or list of str, optional): Explicit `.obs` key(s) to group ellipses.
            If None, grouping is chosen by priority:

            1. categorical `color`
            2. `edge_color`
            3. `marker_shape`
            4. otherwise raises ValueError

        ellipse_cmap (str, list, or dict): Ellipse color mapping.

            - `"default"`: if grouping uses categorical `color` or `edge_color`, ellipses reuse
              those colors; if grouping uses `marker_shape`, ellipses use `get_color()`.
            - list: colors assigned to sorted group labels.
            - dict: `{label: color}` mapping.
            - str: matplotlib colormap name (used to generate a palette across groups).

        ellipse_kwargs (dict, optional): Extra keyword arguments passed to the ellipse patch.

        force (bool): If True, recompute UMAP even if cached.
        return_fit (bool): If True, return the fitted UMAP object.
        **kwargs: Extra keyword arguments passed to `ax.scatter()`.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the UMAP plot.
        fit_umap (umap.UMAP): The fitted UMAP object (only if `return_fit=True`).

    Raises:
        AssertionError: If `n_components=3` and the axis is not 3D.
        ValueError: If `edge_color` is continuous (use `color=` for abundance instead).
        ValueError: If `marker_shape` is not a categorical `.obs` key.
        ValueError: If `add_ellipses=True` but no categorical grouping is available.

    Note:
        - If `color` is continuous (abundance), a colorbar is shown automatically.
        - `edge_color` and `marker_shape` are categorical only.
        - Use `classes=` only for backwards compatibility; prefer `color=`.

    Example:
        Plot by treatment group with default palette, using custom UMAP parameters:
            ```python
            umap_params = {'n_neighbors': 10, 'min_dist': 0.1}
            plot_umap(ax, pdata, color='treatment', umap_params=umap_params)
            ```

        Plot by protein abundance (continuous coloring):
            ```python
            plot_umap(ax, pdata, color='P12345', cmap='plasma')
            ```

        Plot with custom palette:
            ```python
            color_palette = {'ctrl': '#CCCCCC', 'treated': '#E41A1C'}
            edge_palette = {'wt': '#000000', 'mut': '#377EB8'}
            
            plot_umap(ax, pdata, color='group', edge_color='treatment', cmap=color_palette, edge_cmap=edge_palette)
            ```

        Marker shapes by categorical key:
            ```python
            shape_map = {"WT": "o", "MUT": "s"}
            plot_umap(ax, pdata, color="treatment", marker_shape="genotype", shape_cmap=shape_map)
            ```

        Add ellipses grouped explicitly (useful when `color` is continuous):
            ```python
            ellipse_colors = {"WT": "#000000", "MUT": "#377EB8"}
            plot_umap(
                ax, pdata,
                color="UBE4B", cmap="viridis",
                marker_shape="genotype",
                add_ellipses=True,
                ellipse_group="genotype",
                ellipse_cmap=ellipse_colors,
                ellipse_kwargs={"alpha": 0.10, "lw": 1.5},
            )
            ```

        Plot a 3D UMAP:
            ```python
            umap_params = {'n_components':3}
            ax = fig.add_subplot(111, projection='3d')
            plot_umap(ax, pdata, color='treatment', umap_params=umap_params)
            ```
            
    """
    default_umap_params = {'n_components': 2, 'random_state': 42}
    umap_param = {**default_umap_params, **(umap_params if umap_params else {})}
    
    if umap_param['n_components'] == 3:
        assert ax.name == '3d', "The ax must be a 3D projection, please define projection='3d'"

    # check deprecated classes argument
    if classes is not None and color is None:
        print(f"{utils.format_log_prefix('warn')} `classes` is deprecated; use `color=` instead.")
        color = classes
    elif classes is not None and color is not None:
        print(f"{utils.format_log_prefix('warn')} Both `classes` and `color` were provided; using `color` and ignoring `classes`.")

    adata = utils.get_adata(pdata, on)
 
    if force == False:
        if 'X_umap' in adata.obsm.keys():
            print(f'{utils.format_log_prefix("warn")} UMAP already exists in {on} data, using existing UMAP. Run with `force=True` to recompute.')
        else:
            pdata.umap(on=on, layer=layer, **umap_param)
    else:
        print(f'UMAP calculation forced, re-calculating UMAP')
        pdata.umap(on=on, layer=layer, force_neighbors=True, **umap_param)

    Xt = adata.obsm['X_umap']
    umap = adata.uns['umap']
    mask = _resolve_subset_mask(adata, subset_mask)
    obs_names_plot = adata.obs_names[mask]

    n_comp = umap_param["n_components"]

    pc_idx = [0] if n_comp == 1 else ([0, 1] if n_comp == 2 else [0, 1, 2])
    dim_labels = ["UMAP 1"] if n_comp == 1 else (["UMAP 1", "UMAP 2"] if n_comp == 2 else ["UMAP 1", "UMAP 2", "UMAP 3"])

    y_1d = np.arange(np.sum(mask)) if n_comp == 1 else None

    if label_column and label_column in pdata.summary.columns:
        label_series = pdata.summary.loc[obs_names_plot, label_column]
    else:
        label_series = obs_names_plot

    ax = _plot_embedding_scatter(ax=ax, adata=adata, Xt=Xt, mask=mask, obs_names_plot=obs_names_plot,
        color=color, edge_color=edge_color, marker_shape=marker_shape, layer=layer, cmap=cmap, edge_cmap=edge_cmap, shape_cmap=shape_cmap,
        edge_lw=edge_lw, s=s, alpha=alpha, text_size=text_size, 
        axis_prefix="UMAP", dim_labels=dim_labels, pc_idx=pc_idx, y_1d=y_1d,
        show_labels=show_labels, label_series=label_series,
        add_ellipses=add_ellipses, ellipse_kwargs=ellipse_kwargs, ellipse_group=ellipse_group, ellipse_cmap=ellipse_cmap, plot_confidence_ellipse=_plot_confidence_ellipse,
        **kwargs,
    )

    if return_fit:
        return ax, umap
    else:
        return ax

def plot_pca_scree(ax, pca):
    """
    Plot a scree plot of explained variance from PCA.

    This function visualizes the proportion of variance explained by each
    principal component as a bar chart, helping to assess how many PCs are
    meaningful.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot the scree plot.

        pca (sklearn.decomposition.PCA or dict): The fitted PCA object, or a
            dictionary from `.uns` with key `"variance_ratio"`.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the scree plot.

    Example:
        Basic usage with fitted PCA, first run PCA:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt
            fig, ax = plt.subplots()
            ax, pca = scplt.plot_pca(ax, pdata, classes=["cellline", "treatment"], plot_pc=[1, 2])  # run PCA and plot
            ax = scplt.plot_pca_scree(ax, pca)  # scree plot
            ```

        If PCA has already been run, use cached PCA results from `.uns`:
            ```python
            scplt.plot_pca_scree(ax, pdata.prot.uns["pca"])
            ```
    """
    if isinstance(pca, dict):
        variance_ratio = np.array(pca["variance_ratio"])
        n_components = len(variance_ratio)
    else:
        variance_ratio = pca.explained_variance_ratio_
        n_components = pca.n_components_

    PC_values = np.arange(1, n_components + 1)
    cumulative = np.cumsum(variance_ratio)

    ax.plot(PC_values, variance_ratio, 'o-', linewidth=2, label='Explained Variance', color='blue')
    ax.plot(PC_values, cumulative, 'o--', linewidth=2, label='Cumulative Variance', color='gray')
    ax.set_title('Scree Plot')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    
    return ax

def plot_clustermap(ax, pdata, on='prot', classes=None, layer="X", x_label='accession', namelist=None, lut=None, log2=True,
                    cmap="coolwarm", figsize=(6, 10), force=False, impute=None, order=None, **kwargs):
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
        **kwargs: Additional keyword arguments passed to `seaborn.clustermap`.

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
        Cluster a subset of features with custom annotations:
            ```python
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            scplt.plot_clustermap(
                ax,
                pdata,
                classes=["cell_line", "condition", "treatment", "duration"],
                impute="row_min",
                z_score=0,
                center=0,
                linewidth=0,
                figsize=(10, 6),
                colors_ratio=0.04,
                x_label="gene",
                force=True,
            )
            ```

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

def plot_volcano(ax, pdata=None, values=None, method='ttest', fold_change_mode='mean', label=5,
                 label_type='Gene', color=None, alpha=0.5, pval=0.05, log2fc=1, linewidth=0.5,
                 fontsize=8, no_marks=False, classes=None, de_data=None, return_df=False, 
                 group_annot=True, group_annot_kwargs=None, group1_kwargs=None, group2_kwargs=None, up_kwargs=None, down_kwargs=None,**kwargs):
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
        pval (float): P-value threshold for significance. Default is 0.05.
        log2fc (float): Log2 fold change threshold for significance. Default is 1.
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
        **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.scatter`.

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
            values = [
                {"cellline": "HCT116", "treatment": "DMSO"},
                {"cellline": "HCT116", "treatment": "DrugX"}
            ]
            colors = sns.color_palette("Paired")[4:6]
            color_dict = dict(zip(['downregulated', 'upregulated'], colors))
            ax, df = plot_volcano(ax, pdata, classes="cellline", values=values)
            ```
        Legacy input:
            ```python
            ax, df = plot_volcano(ax, pdata, classes="cellline", values=["A", "B"], color=color_dict)
            add_volcano_legend(ax)
            ```
        To tweak styling:

            Move positions up/down and tweak styling:
            ```python
            plot_volcano(
                ax, pdata, values=values, classes=classes,
                group_annot_kwargs={"pos": {"group1_xy": (0.98, 1.10), "group2_xy": (0.02, 1.10)}},
                up_kwargs={"fontsize": 9},
                down_kwargs={"fontsize": 9},
            )
            ```
            Remove the bbox but keep text:
            ```python
            plot_volcano(
                ax, pdata, values=values, classes=classes,
                group_annot_kwargs={"bbox": None},
            )
            ```
            Turn off all text:
            ```python
            plot_volcano(ax, pdata, values=values, classes=classes, group_annot=False)
            ```


    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    import matplotlib.patheffects as PathEffects

    if de_data is None and pdata is None:
        raise ValueError("Either de_data or pdata must be provided.")

    if de_data is not None:
        volcano_df = de_data.copy()
    else:
        if values is None:
          raise ValueError("If pdata is provided, values must also be provided.")
        if isinstance(values, list) and isinstance(values[0], dict):
          volcano_df = pdata.de(values=values, method=method, pval=pval, log2fc=log2fc, fold_change_mode=fold_change_mode)
        else:
            volcano_df = pdata.de(class_type=classes, values=values, method=method, pval=pval, log2fc=log2fc, fold_change_mode=fold_change_mode)

    df = volcano_df.copy()
    volcano_df = volcano_df.dropna(subset=['p_value']).copy()
    volcano_df = volcano_df[volcano_df["significance"] != "not comparable"]

    default_color = {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}
    if color:
        default_color.update(color)
    elif no_marks:
        default_color = {k: 'grey' for k in default_color}

    scatter_kwargs = dict(s=20, edgecolors='none')
    scatter_kwargs.update(kwargs)
    colors = volcano_df['significance'].astype(str).map(default_color)

    ax.scatter(volcano_df['log2fc'], volcano_df['-log10(p_value)'],
               c=colors, alpha=alpha, **scatter_kwargs)

    ax.axhline(-np.log10(pval), color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(-log2fc, color='black', linestyle='--', linewidth=linewidth)

    ax.set_xlabel('$log_{2}$ fold change')
    ax.set_ylabel('-$log_{10}$ p value')

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
                           label_df.iloc[i]['-log10(p_value)'],
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

def plot_volcano_adata(ax, adata=None, values=None, class_type=None, de_data=None,
    gene_col=None, method='ttest', fold_change_mode='mean', layer='X', label=5, fontsize=8,
    alpha=0.5, color=None, linewidth=0.5, pval=0.05, log2fc=1.0, no_marks=False,
    return_df=False, **kwargs
):
    """
    Volcano plot for AnnData with the *same API behavior* as pdata.plot_volcano.

    Required:
        - Either `de_data` OR (`adata`, `values`, `class_type`).
        
    Supports:
        - Dictionary-style values: [{"cellline":"HCT116","tx":"DMSO"}, {...}]
        - Legacy-style values: ["A","B"]
        - Legacy multi-col values: [["HCT116","DMSO"], ["HCT116","DrugX"]]

    Produces: identical volcano to pAnnData version.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    import matplotlib.patheffects as PathEffects

    if de_data is not None:
        df = de_data.copy()
        # For labels: user must supply labels manually if needed
        group1_label = df.attrs.get("group1_label", None)
        group2_label = df.attrs.get("group2_label", None)

    else:
        if adata is None or values is None:
            raise ValueError("When de_data is not provided, must supply adata and values.")

        df = utils.de_adata(adata=adata, values=values, class_type=class_type,
            method=method, fold_change_mode=fold_change_mode, layer=layer,
            pval=pval, log2fc=log2fc, gene_col=gene_col
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

    # volcano plotting
    volcano_df = df.dropna(subset=['p_value']).copy()
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
        volcano_df['-log10(p_value)'],
        c=colors, alpha=alpha,
        **scatter_kwargs
    )

    # threshold lines
    ax.axhline(-np.log10(pval), color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(-log2fc, color='black', linestyle='--', linewidth=linewidth)

    ax.set_xlabel('$log_{2}$ fold change')
    ax.set_ylabel('-$log_{10}$ p value')

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
                row['log2fc'], row['-log10(p_value)'],
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

def add_volcano_legend(ax, colors=None):
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

def mark_volcano(ax, volcano_df, label, label_color="black", text_color=None, label_type='Gene', s=10, alpha=1, show_names=True, fontsize=8, return_texts=False):
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
        return_texts (bool): Whether to return the list of created text artists.
            This is useful when labeling multiple groups and performing a single
            global `adjust_text()` call at the end.

    Returns:
        ax (matplotlib.axes.Axes): Axis with the highlighted volcano plot.

    Example:
        Highlight specific features on a volcano plot:
            ```python
            fig, ax = plt.subplots()
            ax, df = scplt.plot_volcano(ax, pdata, classes="treatment", values=["ctrl", "drug"])
            ax = scplt.mark_volcano(
                ax, df, label=["P11247", "O35639", "F6ZDS4"],
                label_color="red", s=10, alpha=1, show_names=True
            )
            ```

    Note:
        This function works especially well in combination with
        `plot_volcano(..., no_marks=True)` to render all points in grey,
        followed by `mark_volcano()` to selectively highlight features of interest.
    """
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

        ax.scatter(match_df['log2fc'], match_df['-log10(p_value)'],
                   c=color, s=s, alpha=alpha, edgecolors='none')

        if show_names:
            texts = []
            for idx, row in match_df.iterrows():
                if label_type == "Gene" and "Genes" in volcano_df.columns:
                    text = row.get("Genes", idx)
                else:
                    text = idx

                txt = ax.text(row['log2fc'], row['-log10(p_value)'],
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
    ax, volcano_df, label, color=None, text_color=None, label_type="Gene", s=10, alpha=1, show_names=True, fontsize=8, return_texts=False,):
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
        return_texts (bool): Whether to return the list of created text artists.
            This is useful when labeling multiple groups and performing a single
            global `adjust_text()` call at the end.

    Returns:
        matplotlib.axes.Axes: Axis with highlighted points if `return_texts=False`.
        tuple (matplotlib.axes.Axes, list): Returned if `return_texts=True`,
            where the list contains the text artists for further adjustment.

    Example:
        Highlight specific features on a volcano plot using significance colors:

            ```python
            fig, ax = plt.subplots()
            ax, df = scplt.plot_volcano(
                ax, pdata, classes="treatment", values=["ctrl", "drug"]
            )

            custom_prot = ['Snca','Sox2']
            custom_dict = {"downregulated": "#1F2CCF"}
            ax = scplt.mark_volcano_by_significance(
                ax, df, label=custom_prot, color=custom_dict, show_names=False
            )
            ```

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
            match_df["-log10(p_value)"],
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
                    row["-log10(p_value)"],
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

def volcano_adjust_and_outline_texts(texts, expand=(2, 2), arrowprops=dict(arrowstyle='->', color='k', lw=0.8), linewidth=3, outline_color="w",):
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


def plot_rankquant(ax, pdata, classes = None, layer = "X", on = 'protein', cmap=['Blues'], color=['blue'], order = None, s=20, alpha=0.2, calpha=1, exp_alpha = 70, debug = False):
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
        Plot rank abundance grouped by sample size:
            ```python
            import seaborn as sns
            colors = sns.color_palette("Blues", 4)
            cmaps = ["Blues", "Reds", "Greens", "Oranges"]
            order = ["sc", "5k", "10k", "20k"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax = scplt.plot_rankquant(
                ax, pdata_filter, classes="size",
                order=order,
                cmap=cmaps, color=colors, calpha=1, alpha=0.005
            )
            ```

        Format the plot better:
            ```python
            plt.ylabel("Abundance")
            ax.set_ylim(10**ylims[0], 10**ylims[1])
            legend_patches = [
                mpatches.Patch(color=color, label=label)
                for color, label in zip(colors, order)
            ]
            plt.legend(
                handles=legend_patches, bbox_to_anchor=(0.75, 1),
                loc=2, borderaxespad=0., frameon=False
            )
            ```

        Highlight specific points on the rank-quant plot:
            ```python
            scplt.mark_rankquant(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], show_label=True,
                color="darkorange", label_type="gene"
            )
            ```

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

def mark_rankquant(plot, pdata, mark_df, class_values, layer = "X", on = 'protein', color='red', s=10,alpha=1, show_label=True, label_type='accession'):
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
              prot_sc_df = scutils.get_upset_query(size_upset, present=["sc"], absent=["5k", "10k", "20k"])
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
        Plot rank abundance and highlight specific proteins:
            ```python
            fig, ax = plt.subplots()
            ax = scplt.plot_rankquant(
                ax, pdata_filter, classes="size", order=order,
                cmap=cmaps, color=colors, s=10, calpha=1, alpha=0.005
            )
            size_upset = scutils.get_upset_contents(pdata_filter, classes="size")
            prot_sc_df = scutils.get_upset_query(
                size_upset, present=["sc"], absent=["5k", "10k", "20k"]
            )
            scplt.mark_rankquant(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], show_label=True,
                color="darkorange", label_type="gene"
            )
            ```python

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

def plot_venn(ax, pdata, classes, set_colors = 'default', weighted=False, return_contents = False, label_order=None, fixed_subset_sizes=None, **kwargs):
    """
    Plot a Venn diagram of shared proteins or peptides across groups.

    This function generates a 2- or 3-set Venn diagram based on presence/absence
    data across specified sample-level classes. For more than 3 sets, use
    `plot_upset()` instead.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str): Sample-level classes to partition proteins
            or peptides into sets.
        set_colors (str or list of str): Colors for the sets.
            
            - `"default"`: use internal color palette.
            - list of str: custom color list with length equal to the number of sets.

        weighted (bool): If True, circle/region areas are proportional to set sizes (area-weighted). If False, draws an unweighted Venn (equal-sized regions).
        return_contents (bool): If True, return both the axis and the underlying
            set contents used for plotting.
        label_order (list of str, optional): Custom order of set labels. Must
            contain the same elements as `classes`.
        **kwargs: Additional keyword arguments passed to matplotlib-venn functions.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the Venn diagram. Returned if `return_contents=False`

        tuple (matplotlib.axes.Axes, dict): Returned if `return_contents=True`.
        The dictionary maps class labels to sets of feature identifiers.

    Raises:
        ValueError: If number of sets is not 2 or 3.
        ValueError: If `label_order` does not contain the same elements as `classes`.
        ValueError: If custom `set_colors` length does not match number of sets.

    Example:
        Plot a 2-set Venn diagram of shared proteins:
            ```python
            fig, ax = plt.subplots()
            scplt.plot_venn(
                ax, pdata_1mo_snpc, classes="sample",
                set_colors=["#1f77b4", "#ff7f0e"]
            )
            ```

        Plot a weighted set by counts:
            ```python
            fig, ax = plt.subplots(figsize=(3, 3))
            scplt.plot_venn(
                ax, pdata, classes='treatment',
                weighted=True)
            ```

        Plot a weighted set by specifying a fixed subset size:
            ```python
            fig, ax = plt.subplots(figsize=(3, 3))
            scplt.plot_venn(
                ax, pdata, classes='treatment',
                weighted=True, fixed_subset_sizes=(1,1,3))
            ```            

    See Also:
        plot_upset: Plot an UpSet diagram for >3 sets.  
        plot_rankquant: Rank-based visualization of protein/peptide distributions.
    """
    upset_contents = utils.get_upset_contents(pdata, classes, upsetForm=False)

    num_keys = len(upset_contents)
    if set_colors == 'default':
        set_colors = get_color('colors', n=num_keys)
    elif len(set_colors) != num_keys:
        raise ValueError("The number of colors provided must match the number of sets.")
    
    if label_order is not None:
        if set(label_order) != set(upset_contents.keys()):
            raise ValueError("`label_order` must contain the same elements as `classes`.")
        set_labels = label_order
        set_list = [set(upset_contents[label]) for label in set_labels]
    else:
        set_labels = list(upset_contents.keys())
        set_list = [set(value) for value in upset_contents.values()]

    alpha = kwargs.pop('alpha', 0.5)

        # New API (matplotlib-venn ≥ 0.12)
    try:
        from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm as Venn2Layout
        from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm as Venn3Layout
        from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
        USE_LAYOUT = True
    except ImportError:
        # Older API (no layout subpackage)
        from matplotlib_venn import venn2_unweighted, venn3_unweighted, venn2_circles, venn3_circles
        USE_LAYOUT = False

    if weighted:
        venn_functions = {
            2: lambda: (venn2(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=alpha,
                                layout_algorithm=(Venn2Layout(fixed_subset_sizes=fixed_subset_sizes) if fixed_subset_sizes is not None else None), **kwargs),
                        venn2_circles(subsets=fixed_subset_sizes if fixed_subset_sizes is not None else set_list, ax = ax, linewidth=1)),
            3: lambda: (venn3(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=alpha,
                                layout_algorithm=(Venn3Layout(fixed_subset_sizes=fixed_subset_sizes) if fixed_subset_sizes is not None else None), **kwargs),
                        venn3_circles(subsets=fixed_subset_sizes if fixed_subset_sizes is not None else set_list, ax = ax, linewidth=1))
        }
    else:
        if USE_LAYOUT:
            venn_functions = {
                2: lambda: (venn2(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=alpha, layout_algorithm=Venn2Layout(fixed_subset_sizes=(1,1,1)), **kwargs),
                            venn2_circles(subsets=(1, 1, 1), ax = ax,  linewidth=1)),
                3: lambda: (venn3(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=alpha, layout_algorithm=Venn3Layout(fixed_subset_sizes=(1,1,1,1,1,1,1)), **kwargs),
                            venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), ax = ax, linewidth=1))
            }
        else:
            venn_functions = {
                2: lambda: (venn2_unweighted(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=alpha, **kwargs),
                            venn2_circles(subsets=(1, 1, 1), ax = ax, linewidth=1)),
                3: lambda: (venn3_unweighted(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=alpha, **kwargs),
                            venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), ax = ax, linewidth=1)) }

    if num_keys in venn_functions:
        v, c = venn_functions[num_keys]()
    else:
        raise ValueError("Venn diagrams only accept either 2 or 3 sets. For more than 3 sets, use the plot_upset function.")

    if return_contents:
        return ax, upset_contents
    return ax

def plot_upset(pdata, classes, return_contents = False, **kwargs):
    """
    Plot an UpSet diagram of shared proteins or peptides across groups.

    This function generates an UpSet plot for >2 sets based on presence/absence
    data across specified sample-level classes. Uses the `upsetplot` package
    for visualization.

    Args:
        pdata (pAnnData): Input pAnnData object.

        classes (str or list of str): Sample-level classes to partition proteins
            or peptides into sets.

        return_contents (bool): If True, return both the UpSet object and the
            underlying set contents used for plotting.

        **kwargs: Additional keyword arguments passed to `upsetplot.UpSet`.  
            See the [upsetplot documentation](https://upsetplot.readthedocs.io/en/stable/)  
            for more details. Common arguments include:

            - `sort_categories_by` (str): How to sort categories. Options are
              `"cardinality"`, `"input"`, `"-cardinality"`, or `"-input"`.
            - `min_subset_size` (int): Minimum subset size to display.

    Returns:
        upset (upsetplot.UpSet): The UpSet plot object.

        tuple (upsetplot.UpSet, pandas.DataFrame): Returned if
        `return_contents=True`. The DataFrame contains set membership as a
        multi-index.

    Example:
        Basic usage with set size categories:
            ```python
            upplot, size_upset = scplt.plot_upset(
                pdata_filter, classes="size", sort_categories_by="-input"
            )
            uplot = upplot.plot()
            uplot["intersections"].set_ylabel("Subset size")
            uplot["totals"].set_xlabel("Protein count")
            plt.show()
            ```

        Optional styling of the plot can also be done:
            ```python
            upplot.style_subsets(
                present=["sc"], absent=["2k", "5k", "10k", "20k"],
                edgecolor="black", facecolor="darkorange", linewidth=2, label="sc only"
            )
            upplot.style_subsets(
                absent=["sc"], present=["2k", "5k", "10k", "20k"],
                edgecolor="white", facecolor="#7F7F7F", linewidth=2, label="in all but sc"
            )
            ```

    See Also:
        plot_venn: Plot a Venn diagram for 2 to 3 sets.  
        plot_rankquant: Rank-based visualization of protein/peptide distributions.
    """

    upset_contents = utils.get_upset_contents(pdata, classes = classes)
    upplot = upsetplot.UpSet(upset_contents, subset_size="count", show_counts=True, facecolor = 'black', **kwargs)

    if return_contents:
        return upplot, upset_contents
    else:
        return upplot

def plot_abundance_2D(ax,data,cases,genes='all', cmap='Blues',color=['blue'],s=20,alpha=[0.2,1],calpha=1):
    
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

def plot_raincloud(ax,pdata,classes = None, layer = 'X', on = 'protein', order = None, color=['blue'],boxcolor='black',linewidth=0.5, debug = False):
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
        Plot raincloud distributions grouped by sample size:
            ```python
            ax = scplt.plot_raincloud(
                ax, pdata_filter, classes="size",
                order=order, color=colors, linewidth=0.5, debug=False
            )
            scplt.mark_raincloud(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], color="black"
            )
            ```

    See Also:
        mark_raincloud: Highlight specific features on a raincloud plot.  
        plot_rankquant: Alternative distribution visualization using rank abundance.
    """
    adata = utils.get_adata(pdata, on)

    classes_list = utils.get_classlist(adata, classes = classes, order = order)
    data_X = []

    for j, class_value in enumerate(classes_list):
        rank_data = utils.resolve_class_filter(adata, classes, class_value, debug=True)

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

def mark_raincloud(plot,pdata,mark_df,class_values,layer = "X", on = 'protein',lowest_index=0,color='red',s=10,alpha=1):
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
        Highlight specific proteins on a raincloud plot:
            ```python
            ax = scplt.plot_raincloud(
                ax, pdata_filter, classes="size", order=order,
                color=colors, linewidth=0.5
            )
            scplt.mark_raincloud(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], color="black"
            )
            ```

    See Also:
        plot_raincloud: Generate raincloud plots with distributions per group.  
        plot_rankquant: Alternative distribution visualization using rank abundance.
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
    return plot
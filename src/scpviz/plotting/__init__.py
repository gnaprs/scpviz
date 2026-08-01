"""
This module provides a collection of plotting utilities for visualizing
protein and peptide abundance data, quality control metrics, and results of
statistical analyses. Functions are organized into categories based on their
purpose, with paired "plot" and "mark" functions where applicable.

Functions are written to work seamlessly with the `pAnnData` object structure and metadata conventions in scpviz.

## Convenience Plotting Wrappers
    get_color: Generate a list of colors, a colormap, or a palette from package defaults.
    shift_legend: Reposition an axis legend outside the plot while maintaining figure size.
    plot_significance: Add a simple significance bar + label to an axis.
    plot_summary: Bar plots summarizing sample-level metadata (e.g. protein counts).

## Distribution and Abundance Plots

Functions:
    plot_cv: Violin plots of coefficient of variation (CV) across groups.
    plot_abundance: Violin/box/strip plots of protein or peptide abundance.
    plot_abundance_housekeeping: Plot abundance of housekeeping proteins.
    plot_abundance_boxgrid: Multi-panel abundance summary grids (box/bar/violin/line).
    annotate_abundance_boxgrid_significance: Pairwise test brackets on boxgrid panels.
    plot_abundance_2D: 2D scatter of abundance between two case groups.
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
    resolve_marker_shapes: Helper function for resolving marker shapes from categorical groupings.

## PCA overlays (loadings + GSEA)

Functions:
    plot_pca_gsea_pathway_vectors: Overlay PCA-GSEA pathways as arrows in PCA space.
    plot_pca_protein_vectors: Overlay protein PCA loadings as arrows in PCA space.
    plot_pca_gsea_bubble: Bubble plot summarizing PCA-GSEA NES/FDR across PCs.
    plot_pca_gsea_heatmap: Heatmap of PCA-GSEA NES across pathways and PCs.

## Clustering and Heatmaps

Functions:
    plot_grouped_heatmap: Curated protein-group blocks × samples with header strips.
    plot_clustered_heatmap: Hierarchically clustered proteins × samples with optional group strip.
    plot_clustermap: Clustered heatmap of proteins/peptides × samples (seaborn; legacy).
    plot_pairwise_correlation: Group- or sample-level pairwise correlation / distance heatmap with annotation bars.

## Differential Expression and Volcano Plots

Functions:
    plot_volcano: Volcano plot of differential expression results.
    plot_volcano_adata: Same as above, but for AnnData objects.
    mark_volcano: Highlight specific features on a volcano plot with a specific color.
    mark_volcano_by_significance: Similar to above, but colored by significance.
    volcano_adjust_and_outline_texts: Adjust text labels for volcano plots after multiple mark_volcanos.
    add_volcano_legend: Add standard legend handles for volcano plots.

## Enrichment Plots

Functions:
    plot_enrichment_svg: Plot STRING enrichment results (forwarded from `enrichment.py`).

## Set Operations

Functions:
    plot_venn: Venn diagrams for 2 to 3 sets.
    plot_upset: UpSet diagrams for >3 sets.

## Notes and Tips
!!! tip
    * Most functions accept a `matplotlib.axes.Axes` as the first argument for flexible subplot integration. `ax` can be defined as such:

    ```python
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4)) # configure size as needed
    ```

    * "Mark" functions are designed to be used following their paired "plot" functions to highlight features of interest.

---
"""

import seaborn as sns
import upsetplot

from scpviz import utils as _utils

# Test compatibility: monkeypatch targets `scplt.utils` / `scplt.upsetplot`
utils = _utils

sns.set_theme(context="paper", style="ticks")

from .abundance import *  # noqa: E402
from .style import *  # noqa: E402
from .correlation import *  # noqa: E402
from .dimreduc import *  # noqa: E402
from .enrichment import *  # noqa: E402
from .heatmap import *  # noqa: E402
from .sets import *  # noqa: E402
from .volcano import *  # noqa: E402

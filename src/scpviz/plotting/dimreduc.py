"""Dimensionality reduction: PCA, UMAP, scree, PCA–GSEA overlays."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
import re

import anndata as ad
import matplotlib.cm as cm
import matplotlib.collections as clt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap.umap_ as umap
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from scpviz import utils

from .style import _get_cmap, _resolve_subset_mask, get_color
from ._pca_gsea_data import (
    N_VECTORS_UNSET,
    _apply_pathway_name_filters,
    _build_pca_gsea_tables,
    _compute_pc_score_df,
    _ensure_pca_gsea_payload,
    _format_pathway_label,
    _resolve_pca_gsea_namelist_pathways,
    _resolve_protein_namelist_genes,
    _select_pca_protein_vectors_split,
    _select_top_pathways,
    _validate_plot_n_vectors,
    _validate_plot_top_n,
    _vector_color_from_cmap,
)

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData

def plot_pca(ax: "plt.Axes", pdata: pAnnData, color=None, edge_color=None, marker_shape=None, classes=None, 
             layer="X", on='protein', cmap='default', edge_cmap="default", shape_cmap="default", edge_lw=0.8,
             s=20, alpha=.8, plot_pc=[1, 2], pca_params=None, subset_mask=None,
             force=False, basis='X_pca', text_size=9, show_labels=False, label_column=None,
             add_ellipses=False, ellipse_group=None, ellipse_cmap='default', ellipse_kwargs=None, 
             return_fit=False, mapping_keys=None, mapping=None, mapping_on_missing: str = "warn",
             **kwargs: Any) -> "plt.Axes | tuple[plt.Axes, dict[str, Any]]":
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

        mapping_keys (list of str, optional): `.obs` columns whose tuple of levels keys `mapping`.
            Must be provided together with ``mapping``.

        mapping (dict, optional): Keys are tuples matching observed metadata combinations; values
            are dicts with optional ``color`` (literal or abundance feature), ``edge_color`` (literal
            only), and ``marker``. Cannot be combined with ``edge_color`` / ``edge_cmap``. When
            ``color=`` is an abundance feature, mapping entries must not include ``color``.

        mapping_on_missing (str): ``"warn"`` (default) prints a log-prefixed message and uses grey
            face with no edge for missing combinations (abundance ``color=``: missing combo keeps
            abundance face, edges off). ``"raise"`` raises if any observed combination is absent from ``mapping``.

        return_fit (bool): If True, also return the fitted PCA object.
        **kwargs (Any): Extra keyword arguments passed to `ax.scatter()`.

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
        PCA on normalized protein data with ellipses, grouped by cell line and condition:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            pdata_norm.pca(on="protein")
            scplt.plot_pca(ax, pdata_norm, classes=["cellline", "condition"], add_ellipses=True)
            plt.show()
            ```

        ![Plot PCA](../../assets/plots/plot_pca.png)

        PCA on single-cell protein data after ``directlfq`` (example uses ``region``; use ``condition`` or other ``.obs`` columns as in your object):
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            pdata_sc.pca(on="protein")
            scplt.plot_pca(
                ax,
                pdata_sc,
                color=["region"],
                cmap={"Cortex": "#D19DCB", "SNpc": "#85BE9E"},
                add_ellipses=True,
            )
            plt.show()
            ```

        ![Plot PCA (single-cell)](../../assets/plots/plot_pca_sc.png)

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

        Tuple-key ``mapping`` (literal face + edge per combination of ``.obs`` columns):
            ```python
            mapping_keys = ["cellline", "condition"]
            mapping = {
                ("A", "ctrl"): {"color": "white", "edge_color": "black"},
                ("A", "treat"): {"color": "white", "edge_color": "blue"},
                ("B", "ctrl"): {"color": "lightgrey", "edge_color": "black"},
                ("B", "treat"): {"color": "lightgrey", "edge_color": "blue"},
            }
            plot_pca(ax, pdata, mapping_keys=mapping_keys, mapping=mapping, force=True)
            ```

        Global abundance face color with per-combination edges (``mapping`` must not set ``color``):
            ```python
            mapping_keys = ["cellline", "condition"]
            mapping = {
                ("A", "ctrl"): {"edge_color": "black"},
                ("A", "treat"): {"edge_color": "steelblue"},
                ("B", "ctrl"): {"edge_color": "black"},
                ("B", "treat"): {"edge_color": "steelblue"},
            }
            plot_pca(ax, pdata, color="UBE4B", cmap="plasma", mapping_keys=mapping_keys, mapping=mapping)
            ```

        Sequential overlays on the same axes (same embedding, using different ``subset_mask``; order matters).
        Replace column names and palettes with your metadata:
            ```python
            line = "LineA"
            cell_line_color = {"LineA": "#4C72B0", "LineB": "#DD8452"}
            cell_line_color_6h = {"LineA": "#9fb8d9", "LineB": "#e8b896"}

            mask_dark = (
                (pdata.summary["treatment"] == "Drug")
                & (pdata.summary["cell_line"] == line)
                & (pdata.summary["duration"] == "24hr")
            )
            mask_light = (
                (pdata.summary["treatment"] == "Drug")
                & (pdata.summary["cell_line"] == line)
                & (pdata.summary["duration"] == "6hr")
            )
            mask_ctrl = (
                (pdata.summary["treatment"] == "Vehicle")
                & (pdata.summary["cell_line"] == line)
            )

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")

            ax, _ = plot_pca(
                ax,
                pdata,
                color="cell_line",
                cmap=cell_line_color,
                edge_color="duration",
                edge_cmap={"6hr": "grey", "24hr": "black"},
                plot_pc=[1, 2, 3],
                subset_mask=mask_dark,
                return_fit=True,
                force=True,
            )
            ax, _ = plot_pca(
                ax,
                pdata,
                color="cell_line",
                cmap=cell_line_color_6h,
                edge_color="duration",
                edge_cmap={"6hr": "grey", "24hr": "black"},
                plot_pc=[1, 2, 3],
                subset_mask=mask_light,
                return_fit=True,
            )
            plot_pca(
                ax,
                pdata,
                color="cell_line",
                cmap={k: "white" for k in cell_line_color},
                plot_pc=[1, 2, 3],
                edge_color="cell_line",
                edge_cmap=cell_line_color,
                edge_lw=1.2,
                subset_mask=mask_ctrl,
                force=False,
            )
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
        mapping_keys=mapping_keys, mapping=mapping, mapping_on_missing=mapping_on_missing,
        **kwargs,
    )

    if return_fit:
        return ax, pca
    else:
        return ax

def resolve_plot_colors(
    adata: ad.AnnData, classes: Any, cmap: Any, layer: str = "X"
) -> Any:
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
            cmap_obj = _get_cmap(cmap)
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
            cmap_obj = _get_cmap(cmap)
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
        cmap = _get_cmap(cmap) if isinstance(cmap, str) else cmap

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

def resolve_marker_shapes(
    adata: ad.AnnData, marker_shape: Any, shape_cmap: Any = "default"
) -> Any:
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


def _resolve_embedding_style_mapping(
    adata: ad.AnnData,
    *,
    mapping_keys: list[str],
    mapping: dict[tuple, dict[str, Any]],
    mapping_on_missing: str,
    color: Any,
    cmap: Any,
    layer: str,
    marker_shape: Any,
    shape_cmap: Any = "default",
    edge_lw: float = 0.8,
) -> dict[str, Any]:
    """
    Build face / edge / marker arrays for ``mapping_keys`` + ``mapping`` embedding plots.

    Abundance is allowed for face color only (top-level ``color=`` and/or per-entry ``color``);
    edge colors must be literal matplotlib colors (or ``\"none\"``).
    """
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be a dict mapping tuple keys to style dicts.")
    if mapping_on_missing not in ("raise", "warn"):
        raise ValueError("mapping_on_missing must be one of: 'raise', 'warn'.")

    def is_abundance_key(k: str) -> bool:
        if k in adata.var_names:
            return True
        if "Genes" in adata.var.columns:
            gene_map = adata.var["Genes"].to_dict()
            return any(gene == k for gene in gene_map.values())
        return False

    missing_cols = [k for k in mapping_keys if k not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"mapping_keys: column(s) not found in adata.obs: {missing_cols}")
    sub = adata.obs[mapping_keys]
    keys_series = pd.Series(
        [tuple(sub.iloc[i].values.tolist()) for i in range(sub.shape[0])],
        index=adata.obs_names,
    )
    n_obs = len(keys_series)

    for mk in mapping.keys():
        if not isinstance(mk, tuple):
            raise ValueError(f"mapping keys must be tuples, got {type(mk).__name__}: {mk!r}")
        if len(mk) != len(mapping_keys):
            raise ValueError(
                f"mapping key {mk!r} has length {len(mk)} but mapping_keys has length {len(mapping_keys)}."
            )

    mapping_has_marker = any("marker" in v for v in mapping.values())
    if mapping_has_marker and marker_shape is not None:
        raise ValueError("Use either mapping entries with 'marker' or marker_shape=, not both.")

    if color is not None:
        color_is_cat_obs = (
            (isinstance(color, list) and all(c in adata.obs.columns for c in color))
            or (isinstance(color, str) and color in adata.obs.columns)
        )
        if color_is_cat_obs:
            raise ValueError(
                "categorical color= (obs column) cannot be combined with mapping=; "
                "use abundance color= or encode categories in mapping['color']."
            )

    global_abundance = color is not None and isinstance(color, str) and is_abundance_key(color)
    if color is not None and not global_abundance:
        raise ValueError(
            f"When using mapping=, color= must be omitted or an abundance feature (gene/protein); got {color!r}."
        )

    if global_abundance:
        for k, v in mapping.items():
            if "color" in v:
                raise ValueError(
                    "When color= is an abundance feature, mapping entries must not include 'color'; "
                    "use mapping only for edge_color / marker."
                )

    # --- resolve style dict per observation ---
    styles_plot: list[dict[str, Any]] = []

    for i in range(n_obs):
        key = keys_series.iloc[i]
        if key in mapping:
            st = dict(mapping[key])
        else:
            ks = tuple(str(x) for x in key)
            st = dict(mapping[ks]) if ks in mapping else None
        if st is None:
            if mapping_on_missing == "raise":
                raise ValueError(
                    f"Observed combination {key} not found in mapping. "
                    f"Known mapping keys: {list(mapping.keys())}"
                )
            else:
                msg = (
                    f"No mapping entry for combination {key!r}; using abundance face color with edges off."
                    if global_abundance
                    else f"No mapping entry for combination {key!r}; using grey face and no edge."
                )
                print(f"{utils.format_log_prefix('warn')} {msg}")
            if global_abundance:
                st = {"edge_color": "none"}
            else:
                st = {"color": "grey", "edge_color": "none"}
        styles_plot.append(st)

    if not global_abundance:
        for k, v in mapping.items():
            if "color" not in v:
                raise ValueError(f"mapping[{k!r}] must include 'color' when color= is not set.")

    for k, v in mapping.items():
        spec = v.get("color")
        if spec is None:
            continue
        if mcolors.is_color_like(spec):
            continue
        if is_abundance_key(str(spec)):
            continue
        raise ValueError(
            f"mapping[{k!r}]['color'] must be a matplotlib color or an abundance feature name, got {spec!r}."
        )

    for k, v in mapping.items():
        ec = v.get("edge_color")
        if ec is None:
            continue
        if is_abundance_key(str(ec)):
            raise ValueError(
                f"mapping[{k!r}]['edge_color'] must be a literal color (abundance-driven edges are not supported)."
            )
        if not (
            ec is None
            or (isinstance(ec, str) and ec.lower() == "none")
            or mcolors.is_color_like(ec)
        ):
            raise ValueError(f"mapping[{k!r}]['edge_color'] is not a valid literal color: {ec!r}")

    # --- face colors ---
    face_cmap_resolved = None
    face_legend = None

    if global_abundance:
        face_mapped, face_cmap_resolved, _fl = resolve_plot_colors(adata, color, cmap, layer=layer)
        face_mapped = np.asarray(face_mapped)
    else:
        abund_cache: dict[str, tuple[np.ndarray, Any]] = {}
        face_kind: str | None = None
        face_hex_list: list[str] = []
        face_float_buf = np.zeros(n_obs, dtype=float)

        for i in range(n_obs):
            spec = styles_plot[i]["color"]
            if mcolors.is_color_like(spec):
                fk = "lit"
                face_hex_list.append(mcolors.to_hex(spec))
            elif is_abundance_key(str(spec)):
                fk = "ab"
                sspec = str(spec)
                if sspec not in abund_cache:
                    col, cmap_res, _leg = resolve_plot_colors(adata, sspec, cmap, layer=layer)
                    col = np.asarray(col).ravel()
                    if cmap_res is None:
                        raise ValueError(
                            f"mapping color {spec!r} resolved as categorical; use a literal color or abundance feature."
                        )
                    abund_cache[sspec] = (col, cmap_res)
                col, cmap_one = abund_cache[sspec]
                face_float_buf[i] = col[i]
                face_cmap_resolved = cmap_one
            else:
                raise ValueError(f"Invalid mapping face color {spec!r}.")

            if face_kind is None:
                face_kind = fk
            elif face_kind != fk:
                raise ValueError(
                    "Cannot mix literal face colors and abundance face colors for different samples "
                    "(including mapping_on_missing fallback)."
                )

        if face_kind == "lit":
            face_mapped = face_hex_list
            face_cmap_resolved = None
            seen: set[tuple] = set()
            face_legend = []
            for i in range(n_obs):
                lab = keys_series.iloc[i]
                if lab in seen:
                    continue
                seen.add(lab)
                face_legend.append(
                    mpatches.Patch(color=face_hex_list[i], label=", ".join(str(x) for x in lab))
                )
        else:
            face_mapped = face_float_buf
            if face_cmap_resolved is None:
                if cmap == "default":
                    face_cmap_resolved = _get_cmap("viridis")
                else:
                    face_cmap_resolved = _get_cmap(cmap) if isinstance(cmap, str) else cmap

    # --- edges ---
    edge_list: list[Any] = []
    for i in range(n_obs):
        st = styles_plot[i]
        ec = st.get("edge_color", "none")
        if ec is None:
            ec = "none"
        if isinstance(ec, str) and ec.lower() == "none":
            edge_list.append("none")
        else:
            edge_list.append(ec)

    edge_mapped = edge_list
    edge_labels = [", ".join(str(x) for x in k) for k in keys_series.values]
    edge_legend = None
    if any(e not in (None, "none") and not (isinstance(e, str) and e.lower() == "none") for e in edge_mapped):
        uniq = {}
        for i, ek in enumerate(edge_labels):
            ec = edge_mapped[i]
            if isinstance(ec, str) and ec.lower() == "none":
                continue
            uniq.setdefault((ek, ec), None)
        edge_legend = [
            mlines.Line2D(
                [], [], linestyle="none", marker="o", color=ec, markerfacecolor="white",
                markeredgecolor=ec, markersize=7, label=ek,
            )
            for (ek, ec) in sorted(uniq.keys(), key=lambda t: t[0])
        ]

    # One legend for mapping when face colors are literal: same keys as edge styling.
    combined_mapping_legend: list[mpatches.Patch] | None = None
    if not global_abundance and face_legend is not None:
        key_to_i: dict[tuple, int] = {}
        for i in range(n_obs):
            key = keys_series.iloc[i]
            if key not in key_to_i:
                key_to_i[key] = i
        combined_mapping_legend = []
        for key in sorted(key_to_i.keys(), key=lambda k: ", ".join(str(x) for x in k)):
            i = key_to_i[key]
            fc = face_hex_list[i]
            ec = edge_list[i]
            lab = ", ".join(str(x) for x in key)
            if isinstance(ec, str) and ec.lower() == "none":
                combined_mapping_legend.append(
                    mpatches.Patch(facecolor=fc, edgecolor="none", linewidth=0, label=lab)
                )
            else:
                combined_mapping_legend.append(
                    mpatches.Patch(
                        facecolor=fc,
                        edgecolor=ec,
                        linewidth=edge_lw,
                        label=lab,
                    )
                )
        face_legend = None
        edge_legend = None

    # --- markers ---
    markers_all = None
    shape_legend = None
    if mapping_has_marker:
        markers_all = np.array([styles_plot[i].get("marker", "o") for i in range(n_obs)], dtype=object)
        shape_map: dict[tuple, str] = {}
        for i in range(n_obs):
            lab = keys_series.iloc[i]
            if lab not in shape_map:
                shape_map[lab] = styles_plot[i].get("marker", "o")
        shape_legend = [
            mlines.Line2D(
                [], [], linestyle="none", marker=shape_map[c], color="black",
                markerfacecolor="black", markeredgecolor="black", markersize=7,
                label=", ".join(str(x) for x in c),
            )
            for c in sorted(shape_map.keys(), key=str)
        ]
    elif marker_shape is not None:
        markers_all, shape_legend, _ = resolve_marker_shapes(
            adata, marker_shape, shape_cmap=shape_cmap
        )

    legend_title = " / ".join(str(k) for k in mapping_keys)

    return {
        "face_mapped": face_mapped,
        "face_cmap_resolved": face_cmap_resolved,
        "face_legend": face_legend,
        "edge_mapped": edge_mapped,
        "edge_legend": edge_legend,
        "markers_all": markers_all,
        "shape_legend": shape_legend,
        "legend_title": legend_title,
        "color_key_legend": None,
        "combined_mapping_legend": combined_mapping_legend,
    }


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

def _plot_embedding_scatter(
    *,
    ax: "plt.Axes",
    adata: ad.AnnData,
    Xt: np.ndarray,
    mask: np.ndarray | pd.Series,
    obs_names_plot: pd.Index,
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
    mapping_keys=None,
    mapping=None,
    mapping_on_missing: str = "warn",
    **kwargs,
) -> "plt.Axes | dict[str, Any]":
    """
    Shared scatter rendering for UMAP/PCA-like embeddings.

    Notes:
      - color can be categorical or continuous (abundance), via resolve_plot_colors
      - edge_color is categorical only
      - marker_shape is categorical only (marker splitting)
    """

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
        n,
    ):
        """
        Build kwargs for a single ax.scatter call (already subset-aligned).

        Returns:
            dict: kwargs for ax.scatter
        """
        kw = dict(base_kwargs)

        if color_key is None:
            kw.setdefault("c", ["grey"] * n)
            kw.pop("cmap", None)
        else:
            kw["c"] = face_plot
            if face_cmap_resolved is not None:
                kw["cmap"] = face_cmap_resolved

        if edge_key is None:
            kw["edgecolors"] = "none"
        else:
            kw["edgecolors"] = edge_plot
            if isinstance(edge_plot, str):
                kw["linewidths"] = 0.0 if edge_plot.lower() == "none" else edge_lw
            else:
                ec_seq = np.asarray(edge_plot, dtype=object).ravel()
                if ec_seq.shape[0] == n:
                    kw["linewidths"] = np.array(
                        [
                            0.0
                            if e is None or (isinstance(e, str) and e.lower() == "none")
                            else float(edge_lw)
                            for e in ec_seq
                        ],
                        dtype=float,
                    )
                else:
                    kw["linewidths"] = edge_lw

        return kw

    def _slice_scatter_kwargs(scatter_kwargs, m):
        """
        Slice array-like scatter kwargs for a subset mask m (boolean array).

        Returns:
            dict: kwargs safe to pass to ax.scatter for that group.
        """
        kw = dict(scatter_kwargs)

        if "c" in kw and isinstance(kw["c"], (list, np.ndarray)):
            c = np.asarray(kw["c"])
            if c.shape[0] == m.shape[0]:
                kw["c"] = c[m]

        if "edgecolors" in kw:
            ec = kw["edgecolors"]

            if ec is None or (isinstance(ec, str) and ec.lower() == "none"):
                return kw

            if isinstance(ec, str):
                return kw

            ec_arr = np.asarray(ec)
            if ec_arr.shape[0] == m.shape[0]:
                kw["edgecolors"] = ec_arr[m]

        if "linewidths" in kw:
            lw = kw["linewidths"]
            if isinstance(lw, (list, np.ndarray)):
                lw_arr = np.asarray(lw).ravel()
                if lw_arr.shape[0] == m.shape[0]:
                    kw["linewidths"] = lw_arr[m]

        return kw

    def _add_continuous_colorbar(ax_cb, color_mapped, cmap_resolved, label, text_size=10):
        """Attach a colorbar if using continuous coloring."""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

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
        cb = ax_cb.figure.colorbar(sm, ax=ax_cb, pad=0.01)
        cb.set_label(label, fontsize=text_size)

    def _legend_title_from_key(key):
        if key is None:
            return None
        if isinstance(key, list):
            return "/".join(str(c).capitalize() for c in key)
        return str(key).capitalize()

    use_mapping = mapping is not None
    if (mapping is None) != (mapping_keys is None):
        raise ValueError("Provide both mapping_keys and mapping together, or omit both.")

    mapping_legend_title = None
    if use_mapping:
        if edge_color is not None:
            raise ValueError(
                "edge_color cannot be used with mapping=; set edge colors inside mapping[...]['edge_color']."
            )
        if edge_cmap != "default":
            raise ValueError("edge_cmap cannot be used with mapping=; define edge colors inside mapping.")
        mp = _resolve_embedding_style_mapping(
            adata,
            mapping_keys=mapping_keys,
            mapping=mapping,
            mapping_on_missing=mapping_on_missing,
            color=color,
            cmap=cmap,
            layer=layer,
            marker_shape=marker_shape,
            shape_cmap=shape_cmap,
            edge_lw=edge_lw,
        )
        face_mapped = mp["face_mapped"]
        face_cmap_resolved = mp["face_cmap_resolved"]
        face_legend = mp["face_legend"]
        edge_mapped = mp["edge_mapped"]
        edge_legend = mp["edge_legend"]
        markers_all = mp["markers_all"]
        shape_legend = mp["shape_legend"]
        mapping_legend_title = mp["legend_title"]
        combined_mapping_legend = mp.get("combined_mapping_legend")
        scatter_color_key = color if color is not None else tuple(mapping_keys)
        scatter_edge_key = tuple(mapping_keys)
        if add_ellipses and ellipse_group is None:
            ellipse_group = mapping_keys
    else:
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
        scatter_color_key = color
        scatter_edge_key = edge_color
        combined_mapping_legend = None

    # plot
    Xt_plot = Xt[mask]
    n_plot = Xt_plot.shape[0]

    face_plot = None if face_mapped is None else np.asarray(face_mapped)[mask]
    edge_plot = None if edge_mapped is None else np.asarray(edge_mapped)[mask]
    markers_plot = None if markers_all is None else np.asarray(markers_all)[mask]

    base_kwargs = dict(s=s, alpha=alpha, **kwargs)

    scatter_kwargs = _build_scatter_kwargs(
        base_kwargs=base_kwargs,
        color_key=scatter_color_key,
        face_plot=face_plot,
        face_all=face_mapped,
        face_cmap_resolved=face_cmap_resolved,
        edge_key=scatter_edge_key,
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
        label=(_legend_title_from_key(color) if color is not None else "Abundance"),
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
            cmap_obj = _get_cmap(ellipse_cmap)
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

    # --- legends (keep separate, stacked outside right edge) ---
    from matplotlib.legend import Legend as _Legend

    legends = []

    def _make_legend(handles, title):
        leg = _Legend(
            ax, handles, [h.get_label() for h in handles],
            title=title,
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=text_size,
            frameon=False,
        )
        ax.add_artist(leg)
        leg.set_clip_on(False)  # ax.add_artist sets clip_path to axes boundary; undo it
        return leg

    if use_mapping and combined_mapping_legend:
        legends.append(_make_legend(combined_mapping_legend, mapping_legend_title))
    elif face_legend:
        legends.append(
            _make_legend(
                face_legend,
                mapping_legend_title if use_mapping else _legend_title_from_key(color),
            )
        )

    if edge_legend:
        legends.append(
            _make_legend(
                edge_legend,
                mapping_legend_title if use_mapping else _legend_title_from_key(edge_color),
            )
        )

    if shape_legend:
        legends.append(_make_legend(shape_legend, _legend_title_from_key(marker_shape)))

    # Stack legends vertically: render once to get heights, then reposition
    if legends:
        fig = ax.get_figure()
        fig.canvas.draw()  # force layout so legend sizes are available

        renderer = fig.canvas.get_renderer()
        ax_height_px = ax.get_window_extent(renderer).height

        # Walk top-to-bottom, accumulating y offset in axes-fraction units
        y_cursor = 1.0  # start at top of axes
        for leg in legends:
            leg_height_px = leg.get_window_extent(renderer).height
            leg_height_ax = leg_height_px / ax_height_px
            # Place this legend so its top aligns with y_cursor
            leg.set_bbox_to_anchor((0.6, y_cursor))
            leg.set_loc("upper left")
            y_cursor -= leg_height_ax + 0.02  # 0.02 gap between legends

    if return_parts:
        return {
            "face_mapped": face_mapped,
            "face_cmap_resolved": face_cmap_resolved,
            "edge_mapped": edge_mapped,
            "markers_all": markers_all,
        }

    return ax

def plot_umap(ax: "plt.Axes", pdata: pAnnData, color=None, edge_color=None, marker_shape=None, classes = None, 
              layer = "X", on = 'protein', cmap='default', edge_cmap="default", shape_cmap="default", show_labels=False, label_column=None,
              s=20, alpha=.8, umap_params={}, text_size = 10, edge_lw=0.8, 
              add_ellipses=False, ellipse_group=None, ellipse_cmap='default', ellipse_kwargs=None, 
              force = False, return_fit=False, subset_mask=None,
              mapping_keys=None, mapping=None, mapping_on_missing: str = "warn",
              **kwargs: Any) -> "plt.Axes | tuple[plt.Axes, dict[str, Any]]":
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

        mapping_keys (list of str, optional): `.obs` columns whose tuple of levels keys `mapping`.
            Must be provided together with ``mapping``.

        mapping (dict, optional): Tuple-keyed style dicts (``color``, ``edge_color``, ``marker``).
            See ``plot_pca`` for semantics; cannot be combined with ``edge_color`` / ``edge_cmap``.

        mapping_on_missing (str): ``"warn"`` (default) or ``"raise"`` (see ``plot_pca``).

        force (bool): If True, recompute UMAP even if cached.
        return_fit (bool): If True, return the fitted UMAP object.
        **kwargs (Any): Extra keyword arguments passed to `ax.scatter()`.

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
        UMAP after ``pca(on="protein")``, colored by sample metadata (example uses ``region`` and cohort-specific ``umap_params``):
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4.5, 4))
            pdata_sc.pca(on="protein")
            scplt.plot_umap(
                ax,
                pdata_sc,
                color=["region"],
                cmap={"Cortex": "#D19DCB", "SNpc": "#85BE9E"},
                force=True,
                umap_params={"min_dist": 0.3, "n_neighbors": 30, "random_state": 42},
                s=10,
                alpha=0.85,
            )
            scplt.shift_legend(ax)
            plt.show()
            ```

        ![Plot UMAP](../../assets/plots/plot_umap.png)

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

        Tuple-key ``mapping`` (literal face + edge per combination of ``.obs`` columns):
            ```python
            umap_params = {"n_neighbors": 10, "min_dist": 0.1}
            mapping_keys = ["cellline", "condition"]
            mapping = {
                ("A", "ctrl"): {"color": "white", "edge_color": "black"},
                ("A", "treat"): {"color": "white", "edge_color": "blue"},
                ("B", "ctrl"): {"color": "lightgrey", "edge_color": "black"},
                ("B", "treat"): {"color": "lightgrey", "edge_color": "blue"},
            }
            plot_umap(
                ax, pdata,
                mapping_keys=mapping_keys,
                mapping=mapping,
                umap_params=umap_params,
                force=True,
            )
            ```

        Global abundance face color with per-combination edges:
            ```python
            umap_params = {"n_neighbors": 10, "min_dist": 0.1}
            mapping_keys = ["cellline", "condition"]
            mapping = {
                ("A", "ctrl"): {"edge_color": "black"},
                ("A", "treat"): {"edge_color": "steelblue"},
                ("B", "ctrl"): {"edge_color": "black"},
                ("B", "treat"): {"edge_color": "steelblue"},
            }
            plot_umap(
                ax, pdata,
                color="UBE4B",
                cmap="plasma",
                mapping_keys=mapping_keys,
                mapping=mapping,
                umap_params=umap_params,
            )
            ```

        Sequential overlays on the same axes (same UMAP, different ``subset_mask``). Replace
        columns and palettes with your metadata; use matching ``umap_params`` and ``force``
        so all layers share one embedding:
            ```python
            umap_params = {"n_neighbors": 10, "min_dist": 0.1}
            line = "LineA"
            cell_line_color = {"LineA": "#4C72B0", "LineB": "#DD8452"}
            cell_line_color_6h = {"LineA": "#9fb8d9", "LineB": "#e8b896"}

            mask_dark = (
                (pdata.summary["treatment"] == "Drug")
                & (pdata.summary["cell_line"] == line)
                & (pdata.summary["duration"] == "24hr")
            )
            mask_light = (
                (pdata.summary["treatment"] == "Drug")
                & (pdata.summary["cell_line"] == line)
                & (pdata.summary["duration"] == "6hr")
            )
            mask_ctrl = (
                (pdata.summary["treatment"] == "Vehicle")
                & (pdata.summary["cell_line"] == line)
            )

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")

            ax, _ = plot_umap(
                ax,
                pdata,
                color="cell_line",
                cmap=cell_line_color,
                edge_color="duration",
                edge_cmap={"6hr": "grey", "24hr": "black"},
                umap_params={**umap_params, "n_components": 3},
                subset_mask=mask_dark,
                return_fit=True,
                force=True,
            )
            ax, _ = plot_umap(
                ax,
                pdata,
                color="cell_line",
                cmap=cell_line_color_6h,
                edge_color="duration",
                edge_cmap={"6hr": "grey", "24hr": "black"},
                umap_params={**umap_params, "n_components": 3},
                subset_mask=mask_light,
                return_fit=True,
                force=False,
            )
            plot_umap(
                ax,
                pdata,
                color="cell_line",
                cmap={k: "white" for k in cell_line_color},
                edge_color="cell_line",
                edge_cmap=cell_line_color,
                edge_lw=1.2,
                umap_params={**umap_params, "n_components": 3},
                subset_mask=mask_ctrl,
                force=False,
            )
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
        mapping_keys=mapping_keys, mapping=mapping, mapping_on_missing=mapping_on_missing,
        **kwargs,
    )

    if return_fit:
        return ax, umap
    else:
        return ax

def plot_pca_scree(ax: "plt.Axes", pca: Any) -> "plt.Axes":
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
        Basic usage with PCA results from ``.uns``:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 3))
            pdata_norm.pca(on="protein")
            scplt.plot_pca_scree(ax, pdata_norm.prot.uns["pca"])
            plt.show()
            ```

        ![Plot PCA scree](../../assets/plots/plot_pca_scree.png)
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

def plot_pca_gsea_pathway_vectors(
    ax,
    pdata: pAnnData,
    on="protein",
    key_added="pca_gsea",
    plot_pc=[1, 2],
    n_vectors=N_VECTORS_UNSET,
    fdr_cutoff=0.1,
    arrow_scale=0.25,
    pca_kwargs=None,
    show_samples=True,
    title_case_labels=True,
    force=False,
    gsea_kwargs=None,
    adjust_labels=True,
    adjust_text_kwargs=None,
    text_positions=None,
    lock_text_positions=False,
    top_n_mode="balanced",
    exclude_pathways=None,
    namelist=None,
    cmap=None,
    xlim=None,
    ylim=None,
    return_df=False,
) -> Any:
    """
    Overlay PCA-GSEA pathways as arrows in a two-dimensional PCA sample space.

    Each arrow encodes normalized enrichment scores (NES) on two principal components taken from
    ``adata.uns[key_added]['results']`` (from ``pca_gsea``). Arrow endpoints are rescaled using the
    current axis limits so pathways remain visible; they are not plotted in the same numeric units as
    sample coordinates. When ``show_samples`` is True, the sample PCA scatter is drawn first via
    ``plot_pca``.

    Args:
        ax (matplotlib.axes.Axes): Target axis (2D).
        pdata (pAnnData): Input object.
        on (str): Data level, ``"protein"`` or ``"peptide"``.
        key_added (str): ``adata.uns`` key for PCA-GSEA results (default ``"pca_gsea"``).
        plot_pc (list of int): Exactly two 1-based PCs, e.g. ``[1, 2]``.
        n_vectors (int, sequence, ``None``, or unset): Caps auto-selected pathways (after ``namelist`` rows).
            Default when ``namelist`` is ``None`` is ``12``; when ``namelist`` is set, default is no extra
            top-N unless you pass ``n_vectors`` explicitly. If an int (>= 1), uses ``top_n_mode`` on rows not
            already chosen by ``namelist``. If ``[nx, ny]``, split-axis top union on that remainder.
            Pass ``n_vectors=None`` with ``namelist`` to plot only listed pathways; pass ``n_vectors`` and
            leave ``namelist`` unset for ranking-only.
        fdr_cutoff (float or None): For **auto-selected** rows: pathway-level FDR filtering (keep if any plotted
            PC has FDR ≤ cutoff) and score gating in ``_compute_pc_score_df``. **Namelist** pathways skip the row
            FDR filter; a **warning** is printed
            per named pathway when ``fdr_cutoff`` is not ``None`` and no plotted PC passes FDR.
        arrow_scale (float): Scale factor for arrow length relative to axis span.
        pca_kwargs (dict or None): Additional arguments passed to ``plot_pca`` when ``show_samples=True``.
        show_samples (bool): If True, plot samples first; if False, draw only axes, grid lines, and arrows.
        title_case_labels (bool): If True, format pathway labels for display (e.g. title case).
        force (bool): If True, re-run ``pca_gsea`` for ``plot_pc``.
        gsea_kwargs (dict or None): Forwarded to ``pca_gsea`` when results are auto-computed.
        adjust_labels (bool): If True, run ``adjust_text`` to reduce label overlap.
        adjust_text_kwargs (dict or None): Extra keyword arguments for ``adjust_text``.
        text_positions (dict or None): Optional manual label positions; keys are pathway raw or display
            strings, values are ``(x, y)`` data coordinates.
        lock_text_positions (bool): If True, labels with entries in ``text_positions`` are not moved by
            ``adjust_text``.
        top_n_mode (str): ``"balanced"`` or ``"max_score"``. Used only when ``n_vectors`` is an int.
        exclude_pathways (str, iterable, or None): Remove pathways matching these names (raw Term, short
            pathway, or library), same as before.
        namelist (list of str or None): Pathways to always include first (matches ``Term`` / pathway_raw or short
            pathway name only, **not** library). Shown even if they fail FDR; ``exclude_pathways`` still applies
            first. Combined with ``n_vectors`` on the remaining rows (namelist first, then auto).
        cmap (dict or None): Per-pathway colors; lookup raw ``Term``, formatted label, then case-insensitive keys.
        xlim (tuple or None): Applied after scatter / empty axes, before arrow scaling (with ``ax.set_aspect("auto")``).
        ylim (tuple or None): Same as ``xlim``.
        return_df (bool): If True, also return a DataFrame with NES, FDR, and label positions.

    Returns:
        matplotlib.axes.Axes, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

    Note:
        May attach ``payload["pathway_loadings"]`` for reuse in the same session.

    TODO:
        Add explicit FDR visual encoding on vector arrows (e.g., color or alpha by FDR).

    Example:
        Default overlay on PC1 vs PC2 with label de-cluttering and return coordinates for a second pass:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots()
            ax, vec_df = scplt.plot_pca_gsea_pathway_vectors(
                ax,
                pdata,
                plot_pc=[1, 2],
                adjust_text_kwargs={"expand": (1.3, 1.3)},
                return_df=True,
            )
            ```

        Reuse label positions from a previous run (e.g. after editing coordinates in ``vec_df``):
            ```python
            manual = {
                row["pathway_raw"]: (row["text_x"], row["text_y"])
                for _, row in vec_df.iterrows()
            }
            ax = scplt.plot_pca_gsea_pathway_vectors(
                ax,
                pdata,
                plot_pc=[1, 2],
                text_positions=manual,
                lock_text_positions=True,
            )
            ```
    """
    plot_pc = list(plot_pc)
    if len(plot_pc) != 2:
        raise ValueError("`plot_pc` must contain exactly two PCs for pathway vectors.")

    _, payload = _ensure_pca_gsea_payload(
        pdata=pdata,
        on=on,
        key_added=key_added,
        requested_pcs=plot_pc,
        force=force,
        gsea_kwargs=gsea_kwargs,
    )
    long_df, matrix_df, fdr_df, missing_pc_keys = _build_pca_gsea_tables(payload=payload, pcs=plot_pc)
    pcx, pcy = f"PC{int(plot_pc[0])}", f"PC{int(plot_pc[1])}"
    if missing_pc_keys:
        raise ValueError(
            f"Requested PCs missing from pca_gsea results: {missing_pc_keys}. "
            f"Please run pca_gsea on these PCs (or set force=True)."
        )

    long_df, matrix_df, fdr_df = _apply_pathway_name_filters(
        long_df=long_df,
        matrix_df=matrix_df,
        fdr_df=fdr_df,
        include_pathways=None,
        exclude_pathways=exclude_pathways,
    )

    if n_vectors is N_VECTORS_UNSET:
        n_vectors = None if namelist is not None else 12
    if namelist is None and n_vectors is None:
        raise ValueError("No pathways to plot: provide `n_vectors`, `namelist`, or both.")

    named_resolver_order = []
    named_resolver_set = set()
    if namelist is not None:
        named_resolver_order = _resolve_pca_gsea_namelist_pathways(matrix_df, long_df, namelist)
        named_resolver_set = set(named_resolver_order)

    named_plot_order = [
        i
        for i in named_resolver_order
        if i in matrix_df.index and matrix_df.loc[i, [pcx, pcy]].notna().any()
    ]

    auto_order = []
    if n_vectors is not None:
        remainder = matrix_df.loc[~matrix_df.index.isin(named_resolver_set)]
        remainder = remainder[[pcx, pcy]]
        if fdr_cutoff is not None:
            _fdr_sub = fdr_df.reindex(remainder.index)[[pcx, pcy]]
            _keep_mask = (_fdr_sub <= float(fdr_cutoff)).any(axis=1)
            remainder = remainder.loc[_keep_mask]
        remainder = remainder.dropna(subset=[pcx, pcy], how="all")
        if not remainder.empty:
            mode, nv = _validate_plot_n_vectors(n_vectors, what="pathways")
            score_df = _compute_pc_score_df(
                matrix_df=remainder[[pcx, pcy]],
                fdr_df=fdr_df.reindex(remainder.index)[[pcx, pcy]],
                fdr_cutoff=fdr_cutoff,
            )
            if mode == "single":
                selected = _select_top_pathways(score_df=score_df, top_n=nv, top_n_mode=top_n_mode)
            else:
                nx, ny = nv
                selected = _select_pca_protein_vectors_split(score_df, pcx, pcy, nx, ny)
            auto_order = [r for r in selected if r not in set(named_plot_order)]

    final_order = []
    seen_f = set()
    for i in named_plot_order:
        if i not in seen_f:
            final_order.append(i)
            seen_f.add(i)
    for i in auto_order:
        if i not in seen_f:
            final_order.append(i)
            seen_f.add(i)

    if not final_order:
        raise ValueError("No pathways to plot: provide `n_vectors`, `namelist`, or both.")

    if fdr_cutoff is not None and named_plot_order:
        fc = float(fdr_cutoff)
        for pr in named_plot_order:
            fx = fdr_df.loc[pr, pcx]
            fy = fdr_df.loc[pr, pcy]
            passes = any(pd.notna(v) and float(v) <= fc for v in (fx, fy))
            if not passes:
                print(
                    f"{utils.format_log_prefix('warn')} Pathway {str(pr)!r}: FDR on {pcx}={fx}, {pcy}={fy}; "
                    f"cutoff={fdr_cutoff}. Showing anyway because `namelist` is explicit."
                )

    matrix_df = matrix_df.loc[final_order]
    fdr_df = fdr_df.reindex(matrix_df.index)
    long_df = long_df[long_df["pathway_raw"].isin(matrix_df.index)].copy()

    meta_pw = long_df.drop_duplicates("pathway_raw").set_index("pathway_raw")
    short_by_raw = meta_pw["pathway"]
    lib_by_raw = meta_pw["library"]

    def _pathway_display_name(pathway_raw_key):
        short = short_by_raw.get(pathway_raw_key, np.nan)
        if pd.isna(short):
            raw_s = str(pathway_raw_key)
            short = raw_s.split("__", 1)[1] if "__" in raw_s else raw_s
        return str(short)

    # Cache derived pathway loading tables for downstream reuse.
    payload["pathway_loadings"] = {"matrix": matrix_df.copy(), "fdr_qval": fdr_df.copy(), "long": long_df.copy()}

    if show_samples:
        if pca_kwargs is None:
            pca_kwargs = {}
        plot_pca(ax=ax, pdata=pdata, on=on, plot_pc=plot_pc, **pca_kwargs)
    else:
        adata = utils.get_adata(pdata, on)
        if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
            raise ValueError("PCA metadata not found. Run `.pca()` before plotting pathway vectors with `show_samples=False`.")
        var = adata.uns["pca"]["variance_ratio"]
        ax.set_xlabel(f"PC{plot_pc[0]} ({var[int(plot_pc[0]) - 1] * 100:.2f}%)")
        ax.set_ylabel(f"PC{plot_pc[1]} ({var[int(plot_pc[1]) - 1] * 100:.2f}%)")
        ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")

    if xlim is not None or ylim is not None:
        ax.set_aspect("auto")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    xl = ax.get_xlim()
    yl = ax.get_ylim()
    xspan = xl[1] - xl[0]
    yspan = yl[1] - yl[0]

    coords = matrix_df[[pcx, pcy]].fillna(0.0).values
    denom = np.max(np.abs(coords))
    if denom == 0:
        denom = 1.0
    x_scale = float(arrow_scale) * xspan / denom
    y_scale = float(arrow_scale) * yspan / denom

    texts = []
    text_rows = []
    text_positions = text_positions or {}
    for pathway, (vx, vy) in matrix_df[[pcx, pcy]].fillna(0.0).iterrows():
        vx, vy = float(vx), float(vy)
        x_end = vx * x_scale
        y_end = vy * y_scale
        label_txt = _format_pathway_label(pathway) if title_case_labels else str(pathway)
        pos = text_positions.get(str(pathway), text_positions.get(label_txt, None))
        text_x, text_y = (x_end, y_end) if pos is None else (float(pos[0]), float(pos[1]))
        color = _vector_color_from_cmap(cmap, str(pathway), label_txt)
        ax.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=0.7,
                lw=1.5,
                mutation_scale=10,
            ),
        )
        ax.update_datalim([(x_end, y_end), (0, 0)])
        txt = ax.text(text_x, text_y, label_txt, fontsize=8, ha="left", va="bottom", color=color)
        if not (lock_text_positions and pos is not None):
            texts.append(txt)
        text_rows.append({
            "pathway_raw": str(pathway),
            "pathway": label_txt,
            "arrow_x": x_end,
            "arrow_y": y_end,
            "text_obj": txt,
        })

    ax.autoscale_view()

    if adjust_labels and len(texts) > 0:
        # By default, do not draw connector lines from text to arrow tips.
        adjust_cfg = {"expand": (1.6, 1.6), "arrowprops": None}
        if adjust_text_kwargs:
            adjust_cfg.update(adjust_text_kwargs)
        adjust_text(texts, ax=ax, **adjust_cfg)

    vector_df = matrix_df[[pcx, pcy]].copy().rename(columns={pcx: "nes_x", pcy: "nes_y"})
    vector_df["pc_x"] = pcx
    vector_df["pc_y"] = pcy
    vector_df["fdr_x"] = fdr_df.reindex(vector_df.index)[pcx].values
    vector_df["fdr_y"] = fdr_df.reindex(vector_df.index)[pcy].values
    vector_df["vector_norm"] = np.sqrt(vector_df["nes_x"] ** 2 + vector_df["nes_y"] ** 2)
    vector_df["pathway_raw"] = vector_df.index.astype(str)
    vector_df["library"] = vector_df["pathway_raw"].map(lib_by_raw)
    miss_lib = vector_df["library"].isna()
    if miss_lib.any():
        vector_df.loc[miss_lib, "library"] = vector_df.loc[miss_lib, "pathway_raw"].map(
            lambda x: x.split("__", 1)[0] if "__" in str(x) else ""
        )
    disp_series = vector_df["pathway_raw"].map(_pathway_display_name)
    if title_case_labels:
        vector_df["pathway"] = disp_series.map(_format_pathway_label)
    else:
        vector_df["pathway"] = disp_series
    vector_df = vector_df.reset_index(drop=True)[
        ["pathway", "pathway_raw", "library", "pc_x", "pc_y", "nes_x", "nes_y", "fdr_x", "fdr_y", "vector_norm"]
    ]
    text_pos_df = pd.DataFrame(
        [
            {
                "pathway_raw": row["pathway_raw"],
                "pathway": row["pathway"],
                "arrow_x": row["arrow_x"],
                "arrow_y": row["arrow_y"],
                "text_x": row["text_obj"].get_position()[0],
                "text_y": row["text_obj"].get_position()[1],
            }
            for row in text_rows
        ]
    )
    vector_df = vector_df.merge(text_pos_df, on=["pathway", "pathway_raw"], how="left")

    ax.set_title(f"PCA pathway vectors ({pcx} vs {pcy})")
    if return_df:
        return ax, vector_df
    return ax

def plot_pca_protein_vectors(
    ax,
    pdata: pAnnData,
    on="protein",
    plot_pc=(1, 2),
    gene_col="Genes",
    n_vectors=N_VECTORS_UNSET,
    arrow_scale=0.25,
    pca_kwargs=None,
    show_samples=True,
    title_case_labels=False,
    adjust_labels=True,
    adjust_text_kwargs=None,
    text_positions=None,
    lock_text_positions=False,
    min_abs_loading_for_top_n=None,
    top_n_mode="balanced",
    exclude_genes=None,
    namelist=None,
    cmap=None,
    xlim=None,
    ylim=None,
    return_df=False,
) -> Any:
    """
    Overlay protein PCA loadings as arrows in a two-dimensional sample PCA space.

    Arrows use feature loadings from ``adata.uns['pca']['PCs']`` (from ``pAnnData.pca``), not GSEA NES.
    Geometry matches ``plot_pca_gsea_pathway_vectors``: each arrow runs from the origin in the direction
    ``(loading_on_PCx, loading_on_PCy)``, with length rescaled from the current axis limits for visibility.
    Labels default to the ``gene_col`` column in ``.var`` when present, otherwise ``.var_names``.

    Args:
        ax (matplotlib.axes.Axes): Target axis (2D).
        pdata (pAnnData): Input object.
        on (str): Data level, ``"protein"`` or ``"peptide"``.
        plot_pc (tuple or list of int): Exactly two 1-based PCs.
        gene_col (str): Column in ``.var`` for display labels; missing column falls back to ``.var_names``.
        n_vectors (int, sequence, ``None``, or unset): Caps **auto-selected** proteins (rows not already taken
            by ``namelist``). Default when ``namelist`` is ``None`` is ``20``; when ``namelist`` is set, default
            is no extra top-N unless you pass ``n_vectors`` explicitly. If an int (>= 1), uses ``top_n_mode``.
            If ``[nx, ny]``, split-axis top union on that remainder. ``min_abs_loading_for_top_n`` gates scores
            on the remainder the same way in int and split modes.
        arrow_scale (float): Scale factor for arrow length relative to axis span.
        pca_kwargs (dict or None): Forwarded to ``plot_pca`` when ``show_samples=True``.
        show_samples (bool): If True, draw the sample PCA scatter first; if False, only axes and arrows.
        title_case_labels (bool): If True, lightly format gene text (underscores to spaces, title case).
        adjust_labels (bool): If True, run ``adjust_text`` to reduce overlap.
        adjust_text_kwargs (dict or None): Extra keyword arguments for ``adjust_text``.
        text_positions (dict or None): Manual label positions keyed by gene or formatted label.
        lock_text_positions (bool): If True, manual positions are excluded from ``adjust_text`` motion.
        min_abs_loading_for_top_n (float or None): If set, ranking scores on a PC are zero when
            ``|loading|`` is below this threshold on that PC.
        top_n_mode (str): ``"balanced"`` or ``"max_score"`` (same selection logic as pathway vectors, using
            absolute loadings instead of NES/FDR scores). Used only when ``n_vectors`` is an int.
        exclude_genes (str, iterable, or None): Remove genes/features matching these strings (gene label or
            ``.var_names`` feature id).
        namelist (list of str or None): Gene labels (matrix row index, exact ``str`` match) to include **first**.
            Duplicates in ``namelist`` are ignored for matching order. Combined with ``n_vectors`` on the
            remaining rows (namelist first, then auto). Genes also listed in ``exclude_genes`` are dropped.
        cmap (dict or None): Map gene label (as in matrix or after ``title_case_labels`` formatting) to a
            matplotlib color; lookup tries raw name, formatted label, then case-insensitive keys. Default
            ``None`` draws arrows and labels in black.
        xlim (tuple or None): If set, applied with ``ax.set_xlim(xlim)`` immediately after the PCA scatter
            (or empty axes) and **before** arrow length scaling, so ``arrow_scale`` matches the visible range.
            When either ``xlim`` or ``ylim`` is set, ``ax.set_aspect("auto")`` is called first so a fixed
            data aspect from ``plot_pca`` (or ``show_samples=False``) does not block the limits.
        ylim (tuple or None): If set, ``ax.set_ylim(ylim)`` at the same stage as ``xlim`` (same note).
        return_df (bool): If True, return ``(ax, vector_df)`` with loadings and arrow/text coordinates.

    Returns:
        matplotlib.axes.Axes, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

    Example:
        Show top protein loadings on PC1 vs PC2 on sample PCA scatter:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(4, 4))
            pdata_norm.pca(on="protein")
            scplt.plot_pca_protein_vectors(ax, pdata_norm, n_vectors=10)
            plt.show()
            ```

        ![Plot PCA protein vectors](../../assets/plots/plot_pca_protein_vectors.png)

        Top-loading genes on PC1 vs PC2 over the sample PCA scatter, returning arrow and text coordinates:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots()
            ax, vec = scplt.plot_pca_protein_vectors(
                ax,
                pdata,
                plot_pc=[1, 2],
                n_vectors=25,
                return_df=True,
            )
            ```

        Split-axis selection: top loadings on PC1 and PC3 separately, then union:
            ```python
            fig, ax = plt.subplots()
            scplt.plot_pca_protein_vectors(
                ax,
                pdata,
                plot_pc=[1, 3],
                n_vectors=[5, 3],
                adjust_labels=False,
            )
            ```

        Explicit genes with colors and axis limits:
            ```python
            fig, ax = plt.subplots()
            scplt.plot_pca_protein_vectors(
                ax,
                pdata,
                plot_pc=[1, 2],
                namelist=["TP53", "EGFR"],
                cmap={"TP53": "crimson", "egfr": "steelblue"},
                xlim=(-6, 6),
                ylim=(-5, 5),
            )
            ```

        Loading arrows only (no sample points) for a compact biplot-style panel:
            ```python
            fig, ax = plt.subplots()
            scplt.plot_pca_protein_vectors(
                ax,
                pdata,
                plot_pc=[1, 2],
                n_vectors=20,
                show_samples=False,
                adjust_labels=False,
            )
            ```
    """
    plot_pc = list(plot_pc)
    if len(plot_pc) != 2:
        raise ValueError("`plot_pc` must contain exactly two PCs for protein loading vectors.")

    def _build_pca_protein_loading_matrix(
        adata: ad.AnnData, plot_pc: list[int], gene_col: str = "Genes"
    ) -> tuple[pd.DataFrame, pd.Series, str, str]:
        """
        Build a gene-by-PC matrix of PCA loadings (one row per gene after collapsing duplicate labels).

        Duplicate resolution matches ``enrichment_functional_pca``: for each gene label, keep the feature
        row with the largest Euclidean norm in the loading plane spanned by the two requested PCs.

        Returns:
            tuple: ``(matrix_df, feature_by_gene, pcx_name, pcy_name)``. Loading columns use labels such as
            ``PC1`` and ``PC2`` matching the requested ``plot_pc`` values.
        """
        if "pca" not in adata.uns or "PCs" not in adata.uns["pca"]:
            raise ValueError("PCA loadings not found. Run `.pca()` on this data layer first.")
        PCs = adata.uns["pca"]["PCs"]
        n_comp, n_feat = PCs.shape
        if n_feat != adata.n_vars:
            raise ValueError(
                f"PCA loading matrix width ({n_feat}) does not match number of variables ({adata.n_vars})."
            )
        pc_a, pc_b = int(plot_pc[0]), int(plot_pc[1])
        for pc in (pc_a, pc_b):
            if pc < 1 or pc > n_comp:
                raise ValueError(
                    f"Invalid PC {pc}: available PCs are 1..{n_comp}."
                )
        col_a, col_b = f"PC{pc_a}", f"PC{pc_b}"
        lx = PCs[pc_a - 1, :]
        ly = PCs[pc_b - 1, :]
        if gene_col in adata.var.columns:
            genes = adata.var[gene_col].astype(str)
        else:
            genes = pd.Series(adata.var_names.astype(str), index=adata.var_names)
        df = pd.DataFrame(
            {
                "feature": adata.var_names.astype(str),
                "gene": genes.values,
                col_a: lx,
                col_b: ly,
            },
            index=adata.var_names.astype(str),
        )
        df = df.dropna(subset=["gene"]).copy()
        df["gene"] = df["gene"].astype(str)
        df = df[df["gene"].str.len() > 0]
        if df.empty:
            raise ValueError("No genes with non-empty labels after resolving `.var` for PCA protein vectors.")
        plane_norm = np.sqrt(df[col_a].astype(float) ** 2 + df[col_b].astype(float) ** 2)
        df["_plane_norm"] = plane_norm
        pick = df.groupby("gene", sort=False)["_plane_norm"].idxmax()
        df = df.loc[pick].drop(columns="_plane_norm")
        matrix_df = df.set_index("gene")[[col_a, col_b]]
        feature_by_gene = df.set_index("gene")["feature"]
        return matrix_df, feature_by_gene, col_a, col_b

    def _apply_gene_name_filters(matrix_df, feature_by_gene, exclude_genes=None):
        """
        Filter protein rows by exclude list (gene label or ``.var_names`` feature id).

        Returns:
            tuple: Filtered ``(matrix_df, feature_by_gene)``.
        """
        if exclude_genes is None:
            return matrix_df, feature_by_gene

        def _to_set(x):
            if isinstance(x, str):
                return {x}
            return {str(v) for v in x}

        exclude_set = _to_set(exclude_genes)
        selected = pd.Series(True, index=matrix_df.index)
        feat = feature_by_gene.reindex(matrix_df.index)
        selected &= ~(matrix_df.index.to_series().isin(exclude_set) | feat.astype(str).isin(exclude_set))
        keep = matrix_df.index[selected]
        matrix_df = matrix_df.loc[keep]
        feature_by_gene = feature_by_gene.reindex(matrix_df.index)
        return matrix_df, feature_by_gene

    def _compute_protein_pc_score_df(matrix_df, min_abs_loading_for_top_n=None):
        """
        Compute per-PC scores from absolute loadings for ranking proteins.

        Score on each PC is ``|loading|``. If ``min_abs_loading_for_top_n`` is set, entries below that
        threshold are zeroed on that PC (similar in role to FDR gating for pathway ranking).
        """
        score_df = matrix_df.abs()
        if min_abs_loading_for_top_n is not None:
            m = float(min_abs_loading_for_top_n)
            score_df = score_df.where(score_df >= m, 0.0)
        return score_df.fillna(0.0)

    adata = utils.get_adata(pdata, on)
    matrix_df, feature_by_gene, pcx, pcy = _build_pca_protein_loading_matrix(
        adata, plot_pc, gene_col=gene_col
    )
    matrix_df, feature_by_gene = _apply_gene_name_filters(
        matrix_df,
        feature_by_gene,
        exclude_genes=exclude_genes,
    )
    if matrix_df.empty:
        raise ValueError("No proteins available after gene name filters.")

    if n_vectors is N_VECTORS_UNSET:
        n_vectors = None if namelist is not None else 20
    if namelist is None and n_vectors is None:
        raise ValueError("No proteins to plot: provide `n_vectors`, `namelist`, or both.")

    named_resolver_order = []
    named_resolver_set = set()
    if namelist is not None:
        named_resolver_order, named_resolver_set = _resolve_protein_namelist_genes(matrix_df, namelist)

    named_plot_order = [
        g
        for g in named_resolver_order
        if g in matrix_df.index and matrix_df.loc[g, [pcx, pcy]].notna().any()
    ]

    auto_order = []
    if n_vectors is not None:
        remainder = matrix_df.loc[~matrix_df.index.isin(named_resolver_set)]
        if not remainder.empty:
            mode, nv = _validate_plot_n_vectors(n_vectors, what="proteins")
            score_df = _compute_protein_pc_score_df(remainder[[pcx, pcy]], min_abs_loading_for_top_n)
            if mode == "single":
                selected = _select_top_pathways(score_df=score_df, top_n=nv, top_n_mode=top_n_mode)
            else:
                nx, ny = nv
                selected = _select_pca_protein_vectors_split(score_df, pcx, pcy, nx, ny)
            auto_order = [r for r in selected if r not in set(named_plot_order)]

    final_order = []
    seen_pf = set()
    for g in named_plot_order:
        if g not in seen_pf:
            final_order.append(g)
            seen_pf.add(g)
    for g in auto_order:
        if g not in seen_pf:
            final_order.append(g)
            seen_pf.add(g)

    if not final_order:
        raise ValueError("No proteins to plot: provide `n_vectors`, `namelist`, or both.")

    matrix_df = matrix_df.loc[final_order]
    feature_by_gene = feature_by_gene.reindex(matrix_df.index)

    if show_samples:
        if pca_kwargs is None:
            pca_kwargs = {}
        plot_pca(ax=ax, pdata=pdata, on=on, plot_pc=plot_pc, **pca_kwargs)
    else:
        if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
            raise ValueError(
                "PCA metadata not found. Run `.pca()` before plotting protein vectors with `show_samples=False`."
            )
        var = adata.uns["pca"]["variance_ratio"]
        ax.set_xlabel(f"PC{plot_pc[0]} ({var[int(plot_pc[0]) - 1] * 100:.2f}%)")
        ax.set_ylabel(f"PC{plot_pc[1]} ({var[int(plot_pc[1]) - 1] * 100:.2f}%)")
        ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")

    if xlim is not None or ylim is not None:
        ax.set_aspect("auto")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    xl = ax.get_xlim()
    yl = ax.get_ylim()
    xspan = xl[1] - xl[0]   # full width of visible x range
    yspan = yl[1] - yl[0]   # full height of visible y range

    coords = matrix_df[[pcx, pcy]].fillna(0.0).values
    denom = np.max(np.abs(coords))
    if denom == 0:
        denom = 1.0

    x_scale = float(arrow_scale) * xspan / denom
    y_scale = float(arrow_scale) * yspan / denom

    texts = []
    text_rows = []
    text_positions = text_positions or {}
    for gene, row in matrix_df[[pcx, pcy]].fillna(0.0).iterrows():
        vx, vy = float(row[pcx]), float(row[pcy])
        x_end = vx * x_scale
        y_end = vy * y_scale
        label_txt = str(gene)
        if title_case_labels:
            label_txt = label_txt.replace("_", " ").title()
        pos = text_positions.get(str(gene), text_positions.get(label_txt, None))
        text_x, text_y = (x_end, y_end) if pos is None else (float(pos[0]), float(pos[1]))
        color = _vector_color_from_cmap(cmap, str(gene), label_txt)
        ax.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(0, 0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        alpha=0.7,
                        lw=1.5,
                        mutation_scale=10,  # controls head size in points, like fontsize
                    ),
                )
        ax.update_datalim([(x_end, y_end), (0, 0)])
        txt = ax.text(text_x, text_y, label_txt, fontsize=8, ha="left", va="bottom", color=color)
        if not (lock_text_positions and pos is not None):
            texts.append(txt)
        text_rows.append(
            {
                "gene": str(gene),
                "arrow_x": x_end,
                "arrow_y": y_end,
                "text_obj": txt,
            }
        )

    ax.autoscale_view()  # ensure the axes limits are updated to match the data

    if adjust_labels and len(texts) > 0:
        adjust_cfg = {"expand": (1.6, 1.6), "arrowprops": None}
        if adjust_text_kwargs:
            adjust_cfg.update(adjust_text_kwargs)
        adjust_text(texts, ax=ax, **adjust_cfg)

    vector_df = matrix_df[[pcx, pcy]].copy().rename(columns={pcx: "load_x", pcy: "load_y"})
    vector_df["pc_x"] = pcx
    vector_df["pc_y"] = pcy
    vector_df["feature"] = feature_by_gene.reindex(matrix_df.index).astype(str).values
    vector_df["vector_norm"] = np.sqrt(vector_df["load_x"] ** 2 + vector_df["load_y"] ** 2)
    vector_df = vector_df.reset_index()
    idx_col = vector_df.columns[0]
    if idx_col != "gene":
        vector_df = vector_df.rename(columns={idx_col: "gene"})

    text_pos_df = pd.DataFrame(
        [
            {
                "gene": row["gene"],
                "arrow_x": row["arrow_x"],
                "arrow_y": row["arrow_y"],
                "text_x": row["text_obj"].get_position()[0],
                "text_y": row["text_obj"].get_position()[1],
            }
            for row in text_rows
        ]
    )
    vector_df = vector_df.merge(text_pos_df, on="gene", how="left")
    vector_df = vector_df[
        ["gene", "feature", "pc_x", "pc_y", "load_x", "load_y", "vector_norm", "arrow_x", "arrow_y", "text_x", "text_y"]
    ]

    ax.set_title(f"PCA protein loading vectors ({pcx} vs {pcy})")
    if return_df:
        return ax, vector_df
    return ax

def plot_pca_gsea_bubble(
    ax,
    pdata: pAnnData,
    on="protein",
    key_added="pca_gsea",
    pcs=None,
    top_n=20,
    fdr_cutoff=0.1,
    size_scale=120.0,
    cmap="coolwarm",
    title_case_labels=True,
    force=False,
    gsea_kwargs=None,
    top_n_mode="balanced",
    include_pathways=None,
    exclude_pathways=None,
    return_df=False,
) -> Any:
    """
    Plot PCA-GSEA results as a bubble chart (principal component versus pathway).

    Bubble color encodes NES; bubble area reflects significance (``-log10(FDR)``). Rows and columns
    are ordered by pathway and PC. If ``pcs`` is omitted, all PCs present in stored results are used.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        pdata (pAnnData): Input object.
        on (str): Data level, ``"protein"`` or ``"peptide"``.
        key_added (str): ``adata.uns`` key for PCA-GSEA results (default ``"pca_gsea"``).
        pcs (list of int or None): 1-based PCs to include; ``None`` uses every PC in stored results.
        top_n (int): Cap on distinct pathways after ranking; must be >= 1 (required).
        fdr_cutoff (float or None): Same meaning as in ``plot_pca_gsea_pathway_vectors`` (default ``0.1``):
            eligibility on at least one PC plus ``top_n`` ranking gate. ``None`` disables both.
        size_scale (float): Multiplier for bubble area from ``-log10(FDR)``.
        cmap (str or Colormap): Colormap for NES-centered coloring.
        title_case_labels (bool): If True, format pathway tick labels for display.
        force (bool): If True, re-run ``pca_gsea`` for the PCs being shown.
        gsea_kwargs (dict or None): Forwarded to ``pca_gsea`` when auto-computing results.
        top_n_mode (str): ``"balanced"`` or ``"max_score"`` (see ``plot_pca_gsea_pathway_vectors``).
        include_pathways (str, iterable, or None): Keep only pathways matching these names.
        exclude_pathways (str, iterable, or None): Remove pathways matching these names.
        return_df (bool): If True, return ``(ax, bubble_df)`` with plot coordinates and sizes.

    Returns:
        matplotlib.axes.Axes, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

    Example:
        Bubble chart for the first three PCs, top 25 pathways by ranking, and return the table used for the plot:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(6, 8))
            ax, df = scplt.plot_pca_gsea_bubble(
                ax,
                pdata,
                pcs=[1, 2, 3],
                top_n=25,
                return_df=True,
            )
            ```

        Stricter FDR cutoff (0.05) and title-case pathway labels on the y-axis:
            ```python
            fig, ax = plt.subplots(figsize=(5, 9))
            scplt.plot_pca_gsea_bubble(
                ax,
                pdata,
                pcs=[1, 2],
                top_n=30,
                fdr_cutoff=0.05,
                title_case_labels=True,
            )
            ```
    """
    top_n = _validate_plot_top_n(top_n, what="pathways")
    requested_pcs = pcs
    if requested_pcs is None:
        adata = utils.get_adata(pdata, on)
        if key_added in adata.uns and "results" in adata.uns[key_added]:
            requested_pcs = [int(str(k).replace("PC", "")) for k in adata.uns[key_added]["results"].keys()]
    _, payload = _ensure_pca_gsea_payload(
        pdata=pdata,
        on=on,
        key_added=key_added,
        requested_pcs=requested_pcs,
        force=force,
        gsea_kwargs=gsea_kwargs,
    )
    long_df, matrix_df, fdr_df, missing_pc_keys = _build_pca_gsea_tables(payload=payload, pcs=pcs)
    if missing_pc_keys:
        print(
            f"{utils.format_log_prefix('warn')} Requested PCs missing from existing pca_gsea results: {missing_pc_keys}. "
            f"Showing NaN columns for unrun PCs. Rerun pca_gsea on these PCs (or set force=True)."
        )

    long_df, matrix_df, fdr_df = _apply_pathway_name_filters(
        long_df=long_df,
        matrix_df=matrix_df,
        fdr_df=fdr_df,
        include_pathways=include_pathways,
        exclude_pathways=exclude_pathways,
    )
    if fdr_cutoff is not None:
        _keep_mask = (fdr_df <= float(fdr_cutoff)).any(axis=1)
        matrix_df = matrix_df.loc[_keep_mask]
    if matrix_df.empty:
        raise ValueError("No pathways available after filtering for bubble plot.")
    sel_pathways = matrix_df.index.tolist()
    long_df = long_df[long_df["pathway_raw"].isin(sel_pathways)].copy()

    score_df = _compute_pc_score_df(
        matrix_df=matrix_df,
        fdr_df=fdr_df.reindex(index=matrix_df.index, columns=matrix_df.columns),
        fdr_cutoff=fdr_cutoff,
    )
    sel = _select_top_pathways(score_df=score_df, top_n=top_n, top_n_mode=top_n_mode)
    long_df = long_df[long_df["pathway_raw"].isin(sel)].copy()

    pathway_order = (
        long_df.assign(abs_nes=long_df["NES"].abs())
        .groupby("pathway_raw")["abs_nes"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    pc_order = sorted(long_df["pc"].unique(), key=lambda x: int(str(x).replace("PC", "")))
    long_df["pc_i"] = long_df["pc"].map({pc: i for i, pc in enumerate(pc_order)})
    long_df["pathway_i"] = long_df["pathway_raw"].map({p: i for i, p in enumerate(pathway_order)})

    fdr_safe = long_df["FDR q-val"].fillna(1.0).clip(lower=1e-300, upper=1.0)
    bubble_size = (-np.log10(fdr_safe)) * float(size_scale)
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    scatter = ax.scatter(
        long_df["pc_i"],
        long_df["pathway_i"],
        s=bubble_size,
        c=long_df["NES"],
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )

    ax.set_xticks(np.arange(len(pc_order)))
    ax.set_xticklabels(pc_order)
    ax.set_yticks(np.arange(len(pathway_order)))
    if title_case_labels:
        ax.set_yticklabels([_format_pathway_label(x) for x in pathway_order])
    else:
        ax.set_yticklabels([str(x).split("__", 1)[1] if "__" in str(x) else str(x) for x in pathway_order])
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Pathway")
    ax.set_title("PCA-GSEA bubble plot")
    plt.colorbar(scatter, ax=ax, label="NES")

    # Bubble size legend for -log10(FDR q-val)
    fdr_reference = np.array([0.1, 0.05, 0.01])
    legend_sizes = (-np.log10(fdr_reference.clip(min=1e-300))) * float(size_scale)
    handles = [
        ax.scatter([], [], s=s, facecolors="none", edgecolors="black", linewidths=0.6, label=f"-log10(FDR)={-np.log10(f):.1f}")
        for s, f in zip(legend_sizes, fdr_reference)
    ]
    ax.legend(handles=handles, title="Bubble size", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    bubble_df = long_df.copy()
    if title_case_labels:
        bubble_df["pathway"] = bubble_df["pathway"].map(_format_pathway_label)
    bubble_df["neg_log10_fdr"] = -np.log10(fdr_safe.values)
    bubble_df["bubble_size"] = bubble_size.values
    bubble_df = bubble_df[
        ["pathway", "pathway_raw", "library", "pc", "NES", "FDR q-val", "neg_log10_fdr", "bubble_size", "pc_i", "pathway_i"]
    ].rename(columns={"pc": "PC"})
    if return_df:
        return ax, bubble_df
    return ax

def plot_pca_gsea_heatmap(
    ax,
    pdata: pAnnData,
    on="protein",
    key_added="pca_gsea",
    pcs=None,
    top_n=30,
    fdr_cutoff=0.1,
    cmap="coolwarm",
    title_case_labels=True,
    force=False,
    gsea_kwargs=None,
    top_n_mode="balanced",
    include_pathways=None,
    exclude_pathways=None,
    return_df=False,
) -> Any:
    """
    Plot a pathway-by-principal-component heatmap of PCA-GSEA NES values.

    Cell color is NES; optional ``top_n`` trimming uses the same FDR-aware scoring as the bubble plot.
    Missing PCs in stored results produce NaN columns and a warning.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        pdata (pAnnData): Input object.
        on (str): Data level, ``"protein"`` or ``"peptide"``.
        key_added (str): ``adata.uns`` key for PCA-GSEA results (default ``"pca_gsea"``).
        pcs (list of int or None): 1-based PCs as columns; ``None`` uses all PCs in stored results.
        top_n (int): Maximum pathways to retain after ranking; must be >= 1 (required).
        fdr_cutoff (float or None): Same meaning as in ``plot_pca_gsea_pathway_vectors`` (default ``0.1``).
        cmap (str or Colormap): Heatmap colormap (diverging around zero is typical).
        title_case_labels (bool): If True, format pathway labels on the axis.
        force (bool): If True, re-run ``pca_gsea`` for the PCs being shown.
        gsea_kwargs (dict or None): Forwarded to ``pca_gsea`` when auto-computing results.
        top_n_mode (str): ``"balanced"`` or ``"max_score"``.
        include_pathways (str, iterable, or None): Keep only pathways matching these names.
        exclude_pathways (str, iterable, or None): Remove pathways matching these names.
        return_df (bool): If True, return ``(ax, heatmap_df)`` with the NES matrix used for plotting
            (pathway index may be formatted when ``title_case_labels=True``).

    Returns:
        matplotlib.axes.Axes, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

    Example:
        Heatmap of NES for four PCs and the 40 top-ranked pathways:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(5, 10))
            scplt.plot_pca_gsea_heatmap(ax, pdata, pcs=[1, 2, 3, 4], top_n=40)
            ```

        Diverging colormap with formatted pathway names on rows:
            ```python
            fig, ax = plt.subplots(figsize=(4, 12))
            scplt.plot_pca_gsea_heatmap(
                ax,
                pdata,
                pcs=[1, 2, 3],
                top_n=50,
                cmap="RdBu_r",
                title_case_labels=True,
            )
            ```
    """
    top_n = _validate_plot_top_n(top_n, what="pathways")
    requested_pcs = pcs
    if requested_pcs is None:
        adata = utils.get_adata(pdata, on)
        if key_added in adata.uns and "results" in adata.uns[key_added]:
            requested_pcs = [int(str(k).replace("PC", "")) for k in adata.uns[key_added]["results"].keys()]
    _, payload = _ensure_pca_gsea_payload(
        pdata=pdata,
        on=on,
        key_added=key_added,
        requested_pcs=requested_pcs,
        force=force,
        gsea_kwargs=gsea_kwargs,
    )
    long_df, matrix_df, fdr_df, missing_pc_keys = _build_pca_gsea_tables(payload=payload, pcs=pcs)
    if missing_pc_keys:
        print(
            f"{utils.format_log_prefix('warn')} Requested PCs missing from existing pca_gsea results: {missing_pc_keys}. "
            f"Showing NaN columns for unrun PCs. Rerun pca_gsea on these PCs (or set force=True)."
        )

    long_df, matrix_df, fdr_df = _apply_pathway_name_filters(
        long_df=long_df,
        matrix_df=matrix_df,
        fdr_df=fdr_df,
        include_pathways=include_pathways,
        exclude_pathways=exclude_pathways,
    )
    if fdr_cutoff is not None:
        _keep_mask = (fdr_df <= float(fdr_cutoff)).any(axis=1)
        matrix_df = matrix_df.loc[_keep_mask]
    matrix_df = matrix_df.dropna(how="all")
    if matrix_df.empty:
        raise ValueError("No pathways available after filtering for heatmap.")

    score_df = _compute_pc_score_df(
        matrix_df=matrix_df,
        fdr_df=fdr_df.reindex(index=matrix_df.index, columns=matrix_df.columns),
        fdr_cutoff=fdr_cutoff,
    )
    selected = _select_top_pathways(score_df=score_df, top_n=top_n, top_n_mode=top_n_mode)
    matrix_df = matrix_df.loc[selected]

    if title_case_labels:
        matrix_plot = matrix_df.copy()
        matrix_plot.index = [_format_pathway_label(x) for x in matrix_plot.index]
    else:
        matrix_plot = matrix_df

    payload["pathway_loadings"] = {"matrix": matrix_df.copy(), "fdr_qval": fdr_df.copy(), "long": long_df.copy()}
    sns.heatmap(matrix_plot, ax=ax, cmap=cmap, center=0, linewidths=0.2, cbar_kws={"label": "NES"})
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Pathway")
    ax.set_title("PCA-GSEA pathway x PC heatmap")
    if return_df:
        return ax, matrix_plot.copy()
    return ax

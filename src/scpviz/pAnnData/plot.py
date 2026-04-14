from __future__ import annotations

from typing import Any

from scpviz import plotting

class PlotMixin:
    def plot_counts(self, classes=None, y='protein_count', **kwargs):
        import seaborn as sns

        df = self.summary # type: ignore #, in base
        if classes is None:
            df=df.reset_index()
            classes = 'index'
        sns.violinplot(data=df, x=classes, y=y, **kwargs)

    def plot_rs(self, figsize=(10, 4)) -> None:
        """
        Visualize connectivity in the RS (protein × peptide) matrix.

        Generates side-by-side histograms:

        - Left: Number of peptides mapped to each protein
        - Right: Number of proteins associated with each peptide

        Args:
            figsize (tuple): Size of the matplotlib figure (default: (10, 4)).

        Returns:
            out (None): No return value; shows the figure interactively or closes it when using a non-interactive backend.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        
        if self.rs is None:
            print("⚠️ No RS matrix to plot.")
            return

        rs = self.rs
        prot_links = rs.getnnz(axis=1)
        pep_links = rs.getnnz(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].hist(prot_links, bins=50, color='gray')
        axes[0].set_title("Peptides per Protein")
        axes[0].set_xlabel("Peptide Count")
        axes[0].set_ylabel("Protein Frequency")

        axes[1].hist(pep_links, bins=50, color='gray')
        axes[1].set_title("Proteins per Peptide")
        axes[1].set_xlabel("Protein Count")
        axes[1].set_ylabel("Peptide Frequency")

        plt.tight_layout()
        backend = matplotlib.get_backend()
        if "agg" in backend.lower():
            # Running headless (e.g. pytest, CI)
            plt.close(fig)
        else:
            plt.show(block=False)

    def plot_abundance(self, ax=None, namelist=None, layer="X",
        on="protein", classes=None, return_df=False, order=None,
        palette=None, log=True, facet=None, height=4,
        aspect=0.5, plot_points=True, x_label="gene", kind="auto", **kwargs: Any,):
        """
        Wrapper for `scpviz.plotting.plot_abundance`.

        Plot abundance of proteins or peptides across samples.

        This function visualizes expression values for selected proteins or peptides
        using violin + box + strip plots, or bar plots when the number of replicates
        per group is small. Supports grouping, faceting, and custom ordering.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on. Ignored if `facet` is used.
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
            log (bool): If True, apply log2 transformation to abundance values. Default is True.
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
            ax (matplotlib.axes.Axes or seaborn.FacetGrid):
                The axis or facet grid containing the plot.
            df (pandas.DataFrame, optional): Returned if `return_df=True`.

        !!! example
            Plot abundance of two selected proteins:
                ```python
                pdata.plot_abundance(ax, namelist=['Slc12a2','Septin6'])
                ```            

        """
        return plotting.plot_abundance(ax=ax,pdata=self,namelist=namelist,
            layer=layer,on=on,classes=classes,return_df=return_df,
            order=order,palette=palette,log=log,facet=facet,
            height=height,aspect=aspect,plot_points=plot_points,x_label=x_label,
            kind=kind,**kwargs,
        )
    
    def plot_abundance_boxgrid(self, namelist=None, ax=None, layer="X", on="protein", classes="Grouping", return_df=False, 
        order=None, plot_type="box", log_scale=False, figsize=(2,2), palette=None, y_min=None, y_max=None,
        label_x=True, show_n=False, global_legend=True, box_kwargs=None, hline_kwargs=None, bar_kwargs=None, bar_error="sd", violin_kwargs=None,
        text_kwargs=None, strip_kwargs=None,):
        """
    Plot abundance values in a one-row panel of boxplots, mean-lines, bars, or violins.

    This function generates a clean horizontal panel, with one subplot per gene,
    using ``plot_type`` to select boxplots (default), mean-lines, bar plots, or
    violin plots. If ``log_scale=True``, abundance values are visualized in log10
    units (with zero or negative values clipped to 0 before transformation). The
    layout is optimized for compact manuscript figure panels and supports custom
    global legends, count annotations, and flexible formatting via keyword
    dictionaries.

    Args:
        namelist (list of str, optional): List of accessions or gene names to plot.
            If None, all available features are considered.
        ax (matplotlib.axes.Axes): Axis to plot on. Generates a new axis if None.
        layer (str): Data layer to use for abundance values. Default is ``"X"``.
        on (str): Data level to plot, either ``"protein"`` or ``"peptide"``.
        classes (str or None): Column in the returned abundance DataFrame to use for
            grouping samples. Defaults to ``"Grouping"``. If None, samples are not
            grouped.
        return_df (bool): If True, returns the DataFrame of replicate and summary values.
        order (list of str, optional): Ordered list to plot by (class order). If None,
            uses the order present in the data.
        plot_type (str): Type of plot, select from one of ``{"box", "line", "bar", "violin"}``.
            Defaults to ``"box"``.
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
            (used when ``plot_type="line"``). Note that ``half_width`` sets the length
            of the mean line.
        bar_kwargs (dict, optional): Additional arguments passed to bar plotting
            (used when ``plot_type="bar"``).
        bar_error (str, optional): Error bar for bar plot. Select from one of
            ``{"sd", "sem", None, <callable>}``, where callable takes a 1D array and
            returns a scalar error. Defaults to ``"sd"``.
        violin_kwargs (dict, optional): Additional arguments passed to ``sns.violinplot``
            (used when ``plot_type="violin"``).
        text_kwargs (dict, optional): Keyword arguments for count labels
            (e.g., fontsize, offset).
        strip_kwargs (dict, optional): Keyword arguments for strip (raw points),
            e.g. ``{"darken_factor": 0.65}``.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        axes (list of matplotlib.axes.Axes): One axis per gene.
        df (pandas.DataFrame, optional): Returned if ``return_df=True``.

    Note:
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
            figsize=(2,2.5),
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
            figsize=(2,2.5),
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
            figsize=(2,2.5),
        )
        plt.show()
        ```

        Violin plots (distribution-focused):
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["Gapdh", "Vcp", "Ahnak"],
            classes="condition",
            plot_type="violin",
            figsize=(2,2.5),
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
            figsize=(2,2.5),
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
        return plotting.plot_abundance_boxgrid(pdata=self, namelist=namelist, ax=ax, layer=layer, on=on,
            classes=classes, return_df=return_df, order=order, plot_type=plot_type, log_scale=log_scale,
            figsize=figsize, palette=palette, y_min=y_min, y_max=y_max,
            label_x=label_x, show_n=show_n, global_legend=global_legend,
            box_kwargs=box_kwargs, hline_kwargs=hline_kwargs, bar_kwargs=bar_kwargs, bar_error=bar_error,
            violin_kwargs=violin_kwargs, text_kwargs=text_kwargs, strip_kwargs=strip_kwargs,
        )

    def plot_pairwise_correlation(
        self,
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
        subset_mask=None,
        show_annotation_legend: bool = True,
        legend_anchor_x: float = 0.3,
        show_ticklabels: bool | None = None,
        ticklabels_auto_max_samples: int = 20,
    ):
        """
        Plot a pairwise proteome correlation heatmap across groups or samples in `.obs`.

        Thin wrapper around :func:`scpviz.plotting.plot_pairwise_correlation`.
        See that function's docstring for full parameter documentation.

        Note:
            The ``order`` argument lists **group labels** when ``show_samples=False``
            (including combined strings such as ``"AS, kd"`` for multi-column ``classes``),
            but lists **observation names** (``.prot.obs_names`` / ``.pep.obs_names``)
            when ``show_samples=True``.

        Returns:
            fig (matplotlib.figure.Figure): The created figure.
            ax (matplotlib.axes.Axes): The heatmap axes.

        Example:
            Sample-level heatmap with x-axis sample names forced on:
            ```python
            fig, ax = pdata.plot_pairwise_correlation(
                classes="cellline",
                show_samples=True,
                show_ticklabels=True,
            )
            ```
        """
        return plotting.plot_pairwise_correlation(
            pdata=self,
            classes=classes,
            on=on,
            layer=layer,
            method=method,
            order=order,
            show_samples=show_samples,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotation_cmap=annotation_cmap,
            figsize=figsize,
            text_size=text_size,
            colorbar_label=colorbar_label,
            annot=annot,
            annot_fmt=annot_fmt,
            annot_size=annot_size,
            title=title,
            force=force,
            subset_mask=subset_mask,
            show_annotation_legend=show_annotation_legend,
            legend_anchor_x=legend_anchor_x,
            show_ticklabels=show_ticklabels,
            ticklabels_auto_max_samples=ticklabels_auto_max_samples,
        )

    def plot_pca_gsea_pathway_vectors(
        self,
        ax,
        on="protein",
        key_added="pca_gsea",
        plot_pc=[1, 2],
        n_vectors=plotting.N_VECTORS_UNSET,
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
    ):
        """
        Overlay PCA-GSEA pathways as arrows in a two-dimensional PCA sample space.

        Delegates to ``scpviz.plotting.plot_pca_gsea_pathway_vectors`` with ``pdata=self``.

        Args:
            ax (matplotlib.axes.Axes): Target axis (2D).
            on (str): Data level, ``"protein"`` or ``"peptide"``.
            key_added (str): ``adata.uns`` key for PCA-GSEA results (default ``"pca_gsea"``).
            plot_pc (list of int): Exactly two 1-based PCs, e.g. ``[1, 2]``.
            n_vectors (int, sequence, ``None``, or unset): Auto top-N on rows not in ``namelist``. Default ``12``
                when ``namelist`` is unset; when ``namelist`` is set, default is no extra top-N unless passed.
            fdr_cutoff (float or None): FDR threshold for the default ranking path; with ``namelist``,
                pathways are shown even if they fail FDR, with a printed warning when applicable.
            arrow_scale (float): Scale factor for arrow length relative to axis span.
            pca_kwargs (dict or None): Additional arguments passed to ``plot_pca`` when ``show_samples=True``.
            show_samples (bool): If True, plot samples first; if False, draw only axes, grid lines, and arrows.
            title_case_labels (bool): If True, format pathway labels for display.
            force (bool): If True, re-run ``pca_gsea`` for ``plot_pc``.
            gsea_kwargs (dict or None): Forwarded to ``pca_gsea`` when results are auto-computed.
            adjust_labels (bool): If True, run ``adjust_text`` to reduce label overlap.
            adjust_text_kwargs (dict or None): Extra keyword arguments for ``adjust_text``.
            text_positions (dict or None): Optional manual label positions.
            lock_text_positions (bool): If True, labels with entries in ``text_positions`` are not moved by
                ``adjust_text``.
            top_n_mode (str): ``"balanced"`` or ``"max_score"``. Used only when ``n_vectors`` is an int.
            exclude_pathways (str, iterable, or None): Remove pathways matching these names.
            namelist (list of str or None): Pathways to plot first; optional extra top-N from ``n_vectors`` on the rest.
            cmap (dict or None): Per-pathway colors (raw, formatted, then case-insensitive keys).
            xlim (tuple or None): X-axis limits applied before arrow scaling (releases fixed aspect first).
            ylim (tuple or None): Y-axis limits, same stage as ``xlim``.
            return_df (bool): If True, also return a DataFrame with NES, FDR, and label positions.

        Returns:
            out (matplotlib.axes.Axes | tuple): The axis, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.
        """
        return plotting.plot_pca_gsea_pathway_vectors(
            ax=ax,
            pdata=self,
            on=on,
            key_added=key_added,
            plot_pc=plot_pc,
            n_vectors=n_vectors,
            fdr_cutoff=fdr_cutoff,
            arrow_scale=arrow_scale,
            pca_kwargs=pca_kwargs,
            show_samples=show_samples,
            title_case_labels=title_case_labels,
            force=force,
            gsea_kwargs=gsea_kwargs,
            adjust_labels=adjust_labels,
            adjust_text_kwargs=adjust_text_kwargs,
            text_positions=text_positions,
            lock_text_positions=lock_text_positions,
            top_n_mode=top_n_mode,
            exclude_pathways=exclude_pathways,
            namelist=namelist,
            cmap=cmap,
            xlim=xlim,
            ylim=ylim,
            return_df=return_df,
        )

    def plot_pca_protein_vectors(
        self,
        ax,
        on="protein",
        plot_pc=(1, 2),
        gene_col="Genes",
        n_vectors=plotting.N_VECTORS_UNSET,
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
    ):
        """
        Overlay protein PCA loadings as arrows in a two-dimensional sample PCA space.

        Delegates to ``scpviz.plotting.plot_pca_protein_vectors`` with ``pdata=self``. Arrows use feature
        loadings from ``adata.uns['pca']['PCs']`` (from ``pAnnData.pca``), not GSEA NES. Geometry matches
        ``plot_pca_gsea_pathway_vectors``: each arrow runs from the origin in the direction
        ``(loading_on_PCx, loading_on_PCy)``, with length rescaled from the current axis limits for visibility.
        Labels default to the ``gene_col`` column in ``.var`` when present, otherwise ``.var_names``.

        Args:
            ax (matplotlib.axes.Axes): Target axis (2D).
            on (str): Data level, ``"protein"`` or ``"peptide"``.
            plot_pc (tuple or list of int): Exactly two 1-based PCs.
            gene_col (str): Column in ``.var`` for display labels; missing column falls back to ``.var_names``.
            n_vectors (int, sequence, ``None``, or unset): Auto top-N on rows not in ``namelist``. Default ``20``
                when ``namelist`` is unset; when ``namelist`` is set, default is no extra top-N unless passed.
                ``min_abs_loading_for_top_n`` gates scores on the remainder.
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
            namelist (list of str or None): Gene labels (matrix index, exact ``str`` match) to include first;
                combined with ``n_vectors`` on the remaining rows. Genes in ``exclude_genes`` are dropped.
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
            out (matplotlib.axes.Axes | tuple): The axis, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

        Example:
            Top-loading genes on PC1 vs PC2 over the sample PCA scatter, returning arrow and text coordinates:
                ```python
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax, vec = pdata.plot_pca_protein_vectors(
                    ax,
                    plot_pc=[1, 2],
                    n_vectors=25,
                    return_df=True,
                )
                ```

            Split-axis selection: top loadings on PC1 and PC3 separately, then union:
                ```python
                fig, ax = plt.subplots()
                pdata.plot_pca_protein_vectors(
                    ax,
                    plot_pc=[1, 3],
                    n_vectors=[5, 3],
                    adjust_labels=False,
                )
                ```

            Explicit genes with colors and axis limits:
                ```python
                fig, ax = plt.subplots()
                pdata.plot_pca_protein_vectors(
                    ax,
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
                pdata.plot_pca_protein_vectors(
                    ax,
                    plot_pc=[1, 2],
                    n_vectors=20,
                    show_samples=False,
                    adjust_labels=False,
                )
                ```
        """
        return plotting.plot_pca_protein_vectors(
            ax=ax,
            pdata=self,
            on=on,
            plot_pc=plot_pc,
            gene_col=gene_col,
            n_vectors=n_vectors,
            arrow_scale=arrow_scale,
            pca_kwargs=pca_kwargs,
            show_samples=show_samples,
            title_case_labels=title_case_labels,
            adjust_labels=adjust_labels,
            adjust_text_kwargs=adjust_text_kwargs,
            text_positions=text_positions,
            lock_text_positions=lock_text_positions,
            min_abs_loading_for_top_n=min_abs_loading_for_top_n,
            top_n_mode=top_n_mode,
            exclude_genes=exclude_genes,
            namelist=namelist,
            cmap=cmap,
            xlim=xlim,
            ylim=ylim,
            return_df=return_df,
        )

    def plot_pca_gsea_bubble(
        self,
        ax,
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
    ):
        """
        Wrapper for `scpviz.plotting.plot_pca_gsea_bubble`.

        Bubble chart of pathways by principal component: color is NES, size reflects FDR.

        Args:
            ax (matplotlib.axes.Axes): Target axis.
            on (str): ``"protein"`` or ``"peptide"``.
            key_added (str): ``adata.uns`` key for PCA-GSEA results.
            pcs (list of int or None): PCs to show; ``None`` uses all stored PCs.
            top_n (int): Cap on pathways after ranking; must be >= 1.
            fdr_cutoff (float or None): FDR threshold (default ``0.1``); see plotting function docstring.
            size_scale (float): Scales bubble area from ``-log10(FDR)``.
            cmap (str or Colormap): Colormap for NES.
            title_case_labels (bool): Format pathway tick labels.
            force (bool): Re-run ``pca_gsea`` when True.
            gsea_kwargs (dict or None): Forwarded to ``pca_gsea`` if run automatically.
            top_n_mode (str): ``"balanced"`` or ``"max_score"``.
            include_pathways (str, iterable, or None): Restrict to these names.
            exclude_pathways (str, iterable, or None): Drop these names.
            return_df (bool): If True, return ``(ax, bubble_df)``.

        Returns:
            out (matplotlib.axes.Axes | tuple): The axis, or ``(ax, pandas.DataFrame)`` when ``return_df=True``.
        """
        return plotting.plot_pca_gsea_bubble(
            ax=ax,
            pdata=self,
            on=on,
            key_added=key_added,
            pcs=pcs,
            top_n=top_n,
            fdr_cutoff=fdr_cutoff,
            size_scale=size_scale,
            cmap=cmap,
            title_case_labels=title_case_labels,
            force=force,
            gsea_kwargs=gsea_kwargs,
            top_n_mode=top_n_mode,
            include_pathways=include_pathways,
            exclude_pathways=exclude_pathways,
            return_df=return_df,
        )

    def plot_pca_gsea_heatmap(
        self,
        ax,
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
    ):
        """
        Wrapper for `scpviz.plotting.plot_pca_gsea_heatmap`.

        Heatmap of NES values (pathways × principal components) from PCA-GSEA results.

        Args:
            ax (matplotlib.axes.Axes): Target axis.
            on (str): ``"protein"`` or ``"peptide"``.
            key_added (str): ``adata.uns`` key for PCA-GSEA results.
            pcs (list of int or None): PCs as columns; ``None`` uses all stored PCs.
            top_n (int): Cap on pathways after ranking; must be >= 1.
            fdr_cutoff (float or None): FDR threshold (default ``0.1``); see plotting function docstring.
            cmap (str or Colormap): Heatmap colormap.
            title_case_labels (bool): Format pathway labels on the axis.
            force (bool): Re-run ``pca_gsea`` when True.
            gsea_kwargs (dict or None): Forwarded to ``pca_gsea`` if run automatically.
            top_n_mode (str): ``"balanced"`` or ``"max_score"``.
            include_pathways (str, iterable, or None): Restrict to these names.
            exclude_pathways (str, iterable, or None): Drop these names.
            return_df (bool): If True, return ``(ax, heatmap_df)`` (see plotting function docstring).

        Returns:
            out (matplotlib.axes.Axes | tuple): The axis, or ``(ax, pandas.DataFrame)`` when ``return_df=True``.
        """
        return plotting.plot_pca_gsea_heatmap(
            ax=ax,
            pdata=self,
            on=on,
            key_added=key_added,
            pcs=pcs,
            top_n=top_n,
            fdr_cutoff=fdr_cutoff,
            cmap=cmap,
            title_case_labels=title_case_labels,
            force=force,
            gsea_kwargs=gsea_kwargs,
            top_n_mode=top_n_mode,
            include_pathways=include_pathways,
            exclude_pathways=exclude_pathways,
            return_df=return_df,
        )
from __future__ import annotations

from typing import Any

from scpviz import plotting

class PlotMixin:
    """
    Convenient plotting wrappers for visualizing a `pAnnData` object.

    Most methods forward to functions in the ``scpviz.plotting`` module, keeping
    axes-level control while binding abundance, metadata, and RS context from
    the parent object.

    Functions:
        plot_counts: Violin plot of per-sample count metrics from ``.summary``.
        plot_rs: Histograms of protein–peptide connectivity in the RS matrix.
        plot_abundance: Abundance strip/box/violin plot for selected features.
        plot_abundance_boxgrid: Faceted abundance box/violin/line/bar grid.
        plot_pairwise_correlation: Pairwise sample correlation heatmap.
        plot_pca_protein_vectors: PCA biplot with protein loading vectors.
        plot_pca_gsea_pathway_vectors: PCA biplot with GSEA pathway vectors.
        plot_pca_gsea_bubble: Bubble plot of PCA–GSEA enrichment results.
        plot_pca_gsea_heatmap: Heatmap of PCA–GSEA enrichment results.
        plot_grouped_heatmap: Grouped abundance heatmap.
        plot_clustered_heatmap: Clustered abundance heatmap.
        plot_clustermap: Overview heatmap with row + column dendrograms.
    """

    def plot_counts(self, classes=None, y='protein_count', **kwargs):
        """
        Violin plot of per-sample count metrics from ``pdata.summary``.

        Args:
            classes: Column in ``summary`` for the x-axis; if ``None``, uses the summary index.
            y (str): Column to plot (default ``"protein_count"``).
            **kwargs: Forwarded to ``seaborn.violinplot``.
        """
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
        Plot abundance of proteins or peptides across samples.

        Thin wrapper around :func:`scpviz.plotting.plot_abundance`.

        This function visualizes expression values for selected proteins or peptides
        using violin + box + strip plots, or bar plots when the number of replicates
        per group is small. Supports grouping, faceting, and custom ordering.

        **Important default behavior:**
        - This method defaults to ``log=True`` (log2-transformed abundances on a linear y-axis).
        - With ``log=False``, abundances remain **raw** and the **y-axis is log10-scaled**, so the
          plot displays log10(abundance) without transforming the underlying values.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on. Ignored if ``facet`` is used.
            namelist (list of str, optional): List of accessions or gene names to plot.
                If None, all available features are considered.
            layer (str): Data layer to use for abundance values. Default is ``"X"``.
            on (str): Data level to plot, either ``"protein"`` or ``"peptide"``.
            classes (str or list of str, optional): ``.obs`` column(s) to use for grouping
                samples. Determines coloring and grouping structure.
            return_df (bool): If True, returns the DataFrame of replicate and summary values.
            order (dict or list, optional): Custom order of classes. For dictionary input,
                keys are class names and values are the ordered categories.
                Example: ``order = {"condition": ["sc", "kd"]}``.
            palette (list or dict, optional): Color palette mapping groups to colors.
            log (bool): If True, apply log2 transformation to abundance values (default here).
                If False, raw values are used and the y-axis is log10-scaled.
            facet (str, optional): ``.obs`` column to facet by, creating multiple subplots.
            height (float): Height of each facet plot. Default is 4.
            aspect (float): Aspect ratio of each facet plot. Default is 0.5.
            plot_points (bool): Whether to overlay stripplot of individual samples.
            x_label (str): Label for the x-axis, either ``"gene"`` or ``"accession"``.
            kind (str): Type of plot. Options:

                - ``"auto"``: Default; uses barplot if groups have ≤ 3 samples, otherwise violin.
                - ``"violin"``: Always use violin + box + strip.
                - ``"bar"``: Always use barplot.

            **kwargs (Any): Additional keyword arguments passed to seaborn plotting functions.

        Returns:
            ax (matplotlib.axes.Axes or seaborn.FacetGrid):
                The axis or facet grid containing the plot.
            df (pandas.DataFrame, optional): Returned if ``return_df=True``.

        !!! example
            Plot abundance of selected marker proteins grouped by cell line and condition:
            ```python
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 4))
            pdata.plot_abundance(
                ax, namelist=["GAPDH", "TUBB", "ACTB"], classes=["cellline", "condition"]
            )
            plt.show()
            ```

            ![Plot abundance](../../assets/plots/plot_abundance.png)
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
        text_kwargs=None, strip_kwargs=None, sig_pairs=None, sig_kwargs=None, nd_kwargs=None):
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
        namelist (list of str, optional): List of accessions or gene names to plot.
            If None, all available features are considered.
        ax (matplotlib.axes.Axes): Axis to plot on. Generates a new axis if None.
        layer (str): Data layer to use for abundance values. Default is ``"X"``.
        on (str): Data level to plot, either ``"protein"`` or ``"peptide"``.
        classes (str or list of str, optional): Column in ``.obs`` or list of columns
            for compound grouping. Defaults to ``"Grouping"``. If None, samples are
            not grouped.
        return_df (bool): If True, returns the DataFrame of replicate and summary values.
        order (list of str, optional): Ordered list to plot by. If None, plots by
            given dataframe order.
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
        hline_kwargs (dict, optional): Styling for mean segments when ``plot_type="line"``.
            Recognized keys include Matplotlib ``hlines`` options plus ``half_width``
            (float, default 0.15): half the segment length in x-axis units; use a
            smaller value when dodged groups would otherwise overlap.
        bar_kwargs (dict, optional): Passed to ``Axes.bar`` when ``plot_type="bar"``
            (e.g. ``width`` in x-axis units; default here is 0.3—decrease when many
            hue levels overlap on one gene tick).
        bar_error (str, optional): Error bar for bar plot. Select from one of
            ``{"sd", "sem", None, <callable>}``, where callable takes a 1D array and
            returns a scalar error. Defaults to ``"sd"``.
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
        return plotting.plot_abundance_boxgrid(pdata=self, namelist=namelist, ax=ax, layer=layer, on=on,
            classes=classes, return_df=return_df, order=order, plot_type=plot_type, log_scale=log_scale,
            figsize=figsize, palette=palette, y_min=y_min, y_max=y_max,
            label_x=label_x, show_n=show_n, global_legend=global_legend,
            box_kwargs=box_kwargs, hline_kwargs=hline_kwargs, bar_kwargs=bar_kwargs, bar_error=bar_error,
            violin_kwargs=violin_kwargs, text_kwargs=text_kwargs, strip_kwargs=strip_kwargs,
            sig_pairs=sig_pairs, sig_kwargs=sig_kwargs, nd_kwargs=nd_kwargs,
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
        Plot a pairwise protein/peptide abundance correlation heatmap across groups or samples in ``.obs``.

        Thin wrapper around :func:`scpviz.plotting.plot_pairwise_correlation`.

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
            classes: ``.obs`` column(s) defining groups — passed to ``pairwise_correlation``.
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
                  build a list of those obs names in the desired sequence.

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
                track in a dedicated GridSpec column right of the colorbar.
            legend_anchor_x: Horizontal anchor for annotation legends inside the legend
                column, in axes coordinates (``0`` = left edge of that column, ``1`` = right).
                Larger values shift legends to the **right**, away from the colorbar.
                Typical values: about ``0.15`` to ``0.45`` (default ``0.3``).
            show_ticklabels: When ``show_samples=True``, controls sample names on the
                **x-axis** only. ``None`` (default) shows ticks if
                ``n_samples <= ticklabels_auto_max_samples`` and otherwise hides them.
                ``True`` / ``False`` force on or off. Ignored when ``show_samples=False``.
            ticklabels_auto_max_samples: When ``show_ticklabels is None`` and
                ``show_samples=True``, sample names are shown only if the sample count is
                at most this value (default ``20``). Must be >= 1.

        Returns:
            ``(fig, ax_heatmap)``.

        Note:
            Heatmap row (y) tick labels are always omitted (symmetric matrix; x-axis labels
            carry sample or group names as applicable).
            The ``order`` argument lists **group labels** when ``show_samples=False``
            (including combined strings such as ``"AS, kd"`` for multi-column ``classes``),
            but lists **observation names** when ``show_samples=True``.

        Raises:
            ValueError: If ``sample_matrix`` is missing when ``show_samples=True``, or if
                ``ticklabels_auto_max_samples`` < 1.

        Example:
            Group-level heatmap (``show_samples=False``, default):
            ```python
            fig, ax = pdata.plot_pairwise_correlation(classes="cellline", method="pearson")
            ```

            ![Plot pairwise correlation](../../assets/plots/plot_pairwise_correlation.png)

            Sample × sample heatmap with x-axis sample names forced on:
            ```python
            fig, ax = pdata.plot_pairwise_correlation(
                classes="cellline",
                show_samples=True,
                show_ticklabels=True,
            )
            ```

            Custom row/column order without recomputing (group labels must match the matrix):
            ```python
            fig, ax = pdata.plot_pairwise_correlation(
                classes=["cellline", "treatment"],
                order=["AS, kd", "BE, sc", "AS, sc", "BE, kd"],
            )
            ```

            Subset of samples and no annotation legends:
            ```python
            mask = pdata.prot.obs["cellline"].eq("AS").to_numpy()
            fig, ax = pdata.plot_pairwise_correlation(
                classes="treatment", subset_mask=mask, show_annotation_legend=False
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

        Thin wrapper around :func:`scpviz.plotting.plot_pca_gsea_pathway_vectors`.

        Each arrow encodes normalized enrichment scores (NES) on two principal components taken from
        ``adata.uns[key_added]['results']`` (from ``pca_gsea``). Arrow endpoints are rescaled using the
        current axis limits so pathways remain visible; they are not plotted in the same numeric units as
        sample coordinates. When ``show_samples`` is True, the sample PCA scatter is drawn first via
        ``plot_pca``.

        Args:
            ax (matplotlib.axes.Axes): Target axis (2D).
            on (str): Data level, ``"protein"`` or ``"peptide"``.
            key_added (str): ``adata.uns`` key for PCA-GSEA results (default ``"pca_gsea"``).
            plot_pc (list of int): Exactly two 1-based PCs, e.g. ``[1, 2]``.
            n_vectors (int, sequence, ``None``, or unset): Caps auto-selected pathways (after ``namelist`` rows).
                Default when ``namelist`` is ``None`` is ``12``; when ``namelist`` is set, default is no extra
                top-N unless you pass ``n_vectors`` explicitly. If an int (>= 1), uses ``top_n_mode`` on rows not
                already chosen by ``namelist``. If ``[nx, ny]``, split-axis top union on that remainder.
            fdr_cutoff (float or None): For **auto-selected** rows: pathway-level FDR filtering and score gating.
                **Namelist** pathways skip the row FDR filter; a **warning** is printed per named pathway when
                ``fdr_cutoff`` is not ``None`` and no plotted PC passes FDR.
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
                pathway, or library).
            namelist (list of str or None): Pathways to always include first. Shown even if they fail FDR;
                ``exclude_pathways`` still applies first. Combined with ``n_vectors`` on the remaining rows.
            cmap (dict or None): Per-pathway colors; lookup raw ``Term``, formatted label, then case-insensitive keys.
            xlim (tuple or None): Applied after scatter / empty axes, before arrow scaling.
            ylim (tuple or None): Same as ``xlim``.
            return_df (bool): If True, also return a DataFrame with NES, FDR, and label positions.

        Returns:
            matplotlib.axes.Axes, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

        Example:
            Default overlay on PC1 vs PC2 with label de-cluttering:
            ```python
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax, vec_df = pdata.plot_pca_gsea_pathway_vectors(
                ax,
                plot_pc=[1, 2],
                adjust_text_kwargs={"expand": (1.3, 1.3)},
                return_df=True,
            )
            ```

            Reuse label positions from a previous run:
            ```python
            manual = {
                row["pathway_raw"]: (row["text_x"], row["text_y"])
                for _, row in vec_df.iterrows()
            }
            ax = pdata.plot_pca_gsea_pathway_vectors(
                ax,
                plot_pc=[1, 2],
                text_positions=manual,
                lock_text_positions=True,
            )
            ```
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

        Thin wrapper around :func:`scpviz.plotting.plot_pca_protein_vectors`.

        Arrows use feature loadings from ``adata.uns['pca']['PCs']`` (from ``pAnnData.pca``), not GSEA NES.
        Geometry matches ``plot_pca_gsea_pathway_vectors``: each arrow runs from the origin in the direction
        ``(loading_on_PCx, loading_on_PCy)``, with length rescaled from the current axis limits for visibility.
        Labels default to the ``gene_col`` column in ``.var`` when present, otherwise ``.var_names``.

        Args:
            ax (matplotlib.axes.Axes): Target axis (2D).
            on (str): Data level, ``"protein"`` or ``"peptide"``.
            plot_pc (tuple or list of int): Exactly two 1-based PCs.
            gene_col (str): Column in ``.var`` for display labels; missing column falls back to ``.var_names``.
            n_vectors (int, sequence, ``None``, or unset): Caps **auto-selected** proteins (rows not already taken
                by ``namelist``). Default when ``namelist`` is ``None`` is ``20``; when ``namelist`` is set, default
                is no extra top-N unless you pass ``n_vectors`` explicitly. If an int (>= 1), uses ``top_n_mode``.
                If ``[nx, ny]``, split-axis top union on that remainder.
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
                Combined with ``n_vectors`` on the remaining rows. Genes in ``exclude_genes`` are dropped.
            cmap (dict or None): Map gene label to a matplotlib color; lookup tries raw name, formatted label,
                then case-insensitive keys. Default ``None`` draws arrows and labels in black.
            xlim (tuple or None): Applied after the PCA scatter and **before** arrow length scaling.
            ylim (tuple or None): Same stage as ``xlim``.
            return_df (bool): If True, return ``(ax, vector_df)`` with loadings and arrow/text coordinates.

        Returns:
            matplotlib.axes.Axes, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

        Example:
            Show top protein loadings on PC1 vs PC2 on sample PCA scatter:
            ```python
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 4))
            pdata.pca(on="protein")
            pdata.plot_pca_protein_vectors(ax, n_vectors=10)
            plt.show()
            ```

            ![Plot PCA protein vectors](../../assets/plots/plot_pca_protein_vectors.png)

            Top-loading genes with returned coordinates:
            ```python
            fig, ax = plt.subplots()
            ax, vec = pdata.plot_pca_protein_vectors(
                ax,
                plot_pc=[1, 2],
                n_vectors=25,
                return_df=True,
            )
            ```

            Split-axis selection on PC1 and PC3:
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

            Loading arrows only (no sample points):
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
        size_scale=0.85,
        size_fdr_cap=5.0,
        pc_pad=0.6,
        cbar_scale=1.0,
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
        Plot PCA-GSEA results as a bubble chart (principal component versus pathway).

        Thin wrapper around :func:`scpviz.plotting.plot_pca_gsea_bubble`.

        Bubble color encodes NES; bubble area reflects significance (``-log10(FDR)``). Rows and columns
        are ordered by pathway and PC. If ``pcs`` is omitted, all PCs present in stored results are used.

        Bubble diameters are sized relative to the current figure/axes and the PC × pathway grid, so
        choose ``figsize`` before calling this function. Marker areas are computed once at plot time
        (they do not auto-update on interactive window resize). Column spacing uses ``pc_pad``;
        pathway rows keep unit pitch with box aspect matched to the data spans. ``cbar_scale``
        scales NES colorbar height (``1`` = default); the size legend stays stacked under it.

        Args:
            ax (matplotlib.axes.Axes): Target axis.
            on (str): Data level, ``"protein"`` or ``"peptide"``.
            key_added (str): ``adata.uns`` key for PCA-GSEA results (default ``"pca_gsea"``).
            pcs (list of int or None): 1-based PCs to include; ``None`` uses every PC in stored results.
            top_n (int): Cap on distinct pathways after ranking; must be >= 1.
            fdr_cutoff (float or None): Same meaning as in ``plot_pca_gsea_pathway_vectors`` (default ``0.1``):
                eligibility on at least one PC plus ``top_n`` ranking gate. ``None`` disables both.
            size_scale (float): Max bubble diameter as a fraction of the smaller cell pitch
                (axes size in points divided by grid count). Default ``0.85`` ≈ nearly fill a cell;
                values ``>1`` allow overlap. This is **not** an absolute points² multiplier.
            size_fdr_cap (float): Clip ``-log10(FDR)`` used for sizing (default ``5.0``, FDR ≈ ``1e-5``).
                Area scales as ``clipped / size_fdr_cap`` of the max area.
            pc_pad (float): Half-spacing between PC columns in data units (also the x-edge margin).
                Column centers are ``2 * pc_pad`` apart. Default ``0.6`` (``0.5`` recovers unit spacing).
            cbar_scale (float): Vertical scale for the NES colorbar relative to the default height
                (default ``1.0``; ``<1`` shorter, ``>1`` taller). Size legend follows the colorbar.
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
            Bubble chart for the first three PCs, top 25 pathways:
            ```python
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 8))
            pdata.plot_pca_gsea_bubble(ax, pcs=[1, 2, 3], top_n=25)
            ```

            Colorbar height via ``cbar_scale`` (``0.5`` / ``1`` / ``1.5``):
            ```python
            for scale in (0.5, 1.0, 1.5):
                fig, ax = plt.subplots(figsize=(6, 8))
                pdata.plot_pca_gsea_bubble(ax, pcs=[1, 2, 3], top_n=25, cbar_scale=scale)
            ```

            Tighter bubbles, or allow more extreme FDR contrast in marker sizes:
            ```python
            fig, ax = plt.subplots(figsize=(6, 8))
            pdata.plot_pca_gsea_bubble(
                ax,
                pcs=[1, 2, 3],
                top_n=25,
                size_scale=0.5,      # smaller max diameter vs cell pitch
                size_fdr_cap=10,     # clip -log10(FDR) at 10 (FDR ≈ 1e-10)
            )
            ```

            Wider gaps between PC columns:
            ```python
            fig, ax = plt.subplots(figsize=(6, 8))
            pdata.plot_pca_gsea_bubble(ax, pcs=[1, 2, 3], top_n=25, pc_pad=0.75)
            ```

            Stricter FDR cutoff and title-case pathway labels:
            ```python
            fig, ax = plt.subplots(figsize=(5, 9))
            pdata.plot_pca_gsea_bubble(
                ax,
                pcs=[1, 2],
                top_n=30,
                fdr_cutoff=0.05,
                title_case_labels=True,
            )
            ```
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
            size_fdr_cap=size_fdr_cap,
            pc_pad=pc_pad,
            cbar_scale=cbar_scale,
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
        Plot a pathway-by-principal-component heatmap of PCA-GSEA NES values.

        Thin wrapper around :func:`scpviz.plotting.plot_pca_gsea_heatmap`.

        Cell color is NES; optional ``top_n`` trimming uses the same FDR-aware scoring as the bubble plot.
        Missing PCs in stored results produce NaN columns and a warning.

        Args:
            ax (matplotlib.axes.Axes): Target axis.
            on (str): Data level, ``"protein"`` or ``"peptide"``.
            key_added (str): ``adata.uns`` key for PCA-GSEA results (default ``"pca_gsea"``).
            pcs (list of int or None): 1-based PCs as columns; ``None`` uses all PCs in stored results.
            top_n (int): Maximum pathways to retain after ranking; must be >= 1.
            fdr_cutoff (float or None): Same meaning as in ``plot_pca_gsea_pathway_vectors`` (default ``0.1``).
            cmap (str or Colormap): Heatmap colormap (diverging around zero is typical).
            title_case_labels (bool): If True, format pathway labels on the axis.
            force (bool): If True, re-run ``pca_gsea`` for the PCs being shown.
            gsea_kwargs (dict or None): Forwarded to ``pca_gsea`` when auto-computing results.
            top_n_mode (str): ``"balanced"`` or ``"max_score"``.
            include_pathways (str, iterable, or None): Keep only pathways matching these names.
            exclude_pathways (str, iterable, or None): Remove pathways matching these names.
            return_df (bool): If True, return ``(ax, heatmap_df)`` with the NES matrix used for plotting.

        Returns:
            matplotlib.axes.Axes, or ``(ax, pandas.DataFrame)`` if ``return_df=True``.

        Example:
            Heatmap of NES for four PCs and the 40 top-ranked pathways:
            ```python
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 10))
            pdata.plot_pca_gsea_heatmap(ax, pcs=[1, 2, 3, 4], top_n=40)
            ```

            Diverging colormap with formatted pathway names on rows:
            ```python
            fig, ax = plt.subplots(figsize=(4, 12))
            pdata.plot_pca_gsea_heatmap(
                ax,
                pcs=[1, 2, 3],
                top_n=50,
                cmap="RdBu_r",
                title_case_labels=True,
            )
            ```
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

    def plot_grouped_heatmap(
        self,
        protein_groups,
        classes,
        on="protein",
        sort_by=None,
        layer="X",
        display_scale="auto",
        group_colors=None,
        header_colors=None,
        cmap=None,
        row_spacing=True,
        column_spacing=True,
        header_spacing=0.06,
        header_height=0.35,
        group_bar_pad=0.25,
        group_bar_width=0.4,
        sample_label_col=None,
        figsize=None,
        text_size=8,
        cbar_scale=1.0,
        legend_width=None,
        auto_log2=True,
        gene_col="Genes",
        separate_legend=False,
        **kwargs,
    ):
        """
        Plot a curated protein-group × sample heatmap with header strips.

        Thin wrapper around :func:`scpviz.plotting.plot_grouped_heatmap`.
        Filter samples on the object beforehand if you need a subset.

        Args:
            protein_groups (dict): ``group_name → list of gene/accession ids``.
            classes (list of str): ``.obs`` columns for headers and sample order.
            on (str): ``"protein"`` or ``"peptide"``.
            sort_by (dict, optional): Per-class category order.
            layer (str): Abundance layer (default ``"X"``).
            display_scale (str): ``"auto"`` / ``"zscore"`` / ``"log"`` / ``"raw"``.
            group_colors (dict, optional): Override colors for group brackets/legend.
            header_colors (dict, optional): Header strip colors. Nested
                ``{class: {category: color}}``, or flat ``{category: color}``
                when ``len(classes)==1``. Each class is its own row (not one
                color per class combination).
            cmap (str, optional): Colormap; ``None`` auto-selects by display scale.
            row_spacing (bool or float): Vertical gaps between protein groups.
                ``True`` (default) uses a half-cell gap; ``False``/``0`` = none;
                float scales that default (same semantics as ``column_spacing``).
            column_spacing (bool or float): Horizontal gaps between sample leaf
                blocks. ``True`` (default) uses the same half-cell thickness as
                ``row_spacing=True``; ``False``/``0`` = none; float scales the
                default.
            header_spacing (float): Space between header strips and the heatmap
                (default ``0.06``).
            header_height (float): Relative GridSpec height for each header strip
                (default ``0.35``); larger = thicker bands vs the heatmap body.
            group_bar_pad (float): Horizontal gap between heatmap and group bars
                (default 0.25).
            group_bar_width (float): Width of colored group bars in sample-column
                units (default ``0.4``).
            sample_label_col (str, optional): ``.obs`` / ``.summary`` column for
                bottom tick labels (e.g. ``"replicate"``). Default uses ``obs_names``.
            figsize (tuple, optional): Figure size; auto if None.
            text_size (int): Base font size for ticks, colorbar, and legends
                (default 8; same convention as ``plot_pairwise_correlation``).
            cbar_scale (float): Vertical scale factor for the colorbar (default
                ``1.0``). Legends stack tightly below it; also applies with
                ``separate_legend=True``.
            legend_width (float, optional): Left margin for colorbar + legends.
                `None` (default) auto-sizes from legend text; pass a float to
                override. Ignored when `separate_legend=True`.
            auto_log2 (bool): In-memory log2 when layer looks linear.
            gene_col (str): ``.var`` gene label column.
            separate_legend (bool): If True, return ``(fig, legend_fig)`` with
                colorbar and legends on a second figure.
            **kwargs: Forwarded (e.g. ``title``).

        Returns:
            fig (matplotlib.figure.Figure): Constructed figure, or
            ``(fig, legend_fig)`` when ``separate_legend=True``.

        Example:
            Grouped pathway heatmap with condition headers:
            ```python
            fig = pdata.plot_grouped_heatmap(
                {
                    "Cell cycle": ["CDK1", "CDK2", "PCNA"],
                    "Stress": ["HSPA1A", "HSP90AA1"],
                },
                classes=["treatment", "cellline"],
                sort_by={"treatment": ["DMSO", "Drug"]},
            )
            ```

            Custom header and group colors (via ``get_color`` or hex).
            Single class may use a flat map; multiple classes nest by class:
            ```python
            from scpviz import plotting as scplt

            c = scplt.get_color("colors", 7)
            fig = pdata.plot_grouped_heatmap(
                {"Cell cycle": ["Cdk1", "Pcna"], "Stress": ["Hspa1a"]},
                classes=["condition"],
                sort_by={"condition": ["SWI", "Uninjured"]},
                group_colors={"Cell cycle": c[0], "Stress": c[1]},
                header_colors={"SWI": "#FF0000", "Uninjured": "#6EDC00"},
            )
            ```

            Spacing, fonts, display scale, and colorbar height:
            ```python
            fig = pdata.plot_grouped_heatmap(
                {"Cell cycle": ["Cdk1", "Pcna"], "Stress": ["Hspa1a"]},
                classes=["condition"],
                sample_label_col="replicate",
                row_spacing=True,
                group_bar_pad=0.15,
                column_spacing=0.5,
                header_spacing=0.06,
                display_scale="zscore",
                text_size=10,
                cbar_scale=0.8,
                legend_width=0.36,
            )
            ```

            Separate legend figure when resizing the heatmap:
            ```python
            fig, legend_fig = pdata.plot_grouped_heatmap(
                {"Cell cycle": ["Cdk1", "Pcna"], "Stress": ["Hspa1a"]},
                classes=["condition"],
                separate_legend=True,
                cbar_scale=1.25,
            )
            ```
        """
        self._check_data(on)
        return plotting.plot_grouped_heatmap(
            self,
            protein_groups,
            classes,
            on=on,
            sort_by=sort_by,
            layer=layer,
            display_scale=display_scale,
            group_colors=group_colors,
            header_colors=header_colors,
            cmap=cmap,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            header_spacing=header_spacing,
            header_height=header_height,
            group_bar_pad=group_bar_pad,
            group_bar_width=group_bar_width,
            sample_label_col=sample_label_col,
            figsize=figsize,
            text_size=text_size,
            cbar_scale=cbar_scale,
            legend_width=legend_width,
            auto_log2=auto_log2,
            gene_col=gene_col,
            separate_legend=separate_legend,
            **kwargs,
        )

    def plot_clustermap(
        self,
        classes,
        namelist=None,
        on="protein",
        layer="X",
        display_scale="auto",
        metric="correlation",
        cor_method="pearson",
        linkage_method="average",
        optimal_ordering=True,
        header_colors=None,
        sample_label_col=None,
        cmap=None,
        figsize=None,
        text_size=8,
        cbar_scale=1.0,
        legend_width=None,
        header_spacing=0.06,
        header_height=0.35,
        dendrogram_linewidth=None,
        show_row_labels=False,
        auto_log2=True,
        gene_col="Genes",
        separate_legend=False,
        **kwargs,
    ):
        """
        Plot an overview clustered heatmap with row and column dendrograms.

        Thin wrapper around :func:`scpviz.plotting.plot_clustermap`. Sample order
        follows the column dendrogram (not metadata sort).

        Args:
            classes (list of str): ``.obs`` columns for header strips.
            namelist (list of str, optional): Gene/accession subset; ``None`` = all.
            on (str): ``"protein"`` or ``"peptide"``.
            layer (str): Abundance layer (default ``"X"``).
            display_scale (str): ``"auto"`` / ``"zscore"`` / ``"log"`` / ``"raw"``.
            metric (str): ``"correlation"`` (default) or ``"euclidean"``.
            cor_method (str): ``"pearson"`` or ``"spearman"``.
            linkage_method (str): Scipy linkage method (default ``"average"``).
            optimal_ordering (bool): Improve leaf order for display.
            header_colors (dict, optional): Header strip colors (same as grouped).
            sample_label_col (str, optional): ``.obs`` column for bottom ticks.
            cmap (str, optional): Colormap; ``None`` auto-selects by scale.
            figsize (tuple, optional): Figure size; auto if None.
            text_size (int): Base font size (default 8).
            cbar_scale (float): Colorbar vertical scale (default ``1.0``).
            legend_width (float, optional): Left margin for colorbar + legends.
            header_spacing (float): Space between headers and heatmap.
            header_height (float): Relative header strip height.
            dendrogram_linewidth (float, optional): Dendrogram line width.
            show_row_labels (bool): Show feature labels (default False).
            auto_log2 (bool): In-memory log2 when layer looks linear.
            gene_col (str): ``.var`` gene label column.
            separate_legend (bool): If True, return ``(fig, legend_fig)``.
            **kwargs: Optional ``title``.

        Returns:
            fig or ``(fig, legend_fig)`` when ``separate_legend=True``.
        """
        self._check_data(on)
        return plotting.plot_clustermap(
            self,
            classes,
            namelist=namelist,
            on=on,
            layer=layer,
            display_scale=display_scale,
            metric=metric,
            cor_method=cor_method,
            linkage_method=linkage_method,
            optimal_ordering=optimal_ordering,
            header_colors=header_colors,
            sample_label_col=sample_label_col,
            cmap=cmap,
            figsize=figsize,
            text_size=text_size,
            cbar_scale=cbar_scale,
            legend_width=legend_width,
            header_spacing=header_spacing,
            header_height=header_height,
            dendrogram_linewidth=dendrogram_linewidth,
            show_row_labels=show_row_labels,
            auto_log2=auto_log2,
            gene_col=gene_col,
            separate_legend=separate_legend,
            **kwargs,
        )

    def plot_clustered_heatmap(
        self,
        classes,
        proteins=None,
        stats_key=None,
        significance_categories=None,
        protein_groups=None,
        on="protein",
        sort_by=None,
        layer="X",
        display_scale="auto",
        metric="correlation",
        cor_method="pearson",
        linkage_method="average",
        optimal_ordering=True,
        show_unassigned=True,
        group_colors=None,
        header_colors=None,
        label_color=None,
        sample_label_col=None,
        cmap=None,
        figsize=None,
        text_size=8,
        cbar_scale=1.0,
        legend_width=None,
        column_spacing=True,
        header_spacing=0.06,
        header_height=0.35,
        dendrogram_linewidth=None,
        auto_log2=True,
        gene_col="Genes",
        separate_legend=False,
        **kwargs,
    ):
        """
        Plot a hierarchically clustered protein/peptide × sample heatmap.

        Thin wrapper around :func:`scpviz.plotting.plot_clustered_heatmap`.
        Provide exactly one of ``proteins`` or ``stats_key``.

        Args:
            classes (list of str): ``.obs`` columns for headers and sample order.
            proteins (list of str, optional): Explicit feature list.
            stats_key (str, optional): ``pdata.stats`` key for a DE volcano table
                (same convention as ``plot_volcano``).
            significance_categories (list of str or str, optional): Categories kept
                for ``stats_key``. Default ``["upregulated", "downregulated"]``.
                A bare string or one-element list selects a single category
                (e.g. ``"not comparable"`` / ``["not comparable"]``).
            protein_groups (dict, optional): Annotation overlay for the color strip.
            on (str): ``"protein"`` or ``"peptide"``.
            sort_by (dict, optional): Per-class category order.
            layer (str): Abundance layer (default ``"X"``).
            display_scale (str): ``"auto"`` / ``"zscore"`` / ``"log"`` / ``"raw"``.
                Auto uses ``log`` when ``"not comparable"`` is selected.
            metric (str): ``"correlation"`` (default) or ``"euclidean"``.
            cor_method (str): ``"pearson"`` or ``"spearman"``; only used when
                ``metric="correlation"``.
            linkage_method (str): Passed to scipy linkage (default ``"average"``).
            optimal_ordering (bool): Improve leaf order for display (default True).
            show_unassigned (bool): If False, drop features not in ``protein_groups``.
            group_colors (dict, optional): Override colors for the group strip/legend.
            header_colors (dict, optional): Header strip colors. Nested
                ``{class: {category: color}}``, or flat ``{category: color}``
                when ``len(classes)==1``. Each class is its own row (not one
                color per class combination).
            label_color (str, optional): Fixed color for all gene row labels
                (e.g. ``"black"``). ``None`` colors labels by ``protein_groups``.
            sample_label_col (str, optional): ``.obs`` / ``.summary`` column for
                bottom tick labels (e.g. ``"replicate"``). Default uses ``obs_names``.
            cmap (str, optional): Colormap; ``None`` auto-selects by display scale.
            figsize (tuple, optional): Figure size; auto if None.
            text_size (int): Base font size for ticks, colorbar, and legends
                (default 8; same convention as ``plot_pairwise_correlation``).
            cbar_scale (float): Vertical scale factor for the colorbar (default
                ``1.0``). Legends stack tightly below it; also applies with
                ``separate_legend=True``.
            legend_width (float, optional): Left margin for colorbar + legends.
                `None` (default) auto-sizes from legend text; pass a float to
                override. Ignored when `separate_legend=True`.
            column_spacing (bool or float): Horizontal gaps between sample leaf
                blocks. ``True`` (default) uses a half-cell gap; ``False``/``0`` =
                none; float scales the default. Same semantics as
                ``plot_grouped_heatmap``.
            header_spacing (float): Space between header strips and the heatmap
                (default ``0.06``).
            header_height (float): Relative GridSpec height for each header strip
                (default ``0.35``). Same semantics as ``plot_grouped_heatmap``.
            dendrogram_linewidth (float, optional): Line width for the cluster
                tree. ``None`` keeps matplotlib's default.
            auto_log2 (bool): In-memory log2 when layer looks linear.
            gene_col (str): ``.var`` gene label column.
            separate_legend (bool): If True, return ``(fig, legend_fig)`` with
                colorbar and legends on a second figure.
            **kwargs: Forwarded (e.g. ``title``).

        Returns:
            fig (matplotlib.figure.Figure): Constructed figure, or
            ``(fig, legend_fig)`` when ``separate_legend=True``.

        Example:
            Cluster an explicit protein list:
            ```python
            fig = pdata.plot_clustered_heatmap(
                classes=["treatment", "cellline"],
                proteins=["CDK1", "PCNA", "HSPA1A", "GAPDH"],
                protein_groups={"Cell cycle": ["CDK1", "PCNA"]},
            )
            ```

            Cluster DE hits from a stored contrast (default up + down):
            ```python
            fig = pdata.plot_clustered_heatmap(
                classes=["condition"],
                stats_key="SWI vs Uninjured",
                sort_by={"condition": ["SWI", "Uninjured"]},
            )
            ```

            Fonts, display scale, colorbar height, and a separate legend:
            ```python
            fig, legend_fig = pdata.plot_clustered_heatmap(
                classes=["condition"],
                stats_key="SWI vs Uninjured",
                significance_categories=["not comparable"],
                sample_label_col="replicate",
                label_color="black",
                display_scale="log",
                text_size=10,
                cbar_scale=0.8,
                legend_width=0.36,
                dendrogram_linewidth=0.8,
                separate_legend=True,
            )
            ```
        """
        self._check_data(on)
        return plotting.plot_clustered_heatmap(
            self,
            classes,
            proteins=proteins,
            stats_key=stats_key,
            significance_categories=significance_categories,
            protein_groups=protein_groups,
            on=on,
            sort_by=sort_by,
            layer=layer,
            display_scale=display_scale,
            metric=metric,
            cor_method=cor_method,
            linkage_method=linkage_method,
            optimal_ordering=optimal_ordering,
            show_unassigned=show_unassigned,
            group_colors=group_colors,
            header_colors=header_colors,
            label_color=label_color,
            sample_label_col=sample_label_col,
            cmap=cmap,
            figsize=figsize,
            text_size=text_size,
            cbar_scale=cbar_scale,
            legend_width=legend_width,
            column_spacing=column_spacing,
            header_spacing=header_spacing,
            header_height=header_height,
            dendrogram_linewidth=dendrogram_linewidth,
            auto_log2=auto_log2,
            gene_col=gene_col,
            separate_legend=separate_legend,
            **kwargs,
        )
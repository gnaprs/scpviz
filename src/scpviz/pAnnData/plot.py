
from scpviz import plotting

class PlotMixin:
    def plot_counts(self, classes=None, y='protein_count', **kwargs):
        import seaborn as sns

        df = self.summary # type: ignore #, in base
        if classes is None:
            df.reset_index()
            classes = 'index'
        sns.violinplot(data=df, x=classes, y=y, **kwargs)

    def plot_rs(self, figsize=(10, 4)):
        """
        Visualize connectivity in the RS (protein × peptide) matrix.

        Generates side-by-side histograms:

        - Left: Number of peptides mapped to each protein
        - Right: Number of proteins associated with each peptide

        Args:
            figsize (tuple): Size of the matplotlib figure (default: (10, 4)).

        Returns:
            None
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
        aspect=0.5, plot_points=True, x_label="gene", kind="auto", **kwargs,):
        """
        Wrapper for `scpviz.plotting.plot_abundance`.

        Plot abundance of proteins or peptides across samples.

        This function visualizes expression values for selected proteins or peptides
        using violin + box + strip plots, or bar plots when the number of replicates
        per group is small. Supports grouping, faceting, and custom ordering.

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

            **kwargs: Additional keyword arguments passed to seaborn plotting functions.

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
        order=None, plot_type="box", log_scale=False, fig_width=1.0, fig_height=2.0, palette=None, y_min=None, y_max=None,
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
        fig_width (float): Width per subplot, in inches.
        fig_height (float): Height of the entire figure, in inches.
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
            fig_width=2,
            fig_height=2.5,
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
            fig_width=2,
            fig_height=2.5,
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
            fig_width=2,
            fig_height=2.5,
        )
        plt.show()
        ```

        Violin plots (distribution-focused):
        ```python
        fig, axes = pdata.plot_abundance_boxgrid(
            namelist=["Gapdh", "Vcp", "Ahnak"],
            classes="condition",
            plot_type="violin",
            fig_width=2,
            fig_height=2.5,
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
            fig_width=2,
            fig_height=2.5,
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
            fig_width=fig_width, fig_height=fig_height, palette=palette, y_min=y_min, y_max=y_max,
            label_x=label_x, show_n=show_n, global_legend=global_legend,
            box_kwargs=box_kwargs, hline_kwargs=hline_kwargs, bar_kwargs=bar_kwargs, bar_error=bar_error,
            violin_kwargs=violin_kwargs, text_kwargs=text_kwargs, strip_kwargs=strip_kwargs,
        )
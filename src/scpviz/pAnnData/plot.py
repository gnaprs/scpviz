
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
    
    def plot_abundance_boxgrid(self,namelist=None,layer='X',on='protein',classes="Grouping",return_df=False,
        box=True,fig_width=1.0,fig_height=2.0,palette=None,y_min=None,y_max=None,label_x=True,show_n=False,
        global_legend=True,boxplot_kwargs=None,hline_kwargs=None,text_kwargs=None,
    ):
        """
        Plot log-scale abundance values in a one-row panel of boxplots or mean-lines.

        This function generates a clean horizontal panel, with one subplot per gene,
        using either boxplots (default) or colored mean-lines. Abundance values are
        visualized on a log10-transformed y-axis, with zero or negative values clipped
        to 0 before transformation. The layout is optimized for compact manuscript
        figure panels and supports custom global legends, count annotations, colored
        mean-lines, and flexible formatting via keyword dictionaries.

        Args:
            pdata (pAnnData): Input pAnnData object.
            namelist (list of str, optional): List of accessions or gene names to plot.
                If None, all available features are considered.
            layer (str): Data layer to use for abundance values. Default is `'X'`.
            on (str): Data level to plot, either `'protein'` or `'peptide'`.
            return_df (bool): If True, returns the DataFrame of replicate and summary values.
            classes (str): Column in ``df`` to use for grouping samples (default: "Grouping").
            box (bool): If True, draw boxplots. If False, draw colored mean-lines.
            fig_width (float): Width per subplot, in inches.
            fig_height (float): Height of the entire figure, in inches.
            palette (dict or list, optional): Color palette for grouping categories. 
                Defaults to ``scplt.get_color('colors', n_classes)``.
            y_min (float or None): Lower y-axis limit in log10 units (e.g., 2 → 10²). If None, inferred.
            y_max (float or None): Upper y-axis limit in log10 units (e.g., 6 → 10⁶). If None, inferred.
            label_x (bool): Whether to display x tick labels inside each subplot.
            show_n (bool): Whether to annotate each subplot with sample counts.
            global_legend (bool): Whether to display a single global legend.
            boxplot_kwargs (dict, optional): Additional arguments passed to ``sns.boxplot``.
            hline_kwargs (dict, optional): Keyword arguments for mean-lines (e.g., color, linewidth).
            text_kwargs (dict, optional): Keyword arguments for count labels (e.g., fontsize, offset).


        Returns:
            fig (matplotlib.figure.Figure): The generated figure.
            axes (list of matplotlib.axes.Axes): One axis per gene.
            df (pandas.DataFrame, optional): Returned if `return_df=True`.
            
        !!! note
            Default customizations for keyword dictionaries:

            Boxplot styling:
            ```python
            boxplot_kwargs = {
                "showcaps"=False,
                "whiskerprops"={"visible": False},
                "showfliers"=False,
                "boxprops"=dict(alpha=0.6, linewidth=1),
                "linewidth"=1,
                "dodge"=True,
            }
            ```

            Mean-line styling (used when ``box=False``):
            ```python
            hline_kwargs = {
                "color"="k",
                "linewidth"=2.0,
                "zorder"=5,
                "half_width"=0.15,
            }
            ```
            *note that `half_width` sets the length of the mean line

            Text annotation styling (used when ``show_n=True``):
            ```python
            text_kwargs = {
                "fontsize"=7,
                "color"="black",
                "ha"="center",
                "va"="bottom",
                "zorder"=10,
                "offset"=0.1,
            }
            ```

        !!! example
            Basic usage:
            ```python
            df = pdata.get_abundance(
                namelist=["Slc17a7", "Camk2a", "Nrgn", "Homer1"],
                classes="group"
            )
            fig, axes = plot_abundance_boxgrid(df, classes="group")
            ```

            Using colored mean-lines with count annotations:
            ```python
            fig, axes = plot_abundance_boxgrid(
                df,
                classes="group",
                box=False,
                show_n=True,
                fig_width=1,
                fig_height=2
            )
            ```

            Customizing appearance:
            ```python
            fig, axes = plot_abundance_boxgrid(
                df,
                classes="group",
                box=False,
                global_legend=True,
                show_n=True,
                y_max=8,
                hline_kwargs={"color": "red", "linewidth": 2},
                text_kwargs={"fontsize": 9, "color": "blue", "offset": 1}
            )
            ```
        """
        return plotting.plot_abundance_boxgrid(pdata=self,namelist=namelist,
            layer=layer,on=on,classes=classes,return_df=return_df,
            palette=palette, box=box,fig_width=fig_width,fig_height=fig_height,
            y_min=y_min,y_max=y_max,label_x=label_x,show_n=show_n,
            global_legend=global_legend,boxplot_kwargs=boxplot_kwargs,
            hline_kwargs=hline_kwargs, text_kwargs=text_kwargs,
        )
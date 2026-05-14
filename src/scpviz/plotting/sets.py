"""Venn and UpSet set-operation plots."""
from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import get_color

def _plotting_pkg_utils():
    """`scpviz.utils` as exposed on `scpviz.plotting` (tests may patch ``scplt.utils``)."""
    import scpviz.plotting as _pkg

    return _pkg.utils

def _plotting_pkg_upsetplot():
    """``upsetplot`` module as exposed on ``scpviz.plotting`` (tests may patch ``scplt.upsetplot``)."""
    import scpviz.plotting as _pkg

    return _pkg.upsetplot

def plot_venn(
    ax,
    pdata,
    classes,
    set_colors="default",
    weighted=False,
    return_contents=False,
    label_order=None,
    fixed_subset_sizes=None,
    **kwargs: Any,
) -> plt.Axes | tuple[plt.Axes, dict[str, set[str]]]:
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
        **kwargs (Any): Additional keyword arguments passed to matplotlib-venn functions.

    Returns:
        The axes containing the Venn diagram, or ``(ax, upset_contents)`` if ``return_contents=True`` (``upset_contents`` maps class labels to sets of feature identifiers).

    Raises:
        ValueError: If number of sets is not 2 or 3.
        ValueError: If `label_order` does not contain the same elements as `classes`.
        ValueError: If custom `set_colors` length does not match number of sets.

    Example:
        Two-set Venn by cell line:
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            fig, ax = plt.subplots(figsize=(3, 3))
            scplt.plot_venn(ax, pdata, classes="cellline")
            plt.show()
            ```

        ![Plot venn](../../assets/plots/plot_venn.png)

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
    upset_contents = _plotting_pkg_utils().get_upset_contents(pdata, classes, upsetForm=False)

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

def plot_upset(
    pdata,
    classes,
    return_contents=False,
    **kwargs: Any,
) -> Any:
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

        **kwargs (Any): Additional keyword arguments passed to `upsetplot.UpSet`.
            See the [upsetplot documentation](https://upsetplot.readthedocs.io/en/stable/)
            for more details. Common arguments include:

            - `sort_categories_by` (str): How to sort categories. Options are
              `"cardinality"`, `"input"`, `"-cardinality"`, or `"-input"`.
            - `min_subset_size` (int): Minimum subset size to display.

    Returns:
        The ``upsetplot.UpSet`` instance, or ``(upset, membership_df)`` if ``return_contents=True`` (membership as a multi-index DataFrame).

    Example:
        UpSet for ``cellline`` and ``condition`` (``show_counts=False`` can help when saving some PNGs with matplotlib / upsetplot):
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt

            upplot = scplt.plot_upset(pdata, classes=["cellline", "condition"], show_counts=False)
            upplot.plot()
            plt.show()
            ```

        ![Plot upset](../../assets/plots/plot_upset.png)

        Highlight disjoint subsets (resolve keys with ``get_upset_contents(..., upsetForm=False)``):
            ```python
            import matplotlib.pyplot as plt
            from scpviz import plotting as scplt
            from scpviz import utils as scu

            keys = list(
                scu.get_upset_contents(pdata, classes=["cellline", "condition"], upsetForm=False).keys()
            )
            be_kd = next((k for k in keys if "BE" in k and "kd" in k), keys[0])
            as_sc = next((k for k in keys if "AS" in k and "sc" in k), keys[-1])
            others = [k for k in keys if k not in (be_kd, as_sc)]

            upplot = scplt.plot_upset(pdata, classes=["cellline", "condition"], show_counts=False)
            upplot.style_subsets(
                present=[be_kd],
                absent=others,
                edgecolor="black",
                facecolor="#E59866",
                linewidth=2,
                label="highlight A",
            )
            upplot.style_subsets(
                present=[as_sc],
                absent=[k for k in keys if k != as_sc],
                edgecolor="black",
                facecolor="#5DADE2",
                linewidth=2,
                label="highlight B",
            )
            upplot.plot()
            plt.show()
            ```

        ![Plot upset styled](../../assets/plots/plot_upset_styled.png)

    See Also:
        plot_venn: Plot a Venn diagram for 2 to 3 sets.  
        plot_rankquant: Rank-based visualization of protein/peptide distributions.
    """

    upset_contents = _plotting_pkg_utils().get_upset_contents(pdata, classes=classes)
    show_counts = kwargs.pop("show_counts", True)
    upplot = _plotting_pkg_upsetplot().UpSet(
        upset_contents,
        subset_size="count",
        show_counts=show_counts,
        facecolor="black",
        **kwargs,
    )

    if return_contents:
        return upplot, upset_contents
    else:
        return upplot

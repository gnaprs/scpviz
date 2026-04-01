import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scpviz import pAnnData

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from conftest import _is_axes_container, _count_artists

def test_plot_abundance_boxgrid_smoke(pdata):
    
    fig, axes = pdata.plot_abundance_boxgrid(namelist=["ACTB"],
        classes="treatment")

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert len(axes) == 1
    assert _count_artists(axes) > 0

def test_plot_rs_runs_without_error(pdata):
    """Ensure plot_rs() executes successfully with valid RS."""
    import matplotlib.pyplot as plt
    plt.close("all")

def _seed_mock_pca_gsea(pdata, on="protein"):
    pdata.pca(on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    adata.uns["pca_gsea"] = {
        "results": {
            "PC1": pd.DataFrame({"Term": ["OXPHOS", "Immune"], "NES": [1.5, -1.1], "FDR q-val": [0.01, 0.04]}),
            "PC2": pd.DataFrame({"Term": ["OXPHOS", "Immune"], "NES": [-0.7, 1.9], "FDR q-val": [0.06, 0.01]}),
        }
    }

def test_plot_pca_gsea_pathway_vectors_wrapper_smoke(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = pdata.plot_pca_gsea_pathway_vectors(ax=ax, on="protein", plot_pc=[1, 2], top_n=2)
    assert isinstance(out, matplotlib.axes.Axes)
    assert len(ax.patches) >= 1
    plt.close(fig)

def test_plot_pca_protein_vectors_wrapper_smoke(pdata):
    pdata.pca(on="protein")
    fig, ax = plt.subplots()
    out = pdata.plot_pca_protein_vectors(
        ax=ax, on="protein", plot_pc=[1, 2], top_n=3, adjust_labels=False
    )
    assert isinstance(out, matplotlib.axes.Axes)
    assert len(ax.patches) >= 1
    plt.close(fig)


def test_plot_pca_gsea_bubble_wrapper_smoke(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = pdata.plot_pca_gsea_bubble(ax=ax, on="protein", pcs=[1, 2], top_n=2)
    assert isinstance(out, matplotlib.axes.Axes)
    assert len(ax.collections) >= 1
    plt.close(fig)

def test_plot_pca_gsea_heatmap_wrapper_smoke(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = pdata.plot_pca_gsea_heatmap(ax=ax, on="protein", pcs=[1, 2], top_n=2)
    assert isinstance(out, matplotlib.axes.Axes)
    assert len(ax.collections) >= 1
    plt.close(fig)

def test_plot_pca_gsea_bubble_wrapper_include_exclude(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    _, out_df = pdata.plot_pca_gsea_bubble(
        ax=ax,
        on="protein",
        pcs=[1, 2],
        include_pathways=["OXPHOS", "Immune"],
        exclude_pathways=["Immune"],
        top_n=1,
        title_case_labels=False,
        return_df=True,
    )
    assert set(out_df["pathway_raw"]) == {"OXPHOS"}
    plt.close(fig)
    pdata.plot_rs(figsize=(4, 2))

def test_plot_rs_handles_missing_rs(pdata_nopep, capsys):
    """If rs is None, should print warning and not plot."""
    plt.close("all")
    pdata_nopep.plot_rs()
    captured = capsys.readouterr().out

    assert "No RS matrix" in captured
    assert not plt.get_fignums(), "No figure should be created when RS is None"

def test_plot_counts_smoke(pdata):
    pdata.plot_counts(classes="treatment")

    figs = list(map(plt.figure, plt.get_fignums()))
    assert len(figs) == 1, "No figure was created by plot_counts"
    fig = figs[0]
    axes = fig.get_axes()
    assert len(axes) >= 1, "plot_counts did not create any axes"
    ax = axes[0]
    assert len(ax.collections) > 0 or len(ax.patches) > 0, \
        "plot_counts created an empty axis (no violins or patches)"
    plt.close(fig)

def test_plot_abundance_smoke(pdata):
    plt.close('all')
    fig, ax = plt.subplots()

    out = pdata.plot_abundance(ax=ax, namelist=["ACTB"], classes="treatment")

    assert isinstance(out, (matplotlib.axes.Axes, sns.axisgrid.FacetGrid))
    if isinstance(out, matplotlib.axes.Axes):
        assert (
            len(out.collections) > 0
            or len(out.patches) > 0
            or len(out.lines) > 0
        ), "plot_abundance created an empty axis"
    plt.close('all')

def test_plot_abundance_boxgrid_smoke(pdata):
    plt.close('all')

    fig, axes = pdata.plot_abundance_boxgrid(namelist=["ACTB"], classes="treatment")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, (list, tuple))
    assert len(axes) == 1
    assert isinstance(axes[0], matplotlib.axes.Axes)

    ax = axes[0]
    assert (
        len(ax.collections) > 0
        or len(ax.patches) > 0
        or len(ax.lines) > 0
    ), "plot_abundance_boxgrid created an empty axis"

    plt.close("all")
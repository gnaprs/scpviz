from scpviz import utils as scutils
from scpviz import plotting as scplt

import pandas as pd
import numpy as np
import pytest
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad

from scpviz.plotting.abundance import (
    _pack_sig_bracket_y,
    _resolve_sig_group_label,
    _x_intervals_overlap,
    annotate_abundance_boxgrid_significance,
)
from conftest import _is_axes_container, _count_artists


def _zero_gene_for_obs_mask(pdata, gene: str, obs_mask) -> None:
    """Set protein layer abundances to zero for one gene across masked samples."""
    adata = scutils.get_adata(pdata, "protein")
    var_idx = int(np.where(adata.var["Genes"] == gene)[0][0])
    adata.X[obs_mask, var_idx] = 0


def _minimal_sig_panel(*, nd_groups=None):
    """Small synthetic panel for annotate_abundance_boxgrid_significance edge cases."""
    fig, ax = plt.subplots()
    sub = pd.DataFrame(
        {
            "treatment": ["sc", "sc", "kd", "kd"],
            "abundance": [1.0, 2.0, 3.0, 4.0],
            "plot_abundance": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return fig, [
        {
            "gene": "G",
            "ax": ax,
            "sub": sub,
            "unique_classes": ["sc", "kd"],
            "x_centers": [0.0, 1.0],
            "nd_groups": set(nd_groups or ()),
        }
    ]

# test get_color

def test_get_color_colors_basic():
    colors = scplt.get_color("colors", 5)
    assert isinstance(colors, list)
    assert len(colors) == 5
    assert all(isinstance(c, str) and c.startswith("#") for c in colors)

def test_get_color_colors_warns_on_repeat():
    with pytest.warns(UserWarning, match="Reusing from the start"):
        colors = scplt.get_color("colors", 10)
    assert len(colors) == 10
    assert colors[0] == colors[7]  # colors repeat from base palette

def test_get_color_cmap_single():
    cmap = scplt.get_color("cmap", 1)
    assert isinstance(cmap, matplotlib.colors.LinearSegmentedColormap)

def test_get_color_cmap_multiple():
    cmaps = scplt.get_color("cmap", 3)
    assert isinstance(cmaps, list)
    assert all(isinstance(c, matplotlib.colors.LinearSegmentedColormap) for c in cmaps)

def test_get_color_palette():
    palette = scplt.get_color("palette")
    assert isinstance(palette, list)
    assert all(isinstance(c, tuple) and len(c) == 3 for c in palette)
    assert np.allclose(np.array(palette).max(), 1.0, atol=0.05)  # RGB normalized to ~1

def test_get_color_show_smoke(monkeypatch):
    # Avoid GUI popup during test runs
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    result = scplt.get_color("show")
    assert result is None

def test_get_color_invalid_type():
    with pytest.raises(ValueError, match="Invalid resource_type"):
        scplt.get_color("invalid")

def test_get_color_colors_missing_n():
    with pytest.raises(ValueError, match="must be specified"):
        scplt.get_color("colors")

# test plot_significance

def test_plot_significance_runs_without_error():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=1.0, h=0.1, x1=0, x2=1, pval=0.05)
    assert len(ax.lines) == 1
    assert len(ax.texts) == 1

def test_plot_significance_with_string_label():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=0.5, h=0.2, x1=0, x2=1, pval="custom")
    text_obj = ax.texts[0]
    assert text_obj.get_text() == "custom"
    assert text_obj.get_ha() == "center"
    assert text_obj.get_va() == "bottom"

@pytest.mark.parametrize("pval,expected", [
    (0.2, "n.s."),      # Not significant
    (0.05, "*"),        # 1 star
    (0.005, "**"),      # 2 stars
    (0.0005, "***"),    # 3 stars
])
def test_plot_significance_numeric_levels(pval, expected):
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=0.5, h=0.2, pval=pval)
    text_label = ax.texts[0].get_text()
    assert text_label.startswith(expected[0])

def test_plot_significance_color_and_fontsize():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=1, h=0.2, col="red", fontsize=16)
    line = ax.lines[0]
    text = ax.texts[0]
    assert line.get_color() == "red"
    assert text.get_fontsize() == 16

def test_plot_significance_identical_x():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=1, h=0.1, x1=1, x2=1, pval=0.01)
    line = ax.lines[0]
    xdata, ydata = line.get_data()
    assert all(x == 1 for x in xdata)
    assert len(xdata) == 4

def test_plot_significance_returns_none():
    fig, ax = plt.subplots()
    result = scplt.plot_significance(ax, y=1, h=0.1, pval="n.s.")
    assert result is None

# Tests for scplt.plot_cv

def test_plot_cv_runs_without_error(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_cv(ax, pdata, classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_cv_returns_dataframe(pdata):
    df = scplt.plot_cv(None, pdata, classes="cellline", on="protein", return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert "Class" in df.columns
    assert "CV" in df.columns
    assert "CV_pct" in df.columns
    assert not df.empty
    assert np.allclose(df["CV_pct"], df["CV"] * 100, equal_nan=True)

def test_plot_cv_respects_custom_order(pdata):
    fig, ax = plt.subplots()
    valid_classes = pdata.prot.obs["treatment"].unique().tolist()
    order = valid_classes[::-1]  # reverse for test
    scplt.plot_cv(ax, pdata, classes="treatment", order=order)
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    # Order may not be identical due to seaborn sorting, but should contain same elements
    assert set(order) == set(xticklabels)

def test_plot_cv_on_peptide(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_cv(ax, pdata, on="peptide", classes="sample")
    assert _is_axes_container(result)

def test_plot_cv_return_df_only(pdata):
    df = scplt.plot_cv(None, pdata, classes="treatment", return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert "CV" in df.columns
    assert "CV_pct" in df.columns
    plt.close("all")

def test_plot_cv_axis_labels_and_no_title(pdata):
    fig, ax = plt.subplots()
    scplt.plot_cv(ax, pdata, classes="treatment")
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == "CV (%)"
    assert ax.get_title() == ""
    plt.close("all")

def test_plot_cv_show_n(pdata):
    fig, ax = plt.subplots()
    scplt.plot_cv(ax, pdata, classes="treatment", show_n=True)
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("n=") for t in texts)
    plt.close("all")

def test_plot_cv_annotate_median(pdata):
    fig, ax = plt.subplots()
    scplt.plot_cv(ax, pdata, classes="treatment", annotate="median")
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("median\n") and t.endswith("%") for t in texts)
    plt.close("all")

def test_plot_cv_annotate_custom_dict(pdata):
    fig, ax = plt.subplots()
    class_val = pdata.prot.obs["treatment"].unique()[0]
    scplt.plot_cv(ax, pdata, classes="treatment", annotate={class_val: "custom label"})
    texts = [t.get_text() for t in ax.texts]
    assert "custom label" in texts
    plt.close("all")

def test_plot_cv_show_n_below_axis(pdata):
    fig, ax = plt.subplots()
    scplt.plot_cv(ax, pdata, classes="treatment", show_n=True)
    n_texts = [t for t in ax.texts if t.get_text().startswith("n=")]
    assert n_texts
    for t in n_texts:
        assert t.get_transform() == ax.get_xaxis_transform()
        assert t.get_position()[1] < 0
    plt.close("all")

def test_plot_cv_show_n_and_annotate_positions(pdata):
    fig, ax = plt.subplots()
    scplt.plot_cv(ax, pdata, classes="treatment", show_n=True, annotate="mean")
    n_texts = [t for t in ax.texts if t.get_text().startswith("n=")]
    stat_texts = [t for t in ax.texts if t.get_text().startswith("mean\n")]
    assert n_texts and stat_texts
    for t in n_texts:
        assert t.get_transform() == ax.get_xaxis_transform()
        assert t.get_position()[1] < 0
    for t in stat_texts:
        assert t.get_transform() == ax.get_xaxis_transform()
        assert t.get_position()[1] > 1.0
    plt.close("all")

def test_plot_cv_invalid_annotate_raises(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="annotate must be"):
        scplt.plot_cv(ax, pdata, classes="treatment", annotate="invalid")
    plt.close("all")

# Tests for scplt.plot_summary
def test_plot_summary_mean_by_class(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_summary(ax, pdata, value="protein_count", classes="treatment", plot_mean=True)
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_summary_per_sample(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_summary(ax, pdata, value="protein_count", classes=None, plot_mean=False)
    assert _is_axes_container(result)
    assert len(result.get_xticklabels()) > 0

def test_plot_summary_multiple_classes(pdata):
    classes = ["cellline", "treatment"]
    fig, ax = plt.subplots()
    result = scplt.plot_summary(ax, pdata, value="protein_count", classes=classes, plot_mean=True)
    # Multiple subplots possible → may return list of Axes
    assert _is_axes_container(result)

def test_plot_summary_raises_without_classes(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Classes must be specified"):
        scplt.plot_summary(ax, pdata, value="protein_count", classes=None, plot_mean=True)

def test_plot_summary_invalid_classes_type(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Invalid 'classes'"):
        scplt.plot_summary(ax, pdata, value="protein_count", classes={}, plot_mean=False)

# Tests for scplt.plot_abundance_housekeeping
def test_plot_abundance_housekeeping_whole_cell(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance_housekeeping(ax, pdata, classes="treatment", loading_control="whole cell")
    # Function returns None, but draws onto provided Axes
    assert _is_axes_container(ax)
    assert _count_artists(ax) >= 0
    assert ax.get_title().lower().startswith("whole cell")

def test_plot_abundance_housekeeping_all(pdata):
    fig, axes = plt.subplots(1, 3)
    result = scplt.plot_abundance_housekeeping(axes, pdata, classes="treatment", loading_control="all")
    # Returns (Figure, Axes array)
    assert isinstance(result, tuple)
    fig_out, axes_out = result
    assert isinstance(fig_out, plt.Figure)
    assert _is_axes_container(axes_out)
    assert axes_out.shape == (3,)
    assert all(ax.get_title() for ax in axes_out)
    plt.close(fig_out)

def test_plot_abundance_housekeeping_invalid_type(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Invalid loading control"):
        scplt.plot_abundance_housekeeping(ax, pdata, classes="treatment", loading_control="invalid")

def test_plot_abundance_housekeeping_no_classes(pdata):
    fig, ax = plt.subplots()
    scplt.plot_abundance_housekeeping(ax, pdata, loading_control="nuclear")
    assert _is_axes_container(ax)
    assert ax.get_title().lower().startswith("nuclear")
    plt.close("all")

# Tests for scplt.plot_abundance
def test_plot_abundance_smoke(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB", "VCL"], classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_abundance_violin(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB", "VCL"], classes="treatment", on="protein", kind='violin')
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_abundance_return_df(pdata):
    df = scplt.plot_abundance(None, pdata, namelist=["ACTB"], classes="cellline", return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert {"x_label_name", "abundance", "class"}.intersection(df.columns)
    assert not df.empty

@pytest.mark.parametrize("log", [True, False])
def test_plot_abundance_violin_mode(pdata, log):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", kind="violin", log=log)
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

@pytest.mark.parametrize("log", [True, False])
def test_plot_abundance_bar_mode(pdata, log):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", kind="bar", log=log)
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

@pytest.mark.parametrize("log", [True, False])
def test_plot_abundance_with_facet(pdata, log):
    result = scplt.plot_abundance(None, pdata, namelist=["ACTB"], classes="treatment", facet="cellline",log=log)
    # FacetGrid should be returned when facet is used
    import seaborn as sns
    assert isinstance(result, sns.FacetGrid)
    plt.close(result.fig)

@pytest.mark.parametrize("kind", ["bar", "violin"])
def test_plot_abundance_facet_multi_panel(pdata, kind):
    """Cover FacetGrid branches when facet has >1 level (bar and violin paths)."""
    kwargs = {}
    if kind == "violin":
        # Facet violin + strip overlay is fragile on headless py3.11 CI; facet path
        # is still exercised without point overlay (see test_plot_abundance_with_facet).
        kwargs["plot_points"] = False
    result = scplt.plot_abundance(
        None,
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        facet="cellline",
        kind=kind,
        log=False,
        **kwargs,
    )
    assert isinstance(result, sns.FacetGrid)
    assert result.col_names is not None and len(result.col_names) > 1
    plt.close(result.fig)

def test_plot_abundance_raises_same_facet_class(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must be different"):
        scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", facet="treatment")

def test_plot_abundance_no_log(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", log=False)
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_abundance_custom_order(pdata):
    fig, ax = plt.subplots()
    order = {"treatment": ["sc", "kd"]}
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", order=order)
    assert _is_axes_container(result)
    plt.close("all")

# Tests for scplt.plot_abundance_boxgrid
def test_plot_abundance_boxgrid_noclass(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes=None
    )
    assert isinstance(fig, plt.Figure)

def test_plot_abundance_boxgrid_smoke(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment"
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert len(axes) == 1
    assert _count_artists(axes) > 0

def test_plot_abundance_boxgrid_multiclass(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes=['treatment', 'cellline']
    )
    assert isinstance(fig, plt.Figure)

def test_plot_abundance_boxgrid_return_df(pdata):
    fig, axes, df = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        return_df=True
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert isinstance(df, pd.DataFrame)
    assert {"gene", "abundance", "log_abundance"}.intersection(df.columns)
    assert not df.empty

def test_plot_abundance_boxgrid_sig_pairs_true(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        sig_pairs=True,
    )
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 1
    assert len(axes[0].lines) >= 1
    plt.close("all")

def test_plot_abundance_boxgrid_sig_pairs_return_stats(pdata):
    fig, axes, df, stats = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        sig_pairs=True,
        return_df=True,
    )
    assert isinstance(stats, pd.DataFrame)
    assert not stats.empty
    assert stats.iloc[0]["status"] == "ok"
    assert "p_value" in stats.columns
    assert np.isfinite(stats.iloc[0]["p_value"])
    plt.close("all")

def test_plot_abundance_boxgrid_sig_pairs_no_return_stats(pdata):
    result = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        sig_pairs=True,
        return_df=False,
    )
    assert len(result) == 2
    plt.close("all")

def test_plot_abundance_boxgrid_sig_pairs_dict_multiclass(pdata):
    fig, axes, df, stats = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes=["cellline", "treatment"],
        sig_pairs=[
            ({"cellline": "BE", "treatment": "sc"}, {"cellline": "BE", "treatment": "kd"}),
        ],
        return_df=True,
    )
    assert stats.iloc[0]["group1"] == "BE_sc"
    assert stats.iloc[0]["group2"] == "BE_kd"
    assert stats.iloc[0]["status"] == "ok"
    plt.close("all")

def test_annotate_abundance_boxgrid_significance_standalone(pdata):
    fig, axes, df = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        return_df=True,
    )
    panel_info = [
        {
            "gene": "ACTB",
            "ax": axes[0],
            "sub": df[df["gene"] == "ACTB"],
            "unique_classes": list(df[df["gene"] == "ACTB"]["treatment"].unique()),
            "x_centers": axes[0].get_xticks().tolist(),
            "nd_groups": set(),
        }
    ]
    stats = scplt.annotate_abundance_boxgrid_significance(
        panel_info,
        True,
        classes="treatment",
        classes_original="treatment",
    )
    assert not stats.empty
    assert stats.iloc[0]["status"] == "ok"
    plt.close("all")

def test_plot_abundance_boxgrid_sig_pairs_requires_classes(pdata):
    with pytest.raises(ValueError, match="requires sample grouping"):
        scplt.plot_abundance_boxgrid(
            pdata,
            namelist=["ACTB"],
            classes=None,
            sig_pairs=True,
        )


# --- _resolve_sig_group_label ---

def test_x_intervals_overlap_touching_and_disjoint():
    assert _x_intervals_overlap(0.0, 1.0, 1.0, 2.0)
    assert _x_intervals_overlap(0.0, 2.0, 0.5, 1.5)
    assert not _x_intervals_overlap(0.0, 1.0, 1.1, 2.0)


def test_pack_sig_bracket_y_dodges_overlapping_spans():
    brackets = [
        {"x1": 0.0, "x2": 2.0, "data_top": 10.0},  # long span
        {"x1": 0.0, "x2": 1.0, "data_top": 10.0},  # short
        {"x1": 1.0, "x2": 2.0, "data_top": 10.0},  # short, touches first short
    ]
    packed = _pack_sig_bracket_y(
        brackets,
        y_range=10.0,
        h=0.3,
        base_offset_frac=0.05,
        spacing_frac=0.08,
    )
    by_span = {(min(b["x1"], b["x2"]), max(b["x1"], b["x2"])): b["y"] for b in packed}
    assert by_span[(0.0, 1.0)] < by_span[(0.0, 2.0)]
    assert by_span[(1.0, 2.0)] < by_span[(0.0, 2.0)]
    # touching short spans also dodge each other
    assert by_span[(0.0, 1.0)] != by_span[(1.0, 2.0)]


def test_pack_sig_bracket_y_allows_disjoint_same_band():
    brackets = [
        {"x1": 0.0, "x2": 1.0, "data_top": 10.0},
        {"x1": 2.0, "x2": 3.0, "data_top": 10.0},
    ]
    packed = _pack_sig_bracket_y(
        brackets,
        y_range=10.0,
        h=0.3,
        base_offset_frac=0.05,
        spacing_frac=0.08,
    )
    assert packed[0]["y"] == pytest.approx(packed[1]["y"])


def test_annotate_sig_bracket_h_stable_under_shared_ylim_inflation():
    """Bracket tick height must not grow when a sibling shared-y axis was padded."""
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4, 3))
    rng = np.random.default_rng(0)
    sub = pd.DataFrame(
        {
            "treatment": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "abundance": np.concatenate(
                [rng.normal(10, 0.5, 4), rng.normal(11, 0.5, 4), rng.normal(12, 0.5, 4)]
            ),
            "plot_abundance": np.concatenate(
                [rng.normal(10, 0.5, 4), rng.normal(11, 0.5, 4), rng.normal(12, 0.5, 4)]
            ),
        }
    )
    for ax in axes:
        ax.set_ylim(8, 14)
        ax.set_xlim(-0.5, 2.5)
    panel_info = [
        {
            "gene": "G1",
            "ax": axes[0],
            "sub": sub,
            "unique_classes": ["a", "b", "c"],
            "x_centers": [0.0, 1.0, 2.0],
            "nd_groups": set(),
        },
        {
            "gene": "G2",
            "ax": axes[1],
            "sub": sub,
            "unique_classes": ["a", "b", "c"],
            "x_centers": [0.0, 1.0, 2.0],
            "nd_groups": set(),
        },
    ]
    layout_y_range = 14.0 - 8.0
    annotate_abundance_boxgrid_significance(
        panel_info,
        True,
        classes="treatment",
        classes_original="treatment",
        layout_y_range=layout_y_range,
    )
    leg_hs = []
    for ax in axes:
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size == 4:
                leg_hs.append(abs(y[1] - y[0]))
    assert leg_hs
    assert max(leg_hs) == pytest.approx(min(leg_hs), rel=1e-6)
    assert max(leg_hs) == pytest.approx(0.03 * layout_y_range, rel=1e-6)
    plt.close(fig)


def test_resolve_sig_group_label_composite_dict():
    assert (
        _resolve_sig_group_label(
            {"cellline": "BE", "treatment": "sc"}, "class", ["cellline", "treatment"]
        )
        == "BE_sc"
    )


def test_resolve_sig_group_label_composite_list():
    assert _resolve_sig_group_label(["BE", "sc"], "class", ["cellline", "treatment"]) == "BE_sc"


def test_resolve_sig_group_label_composite_str():
    assert _resolve_sig_group_label("BE_sc", "class", ["cellline", "treatment"]) == "BE_sc"


def test_resolve_sig_group_label_composite_list_length_mismatch():
    with pytest.raises(ValueError, match="length must match"):
        _resolve_sig_group_label(["BE"], "class", ["cellline", "treatment"])


def test_resolve_sig_group_label_composite_bad_type():
    with pytest.raises(TypeError, match="Group spec must be dict"):
        _resolve_sig_group_label(42, "class", ["cellline", "treatment"])


def test_resolve_sig_group_label_single_column_dict_with_key():
    assert _resolve_sig_group_label({"treatment": "sc"}, "treatment", "treatment") == "sc"


def test_resolve_sig_group_label_single_column_dict_single_entry():
    assert _resolve_sig_group_label({"treatment": "sc"}, "wrong_col", "treatment") == "sc"


def test_resolve_sig_group_label_single_column_dict_missing_key():
    with pytest.raises(ValueError, match="must include column"):
        _resolve_sig_group_label(
            {"cellline": "BE", "foo": "bar"}, "treatment", "treatment"
        )


def test_resolve_sig_group_label_single_column_str():
    assert _resolve_sig_group_label("kd", "treatment", "treatment") == "kd"


# --- annotate_abundance_boxgrid_significance edge cases ---

def test_annotate_abundance_boxgrid_significance_unsupported_method():
    fig, panel_info = _minimal_sig_panel()
    with pytest.raises(ValueError, match="Unsupported sig_test"):
        annotate_abundance_boxgrid_significance(
            panel_info,
            True,
            classes="treatment",
            classes_original="treatment",
            sig_kwargs={"sig_test": "anova"},
        )
    plt.close(fig)


def test_annotate_abundance_boxgrid_significance_sig_pairs_true_all_pairs():
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    sub = pd.DataFrame(
        {
            "treatment": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "abundance": np.concatenate(
                [rng.normal(10, 1, 4), rng.normal(12, 1, 4), rng.normal(14, 1, 4)]
            ),
            "plot_abundance": np.concatenate(
                [rng.normal(10, 1, 4), rng.normal(12, 1, 4), rng.normal(14, 1, 4)]
            ),
        }
    )
    panel_info = [
        {
            "gene": "G",
            "ax": ax,
            "sub": sub,
            "unique_classes": ["a", "b", "c"],
            "x_centers": [0.0, 1.0, 2.0],
            "nd_groups": set(),
        }
    ]
    stats = annotate_abundance_boxgrid_significance(
        panel_info,
        True,
        classes="treatment",
        classes_original="treatment",
    )
    assert len(stats) == 3
    assert set(zip(stats["group1"], stats["group2"])) == {("a", "b"), ("a", "c"), ("b", "c")}
    assert (stats["status"] == "ok").all()
    plt.close(fig)


def test_annotate_abundance_boxgrid_significance_skipped_nd(capsys):
    fig, panel_info = _minimal_sig_panel(nd_groups={"kd"})
    stats = annotate_abundance_boxgrid_significance(
        panel_info,
        [("sc", "kd")],
        classes="treatment",
        classes_original="treatment",
    )
    assert stats.iloc[0]["status"] == "skipped_nd"
    assert "ND group" in stats.iloc[0]["reason"]
    assert "ND group" in capsys.readouterr().out
    plt.close(fig)


def test_annotate_abundance_boxgrid_significance_skipped_pair(capsys):
    fig, panel_info = _minimal_sig_panel()
    stats = annotate_abundance_boxgrid_significance(
        panel_info,
        [("sc", "missing")],
        classes="treatment",
        classes_original="treatment",
    )
    assert stats.iloc[0]["status"] == "skipped_pair"
    assert "Missing x position" in stats.iloc[0]["reason"]
    assert "not on axis" in capsys.readouterr().out
    plt.close(fig)


def test_plot_abundance_boxgrid_sig_pairs_true_multigroup(pdata):
    fig, axes, df, stats = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes=["cellline", "treatment"],
        sig_pairs=True,
        return_df=True,
    )
    n_groups = df["class"].nunique()
    assert len(stats) == n_groups * (n_groups - 1) // 2
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_abundance_boxgrid_nd_linear(pdata):
    adata = scutils.get_adata(pdata, "protein")
    kd_mask = (adata.obs["treatment"] == "kd").to_numpy()
    _zero_gene_for_obs_mask(pdata, "ACTB", kd_mask)
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        log_scale=False,
        nd_kwargs={"nd_label": "ND"},
    )
    assert "ND" in [t.get_text() for t in axes[0].texts]
    plt.close(fig)


def test_plot_abundance_boxgrid_nd_log_scale(pdata):
    adata = scutils.get_adata(pdata, "protein")
    kd_mask = (adata.obs["treatment"] == "kd").to_numpy()
    _zero_gene_for_obs_mask(pdata, "ACTB", kd_mask)
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        log_scale=True,
        nd_kwargs={"nd_label": "ND"},
    )
    assert "ND" in [t.get_text() for t in axes[0].texts]
    plt.close(fig)


@pytest.mark.parametrize("plot_type", ['box', 'line', 'bar', 'violin'])
def test_plot_abundance_boxgrid_modes(pdata, plot_type):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        plot_type=plot_type
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert _count_artists(axes) > 0

@pytest.mark.parametrize("bar_error", ["sd", "sem", None])
def test_plot_abundance_boxgrid_bar_mode(pdata, bar_error):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        plot_type="bar",
        bar_error=bar_error,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert len(axes) == 1
    assert _count_artists(axes[0]) > 0

def test_plot_abundance_boxgrid_multiple_genes(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB", "VCL"],
        classes="treatment"
    )
    assert len(axes) == 2
    assert all(isinstance(ax, plt.Axes) for ax in axes)

@pytest.mark.parametrize("plot_type", ['box', 'line', 'bar', 'violin'])
def test_plot_abundance_boxgrid_show_n(pdata, plot_type):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        show_n=True,
        plot_type=plot_type
    )
    # Should draw at least one Text artist
    texts = [t for ax in axes for t in ax.texts]
    assert isinstance(fig, plt.Figure)

def test_plot_abundance_boxgrid_custom_box_kwargs(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        box_kwargs={"linewidth": 3}
    )
    assert isinstance(fig, plt.Figure)

def test_plot_abundance_boxgrid_custom_hline_kwargs(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        plot_type="line",
        hline_kwargs={"linewidth": 5},
    )
    assert isinstance(fig, plt.Figure)

def test_plot_abundance_boxgrid_custom_text_kwargs(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        plot_type="box",
        show_n=True,
        text_kwargs={"fontsize": 20},
    )
    assert isinstance(fig, plt.Figure)

def test_plot_abundance_boxgrid_label_x_false(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        plot_type="box",
        label_x=False,
    )
    assert isinstance(fig, plt.Figure)

def test_plot_abundance_boxgrid_set_y_limits_logscale(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        plot_type="box",
        log_scale=True,
        y_min=0,
        y_max=4,
    )
    ymin, ymax = axes[0].get_ylim()
    assert ymin == 0
    assert ymax == 4

def test_plot_abundance_boxgrid_global_legend_false(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB"],
        classes="treatment",
        plot_type="box",
        global_legend=False,
    )
    assert fig.legends == []

def test_plot_abundance_boxgrid_missing_group_column_raises_keyerror(pdata):
    with pytest.raises(KeyError, match="does_not_exist"):
        scplt.plot_abundance_boxgrid(
            pdata,
            namelist=["ACTB"],
            classes="does_not_exist",
        )

def test_plot_abundance_boxgrid_multi_gene_axes_independent(pdata):
    fig, axes = scplt.plot_abundance_boxgrid(
        pdata,
        namelist=["ACTB", "VCL"],
        classes="treatment",
        plot_type="box",
    )

    # Each axis should have at least one collection (stripplot points) drawn on it.
    assert len(axes[0].collections) > 0
    assert len(axes[1].collections) > 0

    # Titles should match the gene labels (sanity that both panels were populated)
    assert axes[0].get_title() != ""
    assert axes[1].get_title() != ""


# Tests for scplt.plot_pca
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D projection)

def _seed_mock_pca_gsea(pdata, on="protein"):
    pdata.pca(on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    adata.uns["pca_gsea"] = {
        "results": {
            "PC1": pd.DataFrame(
                {
                    "Term": ["OXPHOS", "Immune", "Synapse"],
                    "NES": [1.8, -1.2, 0.6],
                    "FDR q-val": [0.01, 0.03, 0.2],
                }
            ),
            "PC2": pd.DataFrame(
                {
                    "Term": ["OXPHOS", "Immune", "Synapse"],
                    "NES": [-0.9, 2.1, 0.2],
                    "FDR q-val": [0.08, 0.005, 0.4],
                }
            ),
        }
    }

# --- Basic 2D PCA plot ---
def test_plot_pca_runs_without_error(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0
    plt.close(fig)


# --- Continuous coloring (use a protein/gene expression) ---
def test_plot_pca_continuous_coloring(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, classes="UBE4B", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0
    # Should create a colorbar
    assert len(ax.figure.axes) > 1
    plt.close(fig)


# --- Add ellipses per class ---
def test_plot_pca_add_ellipses(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, classes="treatment", add_ellipses=True)
    assert _is_axes_container(result)
    # Ellipses are patches
    n_patches = len(ax.patches)
    assert n_patches >= 1
    plt.close(fig)


# --- Show sample labels ---
def test_plot_pca_show_labels(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, show_labels=True)
    assert _is_axes_container(result)
    # Text annotations should exist
    assert len(ax.texts) > 0
    plt.close(fig)


# --- Return fitted PCA object ---
def test_plot_pca_return_fit(pdata):
    fig, ax = plt.subplots()
    result, fit = scplt.plot_pca(ax, pdata, return_fit=True)
    assert _is_axes_container(result)
    assert isinstance(fit, dict)
    assert "variance_ratio" in fit
    plt.close(fig)


# --- 3D PCA plotting ---
def test_plot_pca_3d_projection(pdata):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    result = scplt.plot_pca(ax, pdata, plot_pc=[1, 2, 3])
    assert _is_axes_container(result)
    assert _count_artists(result) > 0
    plt.close(fig)


# --- Raises if 3 PCs but 2D axis ---
def test_plot_pca_raises_for_3pc_on_2d(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="3 PCs requested"):
        scplt.plot_pca(ax, pdata, plot_pc=[1, 2, 3])
    plt.close(fig)


# --- Raises if invalid plot_pc input ---
@pytest.mark.parametrize("bad_pc", [None, [1], [1, 2, 3, 4], "12"])
def test_plot_pca_invalid_plot_pc(pdata, bad_pc):
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="plot_pc must be a list"):
        scplt.plot_pca(ax, pdata, plot_pc=bad_pc)
    plt.close(fig)

def test_plot_pca_gsea_pathway_vectors_smoke(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_gsea_pathway_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=2,
        fdr_cutoff=0.1,
    )
    assert _is_axes_container(out)
    assert len(ax.texts) >= 1
    plt.close(fig)

def test_plot_pca_gsea_pathway_vectors_without_samples(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_gsea_pathway_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=2,
        show_samples=False,
    )
    assert _is_axes_container(out)
    assert _count_artists(out) > 0
    plt.close(fig)


def test_plot_pca_gsea_pathway_vectors_xlim_ylim(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    scplt.plot_pca_gsea_pathway_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=2,
        xlim=(-5.0, 5.0),
        ylim=(-4.0, 4.0),
        adjust_labels=False,
    )
    assert ax.get_xlim() == pytest.approx((-5.0, 5.0))
    assert ax.get_ylim() == pytest.approx((-4.0, 4.0))
    plt.close(fig)


def test_plot_pca_gsea_pathway_vectors_namelist_and_cmap(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_gsea_pathway_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        namelist=["OXPHOS", "immune"],
        cmap={"OXPHOS": "red"},
        adjust_labels=False,
    )
    assert _is_axes_container(out)
    plt.close(fig)


def test_plot_pca_gsea_pathway_vectors_namelist_raises(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="namelist"):
        scplt.plot_pca_gsea_pathway_vectors(
            ax=ax,
            pdata=pdata,
            on="protein",
            plot_pc=[1, 2],
            namelist=["__NOT_A_PATHWAY__"],
            adjust_labels=False,
        )
    plt.close(fig)


def test_plot_pca_gsea_pathway_vectors_n_vectors_list(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_gsea_pathway_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=[1, 1],
        adjust_labels=False,
    )
    assert _is_axes_container(out)
    plt.close(fig)

def test_plot_pca_gsea_pathway_vectors_namelist_plus_n_vectors_union(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    _, vec_df = scplt.plot_pca_gsea_pathway_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        namelist=["Synapse"],
        n_vectors=2,
        fdr_cutoff=0.1,
        adjust_labels=False,
        return_df=True,
    )
    assert vec_df.shape[0] >= 2
    assert vec_df.iloc[0]["pathway_raw"] == "Synapse"
    plt.close(fig)


def test_plot_pca_protein_vectors_namelist_plus_n_vectors_union(pdata):
    pdata.pca(on="protein")
    adata = pdata.prot
    g0 = str(adata.var["Genes"].iloc[0]) if "Genes" in adata.var.columns else str(adata.var_names[0])
    fig, ax = plt.subplots()
    _, vec_df = scplt.plot_pca_protein_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        namelist=[g0],
        n_vectors=3,
        adjust_labels=False,
        return_df=True,
    )
    assert vec_df.shape[0] >= 2
    assert vec_df.iloc[0]["gene"] == g0
    plt.close(fig)

def test_plot_pca_protein_vectors_smoke(pdata):
    pdata.pca(on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_protein_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=4,
        adjust_labels=False,
    )
    assert _is_axes_container(out)
    assert len(ax.texts) >= 1
    plt.close(fig)


def test_plot_pca_protein_vectors_return_df_and_namelist(pdata):
    pdata.pca(on="protein")
    fig, ax = plt.subplots()
    adata = pdata.prot
    pick = str(adata.var_names[0])
    if "Genes" in adata.var.columns:
        gene_label = str(adata.var["Genes"].iloc[0])
    else:
        gene_label = pick
    _, vec_df = scplt.plot_pca_protein_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        namelist=[gene_label],
        adjust_labels=False,
        return_df=True,
    )
    assert list(vec_df["feature"]) == [pick]
    assert {"gene", "feature", "load_x", "load_y", "arrow_x", "text_x"}.issubset(vec_df.columns)
    plt.close(fig)


def test_plot_pca_protein_vectors_n_vectors_int_and_list(pdata):
    pdata.pca(on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_protein_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=10,
        adjust_labels=False,
    )
    assert _is_axes_container(out)
    plt.close(fig)
    fig, ax = plt.subplots()
    out = scplt.plot_pca_protein_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=[3, 2],
        adjust_labels=False,
    )
    assert _is_axes_container(out)
    plt.close(fig)


def test_plot_pca_protein_vectors_cmap_and_limits(pdata):
    pdata.pca(on="protein")
    adata = pdata.prot
    g = str(adata.var["Genes"].iloc[0]) if "Genes" in adata.var.columns else str(adata.var_names[0])
    fig, ax = plt.subplots()
    scplt.plot_pca_protein_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=3,
        cmap={g: "red"},
        xlim=(-5.0, 5.0),
        ylim=(-4.0, 4.0),
        adjust_labels=False,
    )
    assert ax.get_xlim() == pytest.approx((-5.0, 5.0))
    assert ax.get_ylim() == pytest.approx((-4.0, 4.0))
    plt.close(fig)


def test_plot_pca_protein_vectors_namelist_empty_raises(pdata):
    pdata.pca(on="protein")
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="namelist"):
        scplt.plot_pca_protein_vectors(
            ax=ax,
            pdata=pdata,
            on="protein",
            plot_pc=[1, 2],
            namelist=["__NOT_A_GENE__"],
            adjust_labels=False,
        )
    plt.close(fig)


def test_plot_pca_protein_vectors_n_vectors_bad_list_raises(pdata):
    pdata.pca(on="protein")
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="length"):
        scplt.plot_pca_protein_vectors(
            ax=ax,
            pdata=pdata,
            on="protein",
            plot_pc=[1, 2],
            n_vectors=[1, 2, 3],
            adjust_labels=False,
        )
    plt.close(fig)


def test_plot_pca_protein_vectors_n_vectors_must_be_positive(pdata):
    pdata.pca(on="protein")
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="No proteins to plot"):
        scplt.plot_pca_protein_vectors(
            ax=ax,
            pdata=pdata,
            on="protein",
            plot_pc=[1, 2],
            n_vectors=None,
            adjust_labels=False,
        )
    with pytest.raises(ValueError, match="at least 1"):
        scplt.plot_pca_protein_vectors(
            ax=ax,
            pdata=pdata,
            on="protein",
            plot_pc=[1, 2],
            n_vectors=0,
            adjust_labels=False,
        )
    plt.close(fig)


def test_plot_pca_gsea_bubble_smoke(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_gsea_bubble(
        ax=ax,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=3,
        fdr_cutoff=0.2,
    )
    assert _is_axes_container(out)
    assert _count_artists(out) > 0
    plt.close(fig)

def test_plot_pca_gsea_heatmap_smoke(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    out = scplt.plot_pca_gsea_heatmap(
        ax=ax,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=3,
        fdr_cutoff=0.2,
    )
    assert _is_axes_container(out)
    assert _count_artists(out) > 0
    assert "pathway_loadings" in pdata.prot.uns["pca_gsea"]
    plt.close(fig)

def test_plot_pca_gsea_pathway_vectors_library_split_and_return_df(pdata):
    pdata.pca(on="protein")
    adata = pdata.prot
    adata.uns["pca_gsea"] = {
        "results": {
            "PC1": pd.DataFrame(
                {
                    "Term": ["KEGG_2026__DNA_REPLICATION", "Reactome_Pathways_2024__IMMUNE_RESPONSE"],
                    "NES": [1.8, -1.2],
                    "FDR q-val": [0.01, 0.03],
                }
            ),
            "PC2": pd.DataFrame(
                {
                    "Term": ["KEGG_2026__DNA_REPLICATION", "Reactome_Pathways_2024__IMMUNE_RESPONSE"],
                    "NES": [-0.9, 2.1],
                    "FDR q-val": [0.08, 0.005],
                }
            ),
        }
    }
    fig, ax = plt.subplots()
    _, vec_df = scplt.plot_pca_gsea_pathway_vectors(
        ax=ax,
        pdata=pdata,
        on="protein",
        plot_pc=[1, 2],
        n_vectors=2,
        return_df=True,
    )
    assert "library" in vec_df.columns
    assert not vec_df["pathway"].str.contains("__").any()
    assert set(vec_df["library"]) == {"KEGG_2026", "Reactome_Pathways_2024"}
    plt.close(fig)

def test_plot_pca_gsea_bubble_include_exclude_pathways(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots()
    _, bubble_df = scplt.plot_pca_gsea_bubble(
        ax=ax,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        include_pathways=["Immune", "OXPHOS"],
        exclude_pathways=["Immune"],
        top_n=1,
        title_case_labels=False,
        return_df=True,
    )
    assert set(bubble_df["pathway_raw"]) == {"OXPHOS"}
    plt.close(fig)


def test_plot_pca_gsea_bubble_figsize_scales_sizes(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig_small, ax_small = plt.subplots(figsize=(3, 4))
    _, df_small = scplt.plot_pca_gsea_bubble(
        ax=ax_small,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=3,
        fdr_cutoff=0.2,
        return_df=True,
    )
    fig_large, ax_large = plt.subplots(figsize=(9, 12))
    _, df_large = scplt.plot_pca_gsea_bubble(
        ax=ax_large,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=3,
        fdr_cutoff=0.2,
        return_df=True,
    )
    assert df_large["bubble_size"].max() > df_small["bubble_size"].max()
    plt.close(fig_small)
    plt.close(fig_large)


def test_plot_pca_gsea_bubble_size_fdr_cap_and_size_scale(pdata):
    pdata.pca(on="protein")
    adata = pdata.prot
    adata.uns["pca_gsea"] = {
        "results": {
            "PC1": pd.DataFrame(
                {
                    "Term": ["OXPHOS", "Immune"],
                    "NES": [2.0, -1.5],
                    "FDR q-val": [1e-20, 0.01],
                }
            ),
            "PC2": pd.DataFrame(
                {
                    "Term": ["OXPHOS", "Immune"],
                    "NES": [-1.0, 1.8],
                    "FDR q-val": [0.05, 1e-20],
                }
            ),
        }
    }
    fig, ax = plt.subplots(figsize=(4, 5))
    _, df_cap = scplt.plot_pca_gsea_bubble(
        ax=ax,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=2,
        fdr_cutoff=None,
        size_scale=0.85,
        size_fdr_cap=5.0,
        return_df=True,
    )
    # Extreme FDR must not exceed size of an FDR that hits the cap exactly (1e-5).
    max_at_cap = df_cap.loc[df_cap["FDR q-val"] <= 1e-5, "bubble_size"].max()
    assert np.isclose(df_cap["bubble_size"].max(), max_at_cap)
    plt.close(fig)

    fig_a, ax_a = plt.subplots(figsize=(4, 5))
    _, df_a = scplt.plot_pca_gsea_bubble(
        ax=ax_a,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=2,
        fdr_cutoff=None,
        size_scale=0.5,
        return_df=True,
    )
    fig_b, ax_b = plt.subplots(figsize=(4, 5))
    _, df_b = scplt.plot_pca_gsea_bubble(
        ax=ax_b,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=2,
        fdr_cutoff=None,
        size_scale=1.0,
        return_df=True,
    )
    # Area scales with size_scale**2 for the same FDR weight.
    assert np.isclose(df_b["bubble_size"].max() / df_a["bubble_size"].max(), (1.0 / 0.5) ** 2)
    plt.close(fig_a)
    plt.close(fig_b)


def test_plot_pca_gsea_bubble_square_cells_and_padding(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig, ax = plt.subplots(figsize=(6, 4))
    pc_pad = 0.6
    _, bubble_df = scplt.plot_pca_gsea_bubble(
        ax=ax,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=3,
        fdr_cutoff=0.2,
        pc_pad=pc_pad,
        return_df=True,
    )
    n_pcs = bubble_df["pc_i"].nunique()
    n_pathways = bubble_df["pathway_i"].nunique()
    pc_spacing = 2.0 * pc_pad
    row_pad = 0.5
    assert ax.get_xlim() == pytest.approx((-pc_pad, (n_pcs - 1) * pc_spacing + pc_pad))
    assert ax.get_ylim() == pytest.approx((-row_pad, n_pathways - 1 + row_pad))
    span_x = (n_pcs - 1) * pc_spacing + 2.0 * pc_pad
    span_y = (n_pathways - 1) + 2.0 * row_pad
    assert ax.get_box_aspect() == pytest.approx(span_y / span_x)
    # Larger pc_pad → wider x-span for the same number of PCs.
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    scplt.plot_pca_gsea_bubble(
        ax=ax2,
        pdata=pdata,
        on="protein",
        pcs=[1, 2],
        top_n=3,
        fdr_cutoff=0.2,
        pc_pad=0.9,
    )
    assert (ax2.get_xlim()[1] - ax2.get_xlim()[0]) > (ax.get_xlim()[1] - ax.get_xlim()[0])
    plt.close(fig)
    plt.close(fig2)


def test_plot_pca_gsea_bubble_cbar_scale_moves_size_legend(pdata):
    _seed_mock_pca_gsea(pdata, on="protein")
    fig_s, ax_s = plt.subplots(figsize=(5, 6))
    scplt.plot_pca_gsea_bubble(
        ax=ax_s, pdata=pdata, on="protein", pcs=[1, 2], top_n=3, fdr_cutoff=0.2, cbar_scale=0.5
    )
    fig_l, ax_l = plt.subplots(figsize=(5, 6))
    scplt.plot_pca_gsea_bubble(
        ax=ax_l, pdata=pdata, on="protein", pcs=[1, 2], top_n=3, fdr_cutoff=0.2, cbar_scale=1.5
    )
    fig_s.canvas.draw()
    fig_l.canvas.draw()
    cbar_s = [a for a in fig_s.axes if a is not ax_s][0]
    cbar_l = [a for a in fig_l.axes if a is not ax_l][0]
    assert cbar_l.get_position().height > cbar_s.get_position().height
    # Size legend is in colorbar axes coords just below the bar; taller bar → legend lower on figure.
    leg_s = ax_s.get_legend().get_window_extent(fig_s.canvas.get_renderer())
    leg_l = ax_l.get_legend().get_window_extent(fig_l.canvas.get_renderer())
    assert leg_l.y0 < leg_s.y0
    plt.close(fig_s)
    plt.close(fig_l)


# test resolve_plot_color
@pytest.fixture
def dummy_adata():
    obs = pd.DataFrame({
        "sample": ["S1", "S2", "S3", "S4"],
        "treatment": ["ctrl", "ctrl", "drug", "drug"],
        "cellline": ["A", "B", "A", "B"]
    }, index=[f"cell_{i}" for i in range(4)])
    var = pd.DataFrame({
        "Genes": ["ACTB", "GAPDH", "VDAC1"],
    }, index=["P1", "P2", "P3"])
    X = np.random.rand(4, 3)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata

def test_resolve_plot_colors_none(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes=None, cmap="default")
    assert all(c == "grey" for c in colors)
    assert cmap is None
    assert legend and legend[0].get_label() == "All samples"

def test_resolve_plot_colors_single_obs(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes="treatment", cmap="default")
    unique_colors = set(colors)
    assert len(unique_colors) == 2
    assert cmap is None
    assert all(hasattr(p, "get_facecolor") for p in legend)

def test_resolve_plot_colors_multi_obs(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes=["cellline", "treatment"], cmap="default")
    assert isinstance(colors, list)
    assert len(colors) == len(dummy_adata)
    assert cmap is None
    assert all(hasattr(p, "get_label") for p in legend)

def test_resolve_plot_colors_continuous(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes="P1", cmap="viridis", layer="X")
    assert isinstance(colors, np.ndarray)
    assert cmap is not None
    assert legend is None
    assert np.isfinite(colors).all()

def test_resolve_plot_colors_gene_name(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes="ACTB", cmap="viridis", layer="X")
    assert isinstance(colors, np.ndarray)
    assert np.isfinite(colors).all()

@pytest.mark.parametrize("bad_input", ["NotAColumn", 123, ["bad_col"]])
def test_resolve_plot_colors_invalid(dummy_adata, bad_input):
    with pytest.raises(ValueError):
        scplt.resolve_plot_colors(dummy_adata, classes=bad_input, cmap="default")

def test_hollow_edge_legend_handles():
    from scpviz.plotting.dimreduc import _hollow_edge_legend_handles
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    filled = [mpatches.Patch(color="steelblue", label="A"), mpatches.Patch(color="tomato", label="B")]
    hollow = _hollow_edge_legend_handles(filled)
    assert hollow is not None and len(hollow) == 2
    for h in hollow:
        assert isinstance(h, mlines.Line2D)
        assert h.get_markerfacecolor() in ("none", "None")
    assert hollow[0].get_label() == "A"
    assert hollow[1].get_label() == "B"

def test_plot_pca_edge_color_legend_is_hollow(pdata):
    import matplotlib.lines as mlines

    fig, ax = plt.subplots()
    scplt.plot_pca(ax, pdata, color="treatment", edge_color="cellline", force=True)
    edge_handles = []
    for artist in ax.get_children():
        if isinstance(artist, matplotlib.legend.Legend) and artist.get_title().get_text() == "Cellline":
            edge_handles = artist.legend_handles
            break
    assert edge_handles, "expected a Cellline edge-color legend"
    for h in edge_handles:
        assert isinstance(h, mlines.Line2D)
        assert h.get_markerfacecolor() in ("none", "None")

def test_resolve_colorbar_norm_invalid_literal():
    from scpviz.plotting.dimreduc import _resolve_colorbar_norm

    with pytest.raises(ValueError, match="Invalid colorbar_norm"):
        _resolve_colorbar_norm(np.array([1.0, 10.0]), "ln")

def test_resolve_colorbar_norm_log10_decade_limits():
    from scpviz.plotting.dimreduc import _resolve_colorbar_norm
    import matplotlib.colors as mcolors

    norm, tick_base = _resolve_colorbar_norm(np.array([37.0, 840.0]), "log10")
    assert isinstance(norm, mcolors.LogNorm)
    assert norm.vmin == 10.0
    assert norm.vmax == 1000.0
    assert tick_base == 10

def test_resolve_colorbar_norm_log2_power_limits():
    from scpviz.plotting.dimreduc import _resolve_colorbar_norm
    import matplotlib.colors as mcolors

    norm, tick_base = _resolve_colorbar_norm(np.array([5.0, 20.0]), "log2")
    assert isinstance(norm, mcolors.LogNorm)
    assert norm.vmin == 4.0
    assert norm.vmax == 32.0
    assert tick_base == 2

def test_plot_pca_nan_color_layer(pdata):
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    obs_mask = np.zeros(pdata.prot.n_obs, dtype=bool)
    obs_mask[0] = True
    _zero_gene_for_obs_mask(pdata, gene, obs_mask)
    scplt.plot_pca(ax, pdata, color=gene, force=True, nan_color="red")
    assert len(ax.collections) >= 2
    face_colors = ax.collections[0].get_facecolors()
    assert np.allclose(face_colors[0, :3], matplotlib.colors.to_rgb("red"))

def test_plot_pca_colorbar_norm_log10_label(pdata):
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    scplt.plot_pca(ax, pdata, color=gene, colorbar_norm="log10", force=True)
    ylabels = [a.get_ylabel() for a in fig.axes if a.get_ylabel()]
    assert any(f"{gene} Abundance (log10)" in lab for lab in ylabels)

def test_plot_pca_negative_abundance_warns(pdata, capsys):
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    var_idx = int(np.where(pdata.prot.var["Genes"] == gene)[0][0])
    pdata.prot.X[0, var_idx] = -1.0
    scplt.plot_pca(ax, pdata, color=gene, force=True)
    captured = capsys.readouterr()
    assert "negative abundance" in captured.out.lower()

def _combo_mapping_literal(pdata):
    """Unique (cellline, treatment) pairs with simple face/edge styles."""
    adata = pdata.prot
    mapping_keys = ["cellline", "treatment"]
    mapping = {}
    edges = ["black", "blue", "red", "green"]
    for i, row in enumerate(adata.obs[mapping_keys].drop_duplicates().itertuples(index=False, name=None)):
        mapping[tuple(row)] = {"color": "#f0f0f0", "edge_color": edges[i % len(edges)]}
    return mapping_keys, mapping

def test_plot_pca_mapping_literal_face_edge(pdata):
    fig, ax = plt.subplots()
    mapping_keys, mapping = _combo_mapping_literal(pdata)
    result = scplt.plot_pca(
        ax,
        pdata,
        mapping_keys=mapping_keys,
        mapping=mapping,
        force=True,
        on="protein",
    )
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_pca_mapping_abundance_with_edges(pdata):
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    adata = pdata.prot
    mapping_keys = ["cellline", "treatment"]
    mapping = {}
    for row in adata.obs[mapping_keys].drop_duplicates().itertuples(index=False, name=None):
        mapping[tuple(row)] = {"edge_color": "black"}
    result = scplt.plot_pca(
        ax,
        pdata,
        color=gene,
        cmap="plasma",
        mapping_keys=mapping_keys,
        mapping=mapping,
        force=True,
        on="protein",
    )
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_pca_mapping_rejects_edge_color_kw(pdata):
    fig, ax = plt.subplots()
    mapping_keys, mapping = _combo_mapping_literal(pdata)
    with pytest.raises(ValueError, match="edge_color cannot be used with mapping"):
        scplt.plot_pca(
            ax,
            pdata,
            mapping_keys=mapping_keys,
            mapping=mapping,
            edge_color="treatment",
            force=True,
            on="protein",
        )

def test_plot_pca_mapping_raises_missing_combo(pdata):
    fig, ax = plt.subplots()
    mapping_keys = ["cellline", "treatment"]
    mapping = {("nonexistent", "combo"): {"color": "white", "edge_color": "k"}}
    with pytest.raises(ValueError, match="not found in mapping"):
        scplt.plot_pca(
            ax,
            pdata,
            mapping_keys=mapping_keys,
            mapping=mapping,
            force=True,
            on="protein",
            mapping_on_missing="raise",
        )

def test_plot_pca_mapping_incomplete_warn_default(pdata):
    """Missing combos: default warn prints and grey face + no edge (except abundance color=)."""
    fig, ax = plt.subplots()
    adata = pdata.prot
    mapping_keys = ["cellline", "treatment"]
    one = next(adata.obs[mapping_keys].drop_duplicates().itertuples(index=False, name=None))
    mapping = {tuple(one): {"color": "#e0e0e0", "edge_color": "black"}}
    result = scplt.plot_pca(
        ax,
        pdata,
        mapping_keys=mapping_keys,
        mapping=mapping,
        force=True,
        on="protein",
    )
    assert _is_axes_container(result)
    assert len(ax.collections) > 0


def test_plot_pca_mapping_single_key_string(pdata):
    """Single mapping_keys column may use string keys instead of 1-tuples."""
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    adata = pdata.prot
    mapping_keys = ["treatment"]
    mapping = {
        str(lv): {"edge_color": "black"}
        for lv in adata.obs["treatment"].dropna().unique()
    }
    result = scplt.plot_pca(
        ax,
        pdata,
        color=gene,
        cmap="plasma",
        mapping_keys=mapping_keys,
        mapping=mapping,
        force=True,
        on="protein",
    )
    assert _is_axes_container(result)
    assert len(ax.collections) > 0


def test_plot_pca_mapping_single_key_one_tuple_still_works(pdata):
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    adata = pdata.prot
    mapping_keys = ["treatment"]
    mapping = {
        (str(lv),): {"edge_color": "black"}
        for lv in adata.obs["treatment"].dropna().unique()
    }
    result = scplt.plot_pca(
        ax,
        pdata,
        color=gene,
        cmap="plasma",
        mapping_keys=mapping_keys,
        mapping=mapping,
        force=True,
        on="protein",
    )
    assert _is_axes_container(result)


def test_plot_pca_mapping_rejects_string_key_for_multi_column(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must be tuples of length 2"):
        scplt.plot_pca(
            ax,
            pdata,
            mapping_keys=["cellline", "treatment"],
            mapping={"ctrl": {"color": "white", "edge_color": "black"}},
            force=True,
            on="protein",
        )


# tests for plot_umap
def test_plot_umap_runs_without_error(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_umap(ax, pdata, classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_runs_without_class(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_umap(ax, pdata, classes=None, on="protein")
    assert _is_axes_container(result)

def test_plot_umap_forces_recompute(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_umap(ax, pdata, force=True, classes="treatment")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_on_peptide_level(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_umap(ax, pdata, classes="cellline", on="peptide")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_continuous_coloring(pdata):
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    result = scplt.plot_umap(ax, pdata, classes=gene, cmap="plasma")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_3d_projection(pdata):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    result = scplt.plot_umap(
        ax, pdata, classes="treatment", umap_params={"n_components": 3}
    )
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_invalid_on_raises(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Invalid value for 'on'"):
        scplt.plot_umap(ax, pdata, on="invalid_level")

# Tests for scplt.plot_pca_scree
def test_plot_pca_scree_with_dict_input():
    """Ensure scree plot works when pca is a dict (e.g. from .uns['pca'])."""
    fig, ax = plt.subplots()
    pca_dict = {"variance_ratio": np.array([0.4, 0.3, 0.2, 0.1])}
    result = scplt.plot_pca_scree(ax, pca_dict)
    assert _is_axes_container(result)
    # should have at least two lines (variance + cumulative)
    assert len(result.lines) >= 2
    labels = [line.get_label() for line in result.lines]
    assert "Explained Variance" in labels
    assert "Cumulative Variance" in labels

def test_plot_pca_scree_with_sklearn_object():
    """Ensure scree plot works for a real PCA object."""
    from sklearn.decomposition import PCA

    # Generate random data
    X = np.random.randn(20, 5)
    model = PCA(n_components=3, random_state=42).fit(X)

    fig, ax = plt.subplots()
    result = scplt.plot_pca_scree(ax, model)
    assert _is_axes_container(result)
    assert len(result.lines) >= 2
    assert result.get_title() == "Scree Plot"
    assert "Variance" in result.get_ylabel()

def test_plot_pca_scree_handles_single_component():
    """Handle PCA with a single component gracefully."""
    fig, ax = plt.subplots()
    pca_dict = {"variance_ratio": np.array([1.0])}
    result = scplt.plot_pca_scree(ax, pca_dict)
    assert _is_axes_container(result)
    # should produce one point and one cumulative curve
    assert len(result.lines) == 2
    assert result.get_xlabel() == "Principal Component"

def test_plot_pca_scree_with_real_pdata(pdata):
    """Integration test: works with real PCA results from pdata.prot.uns['pca']."""
    # run PCA first
    pdata.pca(on="protein", layer="X")
    pca_dict = pdata.prot.uns["pca"]

    fig, ax = plt.subplots()
    result = scplt.plot_pca_scree(ax, pca_dict)
    assert _is_axes_container(result)
    assert len(result.lines) >= 2
    assert "Scree" in result.get_title()

# Tests for scplt.plot_clustermap

def test_plot_clustermap_runs_basic(pdata):
    """Smoke test: overview clustermap with row + column dendrograms."""
    import matplotlib.figure

    fig = scplt.plot_clustermap(
        pdata, classes=["cellline"], show_row_labels=False
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_plot_clustermap_with_classes(pdata):
    """Smoke test: multi-class headers; clustering stored in pdata.stats."""
    import matplotlib.figure

    fig = scplt.plot_clustermap(
        pdata,
        classes=["cellline", "treatment"],
        show_row_labels=False,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    cluster_key = "prot_X_clustermap"
    assert cluster_key in pdata.stats
    stats = pdata.stats[cluster_key]
    assert "row_order" in stats
    assert "col_order" in stats
    assert stats["col_linkage"] is not None
    assert stats["row_linkage"] is not None
    plt.close(fig)


def test_plot_clustermap_separate_legend(pdata):
    """separate_legend returns (fig, legend_fig)."""
    import matplotlib.figure

    out = scplt.plot_clustermap(
        pdata,
        classes=["cellline"],
        separate_legend=True,
        figsize=(5, 4),
    )
    assert isinstance(out, tuple) and len(out) == 2
    fig, legend_fig = out
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(legend_fig, matplotlib.figure.Figure)
    plt.close(fig)
    plt.close(legend_fig)


def test_plot_clustermap_with_namelist(pdata):
    """Run with a restricted namelist (subset of proteins)."""
    import matplotlib.figure

    some_proteins = pdata.prot.var_names[:5].tolist()
    fig = scplt.plot_clustermap(
        pdata,
        classes=["cellline"],
        namelist=some_proteins,
        show_row_labels=True,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    cluster_key = "prot_X_clustermap"
    assert cluster_key in pdata.stats
    assert pdata.stats[cluster_key]["namelist_used"] != "all_proteins"
    assert len(pdata.stats[cluster_key]["row_order"]) == 5
    plt.close(fig)


def test_plot_clustermap_invalid_on_raises(pdata):
    """Invalid `on` argument should raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Invalid input: on must be"):
        scplt.plot_clustermap(pdata, classes=["cellline"], on="invalid")


def test_impute_row_min_fills_with_protein_min():
    """Clustering impute uses per-row minimum, not median."""
    from scpviz.plotting.clustering import _impute_row_min

    mat = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, 4.0, np.nan],
        ],
        dtype=float,
    )
    imputed, keep = _impute_row_min(mat)
    assert keep.tolist() == [True, True]
    assert imputed[0, 1] == 1.0  # row min of [1, 3]
    assert imputed[1, 2] == 2.0  # row min of [2, 4]


class TestPlotPairwiseCorrelation:
    """Smoke tests for plot_pairwise_correlation (group + sample level)."""

    def test_returns_fig_and_ax(self, pdata):
        import matplotlib.figure
        fig, ax = scplt.plot_pairwise_correlation(pdata, classes="cellline", force=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_heatmap_has_image(self, pdata):
        import matplotlib.image as mimage
        fig, ax = scplt.plot_pairwise_correlation(pdata, classes="cellline", force=True)
        images = [c for c in ax.get_children() if isinstance(c, mimage.AxesImage)]
        assert len(images) >= 1
        plt.close(fig)

    def test_correct_tick_count(self, pdata):
        n_groups = int(pdata.prot.obs["cellline"].nunique())
        fig, ax = scplt.plot_pairwise_correlation(pdata, classes="cellline", force=True)
        assert len(ax.get_xticks()) == n_groups
        assert len(ax.get_yticks()) == 0  # row labels omitted; groups on x-axis only
        plt.close(fig)

    def test_all_methods_run(self, pdata):
        for method in ("pearson", "spearman", "euclidean"):
            fig, ax = scplt.plot_pairwise_correlation(
                pdata, classes="cellline", method=method, force=True
            )
            plt.close(fig)

    def test_list_classes_runs(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes=["cellline", "treatment"], force=True
        )
        plt.close(fig)

    def test_annot_runs(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", annot=True, force=True
        )
        plt.close(fig)

    def test_figsize_override(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", figsize=(12, 12), force=True
        )
        w, h = fig.get_size_inches()
        assert abs(w - 12) < 0.1 and abs(h - 12) < 0.1
        plt.close(fig)

    def test_custom_cmap(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", cmap="viridis", force=True
        )
        plt.close(fig)

    def test_title_set(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", title="Test Title", force=True
        )
        assert fig._suptitle is not None
        assert "Test Title" in fig._suptitle.get_text()
        plt.close(fig)

    def test_wrapper_method_runs(self, pdata):
        import matplotlib.figure
        fig, ax = pdata.plot_pairwise_correlation(classes="cellline", force=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_show_samples_returns_fig_and_ax(self, pdata):
        import matplotlib.figure
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", show_samples=True, force=True
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_show_samples_matrix_size(self, pdata):
        n_samples = pdata.prot.n_obs
        fig, ax = scplt.plot_pairwise_correlation(
            pdata,
            classes="cellline",
            show_samples=True,
            show_ticklabels=True,
            force=True,
        )
        assert len(ax.get_xticks()) == n_samples
        plt.close(fig)

    def test_show_samples_ticks_hidden_auto(self, pdata, capsys):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata,
            classes="cellline",
            show_samples=True,
            show_ticklabels=None,
            ticklabels_auto_max_samples=3,
            force=True,
        )
        captured = capsys.readouterr()
        out_low = captured.out.lower()
        assert ("tick" in out_low) and ("hidden" in out_low or "threshold" in out_low)
        assert len(ax.get_xticks()) == 0
        plt.close(fig)

    def test_show_samples_ticks_forced_on(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata,
            classes="cellline",
            show_samples=True,
            show_ticklabels=True,
            force=True,
        )
        assert len(ax.get_xticks()) == pdata.prot.n_obs
        plt.close(fig)

    def test_show_samples_ticks_forced_off(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata,
            classes="cellline",
            show_samples=True,
            show_ticklabels=False,
            force=True,
        )
        assert len(ax.get_xticks()) == 0
        plt.close(fig)

    def test_show_samples_list_classes(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata,
            classes=["cellline", "treatment"],
            show_samples=True,
            force=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_show_samples_triggers_recompute_if_sample_matrix_missing(self, pdata):
        pdata.pairwise_correlation(
            classes="cellline", compute_sample_matrix=False, force=True
        )
        assert pdata.prot.uns["pairwise_corr"]["sample_matrix"] is None
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", show_samples=True
        )
        assert pdata.prot.uns["pairwise_corr"]["sample_matrix"] is not None
        plt.close(fig)

    def test_show_samples_cache_reused(self, pdata, capsys):
        scplt.plot_pairwise_correlation(
            pdata, classes="cellline", show_samples=True, force=True
        )
        capsys.readouterr()
        scplt.plot_pairwise_correlation(
            pdata, classes="cellline", show_samples=True
        )
        captured = capsys.readouterr()
        assert "cached" in captured.out.lower()
        plt.close("all")

    def test_group_plot_reuses_cache_when_sample_matrix_present(self, pdata):
        scplt.plot_pairwise_correlation(
            pdata, classes="cellline", show_samples=True, force=True
        )
        assert pdata.prot.uns["pairwise_corr"]["sample_matrix"] is not None
        scplt.plot_pairwise_correlation(
            pdata, classes="cellline", show_samples=False, force=False
        )
        assert pdata.prot.uns["pairwise_corr"]["sample_matrix"] is not None
        plt.close("all")

    def test_ticklabels_auto_max_samples_invalid(self, pdata):
        with pytest.raises(ValueError, match="ticklabels_auto_max_samples"):
            scplt.plot_pairwise_correlation(
                pdata,
                classes="cellline",
                show_samples=True,
                ticklabels_auto_max_samples=0,
                force=True,
            )

    def test_force_recomputes(self, pdata):
        scplt.plot_pairwise_correlation(pdata, classes="cellline", force=True)
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", force=True
        )
        assert fig is not None
        plt.close(fig)

    def test_euclidean_vmin_vmax_auto(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", method="euclidean", force=True
        )
        im = next(c for c in ax.get_children() if hasattr(c, "get_clim"))
        vmin, vmax = im.get_clim()
        assert vmin >= -0.01
        plt.close(fig)

    def test_user_display_order(self, pdata):
        scplt.plot_pairwise_correlation(pdata, classes="cellline", force=True)
        uniq = sorted(str(x) for x in pdata.prot.obs["cellline"].unique())
        user_order = list(reversed(uniq))
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", order=user_order, force=False
        )
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == user_order
        plt.close(fig)

    def test_subset_mask_runs(self, pdata):
        n = pdata.prot.n_obs
        mask = np.zeros(n, dtype=bool)
        mask[: max(1, n // 2)] = True
        fig, ax = scplt.plot_pairwise_correlation(
            pdata, classes="cellline", subset_mask=mask, force=True
        )
        plt.close(fig)

    def test_show_annotation_legend_false(self, pdata):
        fig, ax = scplt.plot_pairwise_correlation(
            pdata,
            classes="cellline",
            show_annotation_legend=False,
            force=True,
        )
        assert ax.get_legend() is None
        plt.close(fig)

# test scplt.plot_volcano and related functions
def mock_volcano_df():
    df = pd.DataFrame({
        "log2fc": [2.1, -1.8, 0.5, -0.2],
        "p_value": [0.001, 0.004, 0.2, 0.8],
        "significance": ["upregulated", "downregulated", "not significant", "not significant"],
        "significance_score": [10, -8, 0.5, 0.1],
        "Genes": ["G1", "G2", "G3", "G4"]
    })
    df["-log10(p_value)"] = -np.log10(df["p_value"])
    df.index = [f"P{i}" for i in range(1, len(df) + 1)]  # make index strings like P1, P2, P3, P4
    return df

def test_plot_volcano_with_de_data():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    result = scplt.plot_volcano(ax, de_data=df)
    assert hasattr(result, "scatter"), "❌ Should return a matplotlib Axes"
    plt.close(fig)

def test_plot_volcano_returns_df():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    ax, out_df = scplt.plot_volcano(ax, de_data=df, return_df=True)
    assert isinstance(out_df, pd.DataFrame), "❌ return_df=True should return a DataFrame"
    assert all(col in out_df.columns for col in ["log2fc", "p_value", "significance"]), "❌ Missing expected DE columns"
    plt.close(fig)

def test_plot_volcano_adata_with_de_data():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    result = scplt.plot_volcano_adata(ax, de_data=df)
    assert hasattr(result, "scatter"), "❌ Should return a matplotlib Axes"
    plt.close(fig)

def test_plot_volcano_adata_no_data():
    fix, ax = plt.subplots()
    with pytest.raises(ValueError, match="must supply adata"):
        scplt.plot_volcano_adata(ax, None)

def test_plot_volcano_adata_returns_df():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    ax, out_df = scplt.plot_volcano_adata(ax, de_data=df, return_df=True)
    assert isinstance(out_df, pd.DataFrame), "❌ return_df=True should return a DataFrame"
    assert all(col in out_df.columns for col in ["log2fc", "p_value", "significance"]), "❌ Missing expected DE columns"
    plt.close(fig)

def test_add_volcano_legend_adds_handles():
    fig, ax = plt.subplots()
    scplt.add_volcano_legend(ax)
    legend = ax.get_legend()
    assert legend is not None, "❌ Legend should exist after calling add_volcano_legend()"
    labels = [t.get_text() for t in legend.get_texts()]
    assert {"Up", "Down", "NS"}.issubset(labels), f"❌ Unexpected legend labels: {labels}"
    plt.close(fig)

def test_mark_volcano_highlights_points():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, no_marks=True)
    n_before = len(ax.collections)
    scplt.mark_volcano(ax, df, label=["G1"], label_color="red")
    n_after = len(ax.collections)
    assert n_after > n_before, "❌ mark_volcano should add new scatter points"
    plt.close(fig)

def test_mark_volcano_return_texts():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, no_marks=True)
    texts = scplt.mark_volcano(ax, df, label=["G1"], return_texts=True)
    assert texts is not None, "mark_volcano with return_texts should return some texts"
    plt.close(fig)

def test_mark_volcano_by_significance_highlights_points():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, no_marks=True)
    n_before = len(ax.collections)
    scplt.mark_volcano_by_significance(ax, df, label=["G1"])
    n_after = len(ax.collections)
    assert n_after > n_before, "❌ mark_volcano_by_significance should add new scatter points"
    plt.close(fig)

def test_mark_volcano_by_significance_return_texts():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, no_marks=True)
    texts = scplt.mark_volcano_by_significance(ax, df, label=["G1"], return_texts=True)
    assert texts is not None, "mark_volcano_by_significance with return_texts should return some texts"
    plt.close(fig)

def test_plot_volcano_with_label_list():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, label=["G1", "G2"])
    texts = [t.get_text() for t in ax.texts]
    assert any(g in texts for g in ["G1", "G2"]), "❌ Gene labels should appear on volcano plot"
    plt.close(fig)

def test_plot_volcano_invalid_p_col():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="p_col must be"):
        scplt.plot_volcano(ax, de_data=df, p_col="q_value")
    plt.close(fig)

def test_plot_volcano_adj_p_col_missing_raises():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="adj_p_value"):
        scplt.plot_volcano(ax, de_data=df, p_col="adj_p_value")
    plt.close(fig)

def test_plot_volcano_adj_p_col_with_fdr_data():
    df = mock_volcano_df()
    df["adj_p_value"] = df["p_value"] * 2
    df["-log10(adj_p_value)"] = -np.log10(df["adj_p_value"])
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, p_col="adj_p_value")
    assert "adjusted" in ax.get_ylabel().lower()
    plt.close(fig)

def test_mark_volcano_adj_p_col_missing_raises():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="adj_p_value"):
        scplt.mark_volcano(ax, df, label=["G1"], p_col="adj_p_value")
    plt.close(fig)

def test_mark_volcano_auto_p_col_prefers_adj():
    df = mock_volcano_df()
    df["adj_p_value"] = [0.1, 0.2, 0.3, 0.4]
    df["-log10(adj_p_value)"] = -np.log10(df["adj_p_value"])
    fig, ax = plt.subplots()
    scplt.mark_volcano(ax, df, label=["G1"], show_names=False)
    y_adj = -np.log10(0.1)
    y_raw = -np.log10(0.001)
    y_marked = ax.collections[-1].get_offsets()[0, 1]
    assert np.isclose(y_marked, y_adj)
    assert not np.isclose(y_marked, y_raw)
    plt.close(fig)

# test scplt.plot_rankquant() and related functions

def test_plot_rankquant_runs_without_error(pdata):
    """Ensure rank–quant plot runs and stores rank metrics."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = scplt.plot_rankquant(ax, pdata, classes="cellline", on="protein", alpha=0.1)
    
    # Check expected outputs
    assert hasattr(ax, "scatter"), "❌ plot_rankquant should return a Matplotlib Axes."
    var_cols = pdata.prot.var.columns
    assert any("Average:" in c for c in var_cols), "❌ Missing Average: columns in .var."
    assert any("Rank:" in c for c in var_cols), "❌ Missing Rank: columns in .var."
    
    plt.close(fig)

def test_mark_rankquant_adds_points(pdata):
    """Test that mark_rankquant overlays highlights correctly."""
    # Run rankquant first to populate .var
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = scplt.plot_rankquant(ax, pdata, classes="cellline", on="protein", alpha=0.1)
    n_before = len(ax.collections)

    # Build mock mark_df with at least one Entry from pdata.prot.var_names
    test_entry = pdata.prot.var_names[0]
    mark_df = pd.DataFrame({
        "Entry": [test_entry],
        "Gene Names": ["TESTGENE"]
    })

    # Call mark_rankquant
    scplt.mark_rankquant(
        ax, pdata, mark_df=mark_df,
        class_values=["A549"] if "A549" in pdata.prot.obs["cellline"].unique() else [pdata.prot.obs["cellline"].unique()[0]],
        on="protein",
        show_label=True,
        color="red"
    )

    n_after = len(ax.collections)
    assert n_after > n_before, "❌ mark_rankquant should add new points to the plot."

    plt.close(fig)

def test_plot_rankquant_debug_mode_runs(pdata, capsys):
    """Ensure plot_rankquant runs in debug mode without errors or warnings."""
    fig, ax = plt.subplots(figsize=(3, 2))

    # Run with debug=True — this should print shapes and intermediate data info
    ax = scplt.plot_rankquant(
        ax,
        pdata,
        classes="cellline",
        on="protein",
        alpha=0.1,
        debug=True
    )

    # Capture printed output and ensure something was printed
    captured = capsys.readouterr()
    assert "nsample" in captured.out or "shape" in captured.out, \
        "❌ Debug mode should print internal diagnostics."

    # Validate output integrity
    assert hasattr(ax, "scatter"), "❌ plot_rankquant should return a Matplotlib Axes in debug mode."
    assert any("Rank:" in c for c in pdata.prot.var.columns), \
        "❌ Rank columns should still be computed in debug mode."

    plt.close(fig)

# test plot_venn()
@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_runs_and_returns_contents(monkeypatch):
    """Ensure plot_venn runs without error and returns correct outputs."""
    fig, ax = plt.subplots()
    pdata = object()  # dummy object; utils mock doesn't need real pdata

    # 2-set Venn with default colors
    ax_out, contents = scplt.plot_venn(ax, pdata, classes=["GroupA", "GroupB"], return_contents=True)

    from matplotlib.axes import Axes

    assert isinstance(ax_out, Axes), "Expected Ax"
    assert isinstance(contents, dict)
    assert set(contents.keys()) == {"GroupA", "GroupB"}

    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_invalid_color_length():
    """Ensure plot_venn raises error for mismatched color count."""
    fig, ax = plt.subplots()
    pdata = object()
    with pytest.raises(ValueError):
        scplt.plot_venn(ax, pdata, classes=["GroupA", "GroupB"], set_colors=["#1f77b4"])
    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_invalid_label_order():
    """Ensure label_order mismatch raises ValueError."""
    fig, ax = plt.subplots()
    pdata = object()
    with pytest.raises(ValueError):
        scplt.plot_venn(ax, pdata, classes=["GroupA", "GroupB"], label_order=["Wrong", "Labels"])
    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_invalid_number_of_sets(monkeypatch):
    """Ensure >3 sets raises a ValueError."""
    def mock_get_upset_contents(pdata, classes, upsetForm=False):
        return {"A": {1}, "B": {2}, "C": {3}, "D": {4}}
    monkeypatch.setattr(scplt.utils, "get_upset_contents", mock_get_upset_contents)
    fig, ax = plt.subplots()
    pdata = object()
    with pytest.raises(ValueError):
        scplt.plot_venn(ax, pdata, classes=["A", "B", "C", "D"])
    plt.close(fig)

# test plot_upset
# mock utilities
class DummyUtils:
    """Mock subset of scpviz.utils used by these plotting functions."""
    @staticmethod
    def get_upset_contents(pdata, classes, upsetForm=True):
        # Return simple dummy content mimicking protein sets
        return {
            "GroupA": {"P1", "P2", "P3"},
            "GroupB": {"P2", "P3", "P4"}
        }

class DummyUpSet:
    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs
    def plot(self):
        return {"intersections": "mock_axes", "totals": "mock_axes"}

@pytest.fixture
def mock_upset_utils(monkeypatch):
    """Temporarily replace scplt.utils with DummyUtils for UpSet tests."""
    monkeypatch.setattr(scplt, "utils", DummyUtils())
    yield
    # pytest will restore scplt.utils afterward

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_upset_runs(monkeypatch):
    """Ensure plot_upset runs and returns an UpSet mock."""
    monkeypatch.setattr(scplt, "upsetplot", type("m", (), {"UpSet": DummyUpSet}))
    fig, ax = plt.subplots()
    pdata = object()

    upset_obj = scplt.plot_upset(pdata, classes=["GroupA", "GroupB"])
    assert isinstance(upset_obj, DummyUpSet)
    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_upset_return_contents(monkeypatch):
    """Ensure return_contents=True returns both UpSet and contents."""
    monkeypatch.setattr(scplt, "upsetplot", type("m", (), {"UpSet": DummyUpSet}))
    pdata = object()
    upset_obj, contents = scplt.plot_upset(pdata, classes=["GroupA", "GroupB"], return_contents=True)
    assert isinstance(upset_obj, DummyUpSet)
    assert isinstance(contents, dict)
    assert set(contents.keys()) == {"GroupA", "GroupB"}

# test plot_abundance_2d()
def test_plot_abundance_2D_runs_with_highlight():
    """Ensure plot_abundance_2D runs with gene highlighting."""
    fig, ax = plt.subplots()
    # Build mock abundance DataFrame
    df = pd.DataFrame({
        "Gene Symbol": ["A", "B", "C"],
        "Abundance: cond1": [10, 20, 30],
        "Abundance: cond2": [15, 25, 35],
        "Abundance: cond3": [5, 12, 18],
    })
    cases = [["cond1"], ["cond2"]]
    genes = ["A", "B"]
    ax = scplt.plot_abundance_2D(ax, df.copy(), cases=cases, genes=genes)
    assert hasattr(ax, "scatter"), "plot_abundance_2D should return Matplotlib Axes"
    plt.close(fig)

def test_plot_abundance_2D_runs_all_genes():
    """Ensure plot_abundance_2D runs when genes='all'."""
    fig, ax = plt.subplots()
    df = pd.DataFrame({
        "Gene Symbol": ["X", "Y"],
        "Abundance: case1": [10, 20],
        "Abundance: case2": [15, 30],
    })
    cases = [["case1"], ["case2"]]
    ax = scplt.plot_abundance_2D(ax, df.copy(), cases=cases, genes="all")
    assert hasattr(ax, "scatter")
    plt.close(fig)

# start raincloud tests

class DummyMatrix(np.ndarray):
    def toarray(self):
        return self

class DummyUtilsRain:
    """Mock subset of utils for raincloud tests."""
    @staticmethod
    def get_adata(pdata, on):
        # If pdata already has prot.var (mark_raincloud case)
        if hasattr(pdata, "prot"):
            return pdata.prot
        # Otherwise build dummy AnnData-like object
        class DummyAdata:
            def __init__(self):
                self.var = pd.DataFrame(index=["P1", "P2", "P3"])
                self.obs = pd.DataFrame({"class": ["A", "B", "A"]})
                self.X = np.abs(np.random.randn(3, 3)).view(DummyMatrix)
            def to_df(self):
                return pd.DataFrame(self.X, columns=self.var.index)
        return DummyAdata()

    @staticmethod
    def get_classlist(adata, classes=None, order=None):
        return ["A", "B"]

    @staticmethod
    def resolve_class_filter(adata, classes, class_value, debug=False):
        class DummySubset:
            def __init__(self, X):
                self.X = X
                self.var = pd.DataFrame(index=["P1", "P2", "P3"])
            def to_df(self):
                return pd.DataFrame(self.X, columns=self.var.index)
        X = np.abs(np.random.randn(3, 3)).view(DummyMatrix)
        return DummySubset(X)

@pytest.fixture
def mock_raincloud_utils(monkeypatch):
    """Temporarily replace scplt.utils with DummyUtilsRain inside a test."""
    monkeypatch.setattr(scplt, "utils", DummyUtilsRain())
    yield
    # pytest automatically restores the original after the test exits

# test plot_raincloud
@pytest.mark.usefixtures("mock_raincloud_utils")
def test_plot_raincloud_runs_without_error():
    """Ensure plot_raincloud runs normally and returns Axes."""
    fig, ax = plt.subplots()
    pdata = object()

    ax_out = scplt.plot_raincloud(
        ax, pdata, classes="class", on="protein", color=["blue", "orange"]
    )
    assert hasattr(ax_out, "violinplot"), "❌ Expected Matplotlib Axes returned."

    plt.close(fig)

@pytest.mark.usefixtures("mock_raincloud_utils")
def test_plot_raincloud_debug_mode_returns_data():
    """Ensure debug=True returns both axis and data_X arrays."""
    fig, ax = plt.subplots()
    pdata = object()

    ax_out, data_X = scplt.plot_raincloud(
        ax, pdata, classes="class", debug=True, color=["blue", "orange"]
    )
    assert isinstance(ax_out, plt.Axes)
    assert isinstance(data_X, list)
    assert len(data_X) > 0 and all(isinstance(arr, np.ndarray) for arr in data_X)

    plt.close(fig)

# test raincloud overlay
class DummyPdata:
    """Simple pdata mock with required attributes."""
    def __init__(self):
        self.prot = type("obj", (), {})()
        self.prot.var = pd.DataFrame(
            {"Average: A": [1.2, 2.5, 3.0]}, index=["P1", "P2", "P3"]
        )
    def _check_rankcol(self, on, class_values):
        return True  # No-op check

def test_mark_raincloud_adds_points():
    """Ensure mark_raincloud overlays points successfully."""
    fig, ax = plt.subplots()
    pdata = DummyPdata()

    mark_df = pd.DataFrame({"Entry": ["P1", "P2"], "Gene Names": ["G1", "G2"]})

    scplt.mark_raincloud(
        ax,
        pdata,
        mark_df=mark_df,
        class_values=["A"],
        color="red",
        s=5,
        alpha=0.8,
    )

    # The plot should now have at least one scatter collection
    assert len(ax.collections) > 0, "❌ mark_raincloud should add scatter points."
    plt.close(fig)

# tests for shift_legend()
def test_shift_legend_moves_existing_legend():
    fig, ax = plt.subplots()

    # Create a dummy plot with legend
    ax.plot([0, 1], [0, 1], label="dummy")
    ax.legend(loc='upper right')

    # Capture the legend before shifting
    leg_before = ax.get_legend()
    orig_bbox = leg_before.get_bbox_to_anchor()

    # Apply shift
    scplt.shift_legend(ax, anchor_pos=(0.8, 0.2), loc="lower left")

    leg_after = ax.get_legend()
    new_bbox = leg_after.get_bbox_to_anchor()

    # Check that the legend exists and was moved
    assert leg_after is not None
    assert orig_bbox != new_bbox      # anchor changed

    plt.close(fig)

def test_shift_legend_no_legend_does_nothing():
    fig, ax = plt.subplots()

    # Confirm no legend present
    assert ax.get_legend() is None

    # Should run silently without errors
    scplt.shift_legend(ax)

    # Still no legend
    assert ax.get_legend() is None

    plt.close(fig)
# ---------------------------------------------------------------------------
# plot_grouped_heatmap / plot_clustered_heatmap
# ---------------------------------------------------------------------------

def test_plot_grouped_heatmap_smoke(pdata, capsys):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:4].tolist()
    groups = {
        "A": genes[:2] + ["NOT_A_REAL_GENE_XYZ"],
        "B": genes[2:4],
    }
    fig = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment", "cellline"],
        sort_by={"treatment": ["sc", "kd"]},
        sample_label_col="treatment",
        row_spacing=True,
        group_bar_pad=0.15,
        figsize=(7, 5),
    )
    assert isinstance(fig, plt.Figure)
    captured = capsys.readouterr().out
    assert "NOT_A_REAL_GENE_XYZ" in captured
    assert "WARN" in captured
    plt.close(fig)

    with pytest.raises(KeyError, match="sample_label_col"):
        scplt.plot_grouped_heatmap(
            pdata,
            protein_groups=groups,
            classes=["treatment"],
            sample_label_col="not_a_real_obs_col_xyz",
        )


def test_plot_grouped_heatmap_header_count(pdata):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:3].tolist()
    groups = {"G": genes}
    for n in (1, 2):
        classes = ["treatment", "cellline"][:n]
        fig = scplt.plot_grouped_heatmap(
            pdata, protein_groups=groups, classes=classes, figsize=(6, 4)
        )
        # n header axes + 1 main = n+1 axes with images; blanks not used in grouped
        assert len(fig.axes) >= n + 1
        plt.close(fig)


def test_plot_grouped_heatmap_sort_by_order(pdata):
    from scpviz.plotting.clustering import _compute_sample_order

    classes = ["treatment", "cellline"]
    sort_by = {"treatment": ["sc", "kd"]}
    order = _compute_sample_order(
        pdata.prot.obs, list(pdata.prot.obs_names.astype(str)), classes, sort_by
    )
    treatments = [str(pdata.prot.obs.loc[s, "treatment"]) for s in order]
    if "sc" in treatments and "kd" in treatments:
        last_sc = max(i for i, t in enumerate(treatments) if t == "sc")
        first_kd = min(i for i, t in enumerate(treatments) if t == "kd")
        assert last_sc < first_kd


def test_plot_grouped_heatmap_mixin(pdata):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:2].tolist()
    fig = pdata.plot_grouped_heatmap(
        {"G": genes}, classes=["treatment"], figsize=(6, 4)
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_clustered_heatmap_smoke(pdata, capsys):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:5].tolist()
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        proteins=genes + ["MISSING_CLUSTER_GENE"],
        protein_groups={"SetA": genes[:2]},
        figsize=(7, 5),
    )
    assert isinstance(fig, plt.Figure)
    captured = capsys.readouterr().out
    assert "MISSING_CLUSTER_GENE" in captured
    plt.close(fig)


def test_plot_clustered_heatmap_xor_proteins_stats_key(pdata):
    with pytest.raises(ValueError, match="exactly one"):
        scplt.plot_clustered_heatmap(pdata, classes=["treatment"])
    with pytest.raises(ValueError, match="exactly one"):
        scplt.plot_clustered_heatmap(
            pdata,
            classes=["treatment"],
            proteins=["TUBB"],
            stats_key="nope",
        )


def test_plot_clustered_heatmap_stats_key(pdata):
    # Inject a synthetic DE table shaped like de()/mixed_de output
    accs = list(pdata.prot.var_names[:6].astype(str))
    de = pd.DataFrame(
        {
            "log2fc": [1.5, -1.2, 0.1, 2.0, -0.5, 0.0],
            "p_value": [1e-4, 1e-3, 0.5, 1e-5, 0.2, 0.9],
            "significance": [
                "upregulated",
                "downregulated",
                "not significant",
                "upregulated",
                "not comparable",
                "not significant",
            ],
            "Genes": pdata.prot.var.loc[accs, "Genes"].astype(str).tolist(),
        },
        index=accs,
    )
    key = "test_heatmap_de"
    pdata.stats[key] = de
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment", "cellline"],
        stats_key=key,
        significance_categories=["upregulated", "downregulated"],
        figsize=(7, 5),
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    with pytest.raises(KeyError, match="not found"):
        scplt.plot_clustered_heatmap(
            pdata, classes=["treatment"], stats_key="does_not_exist_xyz"
        )


def test_plot_clustered_heatmap_stats_key_single_category_str(pdata, capsys):
    """Bare string significance_categories must not be iterated as characters."""
    accs = list(pdata.prot.var_names[:6].astype(str))
    de = pd.DataFrame(
        {
            "log2fc": [np.nan, np.nan, 0.1, 2.0, np.nan, 0.0],
            "p_value": [np.nan, np.nan, 0.5, 1e-5, np.nan, 0.9],
            "significance": [
                "not comparable",
                "not comparable",
                "not significant",
                "upregulated",
                "not comparable",
                "not significant",
            ],
        },
        index=accs,
    )
    key = "test_heatmap_de_nc"
    pdata.stats[key] = de
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        stats_key=key,
        significance_categories="not comparable",
        figsize=(6, 4),
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    captured = capsys.readouterr().out
    assert "not comparable" in captured.lower()
    assert "WARN" in captured or "warn" in captured.lower()

    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        stats_key=key,
        significance_categories=["not comparable"],
        label_color="#ff00aa",
        figsize=(6, 4),
    )
    from matplotlib.colors import to_rgba

    target = to_rgba("#ff00aa")
    assert any(
        to_rgba(t.get_color()) == target
        for ax in fig.axes
        for t in ax.get_yticklabels()
        if t.get_text()
    )
    plt.close(fig)


def test_plot_clustered_heatmap_mixed_de_collection_raises(pdata):
    pdata.stats["mixed_collection_fake"] = {
        "contrasts": {"A vs B": pd.DataFrame({"significance": ["upregulated"]})},
        "meta": {},
    }
    with pytest.raises(ValueError, match="mixed_de collection"):
        scplt.plot_clustered_heatmap(
            pdata, classes=["treatment"], stats_key="mixed_collection_fake"
        )


def test_plot_clustered_heatmap_same_sample_order_as_grouped(pdata):
    from scpviz.plotting.clustering import _compute_sample_order

    classes = ["treatment", "cellline"]
    sort_by = {"treatment": ["kd", "sc"]}
    order = _compute_sample_order(
        pdata.prot.obs, list(pdata.prot.obs_names.astype(str)), classes, sort_by
    )
    # Both functions call the same helper — assert helper output is stable/deterministic
    order2 = _compute_sample_order(
        pdata.prot.obs, list(pdata.prot.obs_names.astype(str)), classes, sort_by
    )
    assert order == order2
    assert len(order) == pdata.prot.n_obs


def test_plot_clustered_heatmap_metric_branches(pdata, monkeypatch):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:4].tolist()
    called = {"corr": 0}

    import scpviz.plotting.clustering as hm

    real = hm.correlation_linkage

    def spy(*args, **kwargs):
        called["corr"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(hm, "correlation_linkage", spy)

    fig = scplt.plot_clustered_heatmap(
        pdata, classes=["treatment"], proteins=genes, metric="correlation", figsize=(6, 4)
    )
    assert called["corr"] == 1
    plt.close(fig)

    fig = scplt.plot_clustered_heatmap(
        pdata, classes=["treatment"], proteins=genes, metric="euclidean", figsize=(6, 4)
    )
    assert called["corr"] == 1  # unchanged
    plt.close(fig)

    with pytest.raises(ValueError, match="metric must be"):
        scplt.plot_clustered_heatmap(
            pdata, classes=["treatment"], proteins=genes, metric="cosine"
        )


def test_plot_clustered_heatmap_euclidean_ignores_cor_method(pdata, capsys):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:4].tolist()
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        proteins=genes,
        metric="euclidean",
        cor_method="spearman",
        figsize=(6, 4),
    )
    captured = capsys.readouterr().out
    assert "ignored" in captured.lower() or "cor_method" in captured
    plt.close(fig)


def test_correlation_linkage_zero_variance_raises():
    from scpviz.utils import correlation_linkage

    X = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    with pytest.raises(ValueError, match="zero variance"):
        correlation_linkage(X)


def test_correlation_linkage_adjacent_correlated_rows():
    from scpviz.utils import correlation_linkage
    from scipy.cluster.hierarchy import leaves_list

    rng = np.random.default_rng(0)
    base = rng.normal(size=20)
    X = np.vstack([base, base + 0.01 * rng.normal(size=20), rng.normal(size=20)])
    Z, _ = correlation_linkage(X, optimal_ordering=True)
    leaves = list(leaves_list(Z))
    # first two rows should be adjacent in leaf order
    i0, i1 = leaves.index(0), leaves.index(1)
    assert abs(i0 - i1) == 1


def test_plot_clustered_heatmap_dendrogram_adjacent(pdata):
    # Hand-build three synthetic proteins with clear correlation structure
    from scpviz.utils import correlation_linkage, get_adata_layer
    from scipy.cluster.hierarchy import leaves_list

    adata = pdata.prot
    names = list(adata.var_names[:3].astype(str))
    X = np.asarray(get_adata_layer(adata, "X"), dtype=float).copy()
    # Make row0 and row1 nearly identical patterns; row2 unrelated
    rng = np.random.default_rng(1)
    pattern = rng.normal(size=adata.n_obs)
    X[:, 0] = pattern
    X[:, 1] = pattern + 0.01 * rng.normal(size=adata.n_obs)
    X[:, 2] = rng.normal(size=adata.n_obs)
    adata.layers["X_heat_test"] = X
    # Mark as log-like so auto_log2 does not transform
    adata.uns.setdefault("layer_provenance", {})
    adata.uns["layer_provenance"]["X_heat_test"] = {
        "op": "log_transform",
        "input_layer": "X_raw",
        "base": "2",
    }

    # z-score rows
    mat = X[:, :3].T  # 3 proteins x samples
    mat = (mat - mat.mean(1, keepdims=True)) / mat.std(1, keepdims=True)
    Z, _ = correlation_linkage(mat)
    leaves = list(leaves_list(Z))
    assert abs(leaves.index(0) - leaves.index(1)) == 1

    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        proteins=names,
        layer="X_heat_test",
        metric="correlation",
        figsize=(6, 4),
        auto_log2=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        proteins=names,
        layer="X_heat_test",
        metric="euclidean",
        figsize=(6, 4),
        auto_log2=False,
    )
    plt.close(fig)


def test_heatmap_display_scale_branches(pdata, capsys):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:4].tolist()
    groups = {"A": genes[:2], "B": genes[2:4]}

    with pytest.raises(ValueError, match="display_scale"):
        scplt.plot_grouped_heatmap(
            pdata, protein_groups=groups, classes=["treatment"], display_scale="nope"
        )

    fig = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment"],
        display_scale="log",
        figsize=(6, 4),
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    out = capsys.readouterr().out
    assert "display_scale" in out

    # Explicit RdBu_r with log should warn
    fig = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment"],
        display_scale="log",
        cmap="RdBu_r",
        figsize=(5, 3),
    )
    plt.close(fig)
    out = capsys.readouterr().out
    assert "RdBu_r" in out and "WARN" in out

    accs = list(pdata.prot.var_names[:4].astype(str))
    de = pd.DataFrame(
        {
            "log2fc": [np.nan] * 4,
            "p_value": [np.nan] * 4,
            "significance": ["not comparable"] * 4,
        },
        index=accs,
    )
    key = "test_heatmap_display_scale_nc"
    pdata.stats[key] = de

    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        stats_key=key,
        significance_categories="not comparable",
        figsize=(6, 4),
    )
    plt.close(fig)
    out = capsys.readouterr().out
    assert "display_scale='log'" in out

    # Leaf order stable across display_scale (clustering always on z-scores)
    from scipy.cluster.hierarchy import dendrogram

    captured = {}

    def _capture_dendrogram(*args, **kwargs):
        dn = dendrogram(*args, **kwargs)
        captured.setdefault("leaves", []).append(list(dn["leaves"]))
        return dn

    import scpviz.plotting.clustering as hm

    original = hm.dendrogram if hasattr(hm, "dendrogram") else None
    # dendrogram is imported inside the function; patch scipy symbol used there
    import scipy.cluster.hierarchy as sch

    real_dendrogram = sch.dendrogram

    def wrapped(*a, **k):
        dn = real_dendrogram(*a, **k)
        captured.setdefault("leaves", []).append(list(dn["leaves"]))
        return dn

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sch, "dendrogram", wrapped)
    try:
        for scale in ("zscore", "log", "raw"):
            fig = scplt.plot_clustered_heatmap(
                pdata,
                classes=["treatment"],
                proteins=accs,
                display_scale=scale,
                figsize=(5, 3),
            )
            plt.close(fig)
    finally:
        monkeypatch.undo()

    assert len(captured["leaves"]) == 3
    assert captured["leaves"][0] == captured["leaves"][1] == captured["leaves"][2]


def test_plot_grouped_heatmap_fractional_row_spacing():
    rgba = np.zeros((4, 3, 4))
    rgba[:, :, 3] = 1.0
    row_groups = ["A", "A", "B", "B"]
    labels = ["a1", "a2", "b1", "b2"]
    from scpviz.plotting.clustering import _GROUP_GAP_PX, _ROW_PX, _rasterize_with_gaps

    disp0, *_ = _rasterize_with_gaps(rgba, row_groups, labels, row_spacing=False)
    disp_true, ticks, tick_labs, yr = _rasterize_with_gaps(
        rgba, row_groups, labels, row_spacing=True
    )
    disp_half, *_ = _rasterize_with_gaps(rgba, row_groups, labels, row_spacing=0.5)
    assert disp0.shape[0] == 4 * _ROW_PX
    assert disp_true.shape[0] == 4 * _ROW_PX + _GROUP_GAP_PX
    assert disp_half.shape[0] == 4 * _ROW_PX + int(round(0.5 * _GROUP_GAP_PX))
    assert tick_labs == labels
    assert set(yr) == {"A", "B"}
    assert len(ticks) == 4

    with pytest.raises(ValueError, match="row_spacing"):
        _rasterize_with_gaps(rgba, row_groups, labels, row_spacing=-1)


def test_heatmap_separate_legend(pdata):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:4].tolist()
    groups = {"A": genes[:2], "B": genes[2:4]}
    out = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment"],
        separate_legend=True,
        figsize=(6, 4),
    )
    assert isinstance(out, tuple) and len(out) == 2
    fig, legend_fig = out
    assert isinstance(fig, plt.Figure)
    assert isinstance(legend_fig, plt.Figure)
    assert legend_fig is not fig
    # Legend figure must have drawable axes (colorbar + legend host)
    assert len(legend_fig.axes) >= 2
    # Main heatmap should not keep the left-side colorbar axes
    # (only header + main heatmap axes from gridspec)
    assert len(fig.legends) == 0
    plt.close(fig)
    plt.close(legend_fig)

    genes2 = list(pdata.prot.var_names[:4].astype(str))
    out2 = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        proteins=genes2,
        separate_legend=True,
        figsize=(6, 4),
    )
    fig2, legend_fig2 = out2
    assert isinstance(legend_fig2, plt.Figure)
    assert len(legend_fig2.axes) >= 2
    plt.close(fig2)
    plt.close(legend_fig2)


def test_heatmap_cbar_scale(pdata):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:4].tolist()
    groups = {"A": genes[:2], "B": genes[2:4]}

    fig1 = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment"],
        figsize=(6, 10),
        cbar_scale=1.0,
    )
    fig2 = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment"],
        figsize=(6, 10),
        cbar_scale=1.5,
    )

    def _cbar_height_in(fig):
        for ax in fig.axes:
            if ax.yaxis.label.get_text():
                return ax.get_position().height * fig.get_size_inches()[1]
        raise AssertionError("colorbar axes not found")

    assert _cbar_height_in(fig1) == pytest.approx(1.35, rel=0.05)
    assert _cbar_height_in(fig2) == pytest.approx(1.35 * 1.5, rel=0.05)

    # Tall figsize: consecutive legends use figure-fraction anchors ~0.05 apart
    # (≈0.5 in on a 10 in figure), not stretched by figure height.
    assert len(fig1.legends) >= 2
    y_fracs = [leg.get_bbox_to_anchor()._bbox.y0 for leg in fig1.legends]
    gaps_in = [(y_fracs[i] - y_fracs[i + 1]) * 10.0 for i in range(len(y_fracs) - 1)]
    assert all(0.2 < g < 1.2 for g in gaps_in)

    out = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment"],
        separate_legend=True,
        cbar_scale=0.75,
        figsize=(6, 4),
    )
    fig_s, legend_fig = out
    assert _cbar_height_in(legend_fig) == pytest.approx(1.35 * 0.75, rel=0.08)
    assert len(legend_fig.legends) >= 1
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig_s)
    plt.close(legend_fig)


def test_heatmap_legend_width(pdata):
    from scpviz.plotting.clustering import (
        _LEGEND_WIDTH_MAX,
        _LEGEND_WIDTH_MIN,
        _estimate_legend_width_frac,
    )

    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:4].tolist()
    groups = {"A": genes[:2], "B": genes[2:4]}
    fig_auto = scplt.plot_grouped_heatmap(
        pdata, protein_groups=groups, classes=["treatment"], figsize=(8, 5)
    )
    fig_wide = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=["treatment"],
        figsize=(8, 5),
        legend_width=0.40,
    )
    left_auto = fig_auto.subplotpars.left
    left_wide = fig_wide.subplotpars.left
    # Auto should land in a sensible band and respond to longer labels
    assert _LEGEND_WIDTH_MIN <= left_auto <= _LEGEND_WIDTH_MAX
    assert left_wide == pytest.approx(0.40, abs=0.02)

    long_specs = [
        ("very_long_condition_name", [], ["supercalifragilisticexpialidocious"])
    ]
    short_specs = [("x", [], ["a"])]
    long_w = _estimate_legend_width_frac(8.0, long_specs, text_size=8, cbar_label="z-score")
    short_w = _estimate_legend_width_frac(8.0, short_specs, text_size=8, cbar_label="z-score")
    assert long_w > short_w

    # Grouped left y-tick labels require extra margin vs legends-only
    with_ticks = _estimate_legend_width_frac(
        8.0,
        short_specs,
        text_size=8,
        cbar_label="z-score",
        left_tick_labels=["VERYLONGGENENAME1", "VERYLONGGENENAME2"],
    )
    assert with_ticks > short_w

    with pytest.raises(ValueError, match="legend_width"):
        scplt.plot_grouped_heatmap(
            pdata,
            protein_groups=groups,
            classes=["treatment"],
            legend_width=0,
        )
    plt.close(fig_auto)
    plt.close(fig_wide)


def test_heatmap_header_colors_flat_and_nested(pdata):
    from scpviz.plotting.clustering import _normalize_header_colors

    flat = {"Agg+": "#C64D4A", "Agg-": "#BFBFBF"}
    assert _normalize_header_colors(flat, ["sample"]) == {"sample": flat}
    nested = {"sample": flat, "cellline": {"AS": "#4C72B0"}}
    assert _normalize_header_colors(nested, ["sample", "cellline"]) == nested
    assert _normalize_header_colors(None, ["sample"]) == {}

    with pytest.raises(ValueError, match="Flat header_colors"):
        _normalize_header_colors(flat, ["sample", "cellline"])
    with pytest.raises(ValueError, match="Mixed forms"):
        _normalize_header_colors({"sample": flat, "Agg+": "#C64D4A"}, ["sample"])

    genes = list(pdata.prot.var["Genes"].astype(str).unique()[:4])
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        proteins=genes,
        header_colors={"kd": "#C64D4A", "sc": "#BFBFBF"},
    )
    assert fig is not None
    plt.close(fig)


def test_plot_clustered_heatmap_dendrogram_linewidth(pdata):
    from matplotlib.collections import LineCollection

    genes = list(pdata.prot.var["Genes"].astype(str).unique()[:5])
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=["treatment"],
        proteins=genes,
        dendrogram_linewidth=0.8,
        column_spacing=False,
    )
    # Dendrogram axes: LineCollection only (no images / QuadMesh colorbar)
    dendro_axes = [
        ax
        for ax in fig.axes
        if ax.collections
        and all(isinstance(c, LineCollection) for c in ax.collections)
        and not ax.images
    ]
    assert dendro_axes
    for coll in dendro_axes[0].collections:
        assert np.allclose(np.atleast_1d(coll.get_linewidth()), 0.8)
    plt.close(fig)


def test_heatmap_column_spacing(pdata):
    from scpviz.plotting.clustering import _GROUP_GAP_PX, _rasterize_with_column_gaps

    genes = list(pdata.prot.var["Genes"].astype(str).unique()[:4])
    groups = {"G1": genes[:2], "G2": genes[2:]}
    classes = ["treatment"]
    samples = list(pdata.prot.obs_names.astype(str))
    n = len(samples)
    rgba = np.zeros((2, n, 4), dtype=float)
    labels = [str(s) for s in samples]

    _, _, _, col_off = _rasterize_with_column_gaps(
        rgba, samples, pdata.prot.obs, classes, False, labels
    )
    assert col_off == list(range(n))

    _, _, _, col_on = _rasterize_with_column_gaps(
        rgba, samples, pdata.prot.obs, classes, True, labels
    )
    n_gap_true = sum(1 for x in col_on if x is None)
    assert n_gap_true > 0
    assert len(col_on) > n

    _, _, _, col_half = _rasterize_with_column_gaps(
        rgba, samples, pdata.prot.obs, classes, 0.5, labels
    )
    n_gap_half = sum(1 for x in col_half if x is None)
    n_blocks_gaps = n_gap_true // _GROUP_GAP_PX
    assert n_gap_half == int(round(0.5 * _GROUP_GAP_PX)) * n_blocks_gaps

    with pytest.raises(ValueError, match="column_spacing"):
        _rasterize_with_column_gaps(
            rgba, samples, pdata.prot.obs, classes, -1, labels
        )

    with pytest.raises(ValueError, match="header_spacing"):
        scplt.plot_grouped_heatmap(
            pdata, protein_groups=groups, classes=classes, header_spacing=-0.1
        )

    fig_off = scplt.plot_grouped_heatmap(
        pdata, protein_groups=groups, classes=classes, column_spacing=False
    )
    fig_on = scplt.plot_grouped_heatmap(
        pdata, protein_groups=groups, classes=classes, column_spacing=True
    )
    # Main heatmap is last non-colorbar axes with an image in the gridspec stack
    def _main_ncols(fig):
        imgs = [ax.images[0] for ax in fig.axes if ax.images]
        return imgs[-1].get_array().shape[1]

    assert _main_ncols(fig_on) > _main_ncols(fig_off)
    plt.close(fig_off)
    plt.close(fig_on)

    fig_c = scplt.plot_clustered_heatmap(
        pdata,
        classes=classes,
        proteins=genes,
        column_spacing=0.5,
    )
    assert fig_c is not None
    plt.close(fig_c)


def test_heatmap_header_height_and_group_bar_width(pdata):
    genes = list(pdata.prot.var["Genes"].astype(str).unique()[:4])
    groups = {"G1": genes[:2], "G2": genes[2:]}
    classes = ["treatment"]

    with pytest.raises(ValueError, match="header_height"):
        scplt.plot_grouped_heatmap(
            pdata, protein_groups=groups, classes=classes, header_height=0
        )
    with pytest.raises(ValueError, match="group_bar_width"):
        scplt.plot_grouped_heatmap(
            pdata, protein_groups=groups, classes=classes, group_bar_width=-0.1
        )

    fig = scplt.plot_grouped_heatmap(
        pdata,
        protein_groups=groups,
        classes=classes,
        header_height=0.6,
        group_bar_width=0.8,
        column_spacing=False,
    )
    # Main axes is the last axes that has an image and Rectangle patches for groups
    main = [ax for ax in fig.axes if ax.images and ax.patches][-1]
    bar_widths = [p.get_width() for p in main.patches]
    assert bar_widths and all(np.isclose(w, 0.8) for w in bar_widths)
    plt.close(fig)

    fig_c = scplt.plot_clustered_heatmap(
        pdata, classes=classes, proteins=genes, header_height=0.55
    )
    assert fig_c is not None
    plt.close(fig_c)

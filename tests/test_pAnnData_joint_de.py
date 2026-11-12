"""Joint group-means linear model DE: ``pdata.joint_de`` and ``scpviz.utils.joint_de``."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from anndata import AnnData
from scipy.stats import t as t_dist, ttest_ind

from scpviz import pAnnData
from scpviz.utils import joint_de as jd
from scpviz.utils.de_reporting import format_de_comparison_label
from scpviz.utils.limma import moderated_two_group_test


def _vs(col, test, ref):
    return format_de_comparison_label({col: test}, {col: ref})


# --- synthetic fixtures -----------------------------------------------------------------------

def _groups(sizes):
    labels = [g for g, n in sizes.items() for _ in range(n)]
    return np.array(labels), pd.Index(sorted(sizes))


def _data(sizes, n_prot=40, seed=0, shifts=None, nan_frac=0.0):
    """Log-scale data (samples x proteins) with optional per-group shifts on the first proteins."""
    rng = np.random.default_rng(seed)
    labels, _ = _groups(sizes)
    X = rng.normal(15, 1, (len(labels), n_prot))
    for g, s in (shifts or {}).items():
        X[labels == g, :5] += s
    if nan_frac:
        X[rng.random(X.shape) < nan_frac] = np.nan
    return labels, X


def _codes(labels):
    levels = sorted(set(labels))
    return np.array([levels.index(g) for g in labels]), levels


def _pdata(sizes, n_prot=40, seed=0, shifts=None, nan_frac=0.0, extra_obs=None, log=True):
    """pAnnData whose layer 'X_log2' is registered as log2 (built from a linear layer)."""
    labels, logX = _data(sizes, n_prot, seed, shifts, nan_frac)
    obs = pd.DataFrame({"condition": labels}, index=[f"s{i}" for i in range(len(labels))])
    for k, v in (extra_obs or {}).items():
        obs[k] = v
    var = pd.DataFrame({"Genes": [f"G{i}" for i in range(n_prot)]}, index=[f"P{i}" for i in range(n_prot)])
    p = pAnnData(prot=AnnData(X=2.0 ** logX, obs=obs, var=var))
    if log:
        p.log_transform(base=2, pseudocount=1.0, layer="X")
    return p, labels, logX


def _log_layer(p):
    return p.prot.uns["current_X_layer"] if "current_X_layer" in p.prot.uns else "X_log2"


# --- core maths: closed form == statsmodels OLS --------------------------------------------------

SIZES6 = {"a": 3, "b": 2, "c": 3, "d": 3, "e": 3, "f": 2}  # paper's PISA benchmark: n=16, df=10


def test_fit_group_means_unbalanced_matches_manual():
    labels, X = _data(SIZES6)
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, len(levels))
    assert (fit["df"] == 10).all()
    for k, g in enumerate(levels):
        np.testing.assert_allclose(fit["mean"][:, k], X[labels == g].mean(axis=0))
        assert (fit["n"][:, k] == SIZES6[g]).all()
    sse = sum(((X[labels == g] - X[labels == g].mean(axis=0)) ** 2).sum(axis=0) for g in levels)
    np.testing.assert_allclose(fit["s2"], sse / 10)


def test_contrast_variance_is_inverse_n_sum():
    labels, X = _data(SIZES6)
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, len(levels))
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"x": ("a", "b")})
    res = jd.fit_contrasts(fit, W, "lm")
    v = (res["se"][:, 0] ** 2) / res["s2"]
    np.testing.assert_allclose(v, 1 / 3 + 1 / 2)


@pytest.mark.parametrize("sizes", [{"a": 4, "b": 4, "c": 4}, SIZES6])
def test_lm_matches_statsmodels_ols(sizes):
    labels, X = _data(sizes, n_prot=6, nan_frac=0.1, seed=3)
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, len(levels))
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"x": (levels[1], levels[0])})
    res = jd.fit_contrasts(fit, W, "lm")
    for i in range(X.shape[1]):
        ok = np.isfinite(X[:, i])
        D = (codes[ok][:, None] == np.arange(len(levels))).astype(float)
        D = D[:, D.sum(0) > 0]
        m = sm.OLS(X[ok, i], D).fit()
        assert fit["df"][i] == pytest.approx(m.df_resid)
        assert fit["s2"][i] == pytest.approx(m.scale)
        occupied = [k for k in range(len(levels)) if (codes[ok] == k).any()]
        if levels[0] in [levels[k] for k in occupied] and levels[1] in [levels[k] for k in occupied]:
            c = np.zeros(len(occupied))
            c[occupied.index(1)], c[occupied.index(0)] = 1, -1
            tt = m.t_test(c)
            assert res["effect"][i, 0] == pytest.approx(float(tt.effect[0]))
            assert res["t"][i, 0] == pytest.approx(float(tt.tvalue[0, 0]))
            assert res["p_value"][i, 0] == pytest.approx(float(tt.pvalue))
        else:
            assert np.isnan(res["effect"][i, 0])


def test_two_group_lm_equals_pooled_ttest():
    labels, X = _data({"a": 4, "b": 5}, n_prot=30, nan_frac=0.1, seed=4)
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, 2)
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"x": ("a", "b")})
    res = jd.fit_contrasts(fit, W, "lm")
    ref = ttest_ind(X[labels == "a"], X[labels == "b"], equal_var=True, nan_policy="omit")
    np.testing.assert_allclose(res["t"][:, 0], ref.statistic, equal_nan=True)
    np.testing.assert_allclose(res["p_value"][:, 0], ref.pvalue, equal_nan=True)


def test_two_group_limma_equals_moderated_two_group_test():
    labels, X = _data({"a": 4, "b": 5}, n_prot=60, nan_frac=0.1, seed=5, shifts={"a": 1.0})
    X[labels == "a", 0] = np.nan
    X[labels == "a", 1] = np.nan  # proteins with one group entirely missing
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, 2)
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"x": ("a", "b")})
    res = jd.fit_contrasts(fit, W, "limma")
    ref = moderated_two_group_test(X[labels == "a"], X[labels == "b"])
    for mine, theirs in (("effect", "beta"), ("t", "t"), ("p_value", "p_value")):
        np.testing.assert_allclose(res[mine][:, 0], ref[theirs], equal_nan=True)
    assert res["df_prior"] == pytest.approx(ref["df_prior"])
    assert res["s2_prior"] == pytest.approx(ref["s2_prior"])


# --- joint vs separate: extra groups add residual df ---------------------------------------------

def test_joint_uses_all_groups_for_variance():
    labels, X = _data({"a": 3, "b": 3, "c": 3, "d": 3}, n_prot=10, seed=6)
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, 4)
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"x": ("a", "b")})
    res = jd.fit_contrasts(fit, W, "lm")
    assert (res["df_residual"] == 8).all()
    sep = ttest_ind(X[labels == "a"], X[labels == "b"], equal_var=True)
    np.testing.assert_allclose(res["effect"][:, 0], X[labels == "a"].mean(0) - X[labels == "b"].mean(0))
    assert not np.allclose(res["p_value"][:, 0], sep.pvalue)  # pooled over 4 groups, df 8 vs 4


# --- contrasts: pairs, weights, validation ---------------------------------------------------------

def test_interaction_contrast_is_difference_of_pair_effects():
    labels, X = _data({"nv": 4, "nd": 4, "dv": 4, "dd": 4}, n_prot=8, seed=7)
    codes, levels = _codes(labels)
    ld = {l: {"g": l} for l in levels}
    fit = jd.fit_group_means(X, codes, 4)
    contrasts = {"np": ("nd", "nv"), "dd": ("dd", "dv"),
                 "int": {"nd": 1, "nv": -1, "dd": -1, "dv": 1}}
    W, labs, _ = jd.build_contrasts(levels, ld, None, None, contrasts)
    res = jd.fit_contrasts(fit, W, "lm")
    np.testing.assert_allclose(res["effect"][:, 2], res["effect"][:, 0] - res["effect"][:, 1])
    v = res["se"][:, 2] ** 2 / res["s2"]
    np.testing.assert_allclose(v, 4 * (1 / 4))  # four groups of 4 with unit weights
    # sign reversal flips effect and t, keeps p
    W2, _, _ = jd.build_contrasts(levels, ld, None, None, {"x": ("nv", "nd")})
    r2 = jd.fit_contrasts(fit, W2, "lm")
    np.testing.assert_allclose(r2["effect"][:, 0], -res["effect"][:, 0])
    np.testing.assert_allclose(r2["p_value"][:, 0], res["p_value"][:, 0])


@pytest.mark.parametrize("kwargs,match", [
    (dict(reference="a"), "Provide"),
    (dict(reference="a", comparisons=["b"], contrasts={"x": ("a", "b")}), "not both"),
    (dict(reference="zz", comparisons=["b"]), "Unknown group"),
    (dict(contrasts={"x": ("a", "a")}), "itself"),
    (dict(contrasts={"x": {"a": 1, "b": 1}}), "sum to 0"),
    (dict(contrasts={"x": {"a": 1, "b": -1, "c": 0}}), "non-zero"),
    (dict(contrasts={"x": {"a": 1, "zz": -1}}), "unknown group"),
    (dict(contrasts={"x": {"a": np.inf, "b": -np.inf}}), "finite"),
    (dict(contrasts={}), "No contrasts"),
])
def test_build_contrasts_validation(kwargs, match):
    levels = ["a", "b", "c"]
    args = dict(reference=None, comparisons=None, contrasts=None)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        jd.build_contrasts(levels, {l: {"g": l} for l in levels}, **args)


# --- missingness / estimability ---------------------------------------------------------------------

def test_absent_group_gives_nan_but_contributes_variance_and_prior():
    labels, X = _data({"a": 4, "b": 4, "c": 4}, n_prot=30, seed=8)
    X[labels == "a", 0] = np.nan  # protein 0 lacks group a entirely
    X[labels == "a", 1] = np.nan
    X[labels != "c", 2] = np.nan  # protein 2 only observed in c
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, 3)
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"bc": ("b", "c"), "ab": ("a", "b")})
    res = jd.fit_contrasts(fit, W, "limma")
    assert np.isnan(res["effect"][0, 1]) and np.isfinite(res["effect"][0, 0])  # a-b NaN, b-c fine
    assert res["df_residual"][0] == 6 and res["df_residual"][2] == 3
    assert np.isnan(res["effect"][2]).all() and np.isnan(res["p_value"][2]).all()
    assert np.isfinite(res["s2_post"][2])  # protein 2 still has a posterior variance


def test_group_observed_once_is_estimable_other_groups_supply_df():
    labels, X = _data({"a": 4, "b": 4, "c": 4}, n_prot=20, seed=9)
    X[labels == "a", 0] = [1.0, np.nan, np.nan, np.nan]
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, 3)
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"ab": ("a", "b")})
    res = jd.fit_contrasts(fit, W, "lm")
    assert fit["n"][0, 0] == 1  # 9 observations - 3 groups = 6
    assert res["df_residual"][0] == 6 and np.isfinite(res["p_value"][0, 0])


def test_all_missing_protein_and_zero_variance():
    labels, X = _data({"a": 3, "b": 3}, n_prot=20, seed=10)
    X[:, 0] = np.nan
    X[:, 1] = 5.0  # no variance, no difference
    X[labels == "a", 2] = 9.0
    X[labels == "b", 2] = 5.0  # no variance, real difference
    codes, levels = _codes(labels)
    fit = jd.fit_group_means(X, codes, 2)
    W, _, _ = jd.build_contrasts(levels, {l: {"g": l} for l in levels}, None, None, {"x": ("a", "b")})
    res = jd.fit_contrasts(fit, W, "lm")
    assert np.isnan(res["p_value"][0, 0]) and fit["df"][0] == 0
    assert np.isnan(res["p_value"][1, 0])  # 0/0
    assert res["p_value"][2, 0] == 0.0 and np.isinf(res["t"][2, 0])


def test_no_residual_df_raises():
    X = np.ones((2, 5))
    with pytest.raises(ValueError, match="residual degrees of freedom"):
        jd.fit_contrasts(jd.fit_group_means(X, np.array([0, 1]), 2), np.array([[1.0], [-1.0]]), "lm")


def test_unknown_method_raises():
    fit = jd.fit_group_means(np.ones((4, 3)), np.array([0, 0, 1, 1]), 2)
    with pytest.raises(ValueError, match="Unsupported method"):
        jd.fit_contrasts(fit, np.array([[1.0], [-1.0]]), "bogus")


# --- pdata.joint_de ---------------------------------------------------------------------------------------

SIZES = {"veh": 4, "low": 4, "high": 4}


def test_joint_de_collection_structure_and_detection():
    p, labels, logX = _pdata(SIZES, shifts={"high": 3.0}, nan_frac=0.05)
    res = p.joint_de("condition", reference="veh", comparisons=["low", "high"], layer="X_log2")
    assert set(res) == {"contrasts", "meta", "diagnostics"}
    lab_high = _vs("condition", "high", "veh")
    assert list(res["contrasts"]) == [_vs("condition", "low", "veh"), lab_high]
    t = res["contrasts"][lab_high]
    assert {"log2fc", "se", "p_value", "adj_p_value", "test_statistic", "significance", "Genes",
            "n_high", "n_veh", "-log10(adj_p_value)"} <= set(t.columns)
    assert (t.loc[[f"P{i}" for i in range(5)], "significance"] == "upregulated").sum() >= 4
    assert res["meta"]["estimator"] in {"legacy", "unequal_df"}
    assert t.attrs["limma"]["df_prior"] >= 0
    assert {"df_residual", "s2", "s2_post", "df_total", "n_veh"} <= set(res["diagnostics"].columns)
    assert list(res["diagnostics"].index) == list(p.prot.var_names)  # source protein order
    assert p.stats[next(k for k in p.stats if k.startswith("joint:"))] is res
    assert any("joint_de" in h for h in p.history)


def test_joint_de_log2fc_matches_group_mean_difference_and_methods_share_effect():
    p, labels, logX = _pdata(SIZES, shifts={"high": 3.0})
    a = p.joint_de("condition", reference="veh", comparisons="high", layer="X_log2", method="limma", store=False)
    b = p.joint_de("condition", reference="veh", comparisons="high", layer="X_log2", method="lm", store=False)
    ta, tb = (next(iter(r["contrasts"].values())) for r in (a, b))
    expected = logX[labels == "high"].mean(0) - logX[labels == "veh"].mean(0)
    np.testing.assert_allclose(ta.reindex(p.prot.var_names)["log2fc"], expected, atol=1e-3)  # float32 X
    np.testing.assert_allclose(ta["log2fc"].sort_index(), tb["log2fc"].sort_index())
    assert "limma" in ta.attrs and "limma" not in tb.attrs


def test_joint_de_does_not_mutate_data_or_alter_order():
    p, *_ = _pdata(SIZES, nan_frac=0.05)
    before = np.array(p.prot.X, copy=True)
    layers = set(p.prot.layers)
    p.joint_de("condition", reference="veh", comparisons=["low", "high"], layer="X_log2")
    np.testing.assert_array_equal(np.asarray(p.prot.X), before)
    assert set(p.prot.layers) == layers


def test_joint_de_requires_log_layer():
    p, *_ = _pdata(SIZES, log=False)
    with pytest.raises(ValueError, match="log-transformed"):
        p.joint_de("condition", reference="veh", comparisons="low")


def test_joint_de_list_groupby_and_interaction_contrast():
    sizes = {"NP40_vehicle": 4, "NP40_drug": 4, "DDM_vehicle": 4, "DDM_drug": 4}
    labels = np.array([g for g, n in sizes.items() for _ in range(n)])
    p, _, logX = _pdata(
        sizes, shifts={"NP40_drug": 2.0},
        extra_obs={"detergent": [l.split("_")[0] for l in labels], "treatment": [l.split("_")[1] for l in labels]},
    )
    res = p.joint_de(
        ["detergent", "treatment"], layer="X_log2",
        contrasts={"NP40 drug": ("NP40_drug", "NP40_vehicle"),
                   "DDM drug": ("DDM_drug", "DDM_vehicle"),
                   "interaction": {"NP40_drug": 1, "NP40_vehicle": -1, "DDM_drug": -1, "DDM_vehicle": 1}},
    )
    c = res["contrasts"]
    np.testing.assert_allclose(c["interaction"].sort_index()["log2fc"],
                               (c["NP40 drug"].sort_index()["log2fc"] - c["DDM drug"].sort_index()["log2fc"]))
    assert list(c) == ["NP40 drug", "DDM drug", "interaction"]
    assert (c["interaction"].loc[[f"P{i}" for i in range(5)], "log2fc"] > 1).all()


def test_joint_de_drops_samples_without_group_and_reports():
    p, *_ = _pdata(SIZES)
    p._summary["condition"] = p._summary["condition"].astype(object)
    p._summary.loc[p._summary.index[:2], "condition"] = np.nan  # .summary takes precedence over .obs
    res = p.joint_de("condition", reference="veh", comparisons="low", layer="X_log2", store=False)
    assert res["meta"]["n_samples_dropped"] == 2
    assert res["meta"]["n_samples"]["veh"] == 2


def test_joint_de_masks_zeros_as_missing():
    p, labels, logX = _pdata(SIZES)
    X = np.asarray(p.prot.layers["X_log2"], dtype=float).copy()
    X[0, 0] = 0.0  # log2(0 + 1): a zero in the original data
    p.prot.layers["X_log2"] = X
    res = p.joint_de("condition", reference="veh", comparisons="low", layer="X_log2", store=False)
    assert res["diagnostics"].loc["P0", "n_veh"] == 3


def test_joint_de_log10_layer_reports_log2fc():
    p, labels, logX = _pdata(SIZES, shifts={"high": 3.0})
    p2, _, _ = _pdata(SIZES, shifts={"high": 3.0}, log=False)  # same data, log10 instead
    p2.log_transform(base=10, pseudocount=1.0, layer="X")
    layer10 = [k for k in p2.prot.layers if k.endswith("log10")]
    a = p.joint_de("condition", reference="veh", comparisons="high", layer="X_log2", method="lm", store=False)
    b = p2.joint_de("condition", reference="veh", comparisons="high", layer=layer10[0], method="lm", store=False)
    np.testing.assert_allclose(next(iter(a["contrasts"].values()))["log2fc"].sort_index(),
                               next(iter(b["contrasts"].values()))["log2fc"].sort_index(), rtol=1e-4, atol=1e-5)


def test_joint_de_fdr_scopes():
    p, *_ = _pdata(SIZES, shifts={"high": 2.0, "low": 1.0})
    kw = dict(reference="veh", comparisons=["low", "high"], layer="X_log2", store=False)
    per = p.joint_de("condition", fdr_scope="per_contrast", **kw)["contrasts"]
    glob = p.joint_de("condition", fdr_scope="global", **kw)["contrasts"]
    both = p.joint_de("condition", fdr_scope="both", **kw)["contrasts"]
    from scpviz.utils.stats import bh_adjust_pvalues
    pooled = np.concatenate([t.sort_index()["p_value"].to_numpy() for t in per.values()])
    adj = bh_adjust_pvalues(pooled)
    n = len(next(iter(per.values())))
    for k, (lab, t) in enumerate(glob.items()):
        np.testing.assert_allclose(t.sort_index()["adj_p_value"], adj[k * n:(k + 1) * n])
        np.testing.assert_allclose(both[lab].sort_index()["adj_p_value_global"], adj[k * n:(k + 1) * n])
        np.testing.assert_allclose(per[lab].sort_index()["adj_p_value"],
                                   bh_adjust_pvalues(per[lab].sort_index()["p_value"].to_numpy()))
        np.testing.assert_allclose(both[lab].sort_index()["adj_p_value"], per[lab].sort_index()["adj_p_value"])
        assert "adj_p_value_global" not in per[lab].columns


def test_joint_de_no_fdr_has_no_adjusted_columns():
    p, *_ = _pdata(SIZES)
    res = p.joint_de("condition", reference="veh", comparisons="low", layer="X_log2", correct_fdr=False, store=False)
    assert "adj_p_value" not in next(iter(res["contrasts"].values())).columns


def test_joint_de_store_false_and_custom_key():
    p, *_ = _pdata(SIZES)
    p.joint_de("condition", reference="veh", comparisons="low", layer="X_log2", store=False)
    assert not any(k.startswith("joint:") for k in p.stats)
    p.joint_de("condition", reference="veh", comparisons="low", layer="X_log2", stats_key="mykey")
    assert "mykey" in p.stats


def test_joint_de_needs_two_groups_and_valid_args():
    p, *_ = _pdata({"veh": 4, "low": 4})
    with pytest.raises(ValueError, match="Unsupported method"):
        p.joint_de("condition", reference="veh", comparisons="low", layer="X_log2", method="bogus")
    with pytest.raises(ValueError, match="fdr_scope"):
        p.joint_de("condition", reference="veh", comparisons="low", layer="X_log2", fdr_scope="bogus")
    with pytest.raises(ValueError, match="Metadata column"):
        p.joint_de("nope", reference="veh", comparisons="low", layer="X_log2")


def test_joint_de_prints_prior_once(capsys):
    p, *_ = _pdata(SIZES)
    p.joint_de("condition", reference="veh", comparisons=["low", "high"], layer="X_log2")
    assert capsys.readouterr().out.count("prior estimator") == 1


def test_joint_de_volcano_from_contrast_table():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt
    p, *_ = _pdata(SIZES, shifts={"high": 3.0})
    res = p.joint_de("condition", reference="veh", comparisons=["high"], layer="X_log2")
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=next(iter(res["contrasts"].values())), correct_fdr=True)
    plt.close(fig)


# --- smokes on real imports (PD x2, DIA-NN x2) ---------------------------------------------------------------

def _smoke(p, groupby, reference, comparisons, method):
    q = p.copy()
    q.log_transform(base=2, pseudocount=1.0)
    res = q.joint_de(groupby, reference=reference, comparisons=comparisons, method=method)
    label = _vs(groupby, comparisons[0], reference)
    t = res["contrasts"][label]
    assert np.isfinite(t["p_value"]).any() and np.isfinite(t["adj_p_value"]).any()
    assert len(t) == q.prot.n_vars
    return q, res


@pytest.mark.parametrize("method", ["limma", "lm"])
def test_smoke_pd_cellline(pdata, method):
    levels = sorted(pdata.prot.obs["cellline"].unique()) if "cellline" in pdata.prot.obs else sorted(pdata.summary["cellline"].unique())
    _smoke(pdata, "cellline", levels[0], levels[1:], method)


def test_smoke_pd_two_columns_and_interaction(pdata):
    q = pdata.copy()
    q.log_transform(base=2, pseudocount=1.0)
    s = q.summary
    combos = sorted({f"{a}_{b}" for a, b in zip(s["cellline"], s["treatment"])})
    assert len(combos) >= 4, combos
    a, b, c, d = combos[:4]
    res = q.joint_de(["cellline", "treatment"], contrasts={"pair": (a, b), "int": {a: 1, b: -1, c: -1, d: 1}})
    assert set(res["contrasts"]) == {"pair", "int"}
    assert np.isfinite(res["contrasts"]["int"]["p_value"]).any()


@pytest.mark.parametrize("method", ["limma", "lm"])
def test_smoke_diann_gradient(pdata_diann, method):
    levels = sorted(pdata_diann.summary["gradient"].unique())
    q, res = _smoke(pdata_diann, "gradient", levels[0], levels[1:], method)
    assert res["meta"]["levels"] == levels


def test_smoke_diann_stats_survive_save_load(pdata_diann, tmp_path):
    q, res = _smoke(pdata_diann, "gradient", "24min", ["30min"], "limma")
    path = tmp_path / "joint.pdata"
    q.save(path)
    loaded = pAnnData.load(path)
    key = next(k for k in q.stats if k.startswith("joint:"))
    lab = _vs("gradient", "30min", "24min")
    pd.testing.assert_frame_equal(loaded.stats[key]["contrasts"][lab], q.stats[key]["contrasts"][lab], check_categorical=False)


# --- parity with R limma (lmFit ~0+group, contrasts.fit, eBayes; fixtures in tests/data/joint_limma) -----------
# Regenerate with tests/tools/make_joint_limma_fixtures.R (R is dev-only).

from pathlib import Path

JOINT_DIR = Path(__file__).parent / "data" / "joint_limma"
JOINT_CASES = ["balanced3", "pisa6", "doses4", "missing4"]


def _close(actual, expected, rtol, atol=0.0):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("case", JOINT_CASES)
def test_joint_limma_parity_with_r(case):
    expr = pd.read_csv(JOINT_DIR / f"{case}_expr.csv", index_col=0, na_values="NA")
    group = pd.read_csv(JOINT_DIR / f"{case}_group.csv")["group"].to_numpy()
    C = pd.read_csv(JOINT_DIR / f"{case}_contrasts.csv", index_col=0)
    prior = pd.read_csv(JOINT_DIR / f"{case}_prior.csv")
    codes, levels = _codes(group)
    assert list(C.index) == levels
    fit = jd.fit_group_means(expr.to_numpy().T, codes, len(levels))
    res = jd.fit_contrasts(fit, C.to_numpy(float), "limma")
    _close(res["df_prior"], prior.loc[0, "df_prior"], 1e-5)
    _close(res["s2_prior"], prior.loc[0, "s2_prior"], 1e-5)
    from scpviz.utils.stats import bh_adjust_pvalues
    for j in range(C.shape[1]):
        ref = pd.read_csv(JOINT_DIR / f"{case}_c{j + 1}_result.csv", na_values="NA")
        _close(res["effect"][:, j], ref["coef"], 1e-9, atol=1e-12)
        _close(res["se"][:, j] ** 2 / res["s2_post"], ref["stdev_unscaled"] ** 2, 1e-9)  # v = stdev_unscaled^2
        _close(res["df_residual"], ref["df_residual"], 0)
        _close(res["s2"], ref["s2"], 1e-8, atol=1e-14)
        _close(res["s2_post"], ref["s2_post"], 1e-5, atol=1e-14)
        _close(res["t"][:, j], ref["t"], 1e-5, atol=1e-10)
        _close(res["df_total"], ref["df_total"], 1e-5)
        _close(res["p_value"][:, j], ref["p_value"], 1e-4, atol=1e-12)
        _close(bh_adjust_pvalues(res["p_value"][:, j]), ref["p_adj"], 1e-4, atol=1e-12)


def test_joint_de_volcano_with_weighted_contrast_label():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt
    p, *_ = _pdata({"a_v": 4, "a_d": 4, "b_v": 4, "b_d": 4}, shifts={"a_d": 3.0})
    name = "drug effect: a minus b"  # no " vs " in the label
    res = p.joint_de("condition", layer="X_log2",
                     contrasts={name: {"a_d": 1, "a_v": -1, "b_d": -1, "b_v": 1}})
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=res["contrasts"][name], correct_fdr=True)
    plt.close(fig)


def _volcano_texts(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=table, correct_fdr=True)
    texts = [c.get_text() for c in ax.texts]
    plt.close(fig)
    return texts


def test_joint_de_volcano_title_and_group_labels():
    p, *_ = _pdata({"a_v": 4, "a_d": 4, "b_v": 4, "b_d": 4}, shifts={"a_d": 3.0})
    res = p.joint_de("condition", layer="X_log2", contrasts={
        "a drug vs b drug": ("a_d", "b_d"),                              # contains ' vs ' but names no groups
        "interaction": {"a_d": 1, "a_v": -1, "b_d": -1, "b_v": 1},
    })
    pair = _volcano_texts(res["contrasts"]["a drug vs b drug"])
    assert any(t.startswith("Joint DE | limma") for t in pair) and not any("Mixed DE" in t for t in pair)
    assert "a_d" in pair and "b_d" in pair                               # +1 group right, -1 group left
    assert "a drug" not in pair
    inter = _volcano_texts(res["contrasts"]["interaction"])
    assert any(t.startswith("Joint DE | limma | interaction") for t in inter)
    assert not any(t in ("a_d", "a_v", "b_d", "b_v") for t in inter)     # no group boxes for weighted contrasts


def test_joint_de_volcano_xlabel_delta_only_for_weighted_contrasts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt
    p, *_ = _pdata({"a_v": 4, "a_d": 4, "b_v": 4, "b_d": 4}, shifts={"a_d": 3.0})
    res = p.joint_de("condition", layer="X_log2", contrasts={
        "pair": ("a_d", "a_v"), "interaction": {"a_d": 1, "a_v": -1, "b_d": -1, "b_v": 1}})
    labels = {}
    for name, t in res["contrasts"].items():
        fig, ax = plt.subplots()
        scplt.plot_volcano(ax, de_data=t, correct_fdr=True)
        labels[name] = ax.get_xlabel()
        plt.close(fig)
    assert labels["pair"] == r"$\mathrm{log}_{2}$ fold change"
    assert labels["interaction"] == r"$\Delta$ $\mathrm{log}_{2}$ fold change"

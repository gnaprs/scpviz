"""Tests for pAnnData.mixed_de."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from anndata import AnnData
from scipy.stats import ttest_ind

from scpviz import pAnnData
from scpviz.utils import mixed_de as mixed_de_utils

# Synthetic fixtures

def _build_nested_pdata(
    *,
    rng_seed: int = 42,
    n_donors: int = 4,
    cells_per_stratum: int = 6,
    condition_shift: float = 2.5,
    condition_feature: int = 0,
    donor_shift_scale: float = 0.4,
    noise_scale: float = 1.0,
    n_features: int = 8,
):
    """
    Nested design with known perturbations.

    - ``P{condition_feature}``: +``condition_shift`` on disease cells.
    - All features: +``d * donor_shift_scale`` per donor (no condition interaction).
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for d in range(n_donors):
        for cond in ("control", "disease"):
            for _ in range(cells_per_stratum):
                rows.append(
                    {
                        "donor": f"D{d}",
                        "condition": cond,
                        "batch": f"B{d % 2}",
                        "cell_type": "Astrocyte",
                    }
                )
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    X = rng.normal(scale=noise_scale, size=(len(obs), n_features))
    disease_mask = obs["condition"] == "disease"
    X[disease_mask, condition_feature] += condition_shift
    for d in range(n_donors):
        donor_mask = obs["donor"] == f"D{d}"
        X[donor_mask, :] += d * donor_shift_scale

    var = pd.DataFrame(
        {"Genes": [f"G{i}" for i in range(n_features)]},
        index=[f"P{i}" for i in range(n_features)],
    )
    meta = {
        "condition_feature": f"P{condition_feature}",
        "condition_shift": condition_shift,
        "null_features": [f"P{i}" for i in range(n_features) if i != condition_feature],
    }
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var)), meta

def _build_confounded_pdata(*, rng_seed: int = 7):
    """Donor D0 baseline on P1 with imbalanced cell counts (pseudoreplication trap)."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for d in range(6):
        for cond in ("control", "disease"):
            n = 40 if (d == 0 and cond == "disease") else (2 if d == 0 else 4)
            for _ in range(n):
                rows.append({"donor": f"D{d}", "condition": cond, "cell_type": "Neuron"})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    n_features = 4
    X = rng.normal(scale=0.25, size=(len(obs), n_features))
    X[obs["condition"] == "disease", 0] += 2.0
    X[obs["donor"] == "D0", 1] += 4.0

    var = pd.DataFrame(
        {"Genes": ["TRUE_DE", "DONOR_CONF", "NULL2", "NULL3"]},
        index=[f"P{i}" for i in range(n_features)],
    )
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var))

def _build_donor_only_null_pdata(*, rng_seed: int = 11):
    """Donor shifts only; expression constant within donor across conditions."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for d in range(4):
        for cond in ("control", "disease"):
            for _ in range(8):
                rows.append({"donor": f"D{d}", "condition": cond})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    X = rng.normal(scale=0.5, size=(len(obs), 4))
    for d in range(4):
        X[obs["donor"] == f"D{d}", :] += d * 1.5

    var = pd.DataFrame({"Genes": [f"G{i}" for i in range(4)]}, index=[f"P{i}" for i in range(4)])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var))

def _build_unpaired_donors_pdata(*, cells_per_donor: int = 8):
    """Donors D0–D2 control-only; D3–D5 disease-only (≥3 per arm for unpaired test)."""
    rows = []
    for d in range(6):
        cond = "control" if d < 3 else "disease"
        for _ in range(cells_per_donor):
            rows.append({"donor": f"D{d}", "condition": cond})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    X = np.random.default_rng(3).normal(size=(len(obs), 3))
    X[obs["condition"] == "disease", 0] += 1.5
    var = pd.DataFrame({"Genes": ["G0", "G1", "G2"]}, index=["P0", "P1", "P2"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var))

def _build_k3_pdata(*, level_shifts: dict[str, float] | None = None):
    """Three-level factor with known per-level shifts on P0."""
    level_shifts = level_shifts or {"A": 0.0, "B": 1.0, "C": 2.0}
    rows = []
    for d in range(4):
        for lvl in ("A", "B", "C"):
            for _ in range(5):
                rows.append({"donor": f"D{d}", "group": lvl})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    X = np.random.default_rng(5).normal(scale=0.2, size=(len(obs), 3))
    for lvl, shift in level_shifts.items():
        X[obs["group"] == lvl, 0] += shift
    var = pd.DataFrame({"Genes": ["G0", "G1", "G2"]}, index=["P0", "P1", "P2"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var)), level_shifts

def _build_interaction_pdata(*, stratum_shift: float = 2.0):
    """True condition effect only in layer L5 on P0."""
    rows = []
    for d in range(4):
        for cond in ("control", "disease"):
            for layer in ("L2", "L5"):
                for _ in range(6):
                    rows.append({"donor": f"D{d}", "condition": cond, "layer": layer})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    X = np.random.default_rng(9).normal(scale=0.3, size=(len(obs), 2))
    l5_dis = (obs["condition"] == "disease") & (obs["layer"] == "L5")
    X[l5_dis, 0] += stratum_shift
    var = pd.DataFrame({"Genes": ["G0", "G1"]}, index=["P0", "P1"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var)), stratum_shift

def _build_with_zero_feature(pdata: pAnnData) -> pAnnData:
    """Return new pdata with an additional all-zero protein (avoids .summary copy)."""
    adata = pdata.prot
    X = np.hstack([np.asarray(adata.X), np.zeros((adata.n_obs, 1))])
    var = adata.var.copy()
    var.loc["Pzero"] = {"Genes": "ZERO"}
    return pAnnData(prot=AnnData(X=X, obs=adata.obs.copy(), var=var))

def _permute_condition_within_donor(pdata: pAnnData, *, seed: int = 0) -> None:
    """In-place: shuffle condition labels within each donor."""
    rng = np.random.default_rng(seed)
    obs = pdata.prot.obs
    for donor in obs["donor"].unique():
        idx = obs.index[obs["donor"] == donor]
        shuffled = obs.loc[idx, "condition"].values.copy()
        rng.shuffle(shuffled)
        obs.loc[idx, "condition"] = shuffled

def _naive_cell_level_pvalue(pdata, feature: str) -> float:
    obs = pdata.prot.obs
    X = np.asarray(pdata.prot.X)
    idx = pdata.prot.var_names.get_loc(feature)
    ctrl = obs["condition"].astype(str) == "control"
    dis = obs["condition"].astype(str) == "disease"
    _, pval = ttest_ind(X[dis, idx], X[ctrl, idx], equal_var=False, nan_policy="omit")
    return float(pval)

def _statsmodels_reference_fc_p(
    pdata: pAnnData,
    feature: str,
    *,
    formula: str = "expr ~ condition",
    donor_col: str = "donor",
) -> tuple[float, float]:
    """Hand-fit MixedLM on one feature for parity checks."""
    adata = pdata.prot
    j = adata.var_names.get_loc(feature)
    meta = adata.obs.copy()
    meta["expr"] = np.asarray(adata.X)[:, j]
    meta[donor_col] = meta[donor_col].astype("category")
    meta["condition"] = meta["condition"].astype("category")
    fit = smf.mixedlm(formula, meta, groups=meta[donor_col]).fit(reml=True, disp=False)
    return mixed_de_utils.extract_contrast_result(
        fit, contrast_term="condition", ref="control", test="disease", contrast_at=None
    )

def _build_composition_pdata(*, type_b_shift: float = 3.0):
    """
    Composition shift only: type B higher on P0, no within-type condition effect.

    Control is mostly type A; disease mostly type B (proportion shift).
    """
    rows = []
    for d in range(4):
        for cond in ("control", "disease"):
            n_a, n_b = (18, 2) if cond == "control" else (2, 18)
            for _ in range(n_a):
                rows.append({"donor": f"D{d}", "condition": cond, "cell_type": "A"})
            for _ in range(n_b):
                rows.append({"donor": f"D{d}", "condition": cond, "cell_type": "B"})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    X = np.random.default_rng(13).normal(scale=0.2, size=(len(obs), 2))
    X[obs["cell_type"] == "B", 0] += type_b_shift
    var = pd.DataFrame({"Genes": ["COMP", "NULL"]}, index=["P0", "P1"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var))

def _build_sparse_subset_pdata():
    """P_sparse: abundant in Astrocytes, rare in Microglia."""
    rows = []
    for d in range(4):
        for cond in ("control", "disease"):
            for _ in range(10):
                rows.append({"donor": f"D{d}", "condition": cond, "cell_type": "Astrocyte"})
            for _ in range(10):
                rows.append({"donor": f"D{d}", "condition": cond, "cell_type": "Microglia"})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    rng = np.random.default_rng(17)
    X = rng.normal(size=(len(obs), 3))
    astro = obs["cell_type"] == "Astrocyte"
    micro = obs["cell_type"] == "Microglia"
    X[astro, 0] = rng.normal(loc=5.0, scale=0.5, size=astro.sum())
    X[micro, 0] = np.nan
    detect = rng.random(micro.sum()) > 0.85
    micro_idx = np.where(micro)[0]
    X[micro_idx[detect], 0] = rng.normal(loc=5.0, scale=0.5, size=detect.sum())
    var = pd.DataFrame({"Genes": ["SPARSE", "G1", "G2"]}, index=["P_sparse", "P1", "P2"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var))

def _build_random_slopes_pdata(*, mean_slope: float = 2.0):
    """Donor-specific condition slopes on P0; population mean slope = mean_slope."""
    rows = []
    for d in range(4):
        slope = mean_slope + (d - 1.5) * 0.6
        for cond in ("control", "disease"):
            for _ in range(8):
                rows.append({"donor": f"D{d}", "condition": cond, "_slope": slope})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    X = np.random.default_rng(19).normal(scale=0.2, size=(len(obs), 2))
    for d in range(4):
        mask = (obs["donor"] == f"D{d}") & (obs["condition"] == "disease")
        X[mask, 0] += mean_slope + (d - 1.5) * 0.6
    var = pd.DataFrame({"Genes": ["G0", "G1"]}, index=["P0", "P1"])
    return pAnnData(prot=AnnData(X=X, obs=obs.drop(columns=["_slope"]), var=var)), mean_slope

def _build_adversarial_alpha_pdata(*, shift: float = 3.0, rng_seed: int = 31):
    """Two-group design with alphabetically hostile labels: ``asample`` < ``zsample``.

    Plants ``+shift`` on ``zsample`` cells. Any code path that relies on Patsy's
    default alphabetical baseline without releveling to the contrast *ref* will
    break when ``test='asample'`` (no coefficient for the baseline level).

    Also plants a modest donor random intercept so MixedLM is not estimating a
    near-zero RE variance (flaky across BLAS builds).
    """
    rows = []
    for d in range(4):
        for grp in ("asample", "zsample"):
            for _ in range(8):
                rows.append({"donor": f"D{d}", "group": grp})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    rng = np.random.default_rng(rng_seed)
    X = rng.normal(scale=0.25, size=(len(obs), 3))
    donor_effects = {"D0": 0.7, "D1": -0.5, "D2": 0.3, "D3": -0.5}
    for d, eff in donor_effects.items():
        X[obs["donor"].to_numpy() == d, :] += eff
    X[obs["group"].to_numpy() == "zsample", 0] += shift
    var = pd.DataFrame({"Genes": ["G0", "G1", "G2"]}, index=["P0", "P1", "P2"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var)), shift

def _build_adversarial_alpha_k3_pdata(
    *,
    level_shifts: dict[str, float] | None = None,
    rng_seed: int = 37,
):
    """Three-level factor with hostile names: ``asample`` < ``msample`` < ``zsample``.

    Sorted pairwise order is alphabetical. Patsy baseline defaults to ``asample``,
    so ``one_vs_rest`` with ``focal_level='asample'`` makes *test* the model
    baseline — the exact failure mode that silent-fell-back before.

    Plants a modest donor random intercept so MixedLM is not estimating a
    boundary (near-zero) RE variance — that path is flaky across BLAS builds.
    """
    level_shifts = level_shifts or {
        "asample": 0.0,
        "msample": 1.5,
        "zsample": 3.0,
    }
    rows = []
    for d in range(4):
        for lvl in level_shifts:
            for _ in range(6):
                rows.append({"donor": f"D{d}", "group": lvl})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    rng = np.random.default_rng(rng_seed)
    X = rng.normal(scale=0.2, size=(len(obs), 3))
    donor_effects = {"D0": 0.7, "D1": -0.5, "D2": 0.3, "D3": -0.5}
    for d, eff in donor_effects.items():
        X[obs["donor"].to_numpy() == d, :] += eff
    for lvl, shift in level_shifts.items():
        X[obs["group"].to_numpy() == lvl, 0] += shift
    var = pd.DataFrame({"Genes": ["G0", "G1", "G2"]}, index=["P0", "P1", "P2"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var)), level_shifts

def _build_single_donor_pdata():
    rows = []
    for cond in ("control", "disease"):
        for _ in range(20):
            rows.append({"donor": "D0", "condition": cond})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    X = np.random.default_rng(23).normal(size=(len(obs), 2))
    X[obs["condition"] == "disease", 0] += 1.0
    var = pd.DataFrame({"Genes": ["G0", "G1"]}, index=["P0", "P1"])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var))

def _build_additive_vs_interaction_pdata():
    """Condition effect only in L5; simple additive model dilutes the estimand."""
    return _build_interaction_pdata(stratum_shift=2.5)

@pytest.fixture
def nested_sim():
    return _build_nested_pdata()

@pytest.fixture
def pdata_nested(nested_sim):
    return nested_sim[0]

@pytest.fixture
def pdata_nested_meta(nested_sim):
    return nested_sim[1]

@pytest.fixture
def pdata_confounded():
    return _build_confounded_pdata()

# synthetic ground truth & CI smoke

def test_mixed_de_detects_injected_condition_effect(pdata_nested, pdata_nested_meta):
    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        fixed_covariates=["batch"],
        subset={"cell_type": "Astrocyte"},
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    hit = pdata_nested_meta["condition_feature"]
    assert df.loc[hit, "log2fc"] == pytest.approx(pdata_nested_meta["condition_shift"], abs=0.6)
    assert df.loc[hit, "p_value"] < 0.01
    assert df.loc[hit, "significance"] == "upregulated"
    assert df.attrs["mixed_de"]["n_donors_paired"] == 4

def test_mixed_de_null_proteins_not_significant(pdata_nested, pdata_nested_meta):
    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    for feat in pdata_nested_meta["null_features"]:
        assert abs(df.loc[feat, "log2fc"]) < 0.8, feat
        assert df.loc[feat, "p_value"] > 0.05, feat

def test_mixed_de_permute_condition_null(pdata_nested, pdata_nested_meta):
    """permute condition within donor → no enriched DE."""
    _permute_condition_within_donor(pdata_nested, seed=1)
    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    hit = pdata_nested_meta["condition_feature"]
    assert abs(df.loc[hit, "log2fc"]) < 1.0
    assert df.loc[hit, "p_value"] > 0.05
    assert float(np.nanmedian(df["p_value"])) > 0.2

def test_mixed_de_contrast_swap_invariance(pdata_nested):
    """swapping (test, ref) negates FC; p unchanged."""
    df_fwd = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    df_rev = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("control", "disease"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df_fwd.loc["P0", "log2fc"] == pytest.approx(-df_rev.loc["P0", "log2fc"], rel=1e-6)
    assert df_fwd.loc["P0", "p_value"] == pytest.approx(df_rev.loc["P0", "p_value"], rel=1e-6)

def test_mixed_de_simple_matches_formula_additive(pdata_nested):
    """simple path matches advanced additive formula."""
    df_simple = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        fixed_covariates=["batch"],
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    df_formula = pdata_nested.mixed_de(
        formula="expr ~ condition + batch",
        contrast_term="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    for feat in ["P0", "P1"]:
        assert df_simple.loc[feat, "log2fc"] == pytest.approx(df_formula.loc[feat, "log2fc"], abs=1e-6)
        assert df_simple.loc[feat, "p_value"] == pytest.approx(df_formula.loc[feat, "p_value"], rel=1e-4)

def test_mixed_de_statsmodels_parity(pdata_nested):
    """mixed_de agrees with hand-fit MixedLM on P0."""
    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    ref_fc, ref_p = _statsmodels_reference_fc_p(pdata_nested, "P0")
    assert df.loc["P0", "log2fc"] == pytest.approx(ref_fc, rel=0.02)
    assert df.loc["P0", "p_value"] == pytest.approx(ref_p, rel=0.05)

def test_mixed_de_donor_only_null_lmm(pdata_nested_meta):
    """syn_donor_only_null: donor variation, no condition effect."""
    pdata = _build_donor_only_null_pdata()
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    for feat in df.index:
        assert abs(df.loc[feat, "log2fc"]) < 0.8, feat
        assert df.loc[feat, "p_value"] > 0.05, feat

def test_mixed_de_pseudobulk_vs_naive_on_donor_confounding(pdata_confounded):
    naive_p0 = _naive_cell_level_pvalue(pdata_confounded, "P0")
    naive_p1 = _naive_cell_level_pvalue(pdata_confounded, "P1")
    mixed = pdata_confounded.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert naive_p0 < 0.05
    assert mixed.loc["P0", "p_value"] < 0.05
    assert mixed.loc["P0", "log2fc"] == pytest.approx(2.0, abs=0.5)
    assert naive_p1 < 0.05
    assert mixed.loc["P1", "p_value"] > 0.05

def test_mixed_de_lmm_cells_recovers_condition_effect(pdata_nested, pdata_nested_meta):
    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    hit = pdata_nested_meta["condition_feature"]
    assert df.loc[hit, "log2fc"] == pytest.approx(pdata_nested_meta["condition_shift"], abs=0.8)
    assert df.loc[hit, "p_value"] < 0.05
    assert df.loc[hit, "de_method"] == "mixedlm"

# observation_level routing

def test_mixed_de_auto_routing_cells_vs_pseudobulk():
    pdata_small, _ = _build_nested_pdata(cells_per_stratum=25)  # 4*2*25 = 200
    df_small = pdata_small.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="auto",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df_small.attrs["mixed_de"]["observation_level_used"] == "cells"

    pdata_large, _ = _build_nested_pdata(cells_per_stratum=250)  # 2000 cells
    df_large = pdata_large.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="auto",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df_large.attrs["mixed_de"]["observation_level_used"] == "pseudobulk"

def test_mixed_de_subsample_reproducible():
    pdata, _ = _build_nested_pdata(cells_per_stratum=40)  # 320 cells
    kwargs = dict(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="subsample",
        max_cells_per_stratum=10,
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    df1 = pdata.mixed_de(**kwargs)
    df2 = pdata.mixed_de(**kwargs)
    assert df1.attrs["mixed_de"]["observation_level_used"] == "subsample"
    assert df1.loc["P0", "log2fc"] == pytest.approx(df2.loc["P0", "log2fc"])
    assert df1.loc["P0", "p_value"] == pytest.approx(df2.loc["P0", "p_value"])

def test_mixed_de_cells_vs_pseudobulk_same_sign():
    pdata, _ = _build_nested_pdata(condition_shift=3.0, noise_scale=0.5)
    kwargs = dict(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    df_cells = pdata.mixed_de(observation_level="cells", method="mixedlm", **kwargs)
    df_pb = pdata.mixed_de(observation_level="pseudobulk", **kwargs)
    assert df_cells.loc["P0", "log2fc"] > 0
    assert df_pb.loc["P0", "log2fc"] > 0
    assert df_pb.loc["P0", "p_value"] >= df_cells.loc["P0", "p_value"]

# feature filtering

def test_mixed_de_excludes_all_zero_feature(pdata_nested):
    pdata = _build_with_zero_feature(pdata_nested)
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.1,
        min_cells_detected=3,
    )
    assert "Pzero" not in df.index or np.isnan(df.loc["Pzero", "p_value"])
    n_tested = int(np.isfinite(df["p_value"]).sum())
    assert n_tested == pdata.prot.n_vars - 1

def test_mixed_de_no_features_raises(pdata_nested):
    pdata_nested.prot.X[:] = np.nan
    with pytest.raises(ValueError, match="filter_prot_found"):
        pdata_nested.mixed_de(
            group_col="condition",
            contrast=("disease", "control"),
            donor_col="donor",
            store=False,
        )

def _build_integer_donor_region_pdata(*, rng_seed: int = 7):
    """Integer donor IDs with one Cortex-only donor (mirrors real fig2 layout)."""
    rows = []
    for donor, cortex, snpc in [(1, 19, 33), (2, 9, 9), (3, 10, 18), (4, 2, 0)]:
        rows += [{"donor": donor, "region": "Cortex"}] * cortex
        rows += [{"donor": donor, "region": "SNpc"}] * snpc
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    rng = np.random.default_rng(rng_seed)
    X = rng.normal(size=(len(obs), 4))
    donor_eff = obs["donor"].map({1: 0.5, 2: 0.0, 3: -0.3, 4: 0.1}).astype(float).values
    region_eff = (obs["region"] == "SNpc").astype(float).values * 2.0
    X += donor_eff[:, None] + region_eff[:, None]
    var = pd.DataFrame({"Genes": [f"G{i}" for i in range(4)]}, index=[f"P{i}" for i in range(4)])
    return pAnnData(prot=AnnData(X=X, obs=obs, var=var))

# Log-scale guard (linear norm → log2 → mixed DE)

def test_expr_looks_non_log_heuristic():
    assert mixed_de_utils.expr_looks_non_log(np.array([1e5, 2e5, 3e5]))
    assert not mixed_de_utils.expr_looks_non_log(np.array([0.0, 2.5, 5.0, 12.0]))

def test_mixed_de_auto_log2_linear_layer():
    pdata, _ = _build_nested_pdata(condition_shift=2.5, condition_feature=0)
    pdata.prot.X = np.asarray(pdata.prot.X) * 1e5 + 50_000.0
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        min_detected_fraction=0.0,
        min_cells_detected=1,
        store=False,
    )
    assert df.attrs["mixed_de"]["expr_auto_log2"] is True
    assert abs(df.loc["P0", "log2fc"] - 2.5) < 0.5
    assert df["log2fc"].abs().max() < 20

def test_mixed_de_auto_log2_false_raises_on_linear_layer():
    pdata, _ = _build_nested_pdata()
    pdata.prot.X = np.asarray(pdata.prot.X) * 1e5 + 50_000.0
    with pytest.raises(ValueError, match="log2-transformed"):
        pdata.mixed_de(
            group_col="condition",
            contrast=("disease", "control"),
            donor_col="donor",
            observation_level="pseudobulk",
            auto_log2=False,
            store=False,
            min_detected_fraction=0.0,
            min_cells_detected=1,
        )

def test_mixed_de_registered_log_layer_not_auto_logged():
    pdata, _ = _build_nested_pdata(condition_shift=2.5, condition_feature=0)
    pdata.log_transform(layer="X", base=2, set_X=False)
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        layer="X_log2",
        observation_level="pseudobulk",
        min_detected_fraction=0.0,
        min_cells_detected=1,
        store=False,
    )
    meta = df.attrs["mixed_de"]
    assert meta["layer_is_log"] is True
    assert meta["expr_auto_log2"] is False
    assert np.isfinite(df.loc["P0", "log2fc"])
    assert abs(df.loc["P0", "log2fc"]) < 10

def test_mixed_de_layer_X_resolves_registered_log_via_current_X():
    """Default layer='X' should see log provenance after log_transform(set_X=True)."""
    pdata, _ = _build_nested_pdata(condition_shift=2.5, condition_feature=0)
    pdata.log_transform(layer="X", base=2, set_X=True)
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        min_detected_fraction=0.0,
        min_cells_detected=1,
        store=False,
    )
    meta = df.attrs["mixed_de"]
    assert meta["layer_is_log"] is True
    assert meta["expr_auto_log2"] is False
    assert meta["log_base"] == "log2"

# unpaired donors

def test_mixed_de_values_sugar_mixedlm_when_test_is_alphabetical_first():
    """values=[Cortex, SNpc] must recover the planted region effect via mixedlm.

    Patsy defaults to alphabetical baselines, so Cortex is the model reference.
    values sugar sets contrast=(Cortex, SNpc) → test=Cortex, which has no
    coefficient unless we relevel to ref=SNpc. Before the fix this raised
    'could not locate coefficient' for every feature → silent fit_error fallback
    to pseudobulk and zero DE power.

    Fixture plants +2.0 on SNpc, so log2fc (Cortex − SNpc) ≈ −2.
    """
    pdata = _build_integer_donor_region_pdata()
    df = pdata.mixed_de(
        values=[{"region": "Cortex"}, {"region": "SNpc"}],
        donor_col="donor",
        observation_level="cells",
        method="auto",
        store=False,
        min_detected_fraction=0.0,
        min_cells_detected=1,
    )
    assert df.attrs["mixed_de"]["contrast"] == ("Cortex", "SNpc")
    assert df.attrs["mixed_de"]["reference_levels"] == {"region": "SNpc"}
    n_mixed = int((df["de_method"] == "mixedlm").sum())
    assert n_mixed > 0, (
        f"expected mixedlm successes; got methods={df['de_method'].value_counts().to_dict()} "
        f"failures={df['mixedlm_failure_reason'].value_counts(dropna=False).to_dict()}"
    )
    # All features share the planted +2 SNpc shift in this fixture.
    assert df.loc["P0", "de_method"] == "mixedlm"
    assert df.loc["P0", "log2fc"] == pytest.approx(-2.0, abs=0.8)
    assert df.loc["P0", "p_value"] < 0.05

def test_mixed_de_values_sugar_control_first_recovers_negative_shift(
    pdata_nested, pdata_nested_meta
):
    """values=[control, disease] → log2fc = control − disease = −shift via mixedlm.

    control sorts before disease alphabetically, so this is the same failure mode
    as Cortex-first: test level is the default Patsy baseline.
    """
    shift = pdata_nested_meta["condition_shift"]
    hit = pdata_nested_meta["condition_feature"]
    df = pdata_nested.mixed_de(
        values=[{"condition": "control"}, {"condition": "disease"}],
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df["contrast"].iloc[0] == "control vs disease"
    assert df.attrs["mixed_de"]["contrast"] == ("control", "disease")
    assert df.loc[hit, "de_method"] == "mixedlm"
    assert df.loc[hit, "log2fc"] == pytest.approx(-shift, abs=0.8)
    assert df.loc[hit, "p_value"] < 0.05

def test_mixed_de_adversarial_alpha_labels_all_call_styles():
    """Canary: ``asample``/``zsample`` must not depend on alphabetical Patsy order.

    Covers both ``values=`` orders and both explicit ``contrast=(test, ref)``
    directions under cells + mixedlm, recovering the planted +shift on zsample.
    """
    pdata, shift = _build_adversarial_alpha_pdata()
    cases = [
        # (kwargs, expected_log2fc, expected_label, expected_contrast_tuple)
        (
            {"values": [{"group": "zsample"}, {"group": "asample"}]},
            shift,
            "zsample vs asample",
            ("zsample", "asample"),
        ),
        (
            {"values": [{"group": "asample"}, {"group": "zsample"}]},
            -shift,
            "asample vs zsample",
            ("asample", "zsample"),
        ),
        (
            {"group_col": "group", "contrast": ("zsample", "asample")},
            shift,
            "zsample vs asample",
            ("zsample", "asample"),
        ),
        (
            {"group_col": "group", "contrast": ("asample", "zsample")},
            -shift,
            "asample vs zsample",
            ("asample", "zsample"),
        ),
    ]
    for kwargs, exp_fc, exp_label, exp_contrast in cases:
        df = pdata.mixed_de(
            donor_col="donor",
            observation_level="cells",
            method="mixedlm",
            store=False,
            min_detected_fraction=0.0,
            min_cells_detected=1,
            **kwargs,
        )
        assert df["contrast"].iloc[0] == exp_label, kwargs
        assert df.attrs["mixed_de"]["contrast"] == exp_contrast, kwargs
        assert df.attrs["mixed_de"]["reference_levels"]["group"] == exp_contrast[1], kwargs
        assert df.loc["P0", "de_method"] == "mixedlm", kwargs
        assert df.loc["P0", "log2fc"] == pytest.approx(exp_fc, abs=0.6), kwargs
        assert df.loc["P0", "p_value"] < 0.05, kwargs

def test_mixed_de_adversarial_alpha_pairwise_mixedlm():
    """Pairwise mixedlm with asample/msample/zsample recovers planted level deltas."""
    pdata, shifts = _build_adversarial_alpha_k3_pdata()
    coll = pdata.mixed_de(
        group_col="group",
        contrast_mode="pairwise",
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.0,
        min_cells_detected=1,
    )
    # sorted(levels) → asample, msample, zsample
    expected = {
        "msample vs asample": shifts["msample"] - shifts["asample"],
        "zsample vs asample": shifts["zsample"] - shifts["asample"],
        "zsample vs msample": shifts["zsample"] - shifts["msample"],
    }
    assert set(coll["contrasts"]) == set(expected)
    for label, exp_fc in expected.items():
        df = coll["contrasts"][label]
        assert df.loc["P0", "de_method"] == "mixedlm", label
        assert df.loc["P0", "log2fc"] == pytest.approx(exp_fc, abs=0.6), label
        assert df.loc["P0", "p_value"] < 0.05, label

def test_mixed_de_adversarial_alpha_one_vs_rest_mixedlm_focal_baseline():
    """one_vs_rest with focal=asample (alphabetical baseline) must still work in mixedlm.

    Each contrast has test=asample, which is Patsy's default reference — the exact
    missing-coefficient failure mode for specified/values sugar before the fix.
    """
    pdata, shifts = _build_adversarial_alpha_k3_pdata()
    coll = pdata.mixed_de(
        group_col="group",
        contrast_mode="one_vs_rest",
        focal_level="asample",
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.0,
        min_cells_detected=1,
    )
    expected = {
        "asample vs msample": shifts["asample"] - shifts["msample"],
        "asample vs zsample": shifts["asample"] - shifts["zsample"],
    }
    assert set(coll["contrasts"]) == set(expected)
    for label, exp_fc in expected.items():
        df = coll["contrasts"][label]
        assert df.loc["P0", "de_method"] == "mixedlm", label
        assert df.loc["P0", "log2fc"] == pytest.approx(exp_fc, abs=0.6), label
        assert df.loc["P0", "p_value"] < 0.05, label

def test_mixed_de_adversarial_alpha_one_vs_rest_mixedlm_focal_last():
    """one_vs_rest with focal=zsample (last alphabetically) recovers planted deltas."""
    pdata, shifts = _build_adversarial_alpha_k3_pdata()
    coll = pdata.mixed_de(
        group_col="group",
        contrast_mode="one_vs_rest",
        focal_level="zsample",
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.0,
        min_cells_detected=1,
    )
    expected = {
        "zsample vs asample": shifts["zsample"] - shifts["asample"],
        "zsample vs msample": shifts["zsample"] - shifts["msample"],
    }
    assert set(coll["contrasts"]) == set(expected)
    for label, exp_fc in expected.items():
        df = coll["contrasts"][label]
        assert df.loc["P0", "de_method"] == "mixedlm", label
        assert df.loc["P0", "log2fc"] == pytest.approx(exp_fc, abs=0.6), label
        assert df.loc["P0", "p_value"] < 0.05, label

def test_merge_contrast_reference_levels_defaults_to_ref():
    merged = mixed_de_utils.merge_contrast_reference_levels(
        None,
        contrast_term="region",
        contrast=("Cortex", "SNpc"),
        contrast_mode="specified",
    )
    assert merged == {"region": "SNpc"}
    merged_user = mixed_de_utils.merge_contrast_reference_levels(
        {"region": "Cortex"},
        contrast_term="region",
        contrast=("Cortex", "SNpc"),
        contrast_mode="specified",
    )
    assert merged_user == {"region": "Cortex"}

def test_extract_contrast_result_when_test_is_model_baseline():
    """If test level is the model reference, return -coef(ref)."""
    rng = np.random.default_rng(1)
    n = 80
    region = np.array(["Cortex"] * 40 + ["SNpc"] * 40)
    donor = np.array([f"D{i % 4}" for i in range(n)])
    expr = rng.normal(size=n) + np.where(region == "Cortex", 2.0, 0.0)
    df = pd.DataFrame({"expr": expr, "region": region, "donor": donor})
    # Alphabetical categories → Cortex is baseline; contrast wants Cortex vs SNpc.
    df["region"] = pd.Categorical(df["region"], categories=["Cortex", "SNpc"])
    fit = smf.mixedlm("expr ~ region", df, groups=df["donor"]).fit(reml=True, disp=False)
    log2fc, pval = mixed_de_utils.extract_contrast_result(
        fit,
        contrast_term="region",
        ref="SNpc",
        test="Cortex",
        contrast_at=None,
    )
    assert log2fc == pytest.approx(2.0, abs=0.5)
    assert np.isfinite(pval)

def test_mixed_de_integer_donor_pairing_detected():
    """Integer ``donor`` column must not break paired-donor preflight.

    Also recovers the planted +2 SNpc effect under values sugar (Cortex vs SNpc
    → log2fc ≈ −2) on the pseudobulk path used by the fig2-like layout.
    """
    pdata = _build_integer_donor_region_pdata()
    df = pdata.mixed_de(
        values=[{"region": "Cortex"}, {"region": "SNpc"}],
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.0,
        min_cells_detected=1,
    )
    meta = df.attrs["mixed_de"]
    assert meta["n_donors_total"] == 4
    assert meta["n_donors_paired"] == 3
    assert meta["n_donors_Cortex_only"] == 1
    assert df["de_method"].eq("pseudobulk_paired").all()
    assert df.loc["P0", "log2fc"] == pytest.approx(-2.0, abs=0.5)
    assert df.loc["P0", "p_value"] < 0.05

def test_mixed_de_unpaired_donors_preflight():
    pdata = _build_unpaired_donors_pdata()
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df.attrs["mixed_de"]["n_donors_paired"] == 0
    assert df.loc["P0", "de_method"] == "pseudobulk_unpaired"
    assert np.isfinite(df.loc["P0", "p_value"])

def test_mixed_de_require_paired_donors_errors():
    pdata = _build_unpaired_donors_pdata()
    with pytest.raises(ValueError, match="require_paired_donors"):
        pdata.mixed_de(
            group_col="condition",
            contrast=("disease", "control"),
            donor_col="donor",
            require_paired_donors=True,
            store=False,
        )

# contrast_mode & fdr_scope

def test_mixed_de_pairwise_k3_hand_computed_fcs():
    pdata, shifts = _build_k3_pdata()
    coll = pdata.mixed_de(
        group_col="group",
        contrast_mode="pairwise",
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert len(coll["contrasts"]) == 3
    expected = {
        "B vs A": shifts["B"] - shifts["A"],
        "C vs A": shifts["C"] - shifts["A"],
        "C vs B": shifts["C"] - shifts["B"],
    }
    for label, exp_fc in expected.items():
        assert label in coll["contrasts"]
        assert coll["contrasts"][label].loc["P0", "log2fc"] == pytest.approx(exp_fc, abs=0.35)

def test_mixed_de_one_vs_rest():
    pdata, shifts = _build_k3_pdata()
    coll = pdata.mixed_de(
        group_col="group",
        contrast_mode="one_vs_rest",
        focal_level="B",
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert "B vs A" in coll["contrasts"]
    assert "B vs C" in coll["contrasts"]
    assert coll["contrasts"]["B vs A"].loc["P0", "log2fc"] == pytest.approx(
        shifts["B"] - shifts["A"], abs=0.35
    )
    assert coll["contrasts"]["B vs C"].loc["P0", "log2fc"] == pytest.approx(
        shifts["B"] - shifts["C"], abs=0.35
    )

def test_mixed_de_fdr_scope_both():
    pdata, _ = _build_k3_pdata()
    coll = pdata.mixed_de(
        group_col="group",
        contrast_mode="pairwise",
        donor_col="donor",
        observation_level="pseudobulk",
        correct_fdr=True,
        fdr_scope="both",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    for label, df in coll["contrasts"].items():
        assert "adj_p_value" in df.columns
        assert "adj_p_value_global" in df.columns
        assert df["adj_p_value_global"].notna().any()

def test_mixed_de_pairwise_collection(pdata_nested):
    coll = pdata_nested.mixed_de(
        group_col="condition",
        contrast_mode="pairwise",
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
    )
    assert "contrasts" in coll
    assert "meta" in coll

# interaction stratum

def test_mixed_de_interaction_stratum_contrast_at():
    pdata, shift = _build_interaction_pdata()
    kwargs = dict(
        formula="expr ~ condition * layer",
        contrast_term="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    df_l5 = pdata.mixed_de(contrast_at={"layer": "L5"}, **kwargs)
    df_l2 = pdata.mixed_de(contrast_at={"layer": "L2"}, **kwargs)
    assert df_l5.loc["P0", "log2fc"] == pytest.approx(shift, abs=0.5)
    assert abs(df_l2.loc["P0", "log2fc"]) < 0.5

def test_mixed_de_interaction_requires_contrast_at(pdata_nested):
    obs = pdata_nested.prot.obs.copy()
    obs["layer"] = np.where(obs.index.str.endswith(("0", "1", "2")), "L2", "L5")
    pdata_nested.prot.obs = obs
    with pytest.raises(ValueError, match="contrast_at"):
        pdata_nested.mixed_de(
            formula="expr ~ condition * layer",
            contrast_term="condition",
            contrast=("disease", "control"),
            donor_col="donor",
            observation_level="pseudobulk",
            store=False,
        )

# API validation

def test_mixed_de_mutual_exclusion(pdata_nested):
    with pytest.raises(ValueError, match="simple path"):
        pdata_nested.mixed_de(
            group_col="condition",
            contrast=("disease", "control"),
            formula="expr ~ condition",
            contrast_term="condition",
            donor_col="donor",
            store=False,
        )

def test_mixed_de_missing_contrast_term_with_formula(pdata_nested):
    with pytest.raises(ValueError, match="contrast_term"):
        pdata_nested.mixed_de(
            formula="expr ~ condition",
            contrast=("disease", "control"),
            donor_col="donor",
            store=False,
        )

def test_mixed_de_empty_subset_raises(pdata_nested):
    with pytest.raises(ValueError, match="No observations remain"):
        pdata_nested.mixed_de(
            group_col="condition",
            contrast=("disease", "control"),
            donor_col="donor",
            subset={"cell_type": "Microglia"},
            store=False,
        )

def test_mixed_de_contrast_level_absent_after_subset():
    pdata, _ = _build_nested_pdata()
    obs = pdata.prot.obs.copy()
    obs.loc[obs["condition"] == "control", "condition"] = "disease"
    pdata.prot.obs = obs
    with pytest.raises(ValueError, match="Contrast level"):
        pdata.mixed_de(
            group_col="condition",
            contrast=("disease", "control"),
            donor_col="donor",
            store=False,
        )

# Plumbing / integration (existing)

def test_resolve_obs_meta_sample_column_same_index_as_obs():
    """PD-style .summary with Sample column must not wipe obs metadata."""
    obs = pd.DataFrame(
        {
            "Sample": ["Sample"] * 4,
            "cellline": ["BE", "BE", "AS", "AS"],
            "condition": ["kd", "sc", "kd", "sc"],
        },
        index=["F1", "F2", "F3", "F4"],
    )
    adata = AnnData(X=np.zeros((4, 2)), obs=obs)
    summary = obs.copy()
    meta = mixed_de_utils.resolve_obs_meta(
        adata, summary, ["cellline", "condition"]
    )
    assert meta["cellline"].tolist() == ["BE", "BE", "AS", "AS"]
    assert meta["condition"].tolist() == ["kd", "sc", "kd", "sc"]

def test_format_mixed_de_formula_display_helpers():
    assert mixed_de_utils.format_mixed_de_formula_display(
        fixed_formula="expr ~ cellline",
        observation_level="cells",
        random_effects="intercept",
        donor_col="animal",
        group_col="cellline",
        slope_col="cellline",
    ) == "expr ~ cellline + (1 | animal)"
    assert mixed_de_utils.format_mixed_de_formula_display(
        fixed_formula="expr ~ cellline + batch",
        observation_level="cells",
        random_effects="intercept_slope",
        donor_col="animal",
        group_col="cellline",
        slope_col="cellline",
    ) == "expr ~ cellline + batch + (1 + cellline | animal)"
    assert mixed_de_utils.format_mixed_de_formula_display(
        fixed_formula="expr ~ cellline",
        observation_level="pseudobulk",
        random_effects="intercept",
        donor_col="animal",
        group_col="cellline",
        slope_col="cellline",
    ) == "Pseudobulk: donor-averaged by animal × cellline"
    assert mixed_de_utils.format_mixed_de_donor_covariates_line(
        fixed_covariates=None,
        random_effects="intercept",
        donor_col="donor",
        observation_level="cells",
    ) == "Donor blocking: donor (intercept) | Fixed covariates: (none)"
    assert mixed_de_utils.format_mixed_de_donor_covariates_line(
        fixed_covariates=["batch"],
        random_effects="intercept_slope",
        donor_col="animal",
        observation_level="pseudobulk",
    ) == "Donor blocking: animal (pseudobulk averaging) | Fixed covariates: batch"

def test_summarize_per_feature_testing_auto_shows_zero_mixed_model():
    df = pd.DataFrame(
        {
            "de_method": ["pseudobulk_unpaired", "pseudobulk_unpaired", "pseudobulk_unpaired"],
            "p_value": [0.01, 0.02, np.nan],
            "mixedlm_failure_reason": ["not_converged", "not_converged", "not_converged"],
            "de_failure_reason": [None, None, "insufficient_donors_per_arm"],
        }
    )
    testing_line, failure_line = mixed_de_utils.summarize_per_feature_testing(
        df,
        method="auto",
        observation_level="cells",
    )
    assert testing_line == (
        "Per-feature testing (auto): mixed model=0, "
        "pseudobulk (unpaired)=2, failed=1"
    )
    assert failure_line == (
        "Failure details: mixed model did not converge=3, "
        "fewer than 2 donors per group=1"
    )

def test_format_mixed_de_comparing_line():
    line = mixed_de_utils.format_mixed_de_comparing_line(
        comparing="AS vs BE",
        group_sizes_before="6 vs 6 samples",
        path="simple",
        contrast_mode="specified",
    )
    assert line == "Comparing: AS vs BE | 6 vs 6 samples (before filter) | simple / specified"

def test_mixed_de_simple_path(pdata_nested):
    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        fixed_covariates=["batch"],
        subset={"cell_type": "Astrocyte"},
        observation_level="pseudobulk",
        store=False,
    )
    assert df.attrs.get("mixed_de", {}).get("path") == "simple"
    assert np.isfinite(df.loc["P0", "p_value"])

def test_format_comparing_groups_helpers():
    contrasts = [("A", "B", "B vs A"), ("A", "C", "C vs A")]
    assert (
        mixed_de_utils.format_comparing_groups(
            contrast_mode="specified",
            contrasts=contrasts,
            values=[{"region": "Cortex"}, {"region": "SNpc"}],
            focal_level=None,
        )
        == "Cortex vs SNpc"
    )
    assert mixed_de_utils.format_comparing_groups(
        contrast_mode="pairwise",
        contrasts=contrasts,
        values=None,
        focal_level=None,
    ).startswith("pairwise:")
    assert "one_vs_rest" in mixed_de_utils.format_comparing_groups(
        contrast_mode="one_vs_rest",
        contrasts=contrasts,
        values=None,
        focal_level="B",
    )

def test_mixed_de_values_sugar(pdata_nested, pdata_nested_meta):
    """values sugar recovers planted disease effect with correct sign (both paths)."""
    shift = pdata_nested_meta["condition_shift"]
    hit = pdata_nested_meta["condition_feature"]

    df_pb = pdata_nested.mixed_de(
        values=[{"condition": "disease"}, {"condition": "control"}],
        donor_col="donor",
        subset={"cell_type": "Astrocyte"},
        observation_level="pseudobulk",
        store=False,
    )
    assert df_pb["contrast"].iloc[0] == "disease vs control"
    assert df_pb.loc[hit, "log2fc"] == pytest.approx(shift, abs=0.8)

    df_lmm = pdata_nested.mixed_de(
        values=[{"condition": "disease"}, {"condition": "control"}],
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df_lmm.loc[hit, "de_method"] == "mixedlm"
    assert df_lmm.loc[hit, "log2fc"] == pytest.approx(shift, abs=0.8)
    assert df_lmm.loc[hit, "p_value"] < 0.05

def test_mixed_de_values_sugar_stats_key_order():
    pdata = _build_integer_donor_region_pdata()
    pdata.mixed_de(
        values=[{"region": "Cortex"}, {"region": "SNpc"}],
        donor_col="donor",
        observation_level="pseudobulk",
        store=True,
        min_detected_fraction=0.0,
        min_cells_detected=1,
    )
    keys = [k for k in pdata.stats if k.startswith("mixed:")]
    assert any(k.startswith("mixed: Cortex vs SNpc |") for k in keys)

def test_mixed_de_stores_stats_key(pdata_nested, capsys):
    pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        subset={"cell_type": "Astrocyte"},
        observation_level="pseudobulk",
    )
    keys = [k for k in pdata_nested.stats if k.startswith("mixed:")]
    assert len(keys) == 1
    assert "contrast" in pdata_nested.stats[keys[0]].columns
    out = capsys.readouterr().out
    assert "Mixed DE complete" in out
    assert "Comparing:" in out
    assert "before filter" in out
    assert "Formula: Pseudobulk:" in out
    assert "Donor blocking: donor (pseudobulk averaging)" in out
    assert "Mixed DE design diagnostics" in out
    assert "Group sizes after filtering:" in out
    assert "Features:" in out
    assert "Per-feature testing (auto):" in out
    assert "Columns:" in out
    assert ".attrs['mixed_de']" in out
    assert "return_diagnostics=True" not in out
    assert "Upregulated:" in out
    assert "Downregulated:" in out
    assert "Not comparable:" in out
    # No blank lines between USER / INFO / complete headers
    assert "\n\nℹ️" not in out and "INFO] Mixed DE design diagnostics" in out
    assert "\n\n     ✅" not in out

def test_mixed_de_peptide_on(pdata_nested):
    pdata_nested.pep = pdata_nested.prot.copy()
    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        on="peptide",
        observation_level="pseudobulk",
        store=False,
    )
    assert len(df) == pdata_nested.pep.n_vars

def test_plot_volcano_stats_key(pdata_nested):
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt

    pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
    )
    key = next(k for k in pdata_nested.stats if k.startswith("mixed:"))
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, pdata=pdata_nested, stats_key=key, group_annot=False)
    plt.close(fig)

def test_plot_volcano_mixed_de_group_annot_labels(pdata_nested):
    """Group bubbles should resolve from contrast when plotting via de_data=."""
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt

    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        subset={"cell_type": "Astrocyte"},
        observation_level="pseudobulk",
        min_detected_fraction=0.05,
        min_cells_detected=2,
        store=False,
    )
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, correct_fdr=True, group_annot=True, label=[0, 0])
    texts = [t.get_text() for t in ax.texts]
    assert "disease" in texts
    assert "control" in texts
    mixed = next(t for t in ax.texts if str(t.get_text()).startswith("Mixed DE"))
    assert mixed.get_position()[1] == pytest.approx(1.25)
    plt.close(fig)

def test_plot_volcano_mixed_subtitle_pos_and_fontsize(pdata_nested):
    """mixed_xy / mixed_kwargs override the Mixed DE subtitle annotation."""
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt

    df = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        min_detected_fraction=0.05,
        min_cells_detected=2,
        store=False,
    )
    fig, ax = plt.subplots()
    scplt.plot_volcano(
        ax,
        de_data=df,
        correct_fdr=True,
        group_annot=True,
        label=[0, 0],
        group_annot_kwargs={
            "pos": {"mixed_xy": (0.4, 1.25)},
            "mixed_kwargs": {"fontsize": 11},
        },
    )
    mixed = next(t for t in ax.texts if str(t.get_text()).startswith("Mixed DE"))
    assert mixed.get_position() == pytest.approx((0.4, 1.25))
    assert mixed.get_fontsize() == pytest.approx(11)
    plt.close(fig)

def test_mixed_de_values_matches_de_volcano_convention(pdata_nested, pdata_nested_meta):
    """values=[g1, g2] must match de(): log2fc = g1−g2; volcano g1=right, g2=left.

    Same call style as ``pdata.de(values=[...])`` / ``plot_volcano(..., values=...)``:
    positive log2fc (upregulated) means higher in values[0], annotated on the right.
    """
    import matplotlib.pyplot as plt
    from scpviz import plotting as scplt

    hit = pdata_nested_meta["condition_feature"]
    shift = pdata_nested_meta["condition_shift"]
    values = [{"condition": "disease"}, {"condition": "control"}]

    df_values = pdata_nested.mixed_de(
        values=values,
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    # de()-compatible: label and sign
    assert df_values["contrast"].iloc[0] == "disease vs control"
    assert df_values.attrs["mixed_de"]["contrast"] == ("disease", "control")  # (test, ref)
    assert df_values.loc[hit, "log2fc"] == pytest.approx(shift, abs=0.8)

    # Explicit contrast=(test, ref) matches values order for the same comparison
    df_contrast = pdata_nested.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df_contrast["contrast"].iloc[0] == "disease vs control"
    assert df_contrast.loc[hit, "log2fc"] == pytest.approx(df_values.loc[hit, "log2fc"], abs=1e-6)

    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df_values, correct_fdr=True, group_annot=True, label=[0, 0])
    by_text = {t.get_text(): t for t in ax.texts}
    assert "disease" in by_text and "control" in by_text
    # Default volcano: group1 (values[0]) at x=0.98 (right), group2 at x=0.02 (left)
    assert by_text["disease"].get_position()[0] == pytest.approx(0.98)
    assert by_text["control"].get_position()[0] == pytest.approx(0.02)
    plt.close(fig)

def test_resolve_values_sugar_order_matches_de():
    """resolve_values_sugar maps values=[g1, g2] → contrast=(g1, g2)=(test, ref)."""
    group_col, contrast, subset = mixed_de_utils.resolve_values_sugar(
        [{"region": "Cortex"}, {"region": "SNpc"}],
        group_col=None,
        contrast=None,
        subset=None,
    )
    assert group_col == "region"
    assert contrast == ("Cortex", "SNpc")  # (test, ref)
    assert mixed_de_utils.contrast_label_from_pair(*contrast) == "Cortex vs SNpc"
    assert subset is None

def test_resolve_values_sugar_warns_when_contrast_disagrees(capsys):
    group_col, contrast, _ = mixed_de_utils.resolve_values_sugar(
        [{"region": "Cortex"}, {"region": "SNpc"}],
        group_col=None,
        contrast=("SNpc", "Cortex"),  # opposite of values sugar
        subset=None,
    )
    assert group_col == "region"
    assert contrast == ("SNpc", "Cortex")  # explicit wins
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "values" in out.lower()
    assert "contrast" in out.lower()

def test_format_group_sizes_uses_group_col_not_first_dict_key():
    meta = pd.DataFrame(
        {
            "cell_type": ["Astrocyte"] * 5 + ["Astrocyte"] * 3,
            "condition": ["disease"] * 5 + ["control"] * 3,
        }
    )
    sizes = mixed_de_utils.format_group_sizes(
        meta,
        "condition",
        contrast_mode="specified",
        contrasts=[("disease", "control", "disease vs control")],
        values=[
            {"cell_type": "Astrocyte", "condition": "disease"},
            {"cell_type": "Astrocyte", "condition": "control"},
        ],
    )
    assert sizes == "5 vs 3 samples"


# spike monotonicity, composition, filtering, slopes

def test_mixed_de_spike_in_monotonic_neglog10p():
    """syn_spike_in: larger δ → stronger −log10(p) on P0."""
    deltas = [0.8, 1.5, 2.5, 3.5]
    scores = []
    for d in deltas:
        pdata, _ = _build_nested_pdata(condition_shift=d, noise_scale=0.35, cells_per_stratum=10)
        df = pdata.mixed_de(
            group_col="condition",
            contrast=("disease", "control"),
            donor_col="donor",
            observation_level="pseudobulk",
            store=False,
            min_detected_fraction=0.05,
            min_cells_detected=2,
        )
        scores.append(-np.log10(df.loc["P0", "p_value"]))
    assert scores == sorted(scores)
    assert scores[-1] > scores[0] + 0.5

def test_mixed_de_filter_after_subset_excludes_sparse_protein():
    """protein passes globally but fails detection threshold in subset."""
    pdata = _build_sparse_subset_pdata()
    df_all = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.2,
        min_cells_detected=3,
    )
    assert np.isfinite(df_all.loc["P_sparse", "p_value"])

    df_sub = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        subset={"cell_type": "Microglia"},
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.2,
        min_cells_detected=3,
    )
    assert np.isnan(df_sub.loc["P_sparse", "p_value"])
    assert int(np.isfinite(df_sub["p_value"]).sum()) < int(np.isfinite(df_all["p_value"]).sum())

def test_mixed_de_min_detected_fraction_excludes_sparse_feature():
    """globally sparse feature excluded at high min_detected_fraction."""
    pdata = _build_sparse_subset_pdata()
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.8,
        min_cells_detected=3,
    )
    assert np.isnan(df.loc["P_sparse", "p_value"])
    assert np.isfinite(df.loc["P1", "p_value"])

def test_mixed_de_composition_negative_control():
    """proportion shift causes false positive pooled; subset removes it."""
    pdata = _build_composition_pdata()
    df_all = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df_all.loc["P0", "p_value"] < 0.05

    df_a = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        subset={"cell_type": "A"},
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df_a.loc["P0", "p_value"] > 0.05
    assert abs(df_a.loc["P0", "log2fc"]) < 0.5

def test_mixed_de_additive_vs_interaction_estimand_differ():
    """
    Use method='auto' so a non-convergent cell-level LMM falls back to
    pseudobulk (CI platforms differ in MixedLM convergence); the estimand
    contrast still holds.
    """
    pdata, shift = _build_additive_vs_interaction_pdata()
    df_simple = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="cells",
        method="auto",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    df_l5 = pdata.mixed_de(
        formula="expr ~ condition * layer",
        contrast_term="condition",
        contrast=("disease", "control"),
        contrast_at={"layer": "L5"},
        donor_col="donor",
        observation_level="cells",
        method="auto",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert np.isfinite(df_simple.loc["P0", "log2fc"])
    assert df_l5.loc["P0", "log2fc"] == pytest.approx(shift, abs=0.6)
    assert abs(df_simple.loc["P0", "log2fc"]) < abs(df_l5.loc["P0", "log2fc"])
    assert abs(df_simple.loc["P0", "log2fc"]) < shift * 0.75

def test_mixed_de_random_slope_recovers_mean_effect():
    """intercept_slope model recovers population mean condition effect."""
    pdata, mean_slope = _build_random_slopes_pdata()
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        random_effects="intercept_slope",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df.loc["P0", "log2fc"] == pytest.approx(mean_slope, abs=0.8)
    assert df.loc["P0", "p_value"] < 0.05

def test_mixed_de_single_donor_lmm_returns_nan_or_warns(capsys):
    """single donor — no silent meaningful donor LMM inference."""
    pdata = _build_single_donor_pdata()
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="cells",
        method="mixedlm",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert df.attrs["mixed_de"]["n_donors_total"] == 1
    assert not np.isfinite(df.loc["P0", "p_value"]) or df.loc["P0", "p_value"] > 0.05

def test_mixed_de_fdr_scope_per_contrast_differs_from_global():
    """per-contrast and global BH columns both present; can differ per row."""
    pdata, _ = _build_k3_pdata()
    coll = pdata.mixed_de(
        group_col="group",
        contrast_mode="pairwise",
        donor_col="donor",
        observation_level="pseudobulk",
        correct_fdr=True,
        fdr_scope="both",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    for df in coll["contrasts"].values():
        assert "adj_p_value" in df.columns
        assert "adj_p_value_global" in df.columns
        assert df["adj_p_value_global"].notna().any()
    diffs = [
        coll["contrasts"][k].loc["P0", "adj_p_value"]
        != coll["contrasts"][k].loc["P0", "adj_p_value_global"]
        for k in coll["contrasts"]
    ]
    assert any(diffs)

def test_mixed_de_subsample_caps_cells_per_stratum():
    """subsample respects max_cells_per_stratum per donor × condition."""
    pdata, _ = _build_nested_pdata(cells_per_stratum=40)
    adata = pdata.prot
    meta = adata.obs.copy()
    expr = np.asarray(adata.X)
    rng = np.random.default_rng(0)
    meta_sub, _ = mixed_de_utils.subsample_observations(
        meta,
        expr,
        donor_col="donor",
        group_col="condition",
        max_cells_per_stratum=10,
        rng=rng,
    )
    counts = meta_sub.groupby(["donor", "condition"], observed=True).size()
    assert counts.max() <= 10
    assert len(meta_sub) < len(meta)

def test_mixed_de_donor_only_null_naive_also_null():
    """balanced donor mix → even naive cell test is null without condition effect."""
    pdata = _build_donor_only_null_pdata()
    naive_p = _naive_cell_level_pvalue(pdata, "P0")
    df = pdata.mixed_de(
        group_col="condition",
        contrast=("disease", "control"),
        donor_col="donor",
        observation_level="pseudobulk",
        store=False,
        min_detected_fraction=0.05,
        min_cells_detected=2,
    )
    assert naive_p > 0.05
    assert df.loc["P0", "p_value"] > 0.05

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scpviz import pAnnData, utils
import scipy.sparse
from copy import deepcopy

@pytest.fixture
def pdata_preprocessing():
    X = np.array([
        [1,    np.nan, 10,   100, 500, 2.0],
        [2,    20,     np.nan, 200, 500, 2.5],
        [np.nan, 30,   30,   np.nan, 500, 3.0],
        [100,  np.nan, 1000, 500, 500, 2.8],
        [200,  400,     np.nan, np.nan, 500, 2.2],
        [np.nan, 600,  3000, 1500, 500, 2.1],
    ])

    obs = pd.DataFrame({
        "cellline": ["BE", "BE", "BE", "AS", "AS", "AS"],
        "treatment": ["kd", "kd", "kd", "sc", "sc", "sc"]
    }, index=[f"sample{i+1}" for i in range(6)])

    var = pd.DataFrame({
        "Genes": ["GAPDH", "ACTB", "TUBB", "MYH9", "HSP90", "RPLP0"]
    }, index=[f"P{i+1}" for i in range(6)])

    ann = AnnData(X=X, obs=obs, var=var)
    return pAnnData(prot=ann)

# test impute

def test_impute_mean_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(classes=["cellline", "treatment"], method="mean")
    imputed = pdata.prot.X

    # Check all NaNs have been filled
    assert not np.isnan(imputed).any(), "There should be no NaNs after mean group-wise imputation."
    assert np.isclose(imputed[0, 1], 25.0) # BE_kd1 was missing P2 → (20 + 30) / 2 = 25
    assert np.isclose(imputed[1, 2], 20.0) # BE_kd2 missing P3 → (10 + 30) / 2 = 20
    assert np.isclose(imputed[2, 0], 1.5) # BE_kd3 missing P1 → (1 + 2) / 2 = 1.5
    assert np.isclose(imputed[3, 1], 500.0) # AS_sc1 missing P2 → (400 + 600) / 2 = 500

def test_impute_median_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(classes=["cellline", "treatment"], method="median")
    imputed = pdata.prot.X

    # Ensure all imputable NaNs are filled
    assert not np.isnan(imputed).any(), "There should be no NaNs after median group-wise imputation."
    assert np.isclose(imputed[0, 1], 25.0)    # BE_kd1 was missing P2 → median of [20, 30] = 25
    assert np.isclose(imputed[1, 2], 20.0)    # BE_kd2 was missing P3 → median of [10, 30] = 20
    assert np.isclose(imputed[2, 0], 1.5)     # BE_kd3 was missing P1 → median of [1, 2] = 1.5
    assert np.isclose(imputed[3, 1], 500.0)   # AS_sc1 was missing P2 → median of [400, 600] = 500
    assert np.isclose(imputed[4, 2], 2000.0)  # AS_sc2 was missing P3 → median of [1000, 3000] = 2000
    assert np.isclose(imputed[4, 3], 1000.0)  # AS_sc2 was missing P4 → median of [500, 1500] = 1000

def test_impute_min_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(classes=["cellline", "treatment"], method="min")
    imputed = pdata.prot.X

    assert not np.isnan(imputed).any(), "There should be no NaNs after median group-wise imputation."

def test_impute_knn(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(method="knn", n_neighbors=2)
    imputed = pdata.prot.X

    # Check shape and that no NaNs remain
    assert imputed.shape == pdata.prot.shape, "Shape mismatch after KNN imputation."
    assert not np.isnan(imputed).any(), "There should be no NaNs after KNN imputation."

    val = imputed[0, 1]  # P2 for BE_kd1, should be between 20 and 30
    assert 20 <= val <= 30, f"KNN-imputed value {val} out of expected range"

def test_impute_min_sparse(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Convert to sparse before imputation
    from scipy import sparse
    
    pdata.prot.X = sparse.csr_matrix(pdata.prot.X)
    original = pdata_preprocessing.prot.X.toarray() if sparse.issparse(pdata_preprocessing.prot.X) else pdata_preprocessing.prot.X.copy()
    pdata.impute(method="min")
    imputed = pdata.prot.X

    # Check result is still sparse and no NaNs remain
    assert sparse.issparse(imputed), "Output should be sparse after min imputation."
    assert not np.isnan(imputed.toarray()).any(), "There should be no NaNs after min imputation."
    
    group = original[[0, 1, 2], :]
    expected = np.nanmin(group, axis=0)[1]  # min for P2 in BE_kd
    assert np.isclose(imputed.toarray()[0, 1], expected)

def test_impute_invalid_method(pdata_preprocessing):
    pdata = pdata_preprocessing
    with pytest.raises(ValueError, match="Unsupported method"):
        pdata.impute(method="invalid_method")

def test_impute_set_X_overwrites(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Save original .X
    original = pdata.prot.X.copy()
    pdata.impute(method="mean")
    # Check if .X was overwritten
    assert not np.allclose(original, pdata.prot.X), "Expected .X to be updated after imputation."

def test_impute_median_groupwise_skips_allnan_feature(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Set entire column (P6) to NaN in BE_kd group
    pdata.prot.X[0:3, 5] = np.nan

    pdata.impute(classes=["cellline", "treatment"], method="median")
    imputed = pdata.prot.X

    # Check that P6 in BE_kd group is still NaN
    assert np.isnan(imputed[0:3, 5]).all(), "All-NaN feature in group should remain NaN after median imputation."

    # Check that AS_sc group (samples 3–5) still imputed P6 correctly
    assert not np.isnan(imputed[3:, 5]).any(), "Non-empty group should have values imputed."

def test_impute_median_global_skips_allnan_feature(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Set entire column (e.g., P6) to NaN globally
    pdata.prot.X[:, 5] = np.nan

    pdata.impute(method="median")  # global median imputation
    imputed = pdata.prot.X

    # Check that P6 is still all NaN
    assert np.isnan(imputed[:, 5]).all(), "All-NaN feature should remain NaN after global median imputation."

    # Check that other missing values were imputed
    assert not np.isnan(imputed[:, :5]).any(), "All imputable values should be filled."

@pytest.fixture
def pdata_preprocessing_make():
    def _make():
        X = np.array([
            [1,    np.nan, 10,   100, 500, 2.0],
            [2,    20,     np.nan, 200, 500, 2.5],
            [np.nan, 30,   30,   np.nan, 500, 3.0],
            [100,  np.nan, 1000, 500, 500, 2.8],
            [200,  400,     np.nan, np.nan, 500, 2.2],
            [np.nan, 600,  3000, 1500, 500, 2.1],
        ])

        obs = pd.DataFrame({
            "cellline": ["BE", "BE", "BE", "AS", "AS", "AS"],
            "treatment": ["kd", "kd", "kd", "sc", "sc", "sc"]
        }, index=[f"sample{i+1}" for i in range(6)])

        var = pd.DataFrame({
            "Genes": ["GAPDH", "ACTB", "TUBB", "MYH9", "HSP90", "RPLP0"]
        }, index=[f"P{i+1}" for i in range(6)])

        ann = AnnData(X=X, obs=obs, var=var)
        return pAnnData(prot=ann)

    return _make

def test_impute_uses_zeros_as_nans(pdata_preprocessing_make):
    col = 5

    pdata1 = pdata_preprocessing_make()
    pdata1.prot.X[:, col] = 0
    pdata1.impute(method="median", use_zeros_as_nan=True)
    imputed1 = pdata1.prot.X[:, col]

    pdata2 = pdata_preprocessing_make()
    pdata2.prot.X[:, col] = np.nan
    pdata2.impute(method="median")
    imputed2 = pdata2.prot.X[:, col]

    assert np.allclose(imputed1, imputed2, equal_nan=True)
    assert np.isnan(imputed1).all()

def test_impute_knn_groupwise_raises(pdata_preprocessing):
    pdata = pdata_preprocessing
    with pytest.raises(ValueError, match="KNN imputation is not supported for group-wise"):
        pdata.impute(classes='cellline',method="knn", n_neighbors=2)

# pimmslearn uses seaborn._BarPlotter (removed in seaborn>=0.13); skip if import fails
_pimms_available = False
try:
    import pimmslearn
    _pimms_available = True
except Exception:
    pass

@pytest.mark.skipif(not _pimms_available, reason="pimmslearn not importable (seaborn>=0.13 incompatibility in pimmslearn)")
@pytest.mark.parametrize("method", ["pimms_dae", "pimms_vae"])
def test_impute_pimms_vae_global(pdata, monkeypatch, method):
    """Test PIMMS DAE/VAE imputation using mocked AETransformer."""
    class MockAE:
        def __init__(self, *args, **kwargs):
            self.fit_called = False

        def fit(self, df, cuda=False, epochs_max=100):
            self.fit_called = True

        def transform(self, df):
            # trivial "imputed" output
            out = df.copy().astype(float)
            out[out.isna()] = 9.999
            return out

    monkeypatch.setattr(
        "pimmslearn.sklearn.ae_transformer.AETransformer",
        MockAE
    )

    X = pdata.prot.X.copy()
    X[1, :3] = np.nan
    pdata.prot.X = X
    pdata.impute(method=method, on="protein", use_zeros_as_nan=True)   # method is pimms_dae or pimms_vae

    layer = f"X_impute_{method}"
    out = pdata.prot.layers[layer]
    dense = out.toarray() if hasattr(out, "toarray") else out

    expected = (2 ** 9.999) - 1
    assert np.allclose(dense[1, :3], expected)

@pytest.mark.skipif(not _pimms_available, reason="pimmslearn not importable (seaborn>=0.13 incompatibility in pimmslearn)")
def test_impute_pimms_cf_global(pdata, monkeypatch):
    class MockCF:
        def __init__(self, *args, **kwargs):
            self.fit_called = False

        def fit(self, series, cuda=False, epochs_max=20):
            self.fit_called = True
            self.index = series.index  # store MultiIndex

        def transform(self, series):
            # Make a float series
            filled = series.astype(float).copy()

            # Fill only NaNs (this is the imputation behavior we want to test)
            filled[filled.isna()] = 9.999

            # IMPORTANT: return a Series with the full MultiIndex
            return filled.reindex(self.index)


    monkeypatch.setattr(
        "pimmslearn.sklearn.cf_transformer.CollaborativeFilteringTransformer",
        MockCF
    )

    arr = pdata.prot.X.copy()
    arr[1, :3] = np.nan
    pdata.prot.X = arr

    pdata.impute(method="pimms_cf", on="protein")

    layer = "X_impute_pimms_cf"
    out = pdata.prot.layers[layer]

    dense_out = out.toarray() if hasattr(out, "toarray") else out
    fill_value = 9.999
    expected = (2 ** fill_value) - 1
    assert np.allclose(dense_out[1, :3], expected, equal_nan=False)

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_impute_raises_if_layer_not_found(pdata, on):
    with pytest.raises(ValueError, match="Layer 'X_invalid' not found"):
        pdata.impute(on=on, layer="X_invalid")

# test normalize()

def test_normalize_sum(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="sum", set_X=True)
    norm = pdata.prot.X
    row_sums = np.nansum(norm, axis=1)
    assert np.allclose(row_sums, row_sums[0]), "All row sums should match after sum normalization"

def test_normalize_sum_use_nonmissing(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="sum", set_X=False, use_nonmissing=True)
    norm = pdata.prot.layers["X_norm_sum"]
    assert norm.shape == pdata.prot.shape
    assert np.isclose(norm[0, 0], 1.00199, atol=1e-3)
    assert np.isclose(norm[0, 2], 10.0199, atol=1e-3)
    assert np.isclose(norm[0, 4], 500.996, atol=1e-3)

def test_normalize_reference_feature_by_gene_name(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="reference_feature", reference_columns=["GAPDH", "ACTB"], reference_method="mean", set_X=False)
    norm = pdata.prot.layers["X_norm_reference_feature"]
    assert norm.shape == pdata.prot.shape
    assert np.isclose(norm[0, 0], 200.0)
    assert np.isclose(norm[0, 2], 2000.0)
    assert np.isclose(norm[0, 4], 100000.0)

def test_normalize_reference_feature_by_index(pdata_preprocessing):
    pdata_preprocessing.normalize(
        method="reference_feature",
        reference_columns=["P1", "P2"],  # raw indices
        reference_method="mean",
        set_X=False
    )
    norm = pdata_preprocessing.prot.layers["X_norm_reference_feature"]
    assert np.isclose(norm[0, 0], 200.0)

def test_normalize_median_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="median", classes=["cellline", "treatment"], set_X=False)
    norm = pdata.prot.layers["X_norm_median"]

    # Check scaling applied correctly within BE_kd
    assert np.isclose(norm[0, 0], 3.0)
    assert np.isclose(norm[0, 2], 30.0)
    assert np.isclose(norm[0, 5], 6.0)

    # Check that AS_sc group is also scaled correctly
    assert np.isclose(norm[3, 2], 1200.0)
    assert np.isclose(norm[4, 1], 800.0)
    assert np.isclose(norm[5, 2], 3000.0)

def test_normalize_reference_feature_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(
        method="reference_feature",
        reference_columns=["GAPDH", "ACTB"],
        reference_method="mean",
        classes=["cellline"],
        set_X=False
    )
    norm = pdata.prot.layers["X_norm_reference_feature"]
    expected = np.array([
        [2.0,      np.nan,   20.0,   200.0, 1000.0,   4.0  ],  # 2.0 × row 1
        [2.5,      25.0,     np.nan, 250.0,  625.0,   3.125],  # 1.25 × row 2
        [np.nan,   30.0,     30.0,   np.nan, 500.0,   3.0  ],  # 1.0 × row 3
        [200.0,    np.nan, 2000.0,  1000.0, 1000.0,   5.6  ],  # 2.0 × row 4
        [250.0,   500.0,     np.nan,  np.nan, 625.0,  2.75 ],  # 1.25 × row 5
        [np.nan,  600.0,   3000.0,  1500.0,  500.0,   2.1  ],  # 1.0 × row 6
    ])
    np.testing.assert_allclose(norm, expected, rtol=1e-4)

def test_normalize_warns_on_bad_rows(pdata_preprocessing, capsys):
    """Test that normalize() detects bad rows and exits early."""
    pdata = pdata_preprocessing

    # Force >50% NaN in one sample to trigger the warning
    pdata.prot.X[0, :] = np.nan  # first row = completely missing

    # Run normalization (without force=True) — should trigger early return
    pdata.normalize(method="sum")

    # Capture printed output
    captured = capsys.readouterr().out

    # Assert that the warning message was printed
    assert "have >50% missing values" in captured
    assert "Use `force=True` to proceed" in captured

    # Also assert that normalization did not proceed (layer not created)
    assert "X_norm_sum" not in pdata.prot.layers

def test_normalize_force_bad_rows(pdata_preprocessing, capsys):
    """Test that normalize(force=True) proceeds despite bad rows."""
    pdata = pdata_preprocessing
    pdata.prot.X[0, :] = np.nan

    pdata.normalize(method="sum", force=True)

    captured = capsys.readouterr().out
    assert "Proceeding with normalization despite bad rows" in captured
    assert "X_norm_sum" in pdata.prot.layers

def test_normalize_directlfq(pdata):
    pdata.normalize(method='directlfq')
    assert 'X_norm_directlfq' in pdata.prot.layers

def test_normalize_directlfq_strict_true(pdata):
    pdata.normalize(method='directlfq', strict=True)
    assert 'X_norm_directlfq' in pdata.prot.layers

def test_normalize_directlfq_nopep_raises(pdata_nopep):
    """Test that directLFQ raises a clear error when peptide-level data is missing."""
    with pytest.raises(ValueError, match="Peptide-level data not found"):
        pdata_nopep.normalize(method='directlfq')

def test_normalize_robust_scale(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method='robust_scale')
    assert True

def test_normalize_quantile_transform(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method='quantile_transform')
    assert True

def test_normalize_invalid_method(pdata_preprocessing):
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        pdata_preprocessing.normalize(method="unknown_method")

# test de()

@pytest.mark.parametrize("fold_change_mode", ["mean", "pairwise_median"])
@pytest.mark.parametrize("test", ["ttest", "mannwhitneyu", "wilcoxon"])
def test_de_passes_on_valid_inputs(pdata, fold_change_mode, test):
    df = pdata.de(
        values=[
            {"cellline": "BE", "treatment": "kd"},
            {"cellline": "AS", "treatment": "sc"}
        ],
        method=test,
        fold_change_mode=fold_change_mode
    )
    assert isinstance(df, pd.DataFrame)
    assert "p_value" in df.columns
    assert "adj_p_value" not in df.columns
    assert "log2fc" in df.columns
    assert "[{'cellline': 'BE', 'treatment': 'kd'}]" in df.columns
    assert "[{'cellline': 'AS', 'treatment': 'sc'}]" in df.columns

def test_de_raises_on_invalid_foldchange(pdata):
    with pytest.raises(ValueError, match="Unsupported fold_change_mode"):
        pdata.de(
            values=[
                {"cellline": "BE"},
                {"cellline": "AS"}
            ],
            fold_change_mode="bogus"
        )

def test_de_raises_on_single_class(pdata):
    with pytest.raises(ValueError, match="provide two distinct groups"):
        pdata.de(values=[{"cellline": "BE"}, {"cellline": "BE"}])

def test_de_with_pep_pairwise_warns_if_no_pep(pdata_nopep):
    with pytest.raises(ValueError, match="Peptide-level data | required"):
        pdata_nopep.de(
            values=[{"cellline": "BE"}, {"cellline": "AS"}],
            fold_change_mode="pep_pairwise_median"
        )

def test_de_with_pep_pairwise_median(pdata):
    df = pdata.de(
            values=[
                {"cellline": "BE"},
                {"cellline": "AS"}
            ],
            fold_change_mode="pep_pairwise_median"        
    )

    assert isinstance(df, pd.DataFrame)
    assert "p_value" in df.columns
    assert "log2fc" in df.columns    

def test_de_ignores_inf_foldchange_in_annotations(pdata):
    pdata = deepcopy(pdata)

    X_orig = pdata.prot.X
    if scipy.sparse.issparse(X_orig):
        X_dense = X_orig.toarray()
    else:
        X_dense = X_orig.copy()

    # Get index of target protein (e.g., first column)
    prot_name = pdata.prot.var_names[0]
    prot_idx = pdata.prot.var_names.get_loc(prot_name)

    # Get all sample indices for each group
    group1_idx = np.where(pdata.prot.obs["cellline"] == "AS")[0]
    group2_idx = np.where(pdata.prot.obs["cellline"] == "BE")[0]

    # Inject values to force divide-by-zero log2FC
    X_dense[group1_idx, prot_idx] = 1e5  # large value
    X_dense[group2_idx, prot_idx] = 0    # zero

    # Set the modified matrix back to pdata.prot.X
    pdata.prot.X = scipy.sparse.csr_matrix(X_dense)

    # Run DE
    df = pdata.de(values=[{'cellline': 'AS'}, {'cellline':'BE'}], fold_change_mode="mean")

    print(df)

    assert prot_name in df.index, f"{prot_name} not found in DE result index"

    # Sanity check: log2FC should be +inf
    fc_val = df.loc[prot_name, "log2fc"]
    assert np.isnan(fc_val), f"Expected NaN log2fc for not comparable protein, got {fc_val}"
    assert df.loc[prot_name, "significance"] == "not comparable"

def test_de_correct_fdr_adds_adjusted_columns(pdata):
    df = pdata.de(
        values=[{"cellline": "BE"}, {"cellline": "AS"}],
        correct_fdr=True,
    )
    assert "adj_p_value" in df.columns
    assert "-log10(adj_p_value)" in df.columns
    assert "adj_p_value" not in pdata.de(
        values=[{"cellline": "BE"}, {"cellline": "AS"}],
        correct_fdr=False,
    ).columns

def test_de_deprecated_pval_alias(pdata, capsys):
    values = [{"cellline": "BE"}, {"cellline": "AS"}]
    df_threshold = pdata.de(values=values, threshold=0.05)
    out_before = capsys.readouterr().out
    df_pval = pdata.de(values=values, pval=0.05)
    out_after = capsys.readouterr().out
    assert "deprecated" in out_after.lower()
    assert "pval" in out_after.lower()
    assert df_threshold["significance"].equals(df_pval["significance"])

def test_de_equal_var_welch_changes_pvalues(pdata):
    df_student = pdata.de(
        values=[{"cellline": "BE"}, {"cellline": "AS"}],
        equal_var=True,
    )
    df_welch = pdata.de(
        values=[{"cellline": "BE"}, {"cellline": "AS"}],
        equal_var=False,
    )
    # Welch and Student should differ for at least one protein in typical data
    assert not df_student["p_value"].equals(df_welch["p_value"])

# test cv()

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_cv_computes_single_class(pdata, on):
    pdata.cv(classes="cellline", on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    for cls in ["AS", "BE"]:
        key = f"CV: {cls}"
        assert key in var.columns, f"Missing column {key}"
        assert len(var[key]) == adata.shape[1]
        # Allow NaNs but ensure majority of entries are valid
        valid_fraction = np.isfinite(var[key]).mean()
        assert valid_fraction > 0.8, f"Too many NaNs in {key} ({valid_fraction:.2%} valid)"

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_cv_computes_multi_class(pdata, on):
    pdata.cv(classes=["cellline", "treatment"], on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    expected_keys = [f"CV: {a}_{b}" for a in ["AS", "BE"] for b in ["sc", "kd"]]
    for key in expected_keys:
        assert key in var.columns, f"Missing column {key}"
        valid_fraction = np.isfinite(var[key]).mean()
        assert valid_fraction > 0.8, f"Too many NaNs in {key} ({valid_fraction:.2%} valid)"

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_cv_raises_if_layer_not_found(pdata, on):
    with pytest.raises(ValueError, match="Layer 'X_invalid' not found"):
        pdata.cv(classes="cellline", on=on, layer="X_invalid")

# test rank()

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_computes_single_class(pdata, on):
    pdata.rank(classes="cellline", on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    for cls in ["AS", "BE"]:
        avg_key = f"Average: {cls}"
        std_key = f"Stdev: {cls}"
        rank_key = f"Rank: {cls}"

        # Columns should exist
        assert avg_key in var.columns
        assert std_key in var.columns
        assert rank_key in var.columns

        # Should be the same length as n_features
        assert len(var[avg_key]) == adata.shape[1]

        # Mean and Stdev: at least 90% finite
        assert np.isfinite(var[avg_key]).mean() > 0.9
        assert np.isfinite(var[std_key]).mean() > 0.9

        # Rank: Should be numeric or NaN, mostly finite
        assert var[rank_key].dtype.kind in ("f", "i")
        assert np.isfinite(var[rank_key]).mean() > 0.9

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_computes_multi_class(pdata, on):
    pdata.rank(classes=["cellline", "treatment"], on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    for cls in ["AS_kd", "AS_sc", "BE_kd", "BE_sc"]:
        avg_key = f"Average: {cls}"
        std_key = f"Stdev: {cls}"
        rank_key = f"Rank: {cls}"

        assert avg_key in var.columns
        assert std_key in var.columns
        assert rank_key in var.columns
        assert np.isfinite(var[avg_key]).mean() > 0.9
        assert np.isfinite(var[std_key]).mean() > 0.9
        assert np.isfinite(var[rank_key]).mean() > 0.9

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_raises_if_layer_not_found(pdata, on):
    with pytest.raises(ValueError, match="Layer 'X_invalid' not found"):
        pdata.rank(classes="cellline", on=on, layer="X_invalid")

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_updates_history(pdata, on):
    prev_len = len(pdata._history)
    pdata.rank(classes="cellline", on=on)
    assert len(pdata._history) == prev_len + 1
    assert any("Ranked" in entry for entry in pdata._history[-1:])

def test_check_rankcol_raises_on_missing(pdata):
    with pytest.raises(ValueError, match="class_values must be None"):
        pdata._check_rankcol(on="protein", class_values=None)

# test neighbor()

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_neighbor_default_pca(pdata, on):
    if on == 'protein':
        adata_on='prot'
    else:
        adata_on='pep'
    # Delete existing PCA to force rerun
    getattr(pdata, adata_on).uns.pop("pca", None)
    getattr(pdata, adata_on).obsm.pop("X_pca", None)

    # Run neighbor without existing PCA
    pdata.neighbor(on=on)

    adata = getattr(pdata, adata_on)
    assert "neighbors" in adata.uns
    assert "distances" in adata.obsp
    assert "connectivities" in adata.obsp

    # Make sure distances and connectivities are square and match number of obs
    n = adata.shape[0]
    assert adata.obsp["distances"].shape == (n, n)
    assert adata.obsp["connectivities"].shape == (n, n)

def test_neighbor_custom_rep(pdata):
    # Run PCA to generate a valid rep
    pdata.pca(on="protein")
    adata = pdata.prot

    # Simulate an alternate embedding by copying X_pca to a new key
    adata.obsm["X_pca2"] = adata.obsm["X_pca"][:, :10]  # simulate reduced dims

    # Call neighbor with custom rep
    pdata.neighbor(on="protein", use_rep="X_pca2", n_neighbors=5)

    # Assertions
    assert "neighbors" in adata.uns
    assert "distances" in adata.obsp
    assert "connectivities" in adata.obsp
    assert adata.uns["neighbors"]["params"]["n_neighbors"] == 5
    assert adata.uns["neighbors"]["params"]["use_rep"] == "X_pca2"

def test_neighbor_invalid_rep_raises(pdata):
    with pytest.raises(ValueError, match="not found in obsm"):
        pdata.neighbor(on="protein", use_rep="X_fake")

def test_neighbor_with_layer_switch(pdata):
    from unittest.mock import patch
    pdata.prot.layers["dummy_layer"] = pdata.prot.X.copy()

    with patch.object(pdata, "set_X") as mocked:
        pdata.neighbor(on="protein", layer="dummy_layer")
        mocked.assert_called_once_with(layer="dummy_layer", on="protein")

    # Confirm results stored
    adata = pdata.prot
    assert "neighbors" in adata.uns
    assert "distances" in adata.obsp

# test leiden()

def test_leiden_default_layer(pdata):
    # Make sure no neighbors exist beforehand
    pdata.prot.uns.pop("neighbors", None)

    # Run Leiden clustering with default settings
    pdata.leiden(on="protein", layer="X", resolution=0.5)

    # Check results
    assert "leiden" in pdata.prot.obs.columns
    assert pdata.prot.obs["leiden"].notna().all()
    n_clusters = pdata.prot.obs["leiden"].nunique()
    assert n_clusters >= 1, "Leiden returned no labels"
    if n_clusters == 1:
        print("⚠️ Single cluster detected (likely due to small sample size or Scanpy≥1.10).")

from unittest.mock import patch

def test_leiden_custom_layer(pdata):
    # Add dummy layer and remove neighbors to trigger full path
    pdata.prot.layers["dummy"] = pdata.prot.X.copy()
    pdata.prot.uns.pop("neighbors", None)

    # Patch set_X to confirm it's called
    with patch.object(pdata, "set_X") as mocked:
        pdata.leiden(on="protein", layer="dummy", resolution=0.3)
        mocked.assert_called_with(layer="dummy", on="protein")  # Instead of .assert_called_once_with()
        assert mocked.call_count >= 1

    # Check Leiden results
    assert "leiden" in pdata.prot.obs.columns
    assert pdata.prot.obs["leiden"].notna().all()

def test_leiden_peptide_level(pdata):
    # Only run if .pep exists
    if pdata.pep is None:
        return

    pdata.pep.uns.pop("neighbors", None)
    pdata.leiden(on="peptide", layer="X", resolution=0.4)

    assert "leiden" in pdata.pep.obs.columns
    assert pdata.pep.obs["leiden"].notna().all()

# test umap()

def test_umap_runs_with_default_settings(pdata):
    # Precompute neighbors
    # Run UMAP
    pdata.umap(on="protein", layer="X")
    
    # Assert that UMAP coords are created
    assert "X_umap" in pdata.prot.obsm
    assert "umap" in pdata.prot.uns
    assert pdata.prot.obsm["X_umap"].shape[1] == 2

def test_umap_with_custom_umap_params_and_neighbors(pdata):
    # Remove neighbors to force recompute
    pdata.prot.uns.pop("neighbors", None)

    # Run UMAP with full parameter set
    pdata.umap(
        on="protein", layer="X",
        n_neighbors=10,
        n_pcs=20,
        min_dist=0.1,
        spread=1.0,
        metric="cosine",
        random_state=42
    )

    # Confirm UMAP output
    assert "X_umap" in pdata.prot.obsm
    assert "umap" in pdata.prot.uns
    coords = pdata.prot.obsm["X_umap"]
    assert coords.shape[0] == pdata.prot.n_obs
    assert coords.shape[1] == 2

def test_umap_force_neighbors_detects_changed_X(pdata):
    """
    If the underlying .X is modified, then calling
    umap(force_neighbors=True) should recompute the neighbor graph
    and produce a different UMAP embedding.
    """

    pdata.umap(on="protein", layer="X")
    adata = pdata.prot

    orig_pca = adata.obsm["X_pca"].copy()
    orig_umap = adata.obsm["X_umap"].copy()

    # Modify roughly 1/3 of the matrix
    X = adata.X.toarray()
    rng = np.random.default_rng(0)
    X = rng.normal(size=X.shape) * 1e6

    from scipy import sparse
    adata.X = sparse.csr_matrix(X)

    pdata.umap(on="protein", layer="X", force_neighbors=True)
    new_pca = adata.obsm["X_pca"]
    new_umap = adata.obsm["X_umap"]

    assert not np.allclose(orig_pca, new_pca), \
        "PCA did not change after modifying X and recomputing neighbors"
    assert not np.allclose(orig_umap, new_umap), \
        "UMAP embedding did not change after forcing neighbor + PCA recomputation"

from unittest.mock import patch

def test_umap_with_custom_layer_calls_set_X(pdata):
    # Add dummy layer
    pdata.prot.layers["dummy"] = pdata.prot.X.copy()
    pdata.prot.uns.pop("neighbors", None)

    # Patch set_X
    with patch.object(pdata, "set_X") as mocked:
        pdata.umap(on="protein", layer="dummy", n_neighbors=5)

        # Should call set_X before UMAP
        mocked.assert_called_with(layer="dummy", on="protein") 
        assert mocked.call_count >= 1

    # Confirm UMAP results still created
    assert "X_umap" in pdata.prot.obsm

def test_umap_on_peptide_data(pdata):
    # Confirm .pep exists
    assert pdata.pep is not None

    # Run UMAP on peptides
    pdata.umap(on="peptide", layer="X")

    # Confirm storage
    assert "X_umap" in pdata.pep.obsm
    assert "umap" in pdata.pep.uns

# test harmony
@pytest.mark.xfail(reason="Harmony fails on some Python/sklearn versions (n_clusters=0 bug)")
def test_harmony_runs_with_valid_key(pdata):
    # Sanity check
    assert pdata.prot.obs["cellline"].nunique() >= 2, "Cellline column must have ≥2 unique categories"

    if "X_pca" not in pdata.prot.obsm:
        pdata.pca(on="protein")

    # Run harmony
    pdata.harmony(key="cellline", on="protein")

    # Assertions
    assert "X_pca_harmony" in pdata.prot.obsm
    assert pdata.prot.obsm["X_pca_harmony"].shape[1] > 0
    assert "harmony" in "".join(pdata.history)  # optional

def test_harmony_invalid_key_raises(pdata):
    with pytest.raises(ValueError, match="Batch key 'invalid' not found"):
        pdata.harmony(key="invalid", on="protein")

@pytest.mark.xfail(reason="Harmony fails on some Python/sklearn versions (n_clusters=0 bug)")
def test_harmony_triggers_pca_if_missing(pdata):
    # Remove PCA
    pdata.prot.uns.pop("pca", None)
    pdata.prot.obsm.pop("X_pca", None)

    # Confirm PCA is not present
    assert "pca" not in pdata.prot.uns
    assert "X_pca" not in pdata.prot.obsm

    # Run harmony
    pdata.harmony(key="cellline", on="protein")

    # Confirm PCA was re-run
    assert "pca" in pdata.prot.uns
    assert "X_pca" in pdata.prot.obsm
    assert "X_pca_harmony" in pdata.prot.obsm

# test nanmissing values 

def test_nanmissingvalues_masks_exceeding_features(pdata):
    pdata = deepcopy(pdata)
    adata = pdata.prot

    # Convert to dense if needed
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # Inject missing values
    X[:, 0] = np.nan  # 100% missing → should be fully NaN
    X[:4, 1] = np.nan  # <50% missing → should be preserved
    X[:7, 2] = np.nan  # >50% missing → should be fully NaN

    adata.X = X
    pdata.prot = adata

    # Run filter
    pdata.nanmissingvalues(on="protein", limit=0.5)

    X_filtered = pdata.prot.X.toarray() if scipy.sparse.issparse(pdata.prot.X) else pdata.prot.X

    # Column 0 and 2 should be fully NaN
    assert np.all(np.isnan(X_filtered[:, 0]))
    assert np.all(np.isnan(X_filtered[:, 2]))

    # Column 1 should not be fully NaN
    assert not np.all(np.isnan(X_filtered[:, 1]))

def test_nanmissingvalues_supports_peptide(pdata):

    pdata = deepcopy(pdata)
    adata = pdata.pep

    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # Inject 100% missing in col 0
    X[:, 0] = np.nan
    adata.X = X
    pdata.pep = adata

    # Run on peptide
    pdata.nanmissingvalues(on="peptide", limit=0.5)

    X_filtered = pdata.pep.X.toarray() if scipy.sparse.issparse(pdata.pep.X) else pdata.pep.X
    assert np.all(np.isnan(X_filtered[:, 0]))

def test_nanmissingvalues_limit_zero_masks_all_partial_missing(pdata):

    pdata = deepcopy(pdata)
    adata = pdata.prot

    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # Inject some missing in column 3
    X[0, 3] = np.nan
    adata.X = X
    pdata.prot = adata

    # Run with 0.0 threshold
    pdata.nanmissingvalues(on="protein", limit=0.0)

    X_filtered = pdata.prot.X.toarray() if scipy.sparse.issparse(pdata.prot.X) else pdata.prot.X
    assert np.all(np.isnan(X_filtered[:, 3]))

# test clean_X

def test_clean_X_replaces_nans_in_dense_X(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    pdata.clean_X(on="prot", set_to=0, inplace=True)

    X = pdata.prot.X
    data = X.data if scipy.sparse.issparse(X) else X
    assert not np.isnan(data).any()

def test_clean_X_replaces_nans_in_sparse_X(pdata):
    pdata = pdata.copy()
    pdata.prot.X = scipy.sparse.csr_matrix(pdata.prot.X)
    pdata.prot.X.data[0] = np.nan
    pdata.clean_X(on="prot", set_to=7, inplace=True)

    assert not np.isnan(pdata.prot.X.data).any()
    assert 7 in pdata.prot.X.data

def test_clean_X_creates_backup_layer(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    pdata.clean_X(on="prot", set_to=0, backup_layer="X_testbackup")

    assert "X_testbackup" in pdata.prot.layers
    backup = pdata.prot.layers["X_testbackup"]
    data = backup.data if scipy.sparse.issparse(backup) else backup
    assert np.isnan(data).any()


def test_clean_X_returns_cleaned_matrix_when_inplace_false(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    cleaned = pdata.clean_X(on="prot", inplace=False, set_to=-1)

    assert scipy.sparse.issparse(cleaned) or isinstance(cleaned, np.ndarray)
    data = cleaned.data if scipy.sparse.issparse(cleaned) else cleaned
    assert not np.isnan(data).any()
    assert np.any(data == -1)

    # original should still contain NaN
    orig = pdata.prot.X
    orig_data = orig.data if scipy.sparse.issparse(orig) else orig
    assert np.isnan(orig_data).any()


def test_clean_X_to_sparse_returns_sparse_matrix(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    result = pdata.clean_X(on="prot", inplace=False, set_to=99, to_sparse=True)

    assert scipy.sparse.issparse(result)
    assert 99 in result.data

@pytest.mark.xfail(reason="to do")
def test_clean_X_works_on_peptide(pdata):
    pdata = pdata.copy()
    pdata.pep = pdata.prot.copy()
    pdata.pep.X[1, 1] = np.nan
    pdata.clean_X(on="peptide", set_to=42)

    X = pdata.pep.X
    data = X.data if scipy.sparse.issparse(X) else X

    val = pdata.pep.X[1, 1] if not scipy.sparse.issparse(X) else data[0]
    assert not np.isnan(val)
    assert np.any(data == 42)

def test_clean_X_layer_argument(pdata):
    pdata = pdata.copy()
    layer = pdata.prot.X.copy()
    layer[2, 2] = np.nan
    pdata.prot.layers["testlayer"] = layer
    pdata.clean_X(on="prot", layer="testlayer", set_to=777)

    layer_out = pdata.prot.layers["testlayer"]
    if scipy.sparse.issparse(layer_out):
        data = layer_out.data
    else:
        data = layer_out

    assert not np.isnan(data).any()
    assert np.any(data == 777)

class DummyPrerankResult:
    def __init__(self):
        self.res2d = pd.DataFrame({
            "Term": ["GO_Biological_Process_2025__PATHWAY_A", "KEGG_2026__PATHWAY_B"],
            "NES": [1.5, -1.2],
            "FDR q-val": [0.01, 0.05],
        })

class DummySsGSEAResult:
    def __init__(self, sample_names):
        rows = []
        for s in sample_names:
            rows.append({"Name": s, "Term": "PATHWAY_A", "ES": 0.8})
            rows.append({"Name": s, "Term": "PATHWAY_B", "ES": -0.3})
        self.res2d = pd.DataFrame(rows)

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_pca_gsea_default_storage(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)

    # Ensure PCA exists
    pdata.pca(on=on)

    # Ensure gene column exists
    adata.var["Genes"] = [f"GENE{i}" for i in range(adata.n_vars)]

    def mock_prerank(*args, **kwargs):
        return DummyPrerankResult()

    monkeypatch.setattr("gseapy.prerank", mock_prerank)

    pdata.pca_gsea(on=on)

    assert "pca_gsea" in adata.uns
    assert "params" in adata.uns["pca_gsea"]
    assert "results" in adata.uns["pca_gsea"]
    assert "rankings" in adata.uns["pca_gsea"]

    r1 = adata.uns["pca_gsea"]["results"]["PC1"]
    assert "library" in r1.columns
    assert "pathway" in r1.columns
    assert r1["library"].tolist() == ["GO_Biological_Process_2025", "KEGG_2026"]
    assert r1["pathway"].tolist() == ["PATHWAY_A", "PATHWAY_B"]

    assert "PC1" in adata.uns["pca_gsea"]["results"]
    assert "PC2" in adata.uns["pca_gsea"]["results"]
    assert "PC1" in adata.uns["pca_gsea"]["rankings"]
    assert "PC2" in adata.uns["pca_gsea"]["rankings"]

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_pca_gsea_selected_pcs(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)

    pdata.pca(on=on)
    adata.var["Genes"] = [f"GENE{i}" for i in range(adata.n_vars)]

    def mock_prerank(*args, **kwargs):
        return DummyPrerankResult()

    monkeypatch.setattr("gseapy.prerank", mock_prerank)

    pdata.pca_gsea(on=on, pcs=[1])

    assert "pca_gsea" in adata.uns
    assert list(adata.uns["pca_gsea"]["results"].keys()) == ["PC1"]
    assert list(adata.uns["pca_gsea"]["rankings"].keys()) == ["PC1"]

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_pca_gsea_pcs_none_runs_all(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)

    pdata.pca(on=on)
    adata.var["Genes"] = [f"GENE{i}" for i in range(adata.n_vars)]

    def mock_prerank(*args, **kwargs):
        return DummyPrerankResult()

    monkeypatch.setattr("gseapy.prerank", mock_prerank)

    n_pcs = adata.uns["pca"]["PCs"].shape[0]
    pdata.pca_gsea(on=on, pcs=None)

    expected_keys = [f"PC{i}" for i in range(1, n_pcs + 1)]
    assert list(adata.uns["pca_gsea"]["results"].keys()) == expected_keys
    assert list(adata.uns["pca_gsea"]["rankings"].keys()) == expected_keys

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_pca_gsea_missing_gene_column_hard_stop(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)

    pdata.pca(on=on)
    adata.var.drop(columns=["Genes"], errors="ignore", inplace=True)
    adata.uns.pop("pca_gsea", None)

    called = {"prerank": False}

    def mock_prerank(*args, **kwargs):
        called["prerank"] = True
        return DummyPrerankResult()

    monkeypatch.setattr("gseapy.prerank", mock_prerank)

    pdata.pca_gsea(on=on)

    assert "pca_gsea" not in adata.uns
    assert called["prerank"] is False

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_pca_gsea_duplicate_genes_collapsed_by_max(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)

    pdata.pca(on=on)

    # Force duplicate genes
    genes = [f"GENE{i}" for i in range(adata.n_vars)]
    if adata.n_vars >= 3:
        genes[0] = "DUP"
        genes[1] = "DUP"
    adata.var["Genes"] = genes

    # Force known PC1 loadings
    pcs = adata.uns["pca"]["PCs"].copy()
    pcs[0, 0] = 0.2
    pcs[0, 1] = 0.9
    adata.uns["pca"]["PCs"] = pcs

    captured = {}

    def mock_prerank(*args, **kwargs):
        rnk = kwargs["rnk"]
        captured["rnk"] = rnk.copy()
        return DummyPrerankResult()

    monkeypatch.setattr("gseapy.prerank", mock_prerank)

    pdata.pca_gsea(on=on, pcs=[1])

    assert "DUP" in captured["rnk"].index
    assert captured["rnk"]["DUP"] == 0.9
    assert captured["rnk"].index.tolist().count("DUP") == 1

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_ssgsea_default_storage(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)
    adata.var["Genes"] = [f"GENE{i}" for i in range(adata.n_vars)]

    def mock_ssgsea(*args, **kwargs):
        return DummySsGSEAResult(sample_names=adata.obs_names.astype(str))

    monkeypatch.setattr("gseapy.ssgsea", mock_ssgsea)

    pdata.ssgsea(on=on)

    assert "X_ssgsea" in adata.obsm
    assert "ssgsea" in adata.uns
    assert "params" in adata.uns["ssgsea"]
    assert "long_results" in adata.uns["ssgsea"]
    assert "pathway_names" in adata.uns["ssgsea"]

    n = adata.n_obs
    assert adata.obsm["X_ssgsea"].shape[0] == n

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_ssgsea_missing_gene_column_hard_stop(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)
    adata.var.drop(columns=["Genes"], errors="ignore", inplace=True)
    adata.obsm.pop("X_ssgsea", None)
    adata.uns.pop("ssgsea", None)

    called = {"ssgsea": False}

    def mock_ssgsea(*args, **kwargs):
        called["ssgsea"] = True
        return DummySsGSEAResult(sample_names=adata.obs_names.astype(str))

    monkeypatch.setattr("gseapy.ssgsea", mock_ssgsea)

    pdata.ssgsea(on=on)

    assert "X_ssgsea" not in adata.obsm
    assert "ssgsea" not in adata.uns
    assert called["ssgsea"] is False

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_ssgsea_duplicate_genes_collapsed_by_mean(monkeypatch, pdata, on):
    if on == "protein":
        adata_on = "prot"
    else:
        adata_on = "pep"

    adata = getattr(pdata, adata_on)

    # Create duplicate genes
    genes = [f"GENE{i}" for i in range(adata.n_vars)]
    if adata.n_vars >= 3:
        genes[0] = "DUP"
        genes[1] = "DUP"
    adata.var["Genes"] = genes

    # Force known values in first two duplicated rows
    X = adata.X.toarray().copy()
    X[:, 0] = 2.0
    X[:, 1] = 6.0
    adata.X = X

    captured = {}

    def mock_ssgsea(*args, **kwargs):
        data = kwargs["data"]
        captured["data"] = data.copy()
        return DummySsGSEAResult(sample_names=adata.obs_names.astype(str))

    monkeypatch.setattr("gseapy.ssgsea", mock_ssgsea)

    pdata.ssgsea(on=on)

    assert "DUP" in captured["data"].index
    np.testing.assert_allclose(captured["data"].loc["DUP"].values, np.full(adata.n_obs, 4.0))
    assert captured["data"].index.tolist().count("DUP") == 1

# tests for pairwise_correlation

class TestPairwiseCorrelation:

    def test_stores_group_matrix(self, pdata):
        """Group matrix stored in uns after calling pairwise_correlation."""
        pdata.pairwise_correlation(classes="cellline")
        assert "pairwise_corr" in pdata.prot.uns
        result = pdata.prot.uns["pairwise_corr"]
        assert "group_matrix" in result
        df = result["group_matrix"]
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == df.shape[1]
        assert list(df.index) == list(df.columns)

    def test_group_matrix_is_square_and_labeled(self, pdata):
        """group_matrix index and columns match the groups in the classes column."""
        pdata.pairwise_correlation(classes="cellline")
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        expected_groups = sorted(pdata.prot.obs["cellline"].unique().tolist())
        assert list(df.index) == expected_groups
        assert list(df.columns) == expected_groups

    def test_default_order_is_alphabetical(self, pdata):
        """Default group order is alphabetically sorted."""
        pdata.pairwise_correlation(classes="cellline")
        result = pdata.prot.uns["pairwise_corr"]
        expected = sorted(pdata.prot.obs["cellline"].unique().tolist())
        assert result["order"] == expected

    def test_custom_order_respected(self, pdata):
        """Custom order is stored and reflected in group_matrix index."""
        groups = sorted(pdata.prot.obs["cellline"].unique().tolist())
        custom_order = list(reversed(groups))
        pdata.pairwise_correlation(classes="cellline", order=custom_order)
        result = pdata.prot.uns["pairwise_corr"]
        assert result["order"] == custom_order
        assert list(result["group_matrix"].index) == custom_order

    def test_order_unknown_values_warn_and_resolve(self, pdata, capsys):
        """Unknown order entries are dropped with a warning; result still valid."""
        pdata.pairwise_correlation(classes="cellline", order=["nonexistent_group"])
        captured = capsys.readouterr()
        assert "removing" in captured.out.lower()
        result = pdata.prot.uns["pairwise_corr"]
        expected = sorted(pdata.prot.obs["cellline"].unique().tolist())
        assert result["order"] == expected

    def test_order_partial_appends_omitted(self, pdata, capsys):
        """Groups omitted from order are appended alphabetically after user order."""
        groups = sorted(pdata.prot.obs["cellline"].unique().tolist())
        assert len(groups) >= 2
        partial = [groups[1]]
        pdata.pairwise_correlation(classes="cellline", order=partial)
        out = capsys.readouterr()
        assert "appended" in out.out.lower()
        result = pdata.prot.uns["pairwise_corr"]
        assert result["order"] == partial + sorted([g for g in groups if g != groups[1]])

    @pytest.mark.parametrize(
        "extra_kw,exc_type,match_part",
        [
            pytest.param({"method": "cosine"}, ValueError, "method=", id="bad_method"),
            pytest.param(
                {"classes": "nonexistent_column"}, ValueError, "not found", id="bad_classes"
            ),
            pytest.param(
                {"layer": "X_doesnotexist"}, KeyError, None, id="bad_layer"
            ),
        ],
    )
    def test_invalid_args_raise(self, pdata, extra_kw, exc_type, match_part):
        """Unsupported method, missing classes column, or missing layer raises."""
        kw = {"classes": "cellline", **extra_kw}
        if match_part is not None:
            with pytest.raises(exc_type, match=match_part):
                pdata.pairwise_correlation(**kw)
        else:
            with pytest.raises(exc_type):
                pdata.pairwise_correlation(**kw)

    @pytest.mark.parametrize(
        "method,expected_fill",
        [
            pytest.param("pearson", 1.0, id="pearson"),
            pytest.param("spearman", 1.0, id="spearman"),
            pytest.param("euclidean", 0.0, id="euclidean"),
        ],
    )
    def test_group_matrix_diagonal(self, pdata, method, expected_fill):
        """Correlation methods have 1 on diagonal; euclidean distance has 0."""
        pdata.pairwise_correlation(classes="cellline", method=method)
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        diag = np.diag(df.values)
        np.testing.assert_allclose(
            diag, np.full(len(diag), expected_fill), atol=1e-6
        )

    @pytest.mark.parametrize("method", ["pearson", "spearman", "euclidean"])
    def test_group_matrix_symmetric(self, pdata, method):
        """Group-level pearson, spearman, and euclidean matrices are symmetric."""
        pdata.pairwise_correlation(classes="cellline", method=method)
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        np.testing.assert_allclose(df.values, df.values.T, atol=1e-6)

    def test_sample_matrix_none_by_default(self, pdata):
        """sample_matrix is None when compute_sample_matrix=False."""
        pdata.pairwise_correlation(classes="cellline")
        assert pdata.prot.uns["pairwise_corr"]["sample_matrix"] is None

    def test_sample_matrix_computed_when_requested(self, pdata):
        """sample_matrix is a DataFrame when compute_sample_matrix=True."""
        pdata.pairwise_correlation(classes="cellline", compute_sample_matrix=True)
        result = pdata.prot.uns["pairwise_corr"]
        sm = result["sample_matrix"]
        assert isinstance(sm, pd.DataFrame)
        assert sm.shape[0] == sm.shape[1] == pdata.prot.n_obs

    def test_sample_matrix_euclidean_finite_with_missing_abundance(self, pdata):
        """Euclidean sample matrix must stay finite when .X has NaNs (nan_euclidean)."""
        pdata.pairwise_correlation(
            classes="cellline", method="euclidean", compute_sample_matrix=True, force=True
        )
        sm = pdata.prot.uns["pairwise_corr"]["sample_matrix"]
        assert np.isfinite(sm.values).all()

    def test_sample_matrix_sorted_by_order(self, pdata):
        """sample_matrix rows/columns are sorted to match group order."""
        pdata.pairwise_correlation(classes="cellline", compute_sample_matrix=True)
        result = pdata.prot.uns["pairwise_corr"]
        sm = result["sample_matrix"]
        order = result["order"]
        obs = pdata.prot.obs
        group_of_sample = obs["cellline"].to_dict()
        indices_by_group = {g: [] for g in order}
        for i, name in enumerate(sm.index):
            indices_by_group[group_of_sample[name]].append(i)
        prev_max = -1
        for g in order:
            idxs = indices_by_group[g]
            assert min(idxs) > prev_max, f"Group '{g}' not contiguous/ordered in sample_matrix"
            prev_max = max(idxs)

    def test_force_recomputes(self, pdata):
        """force=True recomputes even if uns key already present."""
        pdata.pairwise_correlation(classes="cellline")
        first_result = pdata.prot.uns["pairwise_corr"]["group_matrix"].copy()
        pdata.pairwise_correlation(classes="cellline", force=True)
        second_result = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        pd.testing.assert_frame_equal(first_result, second_result)

    def test_cache_hit_no_recompute(self, pdata, capsys):
        """Second call with same params prints cache hit and appends cache history."""
        pdata.pairwise_correlation(classes="cellline")
        n_after_first = len(pdata.history)
        pdata.pairwise_correlation(classes="cellline")
        captured = capsys.readouterr()
        assert "already computed" in captured.out.lower() or "force=true" in captured.out.lower()
        assert len(pdata.history) == n_after_first + 1
        assert "cached" in pdata.history[-1].lower()

    def test_different_params_triggers_recompute(self, pdata, capsys):
        """Calling with different method triggers recompute with a warning."""
        pdata.pairwise_correlation(classes="cellline", method="pearson")
        pdata.pairwise_correlation(classes="cellline", method="spearman")
        captured = capsys.readouterr()
        assert "recomputing" in captured.out.lower() or "differ" in captured.out.lower()
        assert pdata.prot.uns["pairwise_corr"]["method"] == "spearman"

    def test_peptide_level(self, pdata):
        """Works on peptide level (on='peptide')."""
        pdata.pairwise_correlation(classes="cellline", on="peptide")
        assert "pairwise_corr" in pdata.pep.uns
        df = pdata.pep.uns["pairwise_corr"]["group_matrix"]
        assert df.shape[0] == df.shape[1]

    def test_stored_metadata(self, pdata):
        """uns dict contains all expected metadata keys."""
        pdata.pairwise_correlation(classes="cellline")
        result = pdata.prot.uns["pairwise_corr"]
        for key in (
            "group_matrix",
            "sample_matrix",
            "classes",
            "classes_list",
            "separator",
            "order",
            "method",
            "layer",
            "compute_sample_matrix",
            "n_features_used",
            "n_features_dropped",
            "subset_indices",
        ):
            assert key in result, f"Missing key: {key}"

    def test_n_features_used_plus_dropped_equals_total(self, pdata):
        """n_features_used + n_features_dropped should equal total features."""
        pdata.pairwise_correlation(classes="cellline")
        result = pdata.prot.uns["pairwise_corr"]
        total = pdata.prot.n_vars
        assert result["n_features_used"] + result["n_features_dropped"] == total

    def test_history_appended(self, pdata):
        """History entry is appended after running pairwise_correlation."""
        before = len(pdata.history)
        pdata.pairwise_correlation(classes="cellline")
        assert len(pdata.history) > before
        assert "pairwise_correlation" in pdata.history[-1].lower()

    def test_no_pep_raises(self, pdata_nopep):
        """Raises when on='peptide' but pep is None."""
        with pytest.raises(ValueError):
            pdata_nopep.pairwise_correlation(classes="cellline", on="peptide")

    def test_subset_mask_smaller_sample_matrix(self, pdata):
        """subset_mask limits samples in group and sample matrices."""
        obs = pdata.prot.obs
        mask = (obs["cellline"] == obs["cellline"].iloc[0]).to_numpy()
        assert mask.sum() < pdata.prot.n_obs
        pdata.pairwise_correlation(
            classes="cellline", compute_sample_matrix=True, subset_mask=mask
        )
        result = pdata.prot.uns["pairwise_corr"]
        assert result["subset_indices"] is not None
        sm = result["sample_matrix"]
        assert sm is not None
        n_sub = int(mask.sum())
        assert sm.shape == (n_sub, n_sub)
        assert result["group_matrix"].shape[0] == 1

    def test_subset_mask_bad_length_raises(self, pdata):
        with pytest.raises(ValueError, match="subset_mask"):
            pdata.pairwise_correlation(
                classes="cellline", subset_mask=np.array([True, False])
            )

    # tests for list classes (get_samplenames / comma-space join)

    def test_list_classes_runs(self, pdata):
        """pairwise_correlation accepts a list of classes without error."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        assert "pairwise_corr" in pdata.prot.uns

    def test_list_classes_combined_labels(self, pdata):
        """group_matrix labels use get_samplenames comma-space join for 2+ columns."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        for label in df.index:
            assert ", " in str(label), f"Expected combined label, got: {label!r}"

    def test_list_classes_stores_classes_list(self, pdata):
        """uns stores classes_list as the original list of column names."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        result = pdata.prot.uns["pairwise_corr"]
        assert result["classes_list"] == ["cellline", "treatment"]

    def test_string_classes_stores_classes_list_as_single_item(self, pdata):
        """uns stores classes_list as a single-element list when classes is a str."""
        pdata.pairwise_correlation(classes="cellline")
        result = pdata.prot.uns["pairwise_corr"]
        assert result["classes_list"] == ["cellline"]
        assert result["separator"] is None

    def test_stores_separator(self, pdata):
        """uns stores comma-space separator for multi-column classes."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        assert pdata.prot.uns["pairwise_corr"]["separator"] == ", "

    def test_list_classes_matrix_is_square(self, pdata):
        """group_matrix is square when classes is a list."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        assert df.shape[0] == df.shape[1]

    def test_list_classes_n_groups_correct(self, pdata):
        """Number of groups matches unique combined labels from get_samplenames."""
        expected_labels = sorted(
            set(utils.get_samplenames(pdata.prot, ["cellline", "treatment"]))
        )
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        assert list(df.index) == expected_labels

    def test_list_classes_pearson_diagonal_is_one(self, pdata):
        """Pearson diagonal is 1.0 for list classes."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"], method="pearson")
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        np.testing.assert_allclose(np.diag(df.values), np.ones(len(df)), atol=1e-6)

    def test_list_classes_custom_order(self, pdata):
        """Custom order is respected for list classes (combined labels)."""
        combined = sorted(
            set(utils.get_samplenames(pdata.prot, ["cellline", "treatment"]))
        )
        custom_order = list(reversed(combined))
        pdata.pairwise_correlation(classes=["cellline", "treatment"], order=custom_order)
        df = pdata.prot.uns["pairwise_corr"]["group_matrix"]
        assert list(df.index) == custom_order

    def test_list_classes_invalid_column_raises(self, pdata):
        """Raises ValueError if any column in list classes is not in obs."""
        with pytest.raises(ValueError, match="not found in adata.obs"):
            pdata.pairwise_correlation(classes=["cellline", "nonexistent_col"])

    def test_empty_list_classes_raises(self, pdata):
        """Raises ValueError for empty list classes."""
        with pytest.raises(ValueError, match="non-empty"):
            pdata.pairwise_correlation(classes=[])

    def test_cache_invalidated_str_to_list(self, pdata, capsys):
        """Changing classes from str to list triggers recompute with a warning."""
        pdata.pairwise_correlation(classes="cellline")
        capsys.readouterr()
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        captured = capsys.readouterr()
        assert "recomputing" in captured.out.lower() or "differ" in captured.out.lower()

    def test_cache_invalidated_list_to_str(self, pdata, capsys):
        """Changing classes from list to str triggers recompute with a warning."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        capsys.readouterr()
        pdata.pairwise_correlation(classes="cellline")
        captured = capsys.readouterr()
        assert "recomputing" in captured.out.lower() or "differ" in captured.out.lower()

    def test_cache_hit_list_classes(self, pdata, capsys):
        """Cache hit works correctly when classes is a list."""
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        capsys.readouterr()
        pdata.pairwise_correlation(classes=["cellline", "treatment"])
        captured = capsys.readouterr()
        assert "already computed" in captured.out.lower() or "force=true" in captured.out.lower()

    def test_list_classes_sample_matrix(self, pdata):
        """Sample matrix is computed correctly with list classes."""
        pdata.pairwise_correlation(
            classes=["cellline", "treatment"], compute_sample_matrix=True
        )
        sm = pdata.prot.uns["pairwise_corr"]["sample_matrix"]
        assert isinstance(sm, pd.DataFrame)
        assert sm.shape[0] == sm.shape[1] == pdata.prot.n_obs

class TestLogTransform:
    """log_transform(), provenance, and fixed output layer names."""

    @pytest.mark.parametrize("base,expected_key", [(10, "X_log10"), ("e", "X_loge")])
    def test_log10_and_loge_layers(self, pdata, base, expected_key):
        pdata.log_transform(base=base, set_X=False)
        assert expected_key in pdata.prot.layers

    def test_stores_layer(self, pdata):
        pdata.log_transform(set_X=False)
        assert "X_log2" in pdata.prot.layers

    def test_provenance_registered(self, pdata):
        pdata.log_transform(set_X=False)
        reg = pdata.prot.uns["layer_provenance"]
        assert "X_log2" in reg
        assert reg["X_log2"]["op"] == "log_transform"
        assert reg["X_log2"]["input_layer"] == "X_raw"

    def test_log10_base_metadata(self, pdata):
        pdata.log_transform(base=10, set_X=False)
        assert pdata.prot.uns["layer_provenance"]["X_log10"]["base"] == "10"

    def test_double_log_warns(self, pdata, capsys):
        pdata.log_transform(set_X=True)
        capsys.readouterr()
        pdata.log_transform(layer="X_log2", set_X=False)
        captured = capsys.readouterr()
        assert "already" in captured.out.lower() or "log" in captured.out.lower()

    def test_set_X_updates_X(self, pdata):
        pdata.log_transform(set_X=True)
        X = (
            pdata.prot.X.toarray()
            if scipy.sparse.issparse(pdata.prot.X)
            else pdata.prot.X
        )
        assert np.nanmedian(X) < 100

    def test_set_X_false_leaves_X(self, pdata):
        X_before = (
            pdata.prot.X.toarray().copy()
            if scipy.sparse.issparse(pdata.prot.X)
            else pdata.prot.X.copy()
        )
        pdata.log_transform(set_X=False)
        X_after = (
            pdata.prot.X.toarray()
            if scipy.sparse.issparse(pdata.prot.X)
            else pdata.prot.X
        )
        np.testing.assert_array_equal(X_before, X_after)

    def test_log2_values_correct(self, pdata):
        raw = (
            pdata.prot.X.toarray().copy()
            if scipy.sparse.issparse(pdata.prot.X)
            else pdata.prot.X.copy()
        )
        pdata.log_transform(base=2, pseudocount=1.0, set_X=False)
        result = pdata.prot.layers["X_log2"]
        result = result.toarray() if scipy.sparse.issparse(result) else result
        np.testing.assert_allclose(result, np.log2(raw + 1.0), atol=1e-5)

    def test_invalid_base_raises(self, pdata):
        with pytest.raises(ValueError, match="base="):
            pdata.log_transform(base=3)

    def test_invalid_layer_raises(self, pdata):
        with pytest.raises(KeyError):
            pdata.log_transform(layer="X_doesnotexist")

    def test_history_appended(self, pdata):
        before = len(pdata.history)
        pdata.log_transform(set_X=False)
        assert len(pdata.history) > before

    def test_peptide_level(self, pdata):
        if pdata.pep is None:
            pytest.skip("No peptide data in fixture")
        pdata.log_transform(on="peptide", set_X=False)
        assert "X_log2" in pdata.pep.layers

    def test_collision_suffix_applied(self, pdata):
        pdata.normalize(method="median", set_X=False)
        pdata.log_transform(layer="X", set_X=False)
        pdata.log_transform(layer="X_norm_median", set_X=False)
        assert "X_log2" in pdata.prot.layers
        assert "X_log2_1" in pdata.prot.layers

class TestShowLayerProvenance:
    def test_runs_without_error(self, pdata, capsys):
        pdata.normalize(method="median", set_X=False)
        pdata.log_transform(layer="X_norm_median", set_X=False)
        pdata.show_layer_provenance("X_log2")
        captured = capsys.readouterr()
        assert "log_transform" in captured.out

    def test_full_registry_no_layer_arg(self, pdata, capsys):
        pdata.normalize(method="median", set_X=False)
        pdata.show_layer_provenance()
        captured = capsys.readouterr()
        assert "normalize" in captured.out

    def test_unknown_layer_warns(self, pdata, capsys):
        pdata.normalize(method="median", set_X=False)
        pdata.show_layer_provenance("X_does_not_exist")
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_empty_registry_info(self, pdata, capsys):
        if "layer_provenance" in pdata.prot.uns:
            del pdata.prot.uns["layer_provenance"]
        pdata.show_layer_provenance()
        captured = capsys.readouterr()
        assert "no layer provenance" in captured.out.lower()
        assert "[INFO]" in captured.out

def test_provenance_chain_depth_after_chain(pdata):
    """Full normalize → impute → log chain records resolved input layers."""
    pdata.normalize(method="median")
    pdata.impute(method="min", min_scale=0.1)
    pdata.log_transform()

    reg = pdata.prot.uns["layer_provenance"]

    assert reg["X_norm_median"]["input_layer"] == "X_raw"
    assert reg["X_impute_min"]["input_layer"] == "X_norm_median"
    assert reg["X_log2"]["input_layer"] == "X_impute_min"

def test_chain_walk_depth_greater_than_one(pdata, capsys):
    """show_layer_provenance shows full chain including raw root."""
    pdata.normalize(method="median")
    pdata.impute(method="min", min_scale=0.1)
    pdata.log_transform()
    capsys.readouterr()
    pdata.show_layer_provenance("X_log2")
    captured = capsys.readouterr()
    assert "[3]" in captured.out

def test_explicit_layer_not_resolved(pdata):
    """Explicit layer='X_norm_median' records that name as provenance input."""
    pdata.normalize(method="median")
    pdata.log_transform(layer="X_norm_median", set_X=False)
    reg = pdata.prot.uns["layer_provenance"]
    assert reg["X_log2"]["input_layer"] == "X_norm_median"

def test_two_chains_from_same_raw(pdata):
    """Two normalizations from X_raw both record X_raw as input."""
    pdata.normalize(method="median")
    pdata.normalize(method="sum", layer="X_raw")
    reg = pdata.prot.uns["layer_provenance"]
    assert reg["X_norm_median"]["input_layer"] == "X_raw"
    assert reg["X_norm_sum"]["input_layer"] == "X_raw"

def test_show_provenance_uses_current_X_layer(pdata, capsys):
    pdata.normalize(method="median")
    pdata.log_transform(layer="X_norm_median")
    capsys.readouterr()
    pdata.show_layer_provenance()
    captured = capsys.readouterr()
    assert "Current .X" in captured.out
    assert "X_log2" in captured.out

def test_show_provenance_skips_current_X_when_uns_missing(pdata, capsys):
    """No current_X_layer key → no Current .X section (honest for legacy objects)."""
    pdata.normalize(method="median", set_X=False)
    del pdata.prot.uns["current_X_layer"]
    capsys.readouterr()
    pdata.show_layer_provenance()
    captured = capsys.readouterr()
    assert "Current .X" not in captured.out
    assert "Other layers" in captured.out or "○" in captured.out
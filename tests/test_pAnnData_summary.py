import numpy as np
import pandas as pd
import re
from scipy import sparse

# ----------------------------------------------------------------------
# unit tests for .summary staleness, manual edits, auto-sync behaviour:
def test_summary_stale_flag_after_manual_edit(pdata):
    # update_summary() clears stale flag
    pdata.update_summary(recompute=True)
    assert not pdata._summary_is_stale

    # Manual edit triggers stale flag
    pdata.summary["new_col"] = np.arange(pdata.summary.shape[0])
    assert pdata._summary_is_stale

def test_summary_setter_triggers_sync_back(pdata):
    # Re-assigning .summary = ... triggers sync back
    pdata.update_summary()
    new_summary = pdata.summary.copy()
    new_summary["test_col"] = ["X"] * new_summary.shape[0]
    pdata.summary = new_summary  # triggers sync_back via setter

    # Check that the column was pushed into obs
    assert "test_col" in pdata.prot.obs.columns
    assert (pdata.prot.obs["test_col"] == "X").all()

def test_auto_sync_back_to_obs(pdata):
    # Stale summary triggers auto-sync
    pdata.update_summary()

    # Modify .summary manually
    pdata.summary["condition"] = ["A"] * pdata.summary.shape[0]

    # This should trigger auto sync (check .prot.obs)
    pdata.update_summary(recompute=False)

    assert "condition" in pdata.prot.obs.columns
    assert (pdata.prot.obs["condition"] == "A").all()
    assert not pdata._summary_is_stale

def test_no_sync_when_not_stale(pdata):
    # Clean summary does not sync again
    pdata.update_summary()
    prot_obs_before = pdata.prot.obs.copy()

    # No manual edits, no recompute
    pdata.update_summary(recompute=False)

    pd.testing.assert_frame_equal(prot_obs_before, pdata.prot.obs)

def test_update_metrics_treats_zero_as_missing(pdata):
    pdata.prot = pdata.prot[:2, :3].copy()
    pdata.prot.X = np.array([[1e5, 0.0, np.nan], [0.0, 2e4, 3e4]])
    pdata.update_summary(recompute=True, verbose=False)
    assert pdata.prot.obs["protein_quant"].tolist() == [1 / 3, 2 / 3]
    assert pdata.prot.obs["protein_count"].tolist() == [1, 2]

def test_clean_X_preserves_protein_quant(pdata):
    pdata.update_summary(recompute=True, verbose=False)
    quant_before = pdata.prot.obs["protein_quant"].copy()

    pdata.clean_X(on="prot", set_to=0, inplace=True)

    pd.testing.assert_series_equal(quant_before, pdata.prot.obs["protein_quant"])

def test_directlfq_like_layer_preserves_per_sample_quant(pdata):
    X = pdata.prot.X.toarray().copy()
    X = np.nan_to_num(X, nan=0.0)
    pdata.prot.layers["X_norm_directlfq"] = sparse.csr_matrix(X)
    pdata.set_X("X_norm_directlfq", on="protein")

    quant = pdata.prot.obs["protein_quant"]
    assert quant.nunique() > 1
    assert (quant < 1.0).any()

def test_copy_matches_original_after_directlfq_like_layer(pdata):
    X = pdata.prot.X.toarray().copy()
    X = np.nan_to_num(X, nan=0.0)
    pdata.prot.layers["X_norm_directlfq"] = sparse.csr_matrix(X)
    pdata.set_X("X_norm_directlfq", on="protein")

    pdata_copy = pdata.copy()
    pd.testing.assert_series_equal(
        pdata.prot.obs["protein_quant"],
        pdata_copy.prot.obs["protein_quant"],
    )

    def _repr_quant(obj):
        match = re.search(r"Avg protein quant: ([\d.]+)", repr(obj))
        return float(match.group(1)) if match else None

    assert _repr_quant(pdata) == _repr_quant(pdata_copy)

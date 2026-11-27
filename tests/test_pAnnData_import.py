import pytest
import pandas as pd
import numpy as np
from scipy import sparse
from scpviz import pAnnData
from pathlib import Path

from scpviz.pAnnData.io import _create_pAnnData_from_parts, _safe_strip

def test_safe_strip_dataframe_and_series():
    # --- DataFrame case ---
    df = pd.DataFrame({"A": [" a", "b ", None], "B": [" c ", " d", "e "]})
    out_df = _safe_strip(df)

    # Verify type and values
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.iloc[0, 0] == "a"
    assert out_df.iloc[1, 1] == "d"
    assert out_df.isna().sum().sum() == 1  # preserves None/NaN

    # --- Series case ---
    s = pd.Series([" x ", "y", None])
    out_s = _safe_strip(s)

    assert isinstance(out_s, pd.Series)
    assert list(out_s) == ["x", "y", None]

    # --- Mixed types should remain unchanged ---
    s_mixed = pd.Series([" a", 5, None])
    out_mixed = _safe_strip(s_mixed)
    assert out_mixed.iloc[1] == 5
    assert out_mixed.iloc[0] == "a"

    # --- Should not raise on empty input ---
    empty_df = pd.DataFrame(columns=["A"])
    _ = _safe_strip(empty_df)

def test_import_pd():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')
    pep_file = str(test_dir / 'test_pd_pep.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_pd_noQ():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot-noQ.txt')

    obs_columns = ['sample', 'cellline', 'treatment']    
    with pytest.warns(UserWarning, match="Neither 'Exp. q-value: Combined' nor 'Exp. Protein Group q-value: Combined'"):
        pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, obs_columns=obs_columns)

    # Object should still be valid
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is None
    assert pdata.rs is None

def test_import_pd_excel():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.xlsx')
    pep_file = str(test_dir / 'test_pd_pep.xlsx')

    obs_columns = ['sample', 'cellline', 'treatment']    
    pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)

    # Object should still be valid
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_pd32():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd32_Proteins.txt')

    obs_columns = ['Sample','cellline','ko','condition','duration']
    pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is None
    assert pdata.rs is None

def test_import_pd32_with_pep():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd32_Proteins.txt')
    pep_file = str(test_dir / 'test_pd32_PeptideSequenceGroups.txt')

    obs_columns = ['Sample','cellline','ko','condition','duration']
    pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_pd_drops_peptides_missing_master_accessions():
    """
    Peptides with empty / NaN Master Protein Accessions should be dropped.
    """
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / "test_pd_prot.txt")
    pep_file = str(test_dir / "test_pd_pep-missingMaster.txt")

    pdata = pAnnData.import_data(
        source_type="pd",
        prot_file=prot_file,
        pep_file=pep_file,
        obs_columns=['sample', 'cellline', 'treatment'],
    )

    assert pdata.pep is not None

    col = "Master Protein Accessions"
    assert col in pdata.pep.var.columns

    # Only rows with valid master accessions should remain
    s = pdata.pep.var[col].astype(str).str.strip()
    assert pdata.pep.var[col].notna().all()
    assert (s != "").all()

def test_import_pd_peptide_file_missing_master_accession_column_raises():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / "test_pd_prot.txt")
    pep_file = str(test_dir / "test_pd_pep-missingMasterCol.txt")

    with pytest.raises(KeyError, match="Master Protein Accessions"):
        pAnnData.import_data(
            source_type="pd",
            prot_file=prot_file,
            pep_file=pep_file,
            obs_columns=["sample", "cellline", "treatment"],
        )


def test_import_diann_old():
    # pre diann v1.8.1
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann.tsv')

    obs_columns = ['name','amt','enzyme','date','MS','acquisition','method','gradient','replicate']
    pdata = pAnnData.import_data(source_type='diann', report_file=diann_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_diann_new():
    # post diann v1.8.1
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann.parquet')

    # MP_20250219_OA_DIA_FAIMS_TS25_24min_sc_LCM-Cortex_01.raw
    obs_columns = ['name','date','MS','acquisition','FAIMS','column','gradient','amt','region','replicate']
    pdata = pAnnData.import_data(source_type='diann', report_file=diann_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_diann_old_delimiter():
    # pre diann v1.8.1
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann-delimiter.tsv')

    obs_columns = ['name','amt','enzyme','date','MS','acquisition','method','gradient','replicate']
    pdata = pAnnData.import_data(source_type='diann', report_file=diann_file, obs_columns=obs_columns, delimiter='-')
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_suggest_obs_columns_hyphen_delimiter():
    # pre diann v1.8.1
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann-delimiter.tsv')

    obs_columns = pAnnData.suggest_obs_columns(source_type='diann', source=diann_file)
    assert obs_columns is not None

def test_import_no_obs_uniform():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')

    no_obs_return = pAnnData.import_data(source_type='pd', prot_file=prot_file)
    
    assert no_obs_return is None

def test_import_no_obs_nonuniform():
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann_nonuniform.tsv')

    pdata = pAnnData.import_data(source_type='diann', report_file=diann_file)
    summary = pdata.summary

    assert pdata is not None
    assert 'parsingType' in summary.columns

@pytest.mark.parametrize("on,direction,overwrite", [
    ('protein', 'forward', False),
    ('protein', 'forward', True),
    ('protein', 'reverse', False),
    ('protein', 'reverse', True),
    ('peptide', 'forward', False),
    ('peptide', 'forward', True),
    ('peptide', 'reverse', False),
    ('peptide', 'reverse', True),
])
def test_update_identifier_maps_parametrized(pdata, on, direction, overwrite):
    # Get the current maps
    fwd_map, rev_map = pdata.get_identifier_maps(on=on)

    # Choose dummy test keys depending on direction
    if direction == 'forward':
        test_input = {'TEST_KEY_A': 'TEST_VAL_A'}
        pre_existing_key = 'TEST_KEY_A'
        initial_val = 'OLD_VAL'
        updated_val = 'TEST_VAL_A'
        
        # Inject OLD_VAL into fwd_map[TEST_KEY_A]
        fwd_map[pre_existing_key] = initial_val
        rev_map[initial_val] = pre_existing_key

    else:  # direction == 'reverse'
        test_input = {'TEST_VAL_B': 'TEST_KEY_B'}
        pre_existing_key = 'TEST_VAL_B'
        initial_val = 'OLD_VAL'
        updated_val = 'TEST_KEY_B'
        
        # Inject OLD_VAL into rev_map[TEST_VAL_B]
        rev_map[pre_existing_key] = initial_val
        fwd_map[initial_val] = pre_existing_key


    # Inject a pre-existing key to test overwrite behavior
    fwd_map[pre_existing_key] = initial_val if direction == 'forward' else test_input[pre_existing_key]
    rev_map[initial_val] = pre_existing_key if direction == 'forward' else test_input[pre_existing_key]

    # Run the update
    pdata.update_identifier_maps(test_input, on=on, direction=direction, overwrite=overwrite, verbose=False)

    # Re-fetch the maps
    fwd_map, rev_map = pdata.get_identifier_maps(on=on)

    if overwrite:
        if direction == 'forward':
            assert fwd_map[pre_existing_key] == updated_val
            assert rev_map[updated_val] == pre_existing_key
        else:  # reverse
            assert rev_map[pre_existing_key] == updated_val
            assert fwd_map[updated_val] == pre_existing_key
    else:
        if direction == 'forward':
            assert fwd_map[pre_existing_key] == initial_val
        else:
            assert rev_map[pre_existing_key] == initial_val

    # History message should be added
    history_msgs = getattr(pdata, "_history", [])
    assert any(f"Updated '{on}' ({direction})" in msg for msg in history_msgs)

def test_update_identifier_maps_var_column_sync(pdata):
    # Use the first protein accession in the test data
    acc = pdata.prot.var_names[0]
    old_gene = pdata.prot.var.at[acc, "Genes"]

    # Define a new gene name for that accession
    new_gene = "CUSTOM_GENE_XYZ"
    pdata.update_identifier_maps({acc: new_gene}, on='protein', direction='reverse', overwrite=True, verbose=False)

    # Check that .var["Genes"] was updated
    assert pdata.prot.var.at[acc, "Genes"] == new_gene

    # Check that identifier_map_history was logged
    history = pdata.metadata.get("identifier_map_history", [])
    assert any(
        h.get("on") == "protein" and
        h.get("direction") == "reverse" and
        acc in h.get("updated_var_column", {}).get("accessions", [])
        for h in history
    )

# ----------------------------------------------------------------------
# _build_identifier_maps coverage
def test_build_identifier_maps_protein(pdata):
    """Covers protein mapping branch in _build_identifier_maps."""
    fwd, rev = pdata._build_identifier_maps(pdata.prot, gene_col="Genes")
    # pick a known gene from the dummy file
    gene = list(fwd.keys())[0]
    acc = fwd[gene]
    assert rev[acc] == gene

def test_build_identifier_maps_peptide_warn(monkeypatch, pdata):
    """Forces warning path in _build_identifier_maps (peptide branch exception)."""
    def bad_mapping(_): raise RuntimeError("mock failure")
    monkeypatch.setattr("scpviz.utils.get_pep_prot_mapping", bad_mapping)

    with pytest.warns(UserWarning, match="Could not build peptide-to-protein map"):
        fwd, rev = pdata._build_identifier_maps(pdata.pep)
        assert fwd == {} and rev == {}

# ----------------------------------------------------------------------
# fetch_uniprot during import
def test_import_skips_uniprot_when_fetch_false(monkeypatch, capsys):
    """fetch_uniprot=False should not call UniProt and should leave Genes as NA."""
    called = {"uniprot": False}

    def fail_uniprot(*args, **kwargs):
        called["uniprot"] = True
        raise AssertionError("get_uniprot_fields should not be called")

    monkeypatch.setattr("scpviz.utils.get_uniprot_fields", fail_uniprot)

    prot_X = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    prot_var = pd.DataFrame({"Genes": [pd.NA, "GENE2"]}, index=["P1", "P2"])
    prot_obs = pd.DataFrame({"sample": ["s1", "s2"]})

    pdata = _create_pAnnData_from_parts(
        prot_X,
        None,
        None,
        prot_obs,
        prot_var,
        ["s1", "s2"],
        ["P1", "P2"],
        obs_columns=["sample"],
        fetch_uniprot=False,
    )

    assert not called["uniprot"]
    assert pd.isna(pdata.prot.var.loc["P1", "Genes"])
    assert pdata.prot.var.loc["P2", "Genes"] == "GENE2"
    out = capsys.readouterr().out
    assert "Skipping UniProt gene lookup" in out

# ----------------------------------------------------------------------
# update_missing_genes coverage
def test_update_missing_genes_missing_column(pdata, capsys):
    """Covers branch where gene_col not found."""
    pdata.prot.var.rename(columns={"Genes": "WrongName"}, inplace=True)
    pdata.update_missing_genes(gene_col="Genes", verbose=True)
    out = capsys.readouterr().out
    assert "Column 'Genes' not found" in out

def test_update_missing_genes_no_missing(monkeypatch, pdata, capsys):
    """Covers branch when no missing entries."""
    monkeypatch.setattr("scpviz.utils.get_uniprot_fields", lambda *a, **k: pd.DataFrame())
    # fill all Genes → no missing mask
    pdata.prot.var["Genes"] = pdata.prot.var["Genes"].fillna("TESTGENE")
    pdata.update_missing_genes(verbose=True)
    out = capsys.readouterr().out
    assert "No missing gene names" in out

# ----------------------------------------------------------------------
# search_annotations coverage
def test_search_annotations_case_and_all_matches(pdata):
    """Covers return_all_matches=False branch and case=True logic."""
    df = pdata.search_annotations("some_term", on="protein", return_all_matches=False, case=True)
    assert isinstance(df, pd.DataFrame)
    assert "Matched" in df.columns

def test_validate_passes_on_valid_pdata(pdata, capsys):
    """Ensure validate() returns True and prints success for valid objects."""
    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is True
    assert "is valid" in captured

def test_validate_detects_obs_var_shape_mismatch(pdata, capsys):
    # Introduce duplicated var index to simulate structural inconsistency
    pdata.prot.var.index = pdata.prot.var.index.to_series().replace(pdata.prot.var.index[0], pdata.prot.var.index[1])
    
    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is False
    assert "prot.var has duplicated index" in captured

def test_validate_detects_obs_name_mismatch(pdata, capsys):
    pdata.pep.obs.index = [f"s{i}" for i in range(len(pdata.pep))]
    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is False
    assert "obs_names do not match" in captured

def test_validate_detects_rs_shape_mismatch(pdata, capsys):
    from scipy import sparse
    import numpy as np
    
    bad_rs = sparse.csr_matrix(np.zeros((pdata.prot.shape[1] - 1, pdata.pep.shape[1])))
    pdata._set_RS(bad_rs, validate=False)  # ⬅️ bypass shape check

    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is False
    assert "RS shape mismatch" in captured


# --- obs_columns None placeholders (drop filename tokens at import) ---

PD_OBS = ['sample', 'cellline', 'treatment']
DIANN_OBS = ['name', 'amt', 'enzyme', 'date', 'MS', 'acquisition', 'method', 'gradient', 'replicate']

def _import_pd_obs(obs_columns):
    test_dir = Path(__file__).parent
    return pAnnData.import_data(
        source_type='pd',
        prot_file=str(test_dir / 'test_pd_prot.txt'),
        pep_file=str(test_dir / 'test_pd_pep.txt'),
        obs_columns=obs_columns,
        fetch_uniprot=False,
    )

def _import_diann_obs(obs_columns):
    test_dir = Path(__file__).parent
    return pAnnData.import_data(
        source_type='diann',
        report_file=str(test_dir / 'test_diann.tsv'),
        obs_columns=obs_columns,
        fetch_uniprot=False,
    )

def test_import_pd_obs_columns_none_dropped():
    full = _import_pd_obs(PD_OBS)
    pdata = _import_pd_obs(['sample', None, 'treatment'])

    for df in (pdata.prot.obs, pdata.pep.obs):
        assert 'cellline' not in df.columns
        assert None not in list(df.columns)
        assert list(df.columns[:2]) == ['sample', 'treatment']
        assert df['sample'].tolist() == full.prot.obs['sample'].tolist()
        assert df['treatment'].tolist() == full.prot.obs['treatment'].tolist()
    assert 'cellline' not in pdata.summary.columns

    # column must not come back from .pep.obs when the summary is rebuilt
    pdata.update_summary(recompute=True, verbose=False)
    assert 'cellline' not in pdata.summary.columns
    assert 'cellline' not in pdata.prot.obs.columns
    assert 'cellline' not in pdata.pep.obs.columns
    assert pdata.summary['treatment'].tolist() == full.summary['treatment'].tolist()

def test_import_diann_obs_columns_multiple_none_dropped():
    full = _import_diann_obs(DIANN_OBS)
    obs_columns = ['name', None, 'enzyme', None, None, 'acquisition', 'method', None, 'replicate']
    pdata = _import_diann_obs(obs_columns)
    kept = [c for c in obs_columns if c is not None]
    dropped = {'amt', 'date', 'MS', 'gradient'}

    for df in (pdata.prot.obs, pdata.pep.obs):
        assert list(df.columns[:len(kept)]) == kept  # order preserved
        assert not dropped & set(df.columns)
        for c in kept:
            assert df[c].tolist() == full.prot.obs[c].tolist()

    pdata.update_summary(recompute=True, verbose=False)
    assert not dropped & set(pdata.summary.columns)
    assert not dropped & set(pdata.pep.obs.columns)
    for c in kept:
        assert c in pdata.summary.columns

def test_import_obs_columns_all_none():
    full = _import_pd_obs(PD_OBS)
    pdata = _import_pd_obs([None, None, None])
    # only computed annotation columns remain (none of the filename tokens)
    for df, df_full in ((pdata.prot.obs, full.prot.obs), (pdata.pep.obs, full.pep.obs)):
        assert not set(PD_OBS) & set(df.columns)
        assert list(df.columns) == [c for c in df_full.columns if c not in PD_OBS]
        assert list(df.index) == list(df_full.index)

def test_import_obs_columns_none_length_mismatch_raises():
    with pytest.raises(ValueError):
        _import_pd_obs(['sample', None])
    with pytest.raises(ValueError):
        _import_pd_obs(['sample', None, 'treatment', None])

# --- shared import fixes: empty RS columns, peptide-only, empty-sample warning ---

def test_build_rs_matrix_missing_protein_gets_empty_column(capsys):
    from scpviz.pAnnData.io import _build_rs_matrix
    rs, _ = _build_rs_matrix([["A"], ["A", "B"]], prot_var_names=["B", "C", "A"])
    assert rs.shape == (2, 3)
    assert rs.toarray().tolist() == [[False, False, True], [True, False, True]]
    out = capsys.readouterr().out
    assert "1 protein(s) have no peptides" in out and "C" in out

def test_import_pd_peptide_only():
    test_dir = Path(__file__).parent
    pdata = pAnnData.import_data(source_type='pd', pep_file=str(test_dir / 'test_pd_pep.txt'),
                                 obs_columns=['sample', 'cellline', 'treatment'], fetch_uniprot=False)
    assert pdata.prot is None and pdata.pep is not None and pdata.rs is None

def test_create_from_parts_warns_on_empty_sample(capsys):
    X = sparse.csr_matrix(np.array([[1.0, 2.0], [np.nan, np.nan]]))
    obs = pd.DataFrame({"g": ["a", "b"]})
    pdata = _create_pAnnData_from_parts(
        X, None, None, obs, pd.DataFrame(index=["P1", "P2"]), ["s1", "s2"], ["P1", "P2"],
        obs_columns=["g"], fetch_uniprot=False,
    )
    assert list(pdata.prot.obs_names) == ["s1", "s2"]  # empty sample kept
    out = capsys.readouterr().out
    assert "1 sample(s) have no quantified proteins" in out and "s2" in out

# --- Spectronaut ---

SN_RUNS = ["20260101_a_1.d", "20260101_b_1.d", "20260101_c_1.d"]  # last run is empty

def _sn_write(tmp_path, runs=SN_RUNS, pep_runs=None):
    """Write synthetic protein/peptide Spectronaut pivot reports (q-value block, then quantity block)."""
    nan = np.nan

    def block(ann, runs_, q, x, qname, xname):
        cols = dict(ann)
        for i, r in enumerate(runs_, 1):
            cols[f"[{i}] {r}.{qname}"] = q[r]
        for i, r in enumerate(runs_, 1):
            cols[f"[{i}] {r}.{xname}"] = x[r]
        return pd.DataFrame(cols)

    key = lambda r: r.split("_")[1]  # 'a', 'b', 'c'
    pq = {"a": [0.001, 0.001, 0.5], "b": [0.001, 0.001, 0.5], "c": [0.5, 0.5, 0.5]}
    px = {"a": [10, 5, 1], "b": [20, nan, 2], "c": [nan, nan, nan]}
    prot = block({"PG.ProteinGroups": ["P1;P2", "P3", "P4"], "PG.Genes": ["GA;GB", None, "GD"]}, runs,
                 {r: pq[key(r)] for r in runs}, {r: px[key(r)] for r in runs}, "PG.QValue (Run-Wise)", "PG.Quantity")

    pep_runs = pep_runs or runs
    eq = {"a": [0.001, 0.001, 0.002, 0.001], "b": [0.001, 0.001, 0.002, "Filtered"], "c": ["Filtered"] * 4}
    ex = {"a": [100, 200, 300, 400], "b": [110, 210, 310, "Filtered"], "c": ["Filtered"] * 4}
    pep_ann = {"EG.PrecursorId": ["_AAC[Carbamidomethyl (C)]LK_.2", "_PEPTIDEK_.3", "_M[Oxidation (M)]LK_.2", "_[Acetyl (Protein N-term)]ACK_.2"],
               "PG.ProteinGroups": ["P1;P2", "P1;P2", "P3", "P4"], "PG.Genes": ["GA;GB", "GA;GB", None, "GD"],
               "EG.IonMobility": [0.8, 0.9, 1.0, 1.1]}
    pep = block(pep_ann, pep_runs, {r: eq[key(r)] for r in pep_runs}, {r: ex[key(r)] for r in pep_runs},
                "EG.Qvalue", "EG.TotalQuantity (Settings)")
    p, e = tmp_path / "prot.tsv", tmp_path / "pep.tsv"
    prot.to_csv(p, sep="\t", index=False)
    pep.to_csv(e, sep="\t", index=False)
    return str(p), str(e)

SN_OBS = ["date", "sample", "rep"]

def _sn_import(tmp_path, **kw):
    p, e = _sn_write(tmp_path)
    kw.setdefault("obs_columns", SN_OBS)
    return pAnnData.import_spectronaut(prot_file=p, pep_file=e, fetch_uniprot=False, **kw)

def test_parse_spectronaut_columns():
    from scpviz.pAnnData.io import _parse_spectronaut_columns
    cols = ["PG.ProteinGroups", "[1] r1.d.PG.QValue (Run-Wise)", "[2] r2.raw.EG.TotalQuantity (Settings)",
            "[3] x.y.mzML.PG.Quantity", "[4] plain.PG.Quantity"]
    assert _parse_spectronaut_columns(cols) == [
        ("[1] r1.d.PG.QValue (Run-Wise)", "r1", "PG.QValue"),
        ("[2] r2.raw.EG.TotalQuantity (Settings)", "r2", "EG.TotalQuantity"),
        ("[3] x.y.mzML.PG.Quantity", "x.y", "PG.Quantity"),
        ("[4] plain.PG.Quantity", "plain", "PG.Quantity"),
    ]

def test_import_spectronaut_prot_pep(tmp_path, capsys):
    pdata = _sn_import(tmp_path)
    out = capsys.readouterr().out
    assert pdata.metadata["source"] == "spectronaut"
    assert pdata.metadata["protein_metric"] == "PG.Quantity" and pdata.metadata["peptide_metric"] == "EG.TotalQuantity"
    assert pdata.validate(verbose=False)
    # groups exploded; group string and genes carried to members
    assert list(pdata.prot.var_names) == ["P1", "P2", "P3", "P4"]
    assert pdata.prot.var.loc["P2", "Protein.Group"] == "P1;P2"
    assert pdata.prot.var.loc["P2", "Genes"] == "GA;GB"
    assert pd.isna(pdata.prot.var.loc["P3", "Genes"])
    X = pdata.prot.layers["X_raw"].toarray()
    assert X[:2, 0].tolist() == [10, 20] and X[:2, 1].tolist() == [10, 20]
    assert np.isnan(X[1, 2])
    # sample names / obs from run names (extension stripped); empty run kept and warned
    assert list(pdata.prot.obs_names) == [r[:-2] for r in SN_RUNS]
    assert list(pdata.prot.obs["sample"]) == ["a", "b", "c"]
    assert "no quantified proteins" in out and "20260101_c_1" in out
    # q-values masked where quantity is NaN
    Q = pdata.prot.layers["X_qval"].toarray()
    assert np.isnan(Q[np.isnan(X)]).all() and Q[0, 0] == 0.001
    # peptides: sequences derived, Filtered -> NaN, var names are precursor ids
    assert pdata.pep.var_names[0] == "_AAC[Carbamidomethyl (C)]LK_.2"
    v = pdata.pep.var
    assert v["Modified.Sequence"].tolist() == ["AAC[Carbamidomethyl (C)]LK", "PEPTIDEK", "M[Oxidation (M)]LK", "[Acetyl (Protein N-term)]ACK"]
    assert v["Stripped.Sequence"].tolist() == ["AACLK", "PEPTIDEK", "MLK", "ACK"]
    assert v["Precursor.Charge"].tolist() == [2, 3, 2, 2]
    assert v["Ion.Mobility"].tolist() == [0.8, 0.9, 1.0, 1.1]
    assert "EG.PrecursorId" not in v.columns and "Protein.Group" in v.columns
    PX = pdata.pep.layers["X_raw"].toarray()
    assert np.isnan(PX[1, 3]) and np.isnan(PX[2]).all() and PX[0, 0] == 100
    assert np.isnan(pdata.pep.layers["X_qval"].toarray()[1, 3])
    # RS: a peptide of group P1;P2 maps to both members
    assert pdata.rs.shape == (4, 4)
    assert pdata.rs[:, 0].toarray().ravel().tolist() == [True, True, False, False]
    assert pdata.rs[2, 2] and pdata.rs[3, 3]

def test_import_spectronaut_unquantified_not_significant(tmp_path):
    pdata = _sn_import(tmp_path)
    # P3 in run b: q < 0.01 but quantity NaN -> must not be flagged significant
    assert not pdata.prot.var.loc["P3", "Significant In: 20260101_b_1"]
    assert pdata.prot.var.loc["P1", "Significant In: 20260101_b_1"]

def test_import_spectronaut_pep_runs_reordered(tmp_path):
    p, e = _sn_write(tmp_path, pep_runs=SN_RUNS[::-1])
    pdata = pAnnData.import_spectronaut(prot_file=p, pep_file=e, obs_columns=SN_OBS, fetch_uniprot=False)
    assert pdata.pep.layers["X_raw"].toarray()[0, 0] == 100  # run a still first
    assert list(pdata.pep.obs_names) == list(pdata.prot.obs_names)

def test_import_spectronaut_prot_only_and_pep_only(tmp_path):
    p, e = _sn_write(tmp_path)
    kw = dict(obs_columns=SN_OBS, fetch_uniprot=False)
    a = pAnnData.import_spectronaut(prot_file=p, **kw)
    assert a.pep is None and a.rs is None and a.prot.shape == (3, 4)
    b = pAnnData.import_spectronaut(pep_file=e, **kw)
    assert b.prot is None and b.rs is None and b.pep.shape == (3, 4)
    assert b.pep.var["Modified.Sequence"].iloc[1] == "PEPTIDEK"

def test_import_spectronaut_obs_columns_none_placeholder(tmp_path):
    pdata = _sn_import(tmp_path, obs_columns=["date", None, "rep"])
    assert "sample" not in pdata.prot.obs.columns and "rep" in pdata.prot.obs.columns

def test_import_spectronaut_no_obs_columns_returns_suggestion(tmp_path):
    p, _ = _sn_write(tmp_path)
    assert pAnnData.import_data(source_type="spectronaut", prot_file=p) is None

def test_get_filenames_spectronaut(tmp_path):
    from scpviz.pAnnData.io import get_filenames
    p, _ = _sn_write(tmp_path)
    assert get_filenames(p, "spectronaut") == [r[:-2] for r in SN_RUNS]

def test_import_spectronaut_errors(tmp_path):
    p, e = _sn_write(tmp_path)
    kw = dict(obs_columns=SN_OBS, fetch_uniprot=False)
    with pytest.raises(ValueError, match="At least one"):
        pAnnData.import_spectronaut(**kw)
    with pytest.raises(ValueError, match="Available: .*PG.Quantity"):
        pAnnData.import_spectronaut(prot_file=p, prot_value="PG.Nope", **kw)
    bad = tmp_path / "plain.tsv"
    pd.DataFrame({"PG.ProteinGroups": ["P1"], "x": [1]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="No '"):
        pAnnData.import_spectronaut(prot_file=str(bad), **kw)
    nogroups = tmp_path / "nogroups.tsv"
    pd.read_csv(p, sep="\t").drop(columns="PG.ProteinGroups").to_csv(nogroups, sep="\t", index=False)
    with pytest.raises(ValueError, match="PG.ProteinGroups"):
        pAnnData.import_spectronaut(prot_file=str(nogroups), **kw)
    sub = tmp_path / "two_runs"
    sub.mkdir()
    p2, _ = _sn_write(sub, runs=SN_RUNS[:2])
    with pytest.raises(ValueError, match="different runs"):
        pAnnData.import_spectronaut(prot_file=p2, pep_file=e, **kw)

# --- Spectronaut: edge cases and error branches ---

def _sn_edit(path, fn):
    """Read a synthetic TSV, apply ``fn(df) -> df`` and write it back."""
    df = fn(pd.read_csv(path, sep="\t"))
    df.to_csv(path, sep="\t", index=False)

def _sn_kw():
    return dict(obs_columns=SN_OBS, fetch_uniprot=False)

def test_import_no_input_file_raises_clear_error():
    with pytest.raises(ValueError, match="No input file"):
        pAnnData.import_data(source_type="spectronaut")

def test_import_spectronaut_blank_group_row_dropped(tmp_path, capsys):
    p, e = _sn_write(tmp_path)
    _sn_edit(p, lambda df: df.assign(**{"PG.ProteinGroups": df["PG.ProteinGroups"].where(df.index != 1)}))
    pdata = pAnnData.import_spectronaut(prot_file=p, pep_file=e, **_sn_kw())
    assert "nan" not in list(pdata.prot.var_names) and "P3" not in pdata.prot.var_names
    assert pdata.prot.shape == (3, 3)
    assert "Dropped 1 row(s) with an empty PG.ProteinGroups" in capsys.readouterr().out

def test_import_spectronaut_protein_without_peptides(tmp_path, capsys):
    p, e = _sn_write(tmp_path)
    _sn_edit(e, lambda df: df[df["PG.ProteinGroups"] != "P4"])
    pdata = pAnnData.import_spectronaut(prot_file=p, pep_file=e, **_sn_kw())
    assert "1 protein(s) have no peptides" in capsys.readouterr().out
    assert pdata.rs.shape == (4, 3)
    assert not pdata.rs[list(pdata.prot.var_names).index("P4")].toarray().any()  # empty RS row
    assert pdata.validate(verbose=False)

def test_import_spectronaut_peptide_protein_absent_from_protein_file(tmp_path, capsys):
    p, e = _sn_write(tmp_path)
    _sn_edit(p, lambda df: df[df["PG.ProteinGroups"] != "P4"])
    pdata = pAnnData.import_spectronaut(prot_file=p, pep_file=e, **_sn_kw())
    assert "absent from the protein data" in capsys.readouterr().out
    assert list(pdata.prot.var_names) == ["P1", "P2", "P3"]

def test_import_spectronaut_accession_in_two_groups_keeps_first(tmp_path, capsys):
    p, e = _sn_write(tmp_path)
    _sn_edit(p, lambda df: pd.concat([df, df.iloc[[0]].assign(**{"PG.ProteinGroups": "P1;P9"})], ignore_index=True))
    pdata = pAnnData.import_spectronaut(prot_file=p, pep_file=e, **_sn_kw())
    assert "1 accession(s) occur in more than one protein group" in capsys.readouterr().out
    assert pdata.prot.var_names.is_unique and "P9" in pdata.prot.var_names
    assert pdata.prot.var.loc["P1", "Protein.Group"] == "P1;P2"

def test_import_spectronaut_filtered_in_protein_file_is_nan(tmp_path):
    p, _ = _sn_write(tmp_path)
    xcol = [c for c in pd.read_csv(p, sep="\t", nrows=0).columns if c.endswith("PG.Quantity")][0]
    _sn_edit(p, lambda df: df.astype({xcol: object}).assign(**{xcol: ["Filtered", 5, 1]}))
    pdata = pAnnData.import_spectronaut(prot_file=p, **_sn_kw())
    assert np.isnan(pdata.prot.layers["X_raw"].toarray()[0, 0])

def test_import_spectronaut_missing_genes_column_warns(tmp_path, capsys):
    p, _ = _sn_write(tmp_path)
    _sn_edit(p, lambda df: df.drop(columns="PG.Genes"))
    pdata = pAnnData.import_spectronaut(prot_file=p, **_sn_kw())
    assert "No PG.Genes column" in capsys.readouterr().out
    assert pdata.prot.var["Genes"].isna().all()

def test_import_spectronaut_incomplete_qvalue_columns_warn(tmp_path, capsys):
    p, _ = _sn_write(tmp_path)
    qcol = [c for c in pd.read_csv(p, sep="\t", nrows=0).columns if c.endswith("PG.QValue (Run-Wise)")][0]
    _sn_edit(p, lambda df: df.drop(columns=qcol))
    pdata = pAnnData.import_spectronaut(prot_file=p, **_sn_kw())
    assert "No complete 'PG.QValue' columns" in capsys.readouterr().out
    assert "X_qval" not in pdata.prot.layers

def test_import_spectronaut_optional_peptide_columns_absent(tmp_path):
    p, e = _sn_write(tmp_path)
    _sn_edit(e, lambda df: df.drop(columns=["EG.IonMobility", "PG.Genes"]))
    pdata = pAnnData.import_spectronaut(prot_file=p, pep_file=e, **_sn_kw())
    assert "Ion.Mobility" not in pdata.pep.var.columns and "Genes" not in pdata.pep.var.columns
    assert pdata.validate(verbose=False)

def test_import_spectronaut_nonstandard_precursor_id(tmp_path):
    p, e = _sn_write(tmp_path)
    _sn_edit(e, lambda df: df.assign(**{"EG.PrecursorId": ["WEIRDID"] + df["EG.PrecursorId"].tolist()[1:]}))
    pdata = pAnnData.import_spectronaut(pep_file=e, **_sn_kw())
    v = pdata.pep.var
    assert v.loc["WEIRDID", "Modified.Sequence"] == "WEIRDID" and pd.isna(v.loc["WEIRDID", "Precursor.Charge"])
    assert v["Precursor.Charge"].dropna().tolist() == [3, 2, 2]

def test_import_spectronaut_more_error_paths(tmp_path):
    p, e = _sn_write(tmp_path)
    kw = _sn_kw()
    with pytest.raises(ValueError, match="Available: .*EG.TotalQuantity"):
        pAnnData.import_spectronaut(pep_file=e, pep_value="EG.Nope", **kw)
    noid = tmp_path / "noid.tsv"
    pd.read_csv(e, sep="\t").drop(columns="EG.PrecursorId").to_csv(noid, sep="\t", index=False)
    with pytest.raises(ValueError, match="EG.PrecursorId"):
        pAnnData.import_spectronaut(pep_file=str(noid), **kw)

def test_get_filenames_spectronaut_errors(tmp_path):
    from scpviz.pAnnData.io import get_filenames
    csv = tmp_path / "report.csv"
    csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_filenames(csv, "spectronaut")
    plain = tmp_path / "plain.tsv"
    plain.write_text("PG.ProteinGroups\tx\nP1\t1\n")
    with pytest.raises(ValueError, match="No '"):
        get_filenames(plain, "sn")

import datetime
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scpviz.utils import serialization as ser
from conftest import assert_same_value


def _roundtrip(tmp_path, value):
    pickled = []
    with h5py.File(tmp_path / "v.h5", "w") as f:
        ser.write_value(f, "v", value, pickled=pickled, compression="gzip")
    with h5py.File(tmp_path / "v.h5", "r") as f:
        return ser.read_value(f["v"]), pickled


class Unregistered:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return type(other) is Unregistered and other.x == self.x


def _metrics_df():
    df = pd.DataFrame(index=pd.Index(["P1", "P2", "P3"]))
    df[("BE", "count")] = [1, 2, 3]
    df[("BE", "ratio")] = [0.5, 1.0, np.nan]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _de_df():
    df = pd.DataFrame(
        {
            "Genes": ["A", np.nan, "C"],
            "log2fc": [1.0, -2.0, np.nan],
            "significance": pd.Categorical(["upregulated", "downregulated", "not significant"]),
            "n": pd.array([1, None, 3], dtype="Int64"),
            "flag": [True, False, True],
        },
        index=["P1", "P2", "P3"],
    )
    df.attrs["mixed_de"] = {"contrast_label": "a vs b", "donor": None}
    return df


# Values that must round-trip natively (no pickle), with exact types.
NATIVE_CASES = {
    "none": None,
    "str": "abc",
    "empty_str": "",
    "bool": True,
    "int": 7,
    "float": 1.5,
    "nan": float("nan"),
    "np_float": np.float64(2.5),
    "np_int": np.int32(3),
    "list_str": ["a", "b"],
    "list_int": [1, 2, 3],
    "list_float": [0.1, np.nan],
    "list_bool": [True, False],
    "list_np": [np.float64(1.0), np.float64(2.0)],
    "list_empty": [],
    "list_mixed": ["a", 1, None, 2.5],
    "list_nested": [[1, 2], {"k": "v"}, ("t", 1)],
    "tuple": (1, "a"),
    "set": {"a", "b"},
    "dict_plain": {"b": 1, "a": {"nested": [1, 2]}},
    "dict_order": {"z": 1, "a": 2, "m": 3},
    "dict_empty": {},
    "dict_int_keys": {1: "a", 2: "b"},
    "dict_tuple_keys": {("g1", "count"): 1},
    "dict_slash_key": {"kd/sc": 1, "": 2, ".": 3},
    "dict_repr_key": {"[{'cellline': 'BE'}] vs [{'cellline': 'AS'}]": 1},
    "ndarray_float32": np.arange(6, dtype=np.float32).reshape(2, 3),
    "ndarray_U": np.array(["a", "bb"]),
    "ndarray_obj_str": np.array(["a", None, np.nan, pd.NA], dtype=object),
    "ndarray_bool": np.array([True, False]),
    "ndarray_datetime": np.array(["2026-01-01", "2026-02-01"], dtype="datetime64[D]"),
    "csr_matrix": sparse.csr_matrix(np.eye(3)),
    "csr_array": sparse.csr_array(np.eye(3)),
    "categorical": pd.Categorical(["x", None, "y"], ordered=True),
    "series": pd.Series([1.0, 2.0], index=["a", "b"], name="s"),
    "index": pd.Index(["a", "b"], name="genes"),
    "rangeindex": pd.RangeIndex(0, 5, name="r"),
    "df_de": _de_df(),
    "df_metrics_multiindex": _metrics_df(),
    "df_range_index": pd.DataFrame({"a": [1, 2]}),
    "df_int_columns": pd.DataFrame({0: [1, 2], 1: ["x", "y"]}),
    "df_dup_index": pd.DataFrame({"a": [1, 2]}, index=["x", "x"]),
    "df_multiindex_rows": pd.DataFrame({"a": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])),
    "df_reserved_col": pd.DataFrame({"_index": [1, 2]}),
    "df_empty": pd.DataFrame(),
    "df_mixed_object_col": pd.DataFrame({"mixed": ["a", 1, None, 2.5]}),
    "df_list_cells": pd.DataFrame({"inputGenes": [["A", "B"], ["C"], []], "fdr": [0.1, 0.2, 0.3]}),
    "df_object_floats_with_pdna": pd.DataFrame({"gravy": pd.Series([0.5, pd.NA, -1.2], dtype=object)}),
    "pd_na": pd.NA,
    "timestamp": pd.Timestamp("2026-01-01 10:00", tz="UTC"),
    "datetime": datetime.datetime(2026, 1, 1, 12, 30),
    "date": datetime.date(2026, 1, 1),
    "path": Path("some/file.txt"),
}


@pytest.mark.parametrize("name", list(NATIVE_CASES))
def test_roundtrip_native(tmp_path, name):
    value = NATIVE_CASES[name]
    out, pickled = _roundtrip(tmp_path, value)
    assert pickled == []
    if isinstance(value, set):
        assert out == value and type(out) is set
    else:
        assert_same_value(value, out)


def test_roundtrip_unknown_object_is_pickled_per_value(tmp_path):
    value = {"ok": [1, 2], "odd": Unregistered(3), "df": pd.DataFrame({"cells": [Unregistered(1), "a"]})}
    out, pickled = _roundtrip(tmp_path, value)
    assert out["odd"] == Unregistered(3)
    assert_same_value(value["ok"], out["ok"])
    assert_same_value(value["df"], out["df"])
    # only the offending values fall back to pickle, not the whole dict or column
    assert pickled == ["v['odd']", "v['df']['cells'][0]"]


def test_roundtrip_anndata_all_slots(tmp_path):
    X = np.arange(12, dtype=float).reshape(3, 4)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"group": pd.Categorical(["a", "b", "a"]), "note": ["x", np.nan, "z"]}, index=["s1", "s2", "s3"]),
        var=pd.DataFrame({"Genes": ["G1", None, "G3", "G4"]}, index=["P1", "P2", "P3", "P4"]),
        layers={"X_raw": X * 2, "X_sparse": sparse.csr_matrix(X)},
        obsm={"X_pca": np.ones((3, 2)), "X_ssgsea": pd.DataFrame({"t": [1.0, 2.0, 3.0]}, index=["s1", "s2", "s3"])},
        varm={"PCs": np.zeros((4, 2))},
        obsp={"distances": sparse.csr_matrix(np.eye(3))},
        uns={"metrics": _metrics_df(), "pca": {"variance_ratio": np.array([0.5, 0.2])}, "tuple": (1, 2)},
    )
    out, pickled = _roundtrip(tmp_path, adata)
    assert pickled == []
    assert_same_value(adata, out)


@pytest.mark.parametrize("fixture", ["pdata", "pdata_diann"])
def test_roundtrip_real_prot_pep(tmp_path, fixture, request):
    pdata = request.getfixturevalue(fixture)
    for adata in (pdata.prot, pdata.pep):
        out, pickled = _roundtrip(tmp_path, adata)
        assert pickled == []
        assert_same_value(adata, out)


# --- save() / load() ------------------------------------------------------------

def _handbuilt_pdata():
    """Small pAnnData built without import_data (no metadata['source'])."""
    from scpviz import pAnnData

    rng = np.random.default_rng(0)
    rs = np.zeros((4, 6), dtype=int)
    for j in range(6):
        rs[j % 4, j] = 1
    pdata = pAnnData(prot=rng.random((3, 4)), pep=rng.random((3, 6)), rs=rs)
    pdata.prot.obs_names = pdata.pep.obs_names = ["s1", "s2", "s3"]
    pdata.prot.var_names = [f"P{i}" for i in range(4)]
    pdata.pep.var_names = [f"pep{i}" for i in range(6)]
    pdata.pep.var["Protein.Group"] = [f"P{j % 4}" for j in range(6)]
    pdata.update_summary(recompute=True, verbose=False)
    return pdata


@pytest.fixture
def small_pdata():
    return _handbuilt_pdata()


def test_save_appends_extension_and_load_accepts_bare_path(tmp_path, small_pdata):
    from scpviz import pAnnData
    from conftest import assert_pdata_equal, assert_attrs_classified

    written = small_pdata.save(tmp_path / "run1", verbose=False)
    assert written == tmp_path / "run1.pdata" and written.exists()
    loaded = pAnnData.load(tmp_path / "run1", verbose=False)
    assert_attrs_classified(loaded)
    assert_pdata_equal(small_pdata, loaded, history_tail=1)
    assert "Loaded from run1.pdata" in loaded.history[-1]


def test_save_keeps_explicit_extension(tmp_path, small_pdata):
    assert small_pdata.save(tmp_path / "run1.h5", verbose=False).name == "run1.h5"


def test_save_refuses_overwrite_unless_asked(tmp_path, small_pdata):
    small_pdata.save(tmp_path / "run1", verbose=False)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        small_pdata.save(tmp_path / "run1", verbose=False)
    small_pdata.save(tmp_path / "run1", overwrite=True, verbose=False)
    assert not (tmp_path / "run1.pdata.tmp").exists()


def test_save_has_no_side_effects(tmp_path, small_pdata):
    small_pdata.summary.loc["s1", "note"] = "edited"  # marks summary stale
    assert small_pdata._summary_is_stale
    history_before = list(small_pdata.history)
    obs_before = small_pdata.prot.obs.copy()
    small_pdata.save(tmp_path / "run1", verbose=False)
    assert small_pdata.history == history_before
    assert small_pdata._summary_is_stale
    pd.testing.assert_frame_equal(small_pdata.prot.obs, obs_before)


def test_stale_summary_roundtrips_with_flag(tmp_path, small_pdata):
    from scpviz import pAnnData
    from conftest import assert_pdata_equal

    small_pdata.summary.loc["s1", "note"] = "edited"
    small_pdata.save(tmp_path / "run1", verbose=False)
    loaded = pAnnData.load(tmp_path / "run1", verbose=False)
    assert loaded._summary_is_stale
    assert_pdata_equal(small_pdata, loaded, history_tail=1)
    loaded.summary.loc["s2", "note"] = "x"  # TrackedDataFrame still bound to the loaded object
    assert loaded._summary_is_stale


def test_save_logs_pickled_values(tmp_path, small_pdata, capsys):
    from scpviz import pAnnData

    small_pdata.stats["custom"] = {"obj": Unregistered(1)}
    small_pdata.save(tmp_path / "run1")
    out = capsys.readouterr().out
    assert "stored as pickle" in out and "stats['custom']['obj']" in out
    loaded = pAnnData.load(tmp_path / "run1", verbose=False)
    assert loaded.stats["custom"]["obj"] == Unregistered(1)


def test_load_missing_file_raises(tmp_path):
    from scpviz import pAnnData

    with pytest.raises(FileNotFoundError):
        pAnnData.load(tmp_path / "nope")


def test_load_rejects_non_pdata_file(tmp_path):
    from scpviz import pAnnData

    ad.AnnData(np.ones((2, 2))).write_h5ad(tmp_path / "x.h5ad")
    with pytest.raises(ValueError, match="not a scpviz .pdata file"):
        pAnnData.load(tmp_path / "x.h5ad")


def test_load_rejects_newer_schema(tmp_path, small_pdata):
    from scpviz import pAnnData

    path = small_pdata.save(tmp_path / "run1", verbose=False)
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = ser.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer than this scpviz supports"):
        pAnnData.load(path)


def test_load_warns_on_version_mismatch(tmp_path, small_pdata, capsys):
    from scpviz import pAnnData

    path = small_pdata.save(tmp_path / "run1", verbose=False)
    with h5py.File(path, "a") as f:
        f.attrs["scpviz_version"] = "0.0.1"
    pAnnData.load(path)
    assert "Saved with scpviz v0.0.1" in capsys.readouterr().out


def test_load_missing_source_warns_and_source_arg_sets_it(tmp_path, small_pdata, capsys):
    from scpviz import pAnnData, utils

    path = small_pdata.save(tmp_path / "run1", verbose=False)
    pAnnData.load(path, verbose=False)
    assert "pdata.metadata['source']" in capsys.readouterr().out

    loaded = pAnnData.load(path, source="diann", verbose=False)
    assert loaded.metadata["source"] == "diann"
    assert utils.get_pep_prot_mapping(loaded) == "Protein.Group"


# --- object-level metadata ------------------------------------------------------

def test_handbuilt_metadata_assignment_persists(small_pdata):
    from scpviz import utils

    with pytest.raises(ValueError, match=r"pdata.metadata\['source'\]"):
        utils.get_pep_prot_mapping(small_pdata)
    small_pdata.metadata["source"] = "diann"
    assert utils.get_pep_prot_mapping(small_pdata) == "Protein.Group"


def test_import_stores_metadata_on_object_not_uns(pdata):
    assert pdata.metadata["source"] == "proteomeDiscoverer"
    assert "imported_at" in pdata.metadata
    assert "metadata" not in pdata.prot.uns and "metadata" not in pdata.pep.uns


def test_legacy_uns_metadata_migrates_on_access(small_pdata):
    """Objects from before the move (e.g. unpickled) have metadata in .uns and no _metadata."""
    legacy = {"source": "diann", "imported_at": "2025-01-01"}
    small_pdata.prot.uns["metadata"] = legacy
    small_pdata.pep.uns["metadata"] = legacy
    del small_pdata._metadata
    assert small_pdata.metadata == legacy
    assert "metadata" not in small_pdata.prot.uns and "metadata" not in small_pdata.pep.uns


def test_copy_keeps_metadata_independently(small_pdata):
    small_pdata.metadata["source"] = "diann"
    clone = small_pdata.copy()
    assert clone.metadata == {"source": "diann"}
    clone.metadata["source"] = "maxquant"
    assert small_pdata.metadata["source"] == "diann"


# --- export_h5ad() ---------------------------------------------------------------

def test_export_h5ad_readable_and_warns_about_dropped_and_converted(tmp_path, small_pdata, capsys):
    small_pdata.metadata["source"] = "diann"
    small_pdata.prot.uns["found_metrics_protein"] = _metrics_df().set_axis(small_pdata.prot.var_names[:3])
    small_pdata.stats["de"] = _de_df()
    small_pdata.stats["pair"] = ("a", "b")

    path = small_pdata.export_h5ad(tmp_path / "prot", on="protein")
    out = capsys.readouterr().out
    assert path.name == "prot.h5ad"
    assert "not exported: .pep, .rs, .summary" in out
    assert "MultiIndex columns joined" in out and "DataFrame.attrs dropped" in out and "tuple → list" in out

    adata = ad.read_h5ad(path)
    assert adata.shape == small_pdata.prot.shape
    np.testing.assert_allclose(adata.X, small_pdata.prot.X)
    assert adata.uns["scpviz"]["metadata"]["source"] == "diann"
    assert list(adata.uns["found_metrics_protein"].columns) == ["BE|count", "BE|ratio"]
    assert list(adata.uns["scpviz"]["stats"]["pair"]) == ["a", "b"]
    # the source object is untouched
    assert isinstance(small_pdata.prot.uns["found_metrics_protein"].columns, pd.MultiIndex)
    assert small_pdata.stats["de"].attrs


def test_export_h5ad_peptide_and_overwrite(tmp_path, small_pdata):
    path = small_pdata.export_h5ad(tmp_path / "pep.h5ad", on="peptide", verbose=False)
    assert ad.read_h5ad(path).shape == small_pdata.pep.shape
    with pytest.raises(FileExistsError, match="overwrite=True"):
        small_pdata.export_h5ad(path, on="peptide", verbose=False)
    small_pdata.export_h5ad(path, on="peptide", overwrite=True, verbose=False)


def test_export_h5ad_invalid_on(tmp_path, small_pdata):
    with pytest.raises(ValueError, match="'protein' or 'peptide'"):
        small_pdata.export_h5ad(tmp_path / "x", on="both")


def test_to_h5ad_safe_records_lossy_conversions():
    notes = []
    out = ser.to_h5ad_safe(
        {1: {"a", "b"}, "k/v": [1, "x"], "ts": pd.Timestamp("2026-01-01"), "obj": Unregistered(1)}, "uns", notes
    )
    assert set(out) == {"1", "k|v", "ts", "obj"}
    assert sorted(out["1"]) == ["a", "b"] and out["k|v"] == ["1", "x"] and out["ts"] == "2026-01-01T00:00:00"
    assert out["obj"].startswith("<")
    assert any("key 1" in n for n in notes) and any("set → list" in n for n in notes)
    assert any("mixed-type list" in n for n in notes) and any("stored as repr" in n for n in notes)


def test_to_h5ad_safe_list_of_containers_becomes_writable_dict(tmp_path):
    # e.g. metadata['identifier_map_history'] from update_identifier_maps()
    notes = []
    hist = [{"on": "protein", "input_mapping": {"P1": "G1"}}, {"on": "peptide", "input_mapping": {}}]
    out = ser.to_h5ad_safe({"hist": hist, "nested": [[1, 2], [3]]}, "uns", notes)
    assert out["hist"] == {"0": hist[0], "1": hist[1]}
    assert any("list of containers" in n for n in notes)
    ad.AnnData(X=np.zeros((1, 1)), uns=out).write_h5ad(tmp_path / "x.h5ad")
    assert ad.read_h5ad(tmp_path / "x.h5ad").uns["hist"]["0"]["input_mapping"]["P1"] == "G1"

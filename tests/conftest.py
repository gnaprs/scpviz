import pytest
from scpviz.pAnnData import pAnnData
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import matplotlib

from scpviz.TrackedDataFrame import TrackedDataFrame

import scpviz.setup

matplotlib.use("Agg")

def pytest_configure():
    scpviz.setup.GLOBAL_DEBUG = True

def create_pd_pAnnData():
    """Creates a pd pAnnData object for testing."""
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')
    pep_file = str(test_dir / 'test_pd_pep.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_data(source_type="pd", prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)

    return pdata

@pytest.fixture
def pdata():
    """Fixture to provide a pd pAnnData object."""
    return create_pd_pAnnData()

def create_pd_pAnnData_nopep():
    """Creates a pd pAnnData object for testing with no peptide."""
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_data(source_type="pd", prot_file=prot_file, obs_columns=obs_columns)

    return pdata

@pytest.fixture
def pdata_nopep():
    """Fixture to provide a pd pAnnData object with no peptide."""
    return create_pd_pAnnData_nopep()

def create_diann_pAnnData():
    """Creates a DIA-NN-based pAnnData object with per-sample q-values."""
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / "test_diann.parquet")

    obs_columns = [
        "name", "date", "MS", "acquisition", "FAIMS",
        "column", "gradient", "amt", "region", "replicate",
    ]
    pdata = pAnnData.import_data(
        source_type="diann",
        report_file=diann_file,
        obs_columns=obs_columns,
    )

    # Confirm q-value layers exist (normally DIA-NN always provides these)
    if "X_qval" not in pdata.prot.layers:
        n_obs, n_var = pdata.prot.shape
        pdata.prot.layers["X_qval"] = np.random.rand(n_obs, n_var)

    return pdata

@pytest.fixture
def pdata_diann():
    """DIA-NN–based fixture for tests requiring per-sample q-value data."""
    return create_diann_pAnnData()

# helper for plotting tests
def _is_axes_container(obj):
    """
    Return True if `obj` is a matplotlib Axes or a container (list/array)
    of Axes objects.
    """
    if isinstance(obj, plt.Axes):
        return True
    if isinstance(obj, (list, np.ndarray)):
        return all(isinstance(a, plt.Axes) for a in obj)
    return False

def _count_artists(ax):
    """Return number of drawable artists (patches, lines, collections)."""
    if isinstance(ax, (list, np.ndarray)):
        return sum(len(a.collections) + len(a.patches) + len(a.lines) for a in ax)
    return len(ax.collections) + len(ax.patches) + len(ax.lines)

# helpers for save/load and copy() tests
# Every instance attribute of a pAnnData must be listed here; a new attribute fails
# assert_attrs_classified until it is either persisted by save()/copy() or marked transient.
PDATA_ATTRS = {
    "saved": {"_prot", "_pep", "_rs", "_summary", "_summary_is_stale", "_stats", "_history", "_metadata"},
    "rebuilt": {"_gene_maps_protein", "_protein_maps_peptide"},  # identifier caches, rebuilt lazily
    "transient": set(),
}


def assert_attrs_classified(obj):
    """Fail if `obj` has an instance attribute that save/load and copy() don't account for."""
    known = set().union(*PDATA_ATTRS.values())
    unknown = set(vars(obj)) - known
    assert not unknown, f"Unclassified pAnnData attributes (add to save/load or PDATA_ATTRS): {sorted(unknown)}"


def assert_same_value(a, b, path="value"):
    """Recursively assert `a` and `b` are equal with identical types; report the first mismatch path."""
    assert type(a) is type(b), f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, ad.AnnData):
        assert list(a.obs_names) == list(b.obs_names), f"{path}.obs_names differ"
        assert list(a.var_names) == list(b.var_names), f"{path}.var_names differ"
        for slot in ("X", "obs", "var"):
            assert_same_value(getattr(a, slot), getattr(b, slot), f"{path}.{slot}")
        for slot in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            assert_same_value(dict(getattr(a, slot)), dict(getattr(b, slot)), f"{path}.{slot}")
        assert (a.raw is None) == (b.raw is None), f"{path}.raw presence differs"
        if a.raw is not None:
            assert_same_value(a.raw.to_adata(), b.raw.to_adata(), f"{path}.raw")
    elif isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for k in a:
            assert_same_value(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_value(x, y, f"{path}[{i}]")
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_index_type=True, check_column_type=True, obj=path)
        assert_same_value(a.attrs, b.attrs, f"{path}.attrs")
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, check_index_type=True, obj=path)
        assert_same_value(a.attrs, b.attrs, f"{path}.attrs")
    elif isinstance(a, pd.Index):
        pd.testing.assert_index_equal(a, b, exact=True, obj=path)
    elif isinstance(a, pd.api.extensions.ExtensionArray):
        pd.testing.assert_extension_array_equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{path}: shape {a.shape} != {b.shape}"
        if a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())):
                assert_same_value(x, y, f"{path}.flat[{i}]")
        else:
            np.testing.assert_array_equal(a, b, err_msg=path)
    elif sparse.issparse(a):
        assert a.shape == b.shape and a.dtype == b.dtype, f"{path}: sparse shape/dtype differ"
        np.testing.assert_array_equal(a.toarray(), b.toarray(), err_msg=path)
    elif isinstance(a, (float, np.floating)) and np.isnan(a):
        assert np.isnan(b), f"{path}: {a!r} != {b!r}"
    elif a is None or a is pd.NA:
        pass  # same type already checked
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def assert_pdata_equal(a, b, history_tail=0):
    """
    Assert two pAnnData objects hold the same state (types included).

    `history_tail` entries at the end of `b.history` are ignored (e.g. the "Loaded from" entry).
    """
    assert_same_value(a.prot, b.prot, "prot")
    assert_same_value(a.pep, b.pep, "pep")
    assert_same_value(a.rs, b.rs, "rs")
    assert_same_value(a.stats, b.stats, "stats")
    assert_same_value(a.metadata, b.metadata, "metadata")
    assert_same_value(a.history, b.history[: len(b.history) - history_tail], "history")
    assert type(a._summary) is type(b._summary), "summary: type differs"
    if isinstance(b._summary, TrackedDataFrame):
        assert b._summary._parent is b, "summary: TrackedDataFrame not bound to its parent"
    assert_same_value(pd.DataFrame(a._summary), pd.DataFrame(b._summary), "summary")
    assert a._summary_is_stale == b._summary_is_stale, "summary_is_stale differs"


# skip logic for slow tests
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run tests marked as slow"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")

def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run this test")

"""
Real-data save/load round trips: run an operation on PD / DIA-NN data, save, load, and
assert the loaded object is identical (types included) to the one that was saved.

Network calls (STRING) are stubbed. Slow or optional-dependency operations are marked
``slow`` and skipped when the package is missing; run them locally with ``--runslow``.
"""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from scpviz import pAnnData
from scpviz import plotting as scplt
from conftest import assert_attrs_classified, assert_pdata_equal

PD_DE = [{"cellline": "BE"}, {"cellline": "AS"}]
PD_DE_KEY = "[{'cellline': 'BE'}] vs [{'cellline': 'AS'}]"
DIANN_DE = [{"gradient": "24min"}, {"gradient": "60min"}]
MIXED_DE_KW = dict(observation_level="pseudobulk", min_detected_fraction=0.0, min_cells_detected=1)


def _roundtrip(pdata, tmp_path, capsys):
    """Save + load; assert nothing was pickled and the loaded object matches exactly."""
    capsys.readouterr()
    pdata.save(tmp_path / "rt", overwrite=True)
    out = capsys.readouterr().out
    assert "stored as pickle" not in out, out
    loaded = pAnnData.load(tmp_path / "rt", verbose=False)
    assert_attrs_classified(pdata)
    assert_attrs_classified(loaded)
    assert_pdata_equal(pdata, loaded, history_tail=1)
    return loaded


def _stub_string(monkeypatch):
    """Replace STRING mapping + REST calls with deterministic responses (mappings still cached in .var)."""
    import scpviz.utils as utils_mod

    def fake_map(identifiers, **kwargs):
        accs = list(identifiers)
        return pd.DataFrame({
            "input_identifier": accs,
            "string_identifier": [f"9606.S_{a}" for a in accs],
            "ncbi_taxon_id": ["9606"] * len(accs),
        })

    def fake_post(url, data=None, **kwargs):
        resp = Mock()
        if "ppi_enrichment" in url:
            resp.json.return_value = [{"number_of_nodes": 6, "number_of_edges": 15, "average_node_degree": 5.0,
                                       "local_clustering_coefficient": 0.6, "expected_number_of_edges": 4.2,
                                       "p_value": 1.2e-5}]
        else:
            resp.json.return_value = [{"category": "Process", "term": "GO:0006508", "description": "proteolysis",
                                       "fdr": 1.2e-5, "number_of_genes": 5, "inputGenes": ["A", "B"]}]
        return resp

    monkeypatch.setattr(utils_mod, "get_string_mappings", fake_map)
    monkeypatch.setattr("requests.post", fake_post)


def _local_gene_sets(adata):
    """Gene sets built from the data itself, so gseapy runs offline."""
    genes = adata.var["Genes"].dropna().astype(str).unique().tolist()
    return {"SET_A": genes[:40], "SET_B": genes[40:80], "SET_C": genes[80:120]}


def _close(result):
    plt.close("all")
    return result


# Each op takes (pdata, monkeypatch) and may return a new object (e.g. return_copy filters);
# the returned object (or the mutated input) is what gets round-tripped.
PD_OPS = {
    "annotate_found_protein": lambda p, mp: p.annotate_found(classes="cellline", on="protein", verbose=False),
    "annotate_found_peptide": lambda p, mp: p.annotate_found(classes="cellline", on="peptide", verbose=False),
    "filter_prot": lambda p, mp: p.filter_prot(condition="unique_peptides > 2"),
    "filter_sample": lambda p, mp: p.filter_sample(values={"cellline": "BE"}),
    "filter_rs": lambda p, mp: p.filter_rs(validate_after=False),
    "normalize": lambda p, mp: p.normalize(method="median"),
    "log_transform": lambda p, mp: p.log_transform(set_X=False),
    "impute_mean_classes": lambda p, mp: p.impute(classes=["cellline", "treatment"], method="mean"),
    "impute_knn": lambda p, mp: p.impute(method="knn"),
    "set_X": lambda p, mp: (p.normalize(method="median", set_X=False), p.set_X("X_raw", on="protein")),
    "clean_X": lambda p, mp: p.clean_X(on="prot", set_to=0, inplace=True),
    "pca_protein": lambda p, mp: p.pca(on="protein"),
    "pca_peptide": lambda p, mp: p.pca(on="peptide"),
    "neighbor_umap_leiden": lambda p, mp: (p.neighbor(on="protein"), p.umap(on="protein"),
                                           p.leiden(on="protein", resolution=0.5)),
    "de_ttest": lambda p, mp: p.de(values=PD_DE),
    "de_pep_pairwise_median": lambda p, mp: p.de(values=PD_DE, fold_change_mode="pep_pairwise_median"),
    "pairwise_correlation": lambda p, mp: p.pairwise_correlation(classes="cellline"),
    "cv": lambda p, mp: p.cv(classes="cellline"),
    "rank": lambda p, mp: p.rank(classes="cellline"),
    "clustered_heatmap": lambda p, mp: _close(scplt.plot_clustered_heatmap(
        p, classes=["treatment"], proteins=p.prot.var["Genes"].dropna().astype(str).unique()[:20].tolist())),
    "clustermap": lambda p, mp: _close(p.plot_clustermap(classes=["cellline"])),
    "update_identifier_maps": lambda p, mp: p.update_identifier_maps(
        {p.prot.var_names[0]: "NEWGENE"}, on="protein", direction="reverse", overwrite=True, verbose=False),
    "update_string_mapping": lambda p, mp: p.update_string_mapping({p.prot.var_names[0]: "9606.NEW"}, verbose=False),
    "enrichment_functional": lambda p, mp: (_stub_string(mp), p.enrichment_functional(
        genes=p.prot.var["Genes"].dropna().astype(str).tolist()[:10], from_de=False)),
    "enrichment_ppi": lambda p, mp: (_stub_string(mp), p.enrichment_ppi(
        genes=p.prot.var["Genes"].dropna().astype(str).tolist()[:10])),
    "peptide_properties": lambda p, mp: p.get_peptide_properties(accessions=list(p.prot.var_names[:25])),
}

DIANN_OPS = {
    "annotate_significant": lambda p, mp: p.annotate_significant(classes=["amt"], fdr_threshold=0.05,
                                                                 on="protein", verbose=False),
    "annotate_found": lambda p, mp: p.annotate_found(classes="gradient", on="protein", verbose=False),
    "normalize_impute": lambda p, mp: (p.normalize(method="median"), p.impute(method="mean")),
    "de_ttest": lambda p, mp: p.de(values=DIANN_DE),
    "mixed_de_contrast": lambda p, mp: p.mixed_de(group_col="gradient", contrast=("24min", "60min"),
                                                  donor_col="replicate", store=True, **MIXED_DE_KW),
    "pairwise_correlation": lambda p, mp: p.pairwise_correlation(classes="gradient"),
    "pca": lambda p, mp: p.pca(on="protein"),
}

# Pre-existing failures unrelated to save/load; xfail only on these exact exceptions so a
# serialization error in the same test still fails.
_HARMONY_XFAIL = pytest.mark.xfail(
    raises=InvalidParameterError, strict=False,
    reason="harmony() can't pass nclust; harmonypy picks n_clusters=0 for 12 samples (same xfail as test_pAnnData_analysis)")
_PIMMS_XFAIL = pytest.mark.xfail(
    raises=OSError, strict=False,
    reason="torch DLL (c10.dll) fails to load when imported after scpviz in this env; existing pimms tests fail too")

SLOW_OPS = {
    ("pdata", "harmony"): (None, _HARMONY_XFAIL,
                           lambda p, mp: (p.pca(on="protein"), p.harmony(key="cellline", on="protein"))),
    ("pdata", "ssgsea"): ("gseapy", None, lambda p, mp: p.ssgsea(on="protein", gene_sets=_local_gene_sets(p.prot),
                                                                min_size=5)),
    ("pdata", "pca_gsea"): ("gseapy", None, lambda p, mp: (p.pca(on="protein"), p.pca_gsea(
        on="protein", pcs=[1], gene_sets=_local_gene_sets(p.prot), min_size=5, permutation_num=20))),
    ("pdata", "impute_pimms"): ("pimmslearn", _PIMMS_XFAIL, lambda p, mp: p.impute(method="pimms_dae")),
    ("pdata_diann", "harmony"): (None, _HARMONY_XFAIL,
                                 lambda p, mp: (p.pca(on="protein"), p.harmony(key="gradient", on="protein"))),
    ("pdata_diann", "mixed_de_pairwise"): (None, None, lambda p, mp: p.mixed_de(
        group_col="gradient", contrast_mode="pairwise", donor_col="replicate", store=True, **MIXED_DE_KW)),
}

OP_PARAMS = (
    [pytest.param("pdata", op, id=f"pd-{name}") for name, op in PD_OPS.items()]
    + [pytest.param("pdata_diann", op, id=f"diann-{name}") for name, op in DIANN_OPS.items()]
    + [pytest.param(fx, op, id=f"{'pd' if fx == 'pdata' else 'diann'}-{name}",
                    marks=[pytest.mark.slow] + ([xfail] if xfail else []))
       for (fx, name), (_, xfail, op) in SLOW_OPS.items()]
)
SLOW_DEPS = {f"{'pd' if fx == 'pdata' else 'diann'}-{name}": dep for (fx, name), (dep, _, _) in SLOW_OPS.items()}


@pytest.mark.parametrize("fixture,op", OP_PARAMS)
def test_roundtrip_after_operation(request, monkeypatch, tmp_path, capsys, fixture, op):
    dep = SLOW_DEPS.get(request.node.callspec.id)
    if dep:
        pytest.importorskip(dep)
    pdata = request.getfixturevalue(fixture)
    result = op(pdata, monkeypatch)
    target = result if isinstance(result, pAnnData) else pdata
    _roundtrip(target, tmp_path, capsys)


def _run_chain(pdata, monkeypatch):
    """A realistic analysis session touching most stores (.layers, .obsm, .uns, .stats, metadata)."""
    pdata.annotate_found(classes="cellline", on="protein", verbose=False)
    pdata.normalize(method="median")
    pdata.log_transform(set_X=True)
    pdata.impute(classes=["cellline"], method="mean")
    pdata.pca(on="protein")
    pdata.neighbor(on="protein")
    pdata.umap(on="protein")
    pdata.leiden(on="protein", resolution=0.5)
    pdata.de(values=PD_DE)
    _stub_string(monkeypatch)
    pdata.enrichment_functional(genes=pdata.prot.var["Genes"].dropna().astype(str).tolist()[:10], from_de=False)
    pdata.pairwise_correlation(classes="cellline")
    _close(scplt.plot_clustered_heatmap(pdata, classes=["treatment"], stats_key=PD_DE_KEY))
    pdata.update_identifier_maps({pdata.prot.var_names[0]: "NEWGENE"}, on="protein", direction="reverse",
                                 overwrite=True, verbose=False)
    return pdata


def test_roundtrip_after_analysis_chain_and_loaded_object_still_works(pdata, monkeypatch, tmp_path, capsys):
    _run_chain(pdata, monkeypatch)
    loaded = _roundtrip(pdata, tmp_path, capsys)

    # The loaded object must remain usable, not just equal.
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, pdata=loaded, stats_key=PD_DE_KEY)
    plt.close(fig)
    filtered = loaded.filter_prot_found(group="BE", min_ratio=0.5, on="protein", return_copy=True, verbose=False)
    assert 0 < filtered.prot.n_vars <= loaded.prot.n_vars
    _stub_string(monkeypatch)
    loaded.enrichment_functional(genes=loaded.prot.var["Genes"].dropna().astype(str).tolist()[:5], from_de=False)
    fresh = loaded.de(values=PD_DE)
    numeric = ["log2fc", "p_value", "test_statistic"]  # 'Genes' changed via update_identifier_maps after the first de()
    pd.testing.assert_frame_equal(fresh[numeric], pdata.stats[PD_DE_KEY][numeric])
    assert loaded.prot.uns["current_X_layer"] == pdata.prot.uns["current_X_layer"]
    assert any("Loaded from rt.pdata" in h for h in loaded.history)
    assert "DE for" in loaded.history[-1]


def test_copy_preserves_full_state_after_analysis_chain(pdata, monkeypatch):
    _run_chain(pdata, monkeypatch)
    # leiden() adds obs['leiden'] without refreshing .summary; copy() then re-merges the summary and pushes it
    # back to .obs (so prot.obs['leiden'] reaches pep.obs). Reproduce that sync first to compare like with like.
    pdata.update_summary(recompute=True, verbose=False)
    pdata.update_summary(recompute=True, sync_back=True, verbose=False)
    clone = pdata.copy()
    assert_attrs_classified(clone)
    # copy() re-syncs the summary through the setter, which appends a "Pushed summary values back to obs" entry.
    assert "Pushed summary values back to obs" in clone.history[-1]
    assert_pdata_equal(pdata, clone, history_tail=1)

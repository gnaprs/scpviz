"""
Generate static figures for scpviz API documentation.

Run from the repo root:
    conda activate py311-dev
    python docs/generate_plot_figures.py
    python docs/generate_plot_figures.py --skip-umap   # skips SC import, directlfq, plot_umap.png, and plot_*_sc.png

On Windows, if you see ``UnicodeEncodeError`` from console logging, run with UTF-8 output, for example:
    ``$env:PYTHONIOENCODING='utf-8'; python docs/generate_plot_figures.py``

**UMAP data:** Prefer the large DIA-NN cohort under ``SCPVIZ_UMAP_DATA_ROOT`` (must contain
``data/2505_rna_prot_full/report.tsv`` and ``abc_analysis/file_annotation.csv``). If unset, the
script uses ``<repo>/../3. Results/5. Analysis/2310_PD`` (path relative to the repo root). If those
are missing, it falls back to ``docs/assets/report_sc.parquet`` when present.

When a single-cell bundle loads (large cohort or parquet), the script also writes
``plot_pca_sc.png``, ``plot_pairwise_correlation_sc.png``, ``plot_rankquant_sc.png``, and
``plot_raincloud_sc.png`` (separate from bulk ``plot_pca.png``, ``plot_pairwise_correlation.png``, etc.).

Output: docs/assets/plots/<name>.png at 150 dpi.
Also writes ``plot_abundance_boxgrid_bar.png``, ``plot_abundance_boxgrid_line.png``,
``plot_abundance_boxgrid_violin.png``, ``plot_abundance_boxgrid_custom.png``,
``plot_abundance_boxgrid_significance.png``, and ``plot_abundance_boxgrid_significance_multi.png``
alongside ``plot_abundance_boxgrid.png``,
and ``plot_cv_annotate.png`` / ``plot_cv_custom_annotate.png`` alongside ``plot_cv.png``.
Existing files are overwritten.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from scpviz import pAnnData, plotting as scplt
from scpviz import utils as scutils

ASSETS_DOC = REPO / "docs" / "assets"
TESTS = REPO / "tests"
OUT = REPO / "docs" / "assets" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 150

# Per-protein z-score across samples (samples × proteins), for pairwise correlation on .X-like data.
_PAIRWISE_Z_LAYER = "X_pw_zscore"


def _ensure_pairwise_zscore_layer(pdata_norm: pAnnData, *, layer_out: str = _PAIRWISE_Z_LAYER) -> str:
    """Write ``layer_out``: each protein centered/scaled across samples (current ``X``)."""
    adata = scutils.get_adata(pdata_norm, "protein")
    X = scutils.get_adata_layer(adata, "X")
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sig = np.nanstd(X, axis=0, keepdims=True)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)
    Z = (X - mu) / sig
    adata.layers[layer_out] = Z
    return layer_out


def save_current_fig(name: str) -> None:
    plt.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")


def load_bulk():
    """Return (pdata, pdata_norm, meta) where meta has keys: obs_columns, group2 (condition|treatment)."""
    prot_doc = ASSETS_DOC / "pd32_Proteins.txt"
    pep_doc = ASSETS_DOC / "pd32_PeptideSequenceGroups.txt"
    if prot_doc.is_file() and pep_doc.is_file():
        obs_columns = ["Sample", "cellline", "treatment", "condition", "duration"]
        pdata_raw = pAnnData.import_data(
            source_type="pd",
            prot_file=str(prot_doc),
            pep_file=str(pep_doc),
            obs_columns=obs_columns,
        )
        pdata = pdata_raw.filter_sample(min_prot=8000)
        g2 = "condition"
        cmp_vals = [
            {"cellline": "BE", "condition": "kd"},
            {"cellline": "BE", "condition": "sc"},
        ]
    else:
        prot_file = str(TESTS / "test_pd_prot.txt")
        pep_file = str(TESTS / "test_pd_pep.txt")
        obs_columns = ["sample", "cellline", "treatment"]
        pdata = pAnnData.import_data(
            source_type="pd",
            prot_file=prot_file,
            pep_file=pep_file,
            obs_columns=obs_columns,
        )
        g2 = "treatment"
        cmp_vals = [
            {"cellline": "BE", "treatment": "kd"},
            {"cellline": "BE", "treatment": "sc"},
        ]
    pdata_norm = pdata.copy()
    pdata_norm.normalize(method="median")
    pdata_norm.impute(method="min")
    meta = {"obs_columns": obs_columns, "group2": g2, "comparison_values": cmp_vals}
    return pdata, pdata_norm, meta


def _umap_data_root() -> Path:
    """Root folder containing ``data/2505_rna_prot_full/`` and ``abc_analysis/``."""
    env = os.environ.get("SCPVIZ_UMAP_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO.parent / "3. Results" / "5. Analysis" / "2310_PD").resolve()


def _try_load_large_umap_cohort() -> tuple[pAnnData, dict] | None:
    """
    Load the large DIA-NN + file_annotation pipeline for ``plot_umap`` only.

    Returns ``(pdata_norm, umap_kw)`` or ``None`` if paths are missing or loading fails.
    """
    import pandas as pd

    root = _umap_data_root()
    report_tsv = root / "data" / "2505_rna_prot_full" / "report.tsv"
    ann_csv = root / "abc_analysis" / "file_annotation.csv"
    if not report_tsv.is_file() or not ann_csv.is_file():
        return None

    ann_cols = pd.read_csv(ann_csv, nrows=0).columns
    raw_col = next(
        (c for c in ("RAW FILE", "Raw file", "raw file", "Raw File") if c in ann_cols),
        None,
    )
    if raw_col is None:
        print(
            f"{scutils.format_log_prefix('warn')} file_annotation.csv has no recognizable raw-file column; "
            f"skipping large UMAP cohort."
        )
        return None

    try:
        print(
            f"{scutils.format_log_prefix('user')} Loading large DIA-NN report for UMAP "
            f"({report_tsv.name} under {root})…"
        )
        pdata_all = pAnnData.import_data(
            source_type="diann",
            report_file=str(report_tsv),
        )
        file_annotation = pd.read_csv(ann_csv)
        file_annotation["parsed_filename"] = (
            file_annotation[raw_col].apply(os.path.basename).str.replace(".raw", "", regex=False)
        )
        summary = pdata_all.summary
        summary_files = set(summary.index)
        annotation_files = set(file_annotation["parsed_filename"])
        missing_in_annotation = summary_files - annotation_files
        pdata_f = pdata_all.filter_sample(exclude_file_list=list(missing_in_annotation))

        fa_sub = file_annotation[
            ["parsed_filename", "File Name", "Grouping", "Sub grouping", "Region", "Batch"]
        ]
        fa_sub = fa_sub.set_index("parsed_filename")
        fa_aligned = fa_sub.reindex(pdata_f.summary.index)
        cols_to_add = ["File Name", "Grouping", "Sub grouping", "Region", "Batch"]
        for c in cols_to_add:
            pdata_f.summary[c] = fa_aligned[c]
        pdata_f.update_summary()

        p2 = pdata_f.filter_prot_significant()
        p2 = p2.filter_sample(condition="Grouping != 'ast'")
        p2 = p2.filter_sample(min_prot=1000)
        p2.summary["region"] = p2.summary["Grouping"].replace(
            {"snpc": "SNpc", "ctx": "Cortex"}
        )
        p2.update_summary()

        proc = p2.copy()
        proc = proc.filter_prot_found(min_ratio=0.4, group=["Grouping"], match_any=True)
        proc = proc.filter_prot(valid_genes=True, unique_profiles=True)
        out = proc.copy()
        out.normalize(method="directlfq", on="protein")
    except Exception as exc:
        print(
            f"{scutils.format_log_prefix('warn')} Large UMAP cohort failed ({exc!r}); "
            f"falling back to report_sc.parquet if available."
        )
        return None

    umap_kw: dict = {
        "color": ["region"],
        "cmap": {"Cortex": "#D19DCB", "SNpc": "#85BE9E"},
        "umap_params": {"min_dist": 0.3, "n_neighbors": 30, "random_state": 42},
        "figsize": (4.5, 4),
        "s": 10,
        "alpha": 0.85,
    }
    return out, umap_kw


def _try_load_report_sc_parquet() -> tuple[pAnnData, dict] | None:
    sc_parquet = ASSETS_DOC / "report_sc.parquet"
    if not sc_parquet.is_file():
        return None
    obs_columns_sc = ["date", "acquisition", "size", "cell_type", "replicate"]
    pdata_sc_raw = pAnnData.import_data(
        source_type="diann",
        report_file=str(sc_parquet),
        obs_columns=obs_columns_sc,
    )
    pdata_sc = pdata_sc_raw.filter_sample(condition="cell_type not in ['T2','TEAB35']")
    mapping = {"scartissue": "SWI", "astrocyte": "Uninjured"}
    pdata_sc.summary["condition"] = pdata_sc.summary["cell_type"].replace(mapping)
    pdata_sc.update_summary()
    pdata_sc = pdata_sc.filter_prot_significant()
    pdata_sc = pdata_sc.filter_sample(min_prot=1000)
    pdata_sc = pdata_sc.filter_prot(valid_genes=True, unique_profiles=True)
    pdata_sc = pdata_sc.filter_prot_found(min_ratio=0.4, group=["condition"], match_any=True)
    pdata_sc_norm = pdata_sc.copy()
    pdata_sc_norm.normalize(method="directlfq", on="protein")
    umap_kw: dict = {
        "color": ["condition"],
        "cmap": {"SWI": "#E07B6A", "Uninjured": "#6AB4E0"},
        "umap_params": {"min_dist": 0.3, "n_neighbors": 7, "random_state": 42},
        "figsize": (3, 3),
        "s": 20,
        "alpha": 0.8,
    }
    return pdata_sc_norm, umap_kw


def load_sc_for_umap(*, skip_umap: bool = False) -> tuple[pAnnData, dict] | None:
    """Return ``(pdata_norm, umap_kw)`` for ``plot_umap``, or ``None`` to skip."""
    if skip_umap:
        return None
    large = _try_load_large_umap_cohort()
    if large is not None:
        return large
    return _try_load_report_sc_parquet()


def _mark_df_three_genes(pdata):
    var = pdata.prot.var
    if "Genes" not in var.columns:
        acc = list(var.index[:3])
    else:
        want = ["GAPDH", "TUBB", "ACTB"]
        m = var["Genes"].astype(str).isin(want)
        acc = list(var.index[m][:3])
        if len(acc) < 3:
            acc = list(var.index[:3])
    sub = var.loc[acc].copy()
    sub = sub.reset_index()
    id_col = "index" if "index" in sub.columns else sub.columns[0]
    ren = {id_col: "accession"}
    if "Genes" in sub.columns:
        ren["Genes"] = "gene_primary"
    sub = sub.rename(columns=ren)
    cols = [c for c in ["accession", "gene_primary"] if c in sub.columns]
    return sub[cols]


def _single_cell_supplement_figures(pdata_sc: pAnnData, umap_kw: dict) -> None:
    """
    Extra figures on the single-cell DIA-NN object (separate stems from bulk PD).

    Writes ``plot_pca_sc.png``, ``plot_pairwise_correlation_sc.png``,
    ``plot_rankquant_sc.png``, ``plot_raincloud_sc.png``.
    Expects ``pdata_sc.pca(on='protein')`` to have been run already.
    """
    color_cols = umap_kw.get("color")
    if not color_cols:
        return
    classes_sc = list(color_cols) if isinstance(color_cols, (list, tuple)) else [color_cols]
    cmap_sc = umap_kw.get("cmap", "default")

    fig, ax = plt.subplots(figsize=(4, 4))
    scplt.plot_pca(
        ax,
        pdata_sc,
        color=classes_sc,
        cmap=cmap_sc,
        add_ellipses=True,
    )
    plt.savefig(OUT / "plot_pca_sc.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    pw_layer_sc = _ensure_pairwise_zscore_layer(pdata_sc)
    fig, ax = scplt.plot_pairwise_correlation(
        pdata_sc,
        classes=classes_sc,
        method="pearson",
        show_samples=True,
        layer=pw_layer_sc,
        force=True,
    )
    plt.savefig(OUT / "plot_pairwise_correlation_sc.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    class_list_sc = scutils.get_classlist(pdata_sc.prot, classes_sc)
    sc_rain = [cm.tab10(i % 10) for i in range(len(class_list_sc))]

    fig, ax = plt.subplots(figsize=(4, 4))
    scplt.plot_rankquant(ax, pdata_sc, classes=classes_sc)
    plt.savefig(OUT / "plot_rankquant_sc.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(5, 4))
    scplt.plot_raincloud(ax, pdata_sc, classes=classes_sc, color=sc_rain)
    plt.savefig(OUT / "plot_raincloud_sc.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")


def main(*, skip_umap: bool = False) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    import pandas as pd

    pdata, pdata_norm, meta = load_bulk()
    g2 = meta["group2"]
    classes_2 = ["cellline", g2]
    comparison_values = meta["comparison_values"]

    class_list = scutils.get_classlist(pdata.prot, classes_2)
    rain_colors = [cm.tab10(i % 10) for i in range(len(class_list))]

    # ── style.py ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3))
    scplt.plot_summary(ax, pdata, classes=classes_2)
    save_current_fig("plot_summary")

    fig, ax = plt.subplots(figsize=(2, 3))
    ax.bar([0, 1], [10, 15])
    scplt.plot_significance(ax, 16.0, 1.0, x1=0, x2=1, pval="*")
    save_current_fig("plot_significance")

    # ── abundance.py ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3, 3))
    scplt.plot_cv(ax, pdata, classes=classes_2)
    save_current_fig("plot_cv")

    fig, ax = plt.subplots(figsize=(3, 3))
    scplt.plot_cv(
        ax,
        pdata,
        classes=classes_2,
        show_n=True,
        annotate="median",
        annotate_kwargs={"fontsize": 7},
    )
    save_current_fig("plot_cv_annotate")

    fig, ax = plt.subplots(figsize=(3, 3))
    scplt.plot_cv(
        ax,
        pdata,
        classes=classes_2,
        annotate={"AS_kd": "replicate set A"},
    )
    save_current_fig("plot_cv_custom_annotate")

    fig, ax = plt.subplots(figsize=(4, 4))
    scplt.plot_abundance(
        ax, pdata, namelist=["GAPDH", "TUBB", "ACTB"], classes=classes_2
    )
    save_current_fig("plot_abundance")

    fig, ax = plt.subplots(figsize=(5, 4))
    scplt.plot_abundance_housekeeping(ax, pdata, classes=classes_2)
    save_current_fig("plot_abundance_housekeeping")

    fig, axes = pdata.plot_abundance_boxgrid(
        namelist=["GAPDH", "TUBB", "ACTB"],
        classes=classes_2,
        plot_type="box",
        figsize=(2, 2.5),
    )
    fig.savefig(OUT / "plot_abundance_boxgrid.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    fig, axes = pdata.plot_abundance_boxgrid(
        namelist=["GAPDH", "TUBB", "ACTB"],
        classes=classes_2,
        plot_type="bar",
        bar_error="sd",
        bar_kwargs={"width": 0.14},
        figsize=(2, 2.5),
    )
    fig.savefig(OUT / "plot_abundance_boxgrid_bar.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    fig, axes = pdata.plot_abundance_boxgrid(
        namelist=["GAPDH", "TUBB", "ACTB"],
        classes=classes_2,
        plot_type="line",
        show_n=True,
        hline_kwargs={"half_width": 0.08},
        figsize=(2, 2.5),
    )
    fig.savefig(OUT / "plot_abundance_boxgrid_line.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    fig, axes = pdata.plot_abundance_boxgrid(
        namelist=["GAPDH", "TUBB", "ACTB"],
        classes=classes_2,
        plot_type="violin",
        figsize=(2, 2.5),
    )
    fig.savefig(OUT / "plot_abundance_boxgrid_violin.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    fig, axes = pdata.plot_abundance_boxgrid(
        namelist=["GAPDH", "TUBB", "ACTB"],
        classes=classes_2,
        plot_type="box",
        box_kwargs={"boxprops": {"alpha": 0.45}, "linewidth": 1.2},
        strip_kwargs={"size": 4, "alpha": 0.6},
        figsize=(2, 2.5),
    )
    fig.savefig(OUT / "plot_abundance_boxgrid_custom.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    sig_pairs_doc = [
        ({"cellline": "BE", g2: "sc"}, {"cellline": "BE", g2: "kd"}),
        ({"cellline": "AS", g2: "sc"}, {"cellline": "AS", g2: "kd"}),
    ]
    fig, axes = pdata.plot_abundance_boxgrid(
        namelist=["GAPDH", "TUBB", "ACTB"],
        classes=classes_2,
        plot_type="box",
        figsize=(2, 2.5),
        sig_pairs=sig_pairs_doc,
        sig_kwargs={"fontsize": 8},
    )
    fig.savefig(OUT / "plot_abundance_boxgrid_significance.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    sig_pairs_multi = [
        ({"cellline": "BE", g2: "sc"}, {"cellline": "BE", g2: "kd"}),
        ({"cellline": "BE", g2: "kd"}, {"cellline": "AS", g2: "kd"}),
    ]
    fig, axes = pdata.plot_abundance_boxgrid(
        namelist=["GAPDH", "TUBB", "ACTB"],
        classes=classes_2,
        plot_type="box",
        figsize=(2, 2.5),
        sig_pairs=sig_pairs_multi,
        sig_kwargs={"fontsize": 8},
    )
    fig.savefig(
        OUT / "plot_abundance_boxgrid_significance_multi.png", dpi=DPI, bbox_inches="tight"
    )
    plt.close("all")

    fig, ax = plt.subplots(figsize=(4, 4))
    scplt.plot_rankquant(ax, pdata, classes=classes_2)
    save_current_fig("plot_rankquant")

    mark_df = _mark_df_three_genes(pdata)
    fig, ax = plt.subplots(figsize=(4, 4))
    scplt.plot_rankquant(ax, pdata, classes=classes_2)
    scplt.mark_rankquant(
        ax,
        pdata,
        mark_df=mark_df,
        class_values=class_list[: min(4, len(class_list))],
        color="black",
        label_type="gene",
    )
    save_current_fig("mark_rankquant")

    fig, ax = plt.subplots(figsize=(5, 4))
    scplt.plot_raincloud(ax, pdata, classes=classes_2, color=rain_colors)
    save_current_fig("plot_raincloud")

    fig, ax = plt.subplots(figsize=(5, 4))
    scplt.plot_raincloud(ax, pdata, classes=classes_2, color=rain_colors)
    scplt.mark_raincloud(
        ax,
        pdata,
        mark_df=mark_df,
        class_values=class_list[: min(4, len(class_list))],
        color="black",
    )
    save_current_fig("mark_raincloud")

    # ── dimreduc.py ──────────────────────────────────────────────────────────
    pdata_norm.pca(on="protein")
    fig, ax = plt.subplots(figsize=(4, 4))
    scplt.plot_pca(ax, pdata_norm, classes=classes_2, add_ellipses=True)
    save_current_fig("plot_pca")

    fig, ax = plt.subplots(figsize=(4, 3))
    scplt.plot_pca_scree(ax, pdata_norm.prot.uns["pca"])
    save_current_fig("plot_pca_scree")

    fig, ax = plt.subplots(figsize=(4, 4))
    scplt.plot_pca_protein_vectors(ax, pdata_norm, n_vectors=10)
    save_current_fig("plot_pca_protein_vectors")

    umap_bundle = load_sc_for_umap(skip_umap=skip_umap)
    if umap_bundle is not None:
        pdata_umap, umap_kw = umap_bundle
        pdata_umap.pca(on="protein")
        _single_cell_supplement_figures(pdata_umap, umap_kw)
        fig, ax = plt.subplots(figsize=umap_kw.get("figsize", (3, 3)))
        scplt.plot_umap(
            ax,
            pdata_umap,
            color=umap_kw["color"],
            s=umap_kw.get("s", 20),
            alpha=umap_kw.get("alpha", 0.8),
            cmap=umap_kw["cmap"],
            force=True,
            umap_params=umap_kw.get("umap_params", {}),
        )
        scplt.shift_legend(ax)
        save_current_fig("plot_umap")
    else:
        if skip_umap:
            print(
                f"{scutils.format_log_prefix('info_only')} Skipping plot_umap (--skip-umap)."
            )
        else:
            print(
                f"{scutils.format_log_prefix('warn')} Skipping plot_umap: no large cohort at "
                f"{_umap_data_root()}, no {ASSETS_DOC / 'report_sc.parquet'}, or load failed."
            )

    # ── correlation.py ───────────────────────────────────────────────────────
    pw_layer = _ensure_pairwise_zscore_layer(pdata_norm)
    fig, ax = scplt.plot_pairwise_correlation(
        pdata_norm,
        classes=classes_2,
        method="pearson",
        show_samples=True,
        layer=pw_layer,
        force=True,
    )
    plt.savefig(OUT / "plot_pairwise_correlation.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    _, ax_cm = plt.subplots(figsize=(1, 1))
    g_cm = scplt.plot_clustermap(
        ax_cm,
        pdata_norm,
        on="prot",
        classes=classes_2,
        force=True,
        impute="row_min",
        z_score=0,
        center=0,
        linewidth=0,
        figsize=(10, 6),
    )
    g_cm.savefig(OUT / "plot_clustermap.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    # ── volcano.py ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    ax, volcano_df = scplt.plot_volcano(
        ax, pdata_norm, values=comparison_values, return_df=True
    )
    save_current_fig("plot_volcano")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax, volcano_df = scplt.plot_volcano(
        ax,
        pdata_norm,
        values=comparison_values,
        return_df=True,
        label=[0, 0],
    )
    labels = ["GAPDH", "TUBB", "ACTB"]
    gene_ids = (
        set(volcano_df["Genes"].astype(str))
        if "Genes" in volcano_df.columns
        else set()
    )
    acc_ids = set(volcano_df.index.astype(str))
    ok = [g for g in labels if g in gene_ids or g in acc_ids]
    if not ok:
        ok = list(volcano_df.index[:3])
    scplt.mark_volcano(ax, volcano_df, label=ok[:3])
    save_current_fig("mark_volcano")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax, volcano_df = scplt.plot_volcano(
        ax,
        pdata_norm,
        values=comparison_values,
        return_df=True,
        label=[0, 0],
    )
    up_ids = (
        volcano_df[volcano_df["significance"] == "upregulated"]
        .sort_values("significance_score", ascending=False)
        .head(5)
        .index.tolist()
    )
    down_ids = (
        volcano_df[volcano_df["significance"] == "downregulated"]
        .sort_values("significance_score", ascending=True)
        .head(5)
        .index.tolist()
    )
    scplt.mark_volcano_by_significance(ax, volcano_df, label=up_ids + down_ids)
    save_current_fig("mark_volcano_by_significance")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax, volcano_df = scplt.plot_volcano(
        ax, pdata_norm, values=comparison_values, return_df=True, no_marks=True
    )
    texts = []
    up_df = volcano_df[volcano_df["significance"] == "upregulated"].sort_values(
        "significance_score", ascending=False
    ).head(5)
    down_df = volcano_df[volcano_df["significance"] == "downregulated"].sort_values(
        "significance_score", ascending=True
    ).head(5)
    reds = {
        "upregulated": "#c0392b",
        "downregulated": "lightgrey",
        "not significant": "lightgrey",
    }
    blues = {
        "downregulated": "#2980b9",
        "upregulated": "lightgrey",
        "not significant": "lightgrey",
    }
    greys = {
        "not significant": "#7f8c8d",
        "upregulated": "lightgrey",
        "downregulated": "lightgrey",
    }
    if len(up_df):
        ax, t = scplt.mark_volcano_by_significance(
            ax, volcano_df, label=up_df.index.tolist(), color=reds, return_texts=True
        )
        texts.extend(t)
    if len(down_df):
        ax, t = scplt.mark_volcano_by_significance(
            ax, volcano_df, label=down_df.index.tolist(), color=blues, return_texts=True
        )
        texts.extend(t)
    ns_df = volcano_df[volcano_df["significance"] == "not significant"].head(10)
    if len(ns_df):
        scplt.mark_volcano_by_significance(
            ax,
            volcano_df,
            label=ns_df.index.tolist(),
            color=greys,
            show_names=False,
        )
    if texts:
        scplt.volcano_adjust_and_outline_texts(texts, expand=(2, 2))
    plt.savefig(
        OUT / "volcano_adjust_and_outline_texts.png", dpi=DPI, bbox_inches="tight"
    )
    plt.close("all")

    fig, ax = plt.subplots(figsize=(3, 2))
    scplt.add_volcano_legend(ax)
    save_current_fig("add_volcano_legend")

    # ── sets.py ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3, 3))
    scplt.plot_venn(ax, pdata, classes="cellline")
    save_current_fig("plot_venn")

    upplot = scplt.plot_upset(pdata, classes=classes_2, show_counts=False)
    upplot.plot()
    plt.savefig(OUT / "plot_upset.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    keys = list(
        scutils.get_upset_contents(pdata, classes=classes_2, upsetForm=False).keys()
    )
    if len(keys) >= 4:
        be_kd = next((k for k in keys if "BE" in k and "kd" in k), keys[0])
        as_sc = next((k for k in keys if "AS" in k and "sc" in k), keys[-1])
        others = [k for k in keys if k not in (be_kd, as_sc)]
        upplot_s = scplt.plot_upset(pdata, classes=classes_2, show_counts=False)
        upplot_s.style_subsets(
            present=[be_kd],
            absent=others,
            edgecolor="black",
            facecolor="#E59866",
            linewidth=2,
            label="highlight A",
        )
        upplot_s.style_subsets(
            present=[as_sc],
            absent=[k for k in keys if k != as_sc],
            edgecolor="black",
            facecolor="#5DADE2",
            linewidth=2,
            label="highlight B",
        )
        upplot_s.plot()
        plt.savefig(OUT / "plot_upset_styled.png", dpi=DPI, bbox_inches="tight")
        plt.close("all")
    else:
        print(f"{scutils.format_log_prefix('warn')} Skipping plot_upset_styled: need >=4 upset keys, got {keys!r}")

    print(f"\nDone. Figures saved to {OUT}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate API doc figures under docs/assets/plots/.")
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip single-cell import, directlfq normalization, and plot_umap (much faster).",
    )
    args = parser.parse_args()
    main(skip_umap=args.skip_umap)

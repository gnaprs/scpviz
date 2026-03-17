"""Service functions that wrap scpviz APIs for Dash callbacks."""

from __future__ import annotations

import base64
import json
import re
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from dash import no_update

from scpviz import pAnnData
from scpviz import plotting as scplot
from scpviz.pAnnData.io import get_filenames

from dash_app.state.session import (
    get_pdata,
    get_upload_path,
    set_last_log,
    set_pdata,
    set_upload_path,
)
from dash_app.utils.figures import fig_to_data_uri, new_figure


def _session_tmp_dir(session_id: str) -> Path:
    root = Path(tempfile.gettempdir()) / "scpviz_dash_uploads" / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _refresh_summary_if_needed(pdata: Any) -> None:
    """Sync summary view when marked stale to avoid inconsistent reads/plots."""
    try:
        if getattr(pdata, "_summary_is_stale", False):
            pdata.update_summary(recompute=False, sync_back=False, verbose=False)
    except Exception:
        # Non-fatal: callers still handle plotting/table exceptions upstream.
        pass


def parse_obs_columns(text: str) -> List[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_optional_filters(text: str) -> Optional[List[str]]:
    tokens = parse_obs_columns(text)
    return tokens if tokens else None


def save_upload_contents(session_id: str, upload_key: str, contents: str, filename: str) -> str:
    """Decode Dash upload payload and persist to session temp path."""
    if not contents:
        raise ValueError("Missing upload contents.")
    if not filename:
        raise ValueError("Missing upload filename.")

    _, encoded = contents.split(",", 1)
    decoded = base64.b64decode(encoded)
    path = _session_tmp_dir(session_id) / filename
    path.write_bytes(decoded)
    set_upload_path(session_id, upload_key, str(path))
    return str(path)


def import_data_for_session(
    session_id: str,
    source_type: str,
    obs_columns_text: str,
    delimiter: Optional[str] = None,
) -> Tuple[bool, str]:
    """Run pAnnData import_data from already uploaded files."""
    source_type = (source_type or "").strip().lower()
    obs_columns = parse_obs_columns(obs_columns_text)
    kwargs: Dict[str, Any] = {"source_type": source_type}

    if obs_columns:
        kwargs["obs_columns"] = obs_columns

    if delimiter:
        kwargs["delimiter"] = delimiter

    if source_type in {"pd", "proteomediscoverer", "proteome_discoverer"}:
        prot_file = get_upload_path(session_id, "prot_file")
        pep_file = get_upload_path(session_id, "pep_file")
        if not prot_file:
            return False, "Protein file is required for Proteome Discoverer import."
        kwargs["prot_file"] = prot_file
        if pep_file:
            kwargs["pep_file"] = pep_file
    elif source_type in {"diann", "dia-nn"}:
        report_file = get_upload_path(session_id, "report_file")
        if not report_file:
            return False, "DIA-NN report file is required."
        kwargs["report_file"] = report_file
    else:
        return False, f"Unsupported source_type: {source_type}"

    capture = StringIO()
    try:
        with redirect_stdout(capture):
            pdata = pAnnData.import_data(**kwargs)
    except Exception as exc:  # pragma: no cover - pass-through error formatting
        logs = capture.getvalue().strip()
        return False, f"Import failed: {exc}\n{logs}".strip()

    logs = capture.getvalue().strip()
    if pdata is None:
        return False, (
            "Import returned no object. Provide explicit obs_columns and retry.\n"
            f"{logs}"
        ).strip()

    set_pdata(session_id, pdata)
    set_last_log(session_id, logs)
    return True, f"Import complete. Samples={len(pdata.summary)} | {logs}"


def summary_records(session_id: str, limit: int = 50) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return [], []
    _refresh_summary_if_needed(pdata)
    df = pdata.summary.reset_index().rename(columns={"index": "sample"})
    df = df.head(limit)
    columns = [{"name": c, "id": c} for c in df.columns]
    return df.to_dict("records"), columns


def metadata_columns(session_id: str) -> List[str]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return []
    return list(pdata.summary.columns)


def de_keys(session_id: str) -> List[str]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return []
    return sorted([k for k in pdata.stats.keys() if "vs" in k and not k.endswith(("_up", "_down"))])


def functional_keys(session_id: str) -> List[str]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return []
    return sorted(list(pdata.stats.get("functional", {}).keys()))


def apply_min_protein_filter(session_id: str, min_prot: int) -> Tuple[bool, str]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return False, "No dataset imported."
    filtered = pdata.filter_sample(min_prot=min_prot, return_copy=True)
    set_pdata(session_id, filtered)
    return True, f"Filter applied: min protein_count >= {min_prot}. Samples now={len(filtered.summary)}"


def run_preprocessing(
    session_id: str,
    normalize_method: str,
    impute_method: str,
    layer: str,
) -> Tuple[bool, str]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return False, "No dataset imported."

    capture = StringIO()
    try:
        with redirect_stdout(capture):
            pdata.normalize(method=normalize_method, layer=layer, set_X=True)
            pdata.impute(method=impute_method, layer="X", set_X=True)
    except Exception as exc:
        return False, f"Preprocessing failed: {exc}"
    return True, "Preprocessing complete.\n" + capture.getvalue().strip()


def plot_summary_image(session_id: str, value: str, classes: Optional[List[str]] = None) -> str:
    pdata = get_pdata(session_id)
    if pdata is None:
        return ""
    _refresh_summary_if_needed(pdata)
    with new_figure((6, 4)) as fig:
        ax = fig.add_subplot(111)
        plotted = scplot.plot_summary(ax, pdata, value=value, classes=classes)

        # plot_summary may internally create and return axes on a new figure
        # (e.g. multi-class grouped subplots). Export the actual plotted figure.
        out_fig = fig
        if plotted is not None:
            if hasattr(plotted, "figure"):
                out_fig = plotted.figure
            elif isinstance(plotted, list) and plotted and hasattr(plotted[0], "figure"):
                out_fig = plotted[0].figure
            elif hasattr(plotted, "flat"):
                flat_axes = list(plotted.flat)
                if flat_axes and hasattr(flat_axes[0], "figure"):
                    out_fig = flat_axes[0].figure

        data_uri = fig_to_data_uri(out_fig, fmt="png")
        if out_fig is not fig:
            plt.close(out_fig)
        return data_uri


def plot_cv_image(session_id: str, classes: Optional[List[str]] = None, layer: str = "X") -> str:
    pdata = get_pdata(session_id)
    if pdata is None:
        return ""
    with new_figure((6, 4)) as fig:
        ax = fig.add_subplot(111)
        scplot.plot_cv(ax, pdata, classes=classes, layer=layer)
        return fig_to_data_uri(fig, fmt="png")


def run_embeddings(
    session_id: str,
    layer: str,
    classes: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return False, "No dataset imported."
    capture = StringIO()
    try:
        with redirect_stdout(capture):
            pdata.pca(layer=layer)
            pdata.neighbor(layer=layer)
            pdata.umap(layer=layer)
    except Exception as exc:
        return False, f"Embedding computation failed: {exc}"
    return True, "Embeddings computed.\n" + capture.getvalue().strip()


def plot_pca_image(session_id: str, classes: Optional[List[str]] = None, layer: str = "X") -> str:
    pdata = get_pdata(session_id)
    if pdata is None:
        return ""
    with new_figure((6, 4)) as fig:
        ax = fig.add_subplot(111)
        scplot.plot_pca(ax, pdata, classes=classes, layer=layer, add_ellipses=True)
        return fig_to_data_uri(fig, fmt="png")


def plot_umap_image(session_id: str, classes: Optional[List[str]] = None, layer: str = "X") -> str:
    pdata = get_pdata(session_id)
    if pdata is None:
        return ""
    with new_figure((6, 4)) as fig:
        ax = fig.add_subplot(111)
        scplot.plot_umap(ax, pdata, classes=classes, layer=layer, add_ellipses=True)
        return fig_to_data_uri(fig, fmt="png")


def plot_abundance_image(
    session_id: str,
    genes: List[str],
    classes: Optional[List[str]] = None,
    layer: str = "X",
) -> str:
    pdata = get_pdata(session_id)
    if pdata is None:
        return ""
    with new_figure((7, 4)) as fig:
        ax = fig.add_subplot(111)
        scplot.plot_abundance(ax, pdata, namelist=genes, classes=classes, layer=layer)
        return fig_to_data_uri(fig, fmt="png")


def run_de(
    session_id: str,
    group1_json: str,
    group2_json: str,
    method: str,
    layer: str,
    pval: float,
    log2fc: float,
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return False, "No dataset imported.", None

    try:
        group1 = json.loads(group1_json)
        group2 = json.loads(group2_json)
        if not isinstance(group1, dict) or not isinstance(group2, dict):
            raise ValueError("Both groups must be JSON objects.")
    except Exception as exc:
        return False, f"Invalid group JSON: {exc}", None

    capture = StringIO()
    try:
        with redirect_stdout(capture):
            df = pdata.de(
                values=[group1, group2],
                method=method,
                layer=layer,
                pval=pval,
                log2fc=log2fc,
            )
    except Exception as exc:
        return False, f"DE failed: {exc}", None

    return True, "DE complete.\n" + capture.getvalue().strip(), df


def volcano_plotly_figure(df: pd.DataFrame, pval: float, log2fc: float):
    """Build a Plotly-native volcano chart from DE output."""
    if df is None or df.empty:
        return no_update
    dfp = df.reset_index().copy()
    if "p_value" not in dfp or "log2fc" not in dfp:
        return no_update
    dfp["neg_log10_p"] = -dfp["p_value"].clip(lower=1e-300).apply(np.log10)

    fig = px.scatter(
        dfp,
        x="log2fc",
        y="neg_log10_p",
        color="significance" if "significance" in dfp.columns else None,
        hover_data=[c for c in ["Genes", "p_value", "log2fc"] if c in dfp.columns],
        title="Volcano plot",
    )
    fig.add_vline(x=log2fc, line_dash="dash")
    fig.add_vline(x=-log2fc, line_dash="dash")
    fig.add_hline(y=-np.log10(max(pval, 1e-300)), line_dash="dash")
    fig.update_layout(template="plotly_white", height=460, margin=dict(l=30, r=20, t=50, b=30))
    return fig


def run_functional_enrichment(
    session_id: str,
    de_key: str,
    species: Optional[int],
    top_n: int,
) -> Tuple[bool, str]:
    pdata = get_pdata(session_id)
    if pdata is None:
        return False, "No dataset imported."
    if not de_key:
        return False, "Select a DE key first."

    capture = StringIO()
    try:
        with redirect_stdout(capture):
            pdata.enrichment_functional(
                from_de=True,
                de_key=de_key,
                species=species,
                top_n=top_n,
            )
    except Exception as exc:
        return False, f"Enrichment failed: {exc}"
    return True, "Enrichment complete.\n" + capture.getvalue().strip()


def fetch_enrichment_svg_data_uri(
    session_id: str,
    functional_key: str,
    category: Optional[str] = None,
) -> str:
    """Fetch STRING enrichment SVG directly from stored enrichment metadata."""
    pdata = get_pdata(session_id)
    if pdata is None:
        raise ValueError("No dataset imported.")

    func = pdata.stats.get("functional", {})
    if functional_key not in func:
        raise ValueError(f"Functional key not found: {functional_key}")

    meta = func[functional_key]
    string_ids = meta["string_ids"]
    species_id = meta["species"]
    url = "https://string-db.org/api/svg/enrichmentfigure"
    params = {"identifiers": "%0d".join(string_ids), "species": species_id}
    if category:
        params["category"] = category
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def enrichment_string_url(session_id: str, functional_key: str) -> str:
    """Return STRING website network URL for a stored functional enrichment key."""
    pdata = get_pdata(session_id)
    if pdata is None:
        raise ValueError("No dataset imported.")

    func = pdata.stats.get("functional", {})
    if functional_key not in func:
        raise ValueError(f"Functional key not found: {functional_key}")

    meta = func[functional_key]
    return str(meta.get("string_url", "")).strip()


def summary_dataframe(session_id: str) -> pd.DataFrame:
    """Return current summary table for the active session."""
    pdata = get_pdata(session_id)
    if pdata is None:
        return pd.DataFrame()
    _refresh_summary_if_needed(pdata)
    return pdata.summary.reset_index().rename(columns={"index": "sample"}).copy()


def embedding_dataframe(session_id: str) -> pd.DataFrame:
    """Return embedding coordinates merged with sample metadata."""
    pdata = get_pdata(session_id)
    if pdata is None:
        return pd.DataFrame()

    obs = pdata.prot.obs.copy()
    if obs.index.name is None:
        obs.index.name = "sample"
    out = obs.reset_index()

    pca = pdata.prot.obsm.get("X_pca")
    if pca is not None and getattr(pca, "shape", (0, 0))[1] >= 2:
        out["pca_1"] = pca[:, 0]
        out["pca_2"] = pca[:, 1]

    umap = pdata.prot.obsm.get("X_umap")
    if umap is not None and getattr(umap, "shape", (0, 0))[1] >= 2:
        out["umap_1"] = umap[:, 0]
        out["umap_2"] = umap[:, 1]

    return out


def enrichment_result_dataframe(session_id: str, functional_key: str) -> pd.DataFrame:
    """Return functional enrichment result table for a selected key."""
    pdata = get_pdata(session_id)
    if pdata is None or not functional_key:
        return pd.DataFrame()
    func = pdata.stats.get("functional", {})
    if functional_key not in func:
        return pd.DataFrame()
    result = func[functional_key].get("result")
    if isinstance(result, pd.DataFrame):
        return result.copy()
    return pd.DataFrame()


def infer_obs_columns_from_uploaded_data(
    session_id: str,
    source_type: str,
    delimiter_text: Optional[str],
) -> Tuple[str, str, List[Dict[str, str]]]:
    """Infer obs_columns from run/sample names extracted from uploaded data files."""
    source_norm = (source_type or "").strip().lower()
    if source_norm in {"pd", "proteomediscoverer", "proteome_discoverer"}:
        source_file = get_upload_path(session_id, "prot_file")
        source_kind = "pd"
    elif source_norm in {"diann", "dia-nn"}:
        source_file = get_upload_path(session_id, "report_file")
        source_kind = "diann"
    else:
        source_file = None
        source_kind = source_norm

    if not source_file:
        return "", "Upload an input data file first to infer obs_columns from internal run/sample names.", []

    try:
        internal_names = get_filenames(source_file, source_type=source_kind)
    except Exception as exc:
        return "", f"Could not read run/sample names from uploaded data: {exc}", []

    clean = [str(name).strip() for name in internal_names if str(name).strip()]
    if not clean:
        return "", "No run/sample names were found inside the uploaded dataframe.", []

    delimiter = (delimiter_text or "").strip()
    if not delimiter:
        if source_kind == "pd":
            delimiter = ","
        else:
            candidates = ["_", "-", "."]
            delimiter = max(candidates, key=lambda d: sum(text.count(d) for text in clean))
            if sum(text.count(delimiter) for text in clean) == 0:
                delimiter = "_"

    tokenized = []
    for text in clean:
        parts = [p for p in re.split(re.escape(delimiter), text) if p]
        tokenized.append(parts or [text])

    width = max(len(parts) for parts in tokenized)
    defaults = ["sample", "cellline", "treatment", "replicate", "batch"]
    suggested_cols = [defaults[i] if i < len(defaults) else f"meta_{i + 1}" for i in range(width)]
    token_examples = []
    for i in range(width):
        example = ""
        for parts in tokenized:
            if i < len(parts) and str(parts[i]).strip():
                example = str(parts[i]).strip()
                break
        token_examples.append(example or f"value_{i + 1}")

    rename_rows = [{"token": token_examples[i], "name": col} for i, col in enumerate(suggested_cols)]

    preview_rows = []
    for parts in tokenized[:3]:
        padded = parts + [""] * (width - len(parts))
        preview_rows.append(" | ".join(padded))

    preview_text = (
        f"Inferred from uploaded dataframe (not upload filename).\n"
        f"Detected delimiter: '{delimiter}'\n"
        f"Example parsed internal names:\n- " + "\n- ".join(preview_rows)
    )
    return ",".join(suggested_cols), preview_text, rename_rows


"""Dash callbacks for the scpviz web app."""

from __future__ import annotations

import base64
import io
import json
import uuid
import zipfile
from datetime import datetime

import pandas as pd
from dash import Input, Output, State, callback, dcc, no_update

from dash_app.services.scpviz_service import (
    apply_min_protein_filter,
    de_keys,
    embedding_dataframe,
    enrichment_result_dataframe,
    enrichment_string_url,
    fetch_enrichment_svg_data_uri,
    functional_keys,
    infer_obs_columns_from_uploaded_data,
    import_data_for_session,
    parse_optional_filters,
    parse_obs_columns,
    plot_abundance_image,
    plot_cv_image,
    plot_pca_image,
    plot_summary_image,
    plot_umap_image,
    run_de,
    run_embeddings,
    run_functional_enrichment,
    run_preprocessing,
    save_upload_contents,
    summary_dataframe,
    summary_records,
    volcano_plotly_figure,
)


def _download_df(df: pd.DataFrame, filename: str):
    if df is None or df.empty:
        return no_update
    return dcc.send_data_frame(df.to_csv, filename, index=False)


def _decode_data_uri(data_uri: str) -> tuple[bytes, str]:
    if not data_uri or "," not in data_uri:
        return b"", ""
    header, payload = data_uri.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")
    try:
        return base64.b64decode(payload), mime
    except Exception:
        return b"", mime


def _de_selected_records(selected_data, records):
    if not selected_data or not records:
        return records or []
    points = selected_data.get("points", [])
    if not points:
        return records
    selected_idx = {
        p.get("pointIndex")
        for p in points
        if isinstance(p, dict) and isinstance(p.get("pointIndex"), int)
    }
    if not selected_idx:
        return records
    return [row for i, row in enumerate(records) if i in selected_idx]


def _format_de_records_for_table(records):
    """Decorate DE table rows with UniProt links for the index column."""
    out = []
    for row in records or []:
        item = dict(row)
        if "index" in item and item["index"] is not None:
            accession = str(item["index"]).strip()
            if accession:
                item["index"] = f"[{accession}](https://www.uniprot.org/uniprotkb/{accession}/entry)"
        out.append(item)
    return out


def _de_table_columns(records):
    """Build DE table columns, enabling markdown for index hyperlinks."""
    if not records:
        return no_update
    keys = list(records[0].keys())
    cols = []
    for key in keys:
        col = {"name": key, "id": key}
        if key == "index":
            col["presentation"] = "markdown"
        cols.append(col)
    return cols


def _normalize_classes_input(value):
    """Normalize dropdown/text class inputs into list[str] or None."""
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return items if items else None
    if isinstance(value, str):
        return parse_optional_filters(value or "")
    return None


@callback(
    Output("session-id", "data"),
    Input("url", "pathname"),
    State("session-id", "data"),
    prevent_initial_call=False,
)
def ensure_session_id(_pathname, current_session_id):
    if current_session_id:
        return current_session_id
    return str(uuid.uuid4())


@callback(
    Output("obs-helper-panel", "style"),
    Output("obs-helper-preview", "children"),
    Output("obs-helper-suggest", "value"),
    Output("obs-helper-rename-table", "data"),
    Input("btn-open-obs-helper", "n_clicks"),
    Input("btn-refresh-obs-helper", "n_clicks"),
    Input("upload-prot", "filename"),
    Input("upload-pep", "filename"),
    Input("upload-diann", "filename"),
    State("source-type", "value"),
    State("session-id", "data"),
    State("import-delimiter", "value"),
    prevent_initial_call=True,
)
def update_obs_helper(_open_clicks, _refresh_clicks, _prot_filename, _pep_filename, _diann_filename, source_type, session_id, delimiter):
    suggested, preview, rename_rows = infer_obs_columns_from_uploaded_data(
        session_id=session_id or "",
        source_type=source_type or "",
        delimiter_text=delimiter,
    )
    style = {
        "display": "block",
        "marginTop": "10px",
        "padding": "12px",
        "border": "1px solid #24324c",
        "borderRadius": "10px",
        "backgroundColor": "#0a1428",
    }
    return style, preview, suggested, rename_rows


@callback(
    Output("obs-columns", "value", allow_duplicate=True),
    Output("obs-helper-suggest", "value", allow_duplicate=True),
    Input("btn-apply-obs-helper", "n_clicks"),
    State("obs-helper-rename-table", "data"),
    State("obs-helper-suggest", "value"),
    State("obs-columns", "value"),
    prevent_initial_call=True,
)
def apply_obs_helper(n_clicks, rename_rows, suggested_value, current_value):
    if not n_clicks:
        return no_update, no_update

    row_names = []
    for row in rename_rows or []:
        name = str((row or {}).get("name", "")).strip()
        if name:
            row_names.append(name)

    candidate = ",".join(row_names) if row_names else (suggested_value or "").strip()
    if not candidate:
        fallback = current_value if current_value else no_update
        return fallback, no_update
    return candidate, candidate


@callback(
    Output("obs-helper-suggest", "value", allow_duplicate=True),
    Input("obs-helper-rename-table", "data"),
    prevent_initial_call=True,
)
def sync_obs_helper_text_from_table(rename_rows):
    names = []
    for row in rename_rows or []:
        name = str((row or {}).get("name", "")).strip()
        if name:
            names.append(name)
    return ",".join(names) if names else no_update


@callback(
    Output("upload-prot-msg", "children"),
    Input("upload-prot", "contents"),
    State("upload-prot", "filename"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_upload_prot(contents, filename, session_id):
    if not contents or not session_id:
        return no_update
    try:
        path = save_upload_contents(session_id, "prot_file", contents, filename)
        return f"Saved: {path}"
    except Exception as exc:
        return f"Upload failed: {exc}"


@callback(
    Output("upload-pep-msg", "children"),
    Input("upload-pep", "contents"),
    State("upload-pep", "filename"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_upload_pep(contents, filename, session_id):
    if not contents or not session_id:
        return no_update
    try:
        path = save_upload_contents(session_id, "pep_file", contents, filename)
        return f"Saved: {path}"
    except Exception as exc:
        return f"Upload failed: {exc}"


@callback(
    Output("upload-diann-msg", "children"),
    Input("upload-diann", "contents"),
    State("upload-diann", "filename"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_upload_diann(contents, filename, session_id):
    if not contents or not session_id:
        return no_update
    try:
        path = save_upload_contents(session_id, "report_file", contents, filename)
        return f"Saved: {path}"
    except Exception as exc:
        return f"Upload failed: {exc}"


@callback(
    Output("import-log", "children"),
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Input("btn-import", "n_clicks"),
    State("source-type", "value"),
    State("obs-columns", "value"),
    State("import-delimiter", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_import(n_clicks, source_type, obs_columns_text, delimiter, session_id):
    if not n_clicks or not session_id:
        return no_update, no_update, no_update
    ok, message = import_data_for_session(
        session_id=session_id,
        source_type=source_type,
        obs_columns_text=obs_columns_text or "",
        delimiter=(delimiter or "").strip() or None,
    )
    if not ok:
        return message, [], []
    data, columns = summary_records(session_id)
    return message, data, columns


@callback(
    Output("qc-classes", "options"),
    Output("embed-classes", "options"),
    Output("qc-classes", "value"),
    Output("embed-classes", "value"),
    Output("de-g1-col-1", "options"),
    Output("de-g1-col-2", "options"),
    Output("de-g1-col-3", "options"),
    Output("de-g2-col-1", "options"),
    Output("de-g2-col-2", "options"),
    Output("de-g2-col-3", "options"),
    Input("summary-table", "columns"),
    prevent_initial_call=False,
)
def sync_class_dropdowns(summary_columns):
    if not summary_columns:
        return [], [], [], [], [], [], [], [], [], []

    col_ids = [c.get("id") for c in summary_columns if c.get("id")]
    options = [{"label": c, "value": c} for c in col_ids]
    preferred = [c for c in ["cellline", "condition"] if c in col_ids]
    default_value = preferred if preferred else col_ids[:2]
    return options, options, default_value, default_value, options, options, options, options, options, options


def _option_values_for_column(summary_df, column_name):
    if summary_df is None or summary_df.empty or not column_name or column_name not in summary_df.columns:
        return []
    values = summary_df[column_name].dropna().unique().tolist()
    values = sorted({str(v).strip() for v in values if str(v).strip()})
    return [{"label": v, "value": v} for v in values]


@callback(
    Output("de-g1-val-1", "options"),
    Output("de-g1-val-2", "options"),
    Output("de-g1-val-3", "options"),
    Output("de-g2-val-1", "options"),
    Output("de-g2-val-2", "options"),
    Output("de-g2-val-3", "options"),
    Input("de-g1-col-1", "value"),
    Input("de-g1-col-2", "value"),
    Input("de-g1-col-3", "value"),
    Input("de-g2-col-1", "value"),
    Input("de-g2-col-2", "value"),
    Input("de-g2-col-3", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def sync_de_value_dropdowns(g1c1, g1c2, g1c3, g2c1, g2c2, g2c3, session_id):
    try:
        summary_df = summary_dataframe(session_id) if session_id else pd.DataFrame()
    except Exception:
        summary_df = pd.DataFrame()
    return (
        _option_values_for_column(summary_df, g1c1),
        _option_values_for_column(summary_df, g1c2),
        _option_values_for_column(summary_df, g1c3),
        _option_values_for_column(summary_df, g2c1),
        _option_values_for_column(summary_df, g2c2),
        _option_values_for_column(summary_df, g2c3),
    )


@callback(
    Output("de-group1", "value", allow_duplicate=True),
    Output("de-group2", "value", allow_duplicate=True),
    Input("btn-build-de-json", "n_clicks"),
    State("de-g1-col-1", "value"),
    State("de-g1-val-1", "value"),
    State("de-g1-col-2", "value"),
    State("de-g1-val-2", "value"),
    State("de-g1-col-3", "value"),
    State("de-g1-val-3", "value"),
    State("de-g2-col-1", "value"),
    State("de-g2-val-1", "value"),
    State("de-g2-col-2", "value"),
    State("de-g2-val-2", "value"),
    State("de-g2-col-3", "value"),
    State("de-g2-val-3", "value"),
    prevent_initial_call=True,
)
def build_de_json_from_dropdowns(
    n_clicks,
    g1c1,
    g1v1,
    g1c2,
    g1v2,
    g1c3,
    g1v3,
    g2c1,
    g2v1,
    g2c2,
    g2v2,
    g2c3,
    g2v3,
):
    if not n_clicks:
        return no_update, no_update

    def _build(col_vals):
        out = {}
        for col, val in col_vals:
            if col and val is not None and str(val).strip() != "":
                out[str(col)] = str(val)
        return out

    group1 = _build([(g1c1, g1v1), (g1c2, g1v2), (g1c3, g1v3)])
    group2 = _build([(g2c1, g2v1), (g2c2, g2v2), (g2c3, g2v3)])
    return json.dumps(group1), json.dumps(group2)


@callback(
    Output("qc-log", "children"),
    Output("img-summary", "src"),
    Output("img-cv", "src"),
    Output("qc-summary-table", "data"),
    Output("qc-summary-table", "columns"),
    Input("btn-filter", "n_clicks"),
    Input("btn-qc-refresh", "n_clicks"),
    State("min-prot", "value"),
    State("qc-metric", "value"),
    State("qc-classes", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def update_qc(_btn_filter, _btn_refresh, min_prot, metric, classes_value, session_id):
    if not session_id:
        return no_update, no_update, no_update, no_update, no_update

    log_msgs = []
    summary_img = no_update
    cv_img = no_update

    if min_prot is not None and str(min_prot).strip() != "":
        try:
            min_prot_value = int(float(min_prot))
            if min_prot_value > 0:
                ok, msg = apply_min_protein_filter(session_id, min_prot_value)
                log_msgs.append(msg)
                if not ok:
                    return "\n".join(log_msgs), no_update, no_update, [], []
        except Exception as exc:
            log_msgs.append(f"Filter step failed: {exc}")

    classes = _normalize_classes_input(classes_value)
    metric_value = metric or "protein_count"
    try:
        summary_df = summary_dataframe(session_id)
    except Exception as exc:
        return f"QC summary table failed: {exc}", no_update, no_update, [], []

    if summary_df.empty:
        return "No dataset imported.", no_update, no_update, [], []
    if metric_value not in summary_df.columns:
        fallback = "protein_count" if "protein_count" in summary_df.columns else None
        if fallback is None:
            numeric_cols = summary_df.select_dtypes(include=["number"]).columns.tolist()
            fallback = numeric_cols[0] if numeric_cols else summary_df.columns[0]
        metric_value = fallback

    try:
        summary_img = plot_summary_image(session_id, metric_value, classes=classes)
    except Exception as exc:
        # Fallback without grouping to avoid blank plot on class parsing/grouping issues.
        try:
            summary_img = plot_summary_image(session_id, metric_value, classes=None)
            log_msgs.append(f"Summary plot fallback applied (no grouping): {exc}")
        except Exception as exc2:
            summary_img = no_update
            log_msgs.append(f"Summary plot failed: {exc2}")

    try:
        cv_img = plot_cv_image(session_id, classes=classes, layer="X")
    except Exception as exc:
        cv_img = no_update
        log_msgs.append(f"CV plot failed: {exc}")

    table_cols = [{"name": c, "id": c} for c in summary_df.columns.tolist()]
    table_data = summary_df.to_dict("records")
    return "\n".join(log_msgs) if log_msgs else "QC refreshed.", summary_img, cv_img, table_data, table_cols


@callback(
    Output("prep-log", "children"),
    Input("btn-preprocess", "n_clicks"),
    State("normalize-method", "value"),
    State("impute-method", "value"),
    State("prep-layer", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_prep(n_clicks, norm_method, imp_method, layer, session_id):
    if not n_clicks or not session_id:
        return no_update
    ok, msg = run_preprocessing(
        session_id=session_id,
        normalize_method=norm_method or "median",
        impute_method=imp_method or "min",
        layer=layer or "X",
    )
    return msg


@callback(
    Output("embed-log", "children"),
    Output("img-pca", "src"),
    Output("img-umap", "src"),
    Output("img-abundance", "src"),
    Input("btn-embed", "n_clicks"),
    State("embed-classes", "value"),
    State("embed-layer", "value"),
    State("abundance-genes", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_embedding_plots(n_clicks, classes_text, layer, genes_text, session_id):
    if not n_clicks or not session_id:
        return no_update, no_update, no_update, no_update

    classes = _normalize_classes_input(classes_text)
    genes = parse_obs_columns(genes_text or "")
    ok, msg = run_embeddings(session_id, layer=layer or "X", classes=classes)
    if not ok:
        return msg, no_update, no_update, no_update

    try:
        pca_src = plot_pca_image(session_id, classes=classes, layer=layer or "X")
        umap_src = plot_umap_image(session_id, classes=classes, layer=layer or "X")
        abundance_src = plot_abundance_image(session_id, genes=genes, classes=classes, layer=layer or "X")
    except Exception as exc:
        return f"{msg}\nPlotting failed: {exc}", no_update, no_update, no_update
    return msg, pca_src, umap_src, abundance_src


@callback(
    Output("de-log", "children"),
    Output("de-volcano", "figure"),
    Output("de-table-store", "data"),
    Output("de-table", "data"),
    Output("de-table", "columns"),
    Input("btn-de", "n_clicks"),
    State("de-group1", "value"),
    State("de-group2", "value"),
    State("de-method", "value"),
    State("de-layer", "value"),
    State("de-pval", "value"),
    State("de-log2fc", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_de_callback(n_clicks, group1, group2, method, layer, pval, log2fc, session_id):
    if not n_clicks or not session_id:
        return no_update, no_update, no_update, no_update, no_update

    ok, msg, df = run_de(
        session_id=session_id,
        group1_json=group1 or "{}",
        group2_json=group2 or "{}",
        method=method or "ttest",
        layer=layer or "X",
        pval=float(pval or 0.05),
        log2fc=float(log2fc or 1.0),
    )
    if not ok or df is None:
        return msg, no_update, no_update, no_update, no_update

    fig = volcano_plotly_figure(df, pval=float(pval or 0.05), log2fc=float(log2fc or 1.0))
    records = df.reset_index().to_dict("records")
    table_records = _format_de_records_for_table(records)
    columns = _de_table_columns(records)
    return msg, fig, records, table_records, columns


@callback(
    Output("de-table", "data", allow_duplicate=True),
    Output("de-selection-info", "children"),
    Input("de-volcano", "selectedData"),
    State("de-table-store", "data"),
    prevent_initial_call=True,
)
def update_de_table_from_selection(selected_data, records):
    if not records:
        return no_update, "No DE table available yet."
    filtered = _de_selected_records(selected_data, records)
    if selected_data and selected_data.get("points"):
        return _format_de_records_for_table(filtered), f"Showing {len(filtered)} selected rows from volcano selection."
    return _format_de_records_for_table(records), f"Showing all {len(records)} rows."


@callback(
    Output("de-key-dropdown", "options"),
    Input("btn-refresh-keys", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def refresh_de_key_options(_n_clicks, session_id):
    if not session_id:
        return []
    return [{"label": k, "value": k} for k in de_keys(session_id)]


@callback(
    Output("enrich-log", "children"),
    Input("btn-enrich", "n_clicks"),
    State("de-key-dropdown", "value"),
    State("species-id", "value"),
    State("enrich-topn", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_enrichment_callback(n_clicks, de_key, species, top_n, session_id):
    if not n_clicks or not session_id:
        return no_update
    species_value = int(species) if species else None
    top_n_value = int(top_n) if top_n else 150
    ok, msg = run_functional_enrichment(session_id, de_key=de_key, species=species_value, top_n=top_n_value)
    return msg


@callback(
    Output("functional-key-dropdown", "options"),
    Input("btn-refresh-functional", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def refresh_functional_key_options(_n_clicks, session_id):
    if not session_id:
        return []
    return [{"label": k, "value": k} for k in functional_keys(session_id)]


@callback(
    Output("img-enrichment-svg", "src"),
    Output("string-network-link", "href"),
    Output("string-network-iframe", "src"),
    Input("btn-load-svg", "n_clicks"),
    State("functional-key-dropdown", "value"),
    State("enrich-category", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def load_svg(n_clicks, functional_key, category, session_id):
    if not n_clicks or not session_id or not functional_key:
        return no_update, no_update, no_update
    try:
        svg_data = fetch_enrichment_svg_data_uri(
            session_id=session_id,
            functional_key=functional_key,
            category=(category or "").strip() or None,
        )
        url = enrichment_string_url(session_id, functional_key)
        return svg_data, url, url
    except Exception:
        return no_update, no_update, no_update


@callback(
    Output("download-import-table", "data"),
    Input("btn-download-import-table", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_import_table(n_clicks, session_id):
    if not n_clicks or not session_id:
        return no_update
    return _download_df(summary_dataframe(session_id), "import_summary.csv")


@callback(
    Output("download-qc-table", "data"),
    Input("btn-download-qc-table", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_qc_table(n_clicks, session_id):
    if not n_clicks or not session_id:
        return no_update
    return _download_df(summary_dataframe(session_id), "qc_summary.csv")


@callback(
    Output("download-prep-table", "data"),
    Input("btn-download-prep-table", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_prep_table(n_clicks, session_id):
    if not n_clicks or not session_id:
        return no_update
    return _download_df(summary_dataframe(session_id), "preprocess_summary.csv")


@callback(
    Output("download-embed-table", "data"),
    Input("btn-download-embed-table", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_embed_table(n_clicks, session_id):
    if not n_clicks or not session_id:
        return no_update
    return _download_df(embedding_dataframe(session_id), "embeddings_table.csv")


@callback(
    Output("download-de-table", "data"),
    Input("btn-download-de-table", "n_clicks"),
    State("de-table-store", "data"),
    prevent_initial_call=True,
)
def download_de_table(n_clicks, records):
    if not n_clicks or not records:
        return no_update
    return _download_df(pd.DataFrame(records), "de_results.csv")


@callback(
    Output("download-enrich-table", "data"),
    Input("btn-download-enrich-table", "n_clicks"),
    State("session-id", "data"),
    State("functional-key-dropdown", "value"),
    prevent_initial_call=True,
)
def download_enrich_table(n_clicks, session_id, functional_key):
    if not n_clicks or not session_id or not functional_key:
        return no_update
    return _download_df(enrichment_result_dataframe(session_id, functional_key), "enrichment_results.csv")


@callback(
    Output("download-bundle", "data"),
    Input("btn-download-bundle", "n_clicks"),
    State("session-id", "data"),
    State("functional-key-dropdown", "value"),
    State("de-table-store", "data"),
    State("img-summary", "src"),
    State("img-cv", "src"),
    State("img-pca", "src"),
    State("img-umap", "src"),
    State("img-abundance", "src"),
    State("img-enrichment-svg", "src"),
    State("de-volcano", "figure"),
    prevent_initial_call=True,
)
def download_bundle(
    n_clicks,
    session_id,
    functional_key,
    de_records,
    img_summary,
    img_cv,
    img_pca,
    img_umap,
    img_abundance,
    img_enrichment_svg,
    volcano_fig,
):
    if not n_clicks or not session_id:
        return no_update

    summary_df = summary_dataframe(session_id)
    embed_df = embedding_dataframe(session_id)
    de_df = pd.DataFrame(de_records) if de_records else pd.DataFrame()
    enrich_df = enrichment_result_dataframe(session_id, functional_key) if functional_key else pd.DataFrame()

    def _write_zip(buffer: io.BytesIO):
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            readme_lines = [
                "scpviz analysis export bundle",
                "",
                "This archive contains available outputs from the current Dash session.",
                "",
                "Tables:",
                "- tables/summary.csv: Current session-level sample summary table.",
                "- tables/embeddings.csv: Sample metadata merged with PCA/UMAP coordinates (if computed).",
                "- tables/de_results.csv: Differential expression results from the latest DE run.",
                "- tables/enrichment_results.csv: STRING functional enrichment table for selected functional key.",
                "",
                "Plots:",
                "- plots/qc_summary.png: QC summary plot.",
                "- plots/qc_cv.png: QC coefficient-of-variation plot.",
                "- plots/embed_pca.png: PCA embedding plot.",
                "- plots/embed_umap.png: UMAP embedding plot.",
                "- plots/embed_abundance.png: Abundance plot for selected genes/proteins.",
                "- plots/enrichment.svg: STRING enrichment network SVG.",
                "- plots/de_volcano.json: Plotly volcano figure JSON.",
                "",
                "Note: Only files available at export time are included.",
            ]
            zf.writestr("README.txt", "\n".join(readme_lines))

            if not summary_df.empty:
                zf.writestr("tables/summary.csv", summary_df.to_csv(index=False))
            if not embed_df.empty:
                zf.writestr("tables/embeddings.csv", embed_df.to_csv(index=False))
            if not de_df.empty:
                zf.writestr("tables/de_results.csv", de_df.to_csv(index=False))
            if not enrich_df.empty:
                zf.writestr("tables/enrichment_results.csv", enrich_df.to_csv(index=False))

            img_map = {
                "plots/qc_summary.png": img_summary,
                "plots/qc_cv.png": img_cv,
                "plots/embed_pca.png": img_pca,
                "plots/embed_umap.png": img_umap,
                "plots/embed_abundance.png": img_abundance,
                "plots/enrichment.svg": img_enrichment_svg,
            }
            for name, src in img_map.items():
                data, _mime = _decode_data_uri(src)
                if data:
                    zf.writestr(name, data)

            if volcano_fig:
                zf.writestr("plots/de_volcano.json", json.dumps(volcano_fig, indent=2))

        buffer.seek(0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_bytes(_write_zip, f"scpviz_bundle_{stamp}.zip")


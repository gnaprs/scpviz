"""Dash callbacks for the scpviz web app."""

from __future__ import annotations

import base64
from typing import Literal
import io
import json
import re
import uuid
import zipfile
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, callback_context, dcc, no_update

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
    save_edited_svg_for_session,
    load_edited_svg_for_session,
    summary_dataframe,
    summary_records,
    svg_data_uri_to_markup,
    svg_markup_to_data_uri,
    volcano_svg_markup_from_records,
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


def _coerce_de_row_index(val) -> int | None:
    """Parse an int-like row index from Plotly customdata / pointIndex payloads."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, (float, np.floating)) and np.isfinite(val) and float(val).is_integer():
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if s.isdigit():
            return int(s)
        if s.startswith("-") and s[1:].isdigit():
            return int(s)
    try:
        if hasattr(val, "item"):
            return _coerce_de_row_index(val.item())
    except Exception:
        return None
    return None


def _iter_plotly_customdata_leaves(cd):
    """Yield leaf values; unwrap single-element lists Plotly uses for customdata."""
    if cd is None:
        return
    if isinstance(cd, np.ndarray):
        yield from _iter_plotly_customdata_leaves(cd.ravel().tolist())
        return
    if isinstance(cd, (list, tuple)):
        if len(cd) == 1:
            yield from _iter_plotly_customdata_leaves(cd[0])
        else:
            for item in cd:
                yield from _iter_plotly_customdata_leaves(item)
        return
    yield cd


def _de_row_idx_from_selected_point(point: dict) -> int | None:
    """Resolve de-table-store row index from a Plotly selectedData point."""
    if not isinstance(point, dict):
        return None
    cd = point.get("customdata")
    if cd is not None:
        for item in _iter_plotly_customdata_leaves(cd):
            idx = _coerce_de_row_index(item)
            if idx is not None:
                return idx
    return _coerce_de_row_index(point.get("pointIndex"))


def _volcano_relayout_updates_annotations(relayout_data: dict) -> bool:
    """True only when relayout keys describe annotation geometry/text (not generic pan/zoom)."""
    if not relayout_data:
        return False
    if isinstance(relayout_data.get("annotations"), list):
        return True
    ann_prefix = re.compile(r"^annotations\[\d+\]\.")
    for key in relayout_data.keys():
        key_text = str(key)
        if key_text == "annotations" or ann_prefix.match(key_text):
            return True
    return False


DeSelectionOutcome = Literal["none", "subset", "unresolved"]


def _de_plotly_selection_rows(selected_data, records) -> tuple[list, DeSelectionOutcome]:
    """Map Plotly selectedData/clickData points to DE table rows.

    Returns (rows_for_display, outcome):
    - ``none``: no payload or no points — show full table.
    - ``subset``: at least one point resolved to a valid row index.
    - ``unresolved``: points were present but none matched — show full table; caller may warn.
    """
    if not records:
        return [], "none"
    if not selected_data:
        return records, "none"
    points = selected_data.get("points") or []
    if not points:
        return records, "none"
    selected_idx = set()
    for point in points:
        if not isinstance(point, dict):
            continue
        idx = _de_row_idx_from_selected_point(point)
        if idx is not None and 0 <= idx < len(records):
            selected_idx.add(idx)
    if not selected_idx:
        return records, "unresolved"
    return [row for i, row in enumerate(records) if i in selected_idx], "subset"


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


def _editor_plot_key_label(plot_key: str) -> tuple[str, str]:
    if plot_key == "de":
        return "de", "DE volcano"
    if plot_key == "enrichment":
        return "enrichment", "STRING enrichment"
    return "", "none"


def _normalize_volcano_style(style_data, default_pval=0.05, default_log2fc=1.0):
    style_data = style_data or {}
    colors = style_data.get("colors") or {}
    return {
        "pval": float(style_data.get("pval", default_pval) or default_pval),
        "log2fc": float(style_data.get("log2fc", default_log2fc) or default_log2fc),
        "colors": {
            "up": str(colors.get("up", "#dc2626")),
            "down": str(colors.get("down", "#2563eb")),
            "ns": str(colors.get("ns", "#94a3b8")),
        },
        "font_family": str(style_data.get("font_family", "Arial")),
        "font_size": int(float(style_data.get("font_size", 12) or 12)),
    }


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _de_axis_spans(records):
    coords = []
    for row in records or []:
        point = _volcano_point_xy(row)
        if point is not None:
            coords.append(point)
    if not coords:
        return 1.0, 1.0
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    return max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0)


def _default_label_position(point_x, point_y, index_hint, x_span, y_span):
    patterns = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1.6, 0.2), (-1.6, 0.2), (0.2, 1.6), (0.2, -1.6)]
    dx = max(x_span * 0.045, 0.25)
    dy = max(y_span * 0.06, 0.35)
    px, py = patterns[index_hint % len(patterns)]
    ring = 1.0 + 0.35 * (index_hint // len(patterns))
    return float(point_x) + px * dx * ring, float(point_y) + py * dy * ring


def _dodge_labels(labels, moved_ids, x_span, y_span):
    if not labels:
        return labels
    moved_ids = {str(item) for item in (moved_ids or [])}
    if not moved_ids:
        return labels
    min_dx = max(x_span * 0.03, 0.18)
    min_dy = max(y_span * 0.045, 0.28)
    updated = [dict(item) for item in labels]
    id_to_idx = {str(item.get("id")): idx for idx, item in enumerate(updated)}
    for moved_id in moved_ids:
        idx = id_to_idx.get(str(moved_id))
        if idx is None:
            continue
        target = updated[idx]
        x_val = _as_float(target.get("x"))
        y_val = _as_float(target.get("y"))
        if x_val is None or y_val is None:
            continue
        for _ in range(36):
            shifted = False
            for j, other in enumerate(updated):
                if j == idx:
                    continue
                ox = _as_float(other.get("x"))
                oy = _as_float(other.get("y"))
                if ox is None or oy is None:
                    continue
                if abs(x_val - ox) < min_dx and abs(y_val - oy) < min_dy:
                    y_val += min_dy * (1 if y_val >= oy else -1)
                    x_val += min_dx * 0.35 * (1 if x_val >= ox else -1)
                    shifted = True
            if not shifted:
                break
        target["x"] = x_val
        target["y"] = y_val
    return updated


def _parse_label_tokens(text):
    raw = (text or "").strip()
    if not raw:
        return []
    tokens = [token.strip() for token in re.split(r"[\s,;]+", raw) if token.strip()]
    return list(dict.fromkeys(tokens))


def _row_matches_label_tokens(row, tokens, exact_match=False):
    if not tokens:
        return False
    values = [str(row.get("Genes") or "").lower(), str(row.get("index") or "").lower()]
    for token in tokens:
        query = token.lower()
        if not query:
            continue
        if exact_match and any(query == value for value in values):
            return True
        if (not exact_match) and any(query in value for value in values):
            return True
    return False


def _row_matches_label_cutoffs(row, pval_max, log2fc_min):
    p_val = _as_float(row.get("p_value"))
    fc_val = _as_float(row.get("log2fc"))
    if p_val is None or fc_val is None:
        return False
    if pval_max is None and log2fc_min is None:
        return False
    neg_logp = float(-np.log10(max(float(p_val), 1e-300)))
    p_ok = True if pval_max is None else neg_logp >= float(pval_max)
    fc_ok = True if log2fc_min is None else abs(fc_val) >= float(log2fc_min)
    return p_ok and fc_ok


def _label_priority_key(row, token_hit):
    p_val = _as_float(row.get("p_value"))
    fc_val = _as_float(row.get("log2fc"))
    p_sort = p_val if p_val is not None else float("inf")
    fc_sort = -abs(fc_val) if fc_val is not None else 0.0
    token_rank = 0 if token_hit else 1
    return (token_rank, p_sort, fc_sort)


def _styled_label_text(text):
    safe = escape(str(text or ""))
    return (
        "<span style=\"background-color:rgba(255,255,255,0.92);"
        "border:1px solid #334155;border-radius:8px;padding:2px 6px;"
        "display:inline-block;\">"
        f"{safe}</span>"
    )


def _normalize_annotation_text(raw_text):
    text = str(raw_text or "")
    return re.sub(r"<[^>]+>", "", text).strip()


def _volcano_labels_from_relayout(existing_labels, relayout_data):
    if not relayout_data:
        return existing_labels
    labels = [dict(item) for item in (existing_labels or [])]
    if isinstance(relayout_data.get("annotations"), list):
        rebuilt = []
        for idx, ann in enumerate(relayout_data["annotations"]):
            point_x = _as_float(ann.get("x"))
            point_y = _as_float(ann.get("y"))
            label_x = _as_float(ann.get("ax"))
            label_y = _as_float(ann.get("ay"))
            rebuilt.append(
                {
                    "id": labels[idx].get("id", f"ann-{idx}") if idx < len(labels) else f"ann-{idx}",
                    "x": label_x if label_x is not None else (labels[idx].get("x") if idx < len(labels) else None),
                    "y": label_y if label_y is not None else (labels[idx].get("y") if idx < len(labels) else None),
                    "point_x": point_x if point_x is not None else (labels[idx].get("point_x") if idx < len(labels) else None),
                    "point_y": point_y if point_y is not None else (labels[idx].get("point_y") if idx < len(labels) else None),
                    "text": _normalize_annotation_text(ann.get("text", "")),
                }
            )
        if rebuilt == labels:
            return existing_labels
        return rebuilt
    pattern = re.compile(r"annotations\[(\d+)\]\.(x|y|ax|ay|text)$")
    changed = False
    for key, value in relayout_data.items():
        match = pattern.match(str(key))
        if not match:
            continue
        idx = int(match.group(1))
        field = match.group(2)
        while idx >= len(labels):
            labels.append({"id": f"ann-{len(labels)}", "x": 0.0, "y": 0.0, "point_x": None, "point_y": None, "text": ""})
        if field in {"x", "y", "ax", "ay"}:
            cast_value = _as_float(value)
            if cast_value is None:
                continue
            if field == "x":
                labels[idx]["point_x"] = cast_value
            elif field == "y":
                labels[idx]["point_y"] = cast_value
            elif field == "ax":
                labels[idx]["x"] = cast_value
            elif field == "ay":
                labels[idx]["y"] = cast_value
        elif field == "text":
            labels[idx][field] = _normalize_annotation_text(value)
        changed = True
    return labels if changed else existing_labels


def _build_volcano_figure_from_records(records, style_data, labels, highlight_enabled=True, highlight_color="#16a34a"):
    if not records:
        return no_update
    style = _normalize_volcano_style(style_data)
    df = pd.DataFrame(records).copy()
    if "p_value" not in df.columns or "log2fc" not in df.columns:
        return no_update
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["log2fc"] = pd.to_numeric(df["log2fc"], errors="coerce")
    df["__row_idx"] = list(range(len(df)))
    df = df.dropna(subset=["p_value", "log2fc"])
    if df.empty:
        return no_update
    df["neg_log10_p"] = -df["p_value"].clip(lower=1e-300).apply(np.log10)
    pval_cutoff = float(style["pval"])
    fc_cutoff = float(style["log2fc"])
    up_mask = (df["p_value"] <= pval_cutoff) & (df["log2fc"] >= fc_cutoff)
    down_mask = (df["p_value"] <= pval_cutoff) & (df["log2fc"] <= -fc_cutoff)
    df["sig_custom"] = "not significant"
    df.loc[up_mask, "sig_custom"] = "upregulated"
    df.loc[down_mask, "sig_custom"] = "downregulated"
    fig = px.scatter(
        df,
        x="log2fc",
        y="neg_log10_p",
        color="sig_custom",
        color_discrete_map={
            "upregulated": style["colors"]["up"],
            "downregulated": style["colors"]["down"],
            "not significant": style["colors"]["ns"],
        },
        hover_data=[c for c in ["Genes", "p_value", "log2fc", "index"] if c in df.columns],
        custom_data=["__row_idx"],
        title="Volcano plot",
    )
    fig.add_vline(x=fc_cutoff, line_dash="dash")
    fig.add_vline(x=-fc_cutoff, line_dash="dash")
    fig.add_hline(y=-np.log10(max(pval_cutoff, 1e-300)), line_dash="dash")
    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=30, r=20, t=50, b=30),
        font={"family": style["font_family"], "size": style["font_size"]},
    )
    label_ids = {str(item.get("id")) for item in (labels or [])}
    highlighted_x = []
    highlighted_y = []
    for row_idx, row in enumerate(records or []):
        if str(row_idx) not in label_ids:
            continue
        point = _volcano_point_xy(row)
        if point is None:
            continue
        highlighted_x.append(point[0])
        highlighted_y.append(point[1])
    if highlight_enabled and highlighted_x:
        base_size = 8
        if fig.data:
            try:
                marker_size = getattr(fig.data[0].marker, "size", None)
                if isinstance(marker_size, (int, float)):
                    base_size = marker_size
            except Exception:
                base_size = 8
        fig.add_scatter(
            x=highlighted_x,
            y=highlighted_y,
            mode="markers",
            marker={
                "size": base_size,
                "color": str(highlight_color or "#16a34a"),
                "line": {"color": str(highlight_color or "#16a34a"), "width": 1.5},
                "symbol": "circle",
            },
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
            name="Labeled points",
        )
        # Keep highlighted markers as the last trace so they render on top.
        fig.data = tuple([trace for trace in fig.data[:-1]] + [fig.data[-1]])
    annotations = []
    for item in labels or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        point_x = _as_float(item.get("point_x"))
        point_y = _as_float(item.get("point_y"))
        label_x = _as_float(item.get("x"))
        label_y = _as_float(item.get("y"))
        if label_x is None or label_y is None:
            continue
        if point_x is None:
            point_x = label_x
        if point_y is None:
            point_y = label_y
        annotations.append(
            dict(
                x=point_x,
                y=point_y,
                text=_styled_label_text(text),
                showarrow=True,
                arrowhead=0,
                arrowsize=1,
                arrowwidth=1.4,
                arrowcolor="#64748b",
                ax=label_x,
                ay=label_y,
                axref="x",
                ayref="y",
                font={"family": style["font_family"], "size": style["font_size"]},
                align="left",
            )
        )
    fig.update_layout(annotations=annotations)
    return fig


def _volcano_point_xy(row):
    x_val = pd.to_numeric(pd.Series([row.get("log2fc")]), errors="coerce").iloc[0]
    p_val = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
    if pd.isna(x_val) or pd.isna(p_val):
        return None
    y_val = float(-np.log10(max(float(p_val), 1e-300)))
    return float(x_val), y_val


def _nearest_point(x_val, y_val, records):
    best = None
    best_dist = None
    for row in records or []:
        coords = _volcano_point_xy(row)
        if coords is None:
            continue
        px, py = coords
        dist = (px - float(x_val)) ** 2 + (py - float(y_val)) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (px, py)
    return best


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
    Output("de-volcano-style-store", "data", allow_duplicate=True),
    Output("de-volcano-labels-store", "data", allow_duplicate=True),
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
    State("de-color-up", "value"),
    State("de-color-down", "value"),
    State("de-color-ns", "value"),
    State("de-font-family", "value"),
    State("de-font-size", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_de_callback(
    n_clicks,
    group1,
    group2,
    method,
    layer,
    pval,
    log2fc,
    color_up,
    color_down,
    color_ns,
    font_family,
    font_size,
    session_id,
):
    if not n_clicks or not session_id:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

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
        return msg, no_update, no_update, no_update, no_update, no_update, no_update

    fig = volcano_plotly_figure(df, pval=float(pval or 0.05), log2fc=float(log2fc or 1.0))
    records = df.reset_index().to_dict("records")
    table_records = _format_de_records_for_table(records)
    columns = _de_table_columns(records)
    style_data = _normalize_volcano_style(
        {
            "pval": pval,
            "log2fc": log2fc,
            "colors": {"up": color_up, "down": color_down, "ns": color_ns},
            "font_family": font_family,
            "font_size": font_size,
        }
    )
    return msg, fig, style_data, [], records, table_records, columns


@callback(
    Output("de-label-column", "options"),
    Output("de-label-column", "value"),
    Input("de-table-store", "data"),
    State("de-label-column", "value"),
    prevent_initial_call=True,
)
def update_de_label_column_options(records, current_value):
    if not records:
        return [], no_update
    columns = list(records[0].keys())
    preferred = [c for c in ["Genes", "index", "gene", "symbol"] if c in columns]
    options = [{"label": c, "value": c} for c in columns]
    default_value = current_value if current_value in columns else (preferred[0] if preferred else columns[0])
    return options, default_value


@callback(
    Output("de-volcano-style-store", "data"),
    Output("de-volcano-labels-store", "data", allow_duplicate=True),
    Input("btn-apply-volcano-style", "n_clicks"),
    Input("de-color-up", "value"),
    Input("de-color-down", "value"),
    Input("de-color-ns", "value"),
    Input("de-font-family", "value"),
    Input("de-font-size", "value"),
    State("de-pval", "value"),
    State("de-log2fc", "value"),
    State("de-volcano-labels-store", "data"),
    prevent_initial_call=True,
)
def update_volcano_style_store(_n_clicks, color_up, color_down, color_ns, font_family, font_size, pval, log2fc, labels):
    style_data = _normalize_volcano_style(
        {
            "pval": pval,
            "log2fc": log2fc,
            "colors": {"up": color_up, "down": color_down, "ns": color_ns},
            "font_family": font_family,
            "font_size": font_size,
        }
    )
    return style_data, labels or []


@callback(
    Output("de-volcano-labels-store", "data", allow_duplicate=True),
    Input("btn-add-volcano-label", "n_clicks"),
    Input("btn-add-volcano-label-rules", "n_clicks"),
    Input("btn-clear-volcano-labels", "n_clicks"),
    Input("btn-de", "n_clicks"),
    Input("de-volcano", "relayoutData"),
    State("de-volcano", "selectedData"),
    State("de-volcano", "clickData"),
    State("de-label-list", "value"),
    State("de-label-pval-max", "value"),
    State("de-label-log2fc-min", "value"),
    State("de-label-exact-match-toggle", "value"),
    State("de-label-max-count", "value"),
    State("de-table-store", "data"),
    State("de-label-column", "value"),
    State("de-volcano-labels-store", "data"),
    prevent_initial_call=True,
)
def update_volcano_labels(
    _add_clicks,
    _add_rule_clicks,
    _clear_clicks,
    _de_clicks,
    relayout_data,
    selected_data,
    click_data,
    label_list_text,
    label_pval_max,
    label_log2fc_min,
    label_exact_match_toggle,
    label_max_count,
    records,
    label_column,
    labels,
):
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    labels = [dict(item) for item in (labels or [])]
    if trigger_id in {"btn-clear-volcano-labels", "btn-de"}:
        return []
    if trigger_id == "de-volcano":
        relayout_data = relayout_data or {}
        if not _volcano_relayout_updates_annotations(relayout_data):
            return no_update
        next_labels = _volcano_labels_from_relayout(labels, relayout_data)
        if next_labels is labels or next_labels == labels:
            return no_update
        return next_labels
    if trigger_id == "btn-add-volcano-label":
        if not records:
            return labels
        points = []
        if selected_data and selected_data.get("points"):
            points = selected_data.get("points") or []
        elif click_data and click_data.get("points"):
            points = click_data.get("points") or []
        if not points:
            return labels
        max_labels = int(_as_float(label_max_count) or 40)
        max_labels = max(max_labels, 1)
        if len(labels) >= max_labels:
            return labels
        x_span, y_span = _de_axis_spans(records)
        seen_ids = {str(item.get("id")) for item in labels}
        new_ids = []
        for point in points:
            if len(labels) >= max_labels:
                break
            idx = _de_row_idx_from_selected_point(point) if isinstance(point, dict) else None
            if not isinstance(idx, int) or idx < 0 or idx >= len(records):
                continue
            row = records[idx]
            label_text = str(row.get(label_column) or row.get("Genes") or row.get("index") or f"point-{idx}")
            label_id = str(idx)
            if label_id in seen_ids:
                continue
            point_xy = _volcano_point_xy(row)
            if point_xy is None:
                continue
            point_x, point_y = point_xy
            label_x, label_y = _default_label_position(point_x, point_y, len(labels), x_span, y_span)
            labels.append(
                {
                    "id": label_id,
                    "x": label_x,
                    "y": label_y,
                    "point_x": point_x,
                    "point_y": point_y,
                    "text": label_text,
                }
            )
            seen_ids.add(label_id)
            new_ids.append(label_id)
        return _dodge_labels(labels, new_ids, x_span, y_span)

    if trigger_id == "btn-add-volcano-label-rules":
        if not records:
            return labels
        tokens = _parse_label_tokens(label_list_text)
        exact_match = bool(label_exact_match_toggle and "exact" in label_exact_match_toggle)
        pval_max = _as_float(label_pval_max)
        log2fc_min = _as_float(label_log2fc_min)
        max_labels = int(_as_float(label_max_count) or 40)
        max_labels = max(max_labels, 1)
        if not tokens and pval_max is None and log2fc_min is None:
            return labels
        x_span, y_span = _de_axis_spans(records)
        seen_ids = {str(item.get("id")) for item in labels}
        if len(labels) >= max_labels:
            return labels
        candidates = []
        for idx, row in enumerate(records):
            token_match = _row_matches_label_tokens(row, tokens, exact_match=exact_match)
            cutoff_match = _row_matches_label_cutoffs(row, pval_max, log2fc_min)
            if not (token_match or cutoff_match):
                continue
            label_id = str(idx)
            if label_id in seen_ids:
                continue
            candidates.append((idx, row, token_match))
        candidates.sort(key=lambda item: _label_priority_key(item[1], item[2]))
        new_ids = []
        for idx, row, _token_match in candidates:
            if len(labels) >= max_labels:
                break
            label_id = str(idx)
            point_xy = _volcano_point_xy(row)
            if point_xy is None:
                continue
            point_x, point_y = point_xy
            label_text = str(row.get(label_column) or row.get("Genes") or row.get("index") or f"point-{idx}")
            label_x, label_y = _default_label_position(point_x, point_y, len(labels), x_span, y_span)
            labels.append(
                {
                    "id": label_id,
                    "x": label_x,
                    "y": label_y,
                    "point_x": point_x,
                    "point_y": point_y,
                    "text": label_text,
                }
            )
            seen_ids.add(label_id)
            new_ids.append(label_id)
        return _dodge_labels(labels, new_ids, x_span, y_span)
    return labels


@callback(
    Output("de-label-editor-id", "options"),
    Output("de-label-editor-id", "value"),
    Input("de-volcano-labels-store", "data"),
    State("de-label-editor-id", "value"),
    prevent_initial_call=True,
)
def sync_label_manager_panel(labels, selected_id):
    labels = [dict(item) for item in (labels or [])]
    if not labels:
        return [], None
    options = [{"label": f"{idx + 1}. {str(item.get('text', ''))[:40]}", "value": str(item.get("id", idx))} for idx, item in enumerate(labels)]
    selected = str(selected_id) if selected_id is not None else str(labels[0].get("id", "0"))
    if selected not in {str(opt["value"]) for opt in options}:
        selected = str(labels[0].get("id", "0"))
    return options, selected


@callback(
    Output("de-label-editor-text", "value"),
    Output("de-label-editor-x", "value"),
    Output("de-label-editor-y", "value"),
    Output("de-label-manager-summary", "children"),
    Input("de-label-editor-id", "value"),
    Input("de-volcano-labels-store", "data"),
    prevent_initial_call=True,
)
def hydrate_selected_label(selected_id, labels):
    labels = [dict(item) for item in (labels or [])]
    if not labels:
        return "", 0, 0, "No labels added yet. Select points in the volcano and click 'Add labels from selection/click'."
    selected = str(selected_id) if selected_id is not None else str(labels[0].get("id", "0"))
    current = None
    for item in labels:
        if str(item.get("id")) == selected:
            current = item
            break
    current = current or labels[0]
    return str(current.get("text", "")), current.get("x", 0), current.get("y", 0), f"{len(labels)} label(s) currently on volcano."


@callback(
    Output("de-label-warning", "children"),
    Input("de-volcano-labels-store", "data"),
    Input("de-label-max-count", "value"),
    prevent_initial_call=True,
)
def update_label_count_warning(labels, max_count_value):
    labels = labels or []
    max_count = int(_as_float(max_count_value) or 40)
    max_count = max(max_count, 1)
    n_labels = len(labels)
    if n_labels >= max_count:
        return f"Label limit reached ({n_labels}/{max_count}). Increase 'Max labels' or clear some labels."
    if n_labels >= max(int(max_count * 0.8), 1):
        return f"High label density ({n_labels}/{max_count}). Consider lowering labels for readability."
    return ""


@callback(
    Output("de-volcano-labels-store", "data", allow_duplicate=True),
    Output("de-label-manager-summary", "children", allow_duplicate=True),
    Input("btn-update-volcano-label", "n_clicks"),
    Input("btn-delete-volcano-label", "n_clicks"),
    Input("btn-snap-volcano-labels", "n_clicks"),
    State("de-label-editor-id", "value"),
    State("de-label-editor-text", "value"),
    State("de-label-editor-x", "value"),
    State("de-label-editor-y", "value"),
    State("de-volcano-labels-store", "data"),
    State("de-table-store", "data"),
    prevent_initial_call=True,
)
def mutate_volcano_labels(
    n_update,
    n_delete,
    n_snap,
    selected_id,
    label_text,
    label_x,
    label_y,
    labels,
    records,
):
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    labels = [dict(item) for item in (labels or [])]
    if not labels:
        return labels, "No labels available to modify."
    selected_id = str(selected_id) if selected_id is not None else None
    selected_idx = None
    for idx, item in enumerate(labels):
        if str(item.get("id")) == selected_id:
            selected_idx = idx
            break

    if trigger_id == "btn-update-volcano-label":
        if selected_idx is None:
            return labels, "Select a label first."
        labels[selected_idx]["text"] = str(label_text or "")
        try:
            labels[selected_idx]["x"] = float(label_x)
            labels[selected_idx]["y"] = float(label_y)
        except Exception:
            return labels, "X/Y must be numeric."
        return labels, "Updated selected label."

    if trigger_id == "btn-delete-volcano-label":
        if selected_idx is None:
            return labels, "Select a label first."
        labels.pop(selected_idx)
        return labels, f"Deleted label. {len(labels)} label(s) remaining."

    if trigger_id == "btn-snap-volcano-labels":
        if not records:
            return labels, "No DE points available for snapping."
        x_span, y_span = _de_axis_spans(records)
        moved_ids = []
        updated = 0
        for idx, item in enumerate(labels):
            if selected_idx is not None and idx != selected_idx:
                continue
            nearest = _nearest_point(item.get("x", 0), item.get("y", 0), records)
            if nearest is None:
                continue
            labels[idx]["point_x"] = nearest[0]
            labels[idx]["point_y"] = nearest[1]
            label_x, label_y = _default_label_position(nearest[0], nearest[1], idx, x_span, y_span)
            labels[idx]["x"] = label_x
            labels[idx]["y"] = label_y
            moved_ids.append(labels[idx].get("id"))
            updated += 1
        labels = _dodge_labels(labels, moved_ids, x_span, y_span)
        if selected_idx is not None:
            return labels, "Snapped selected label to nearest point." if updated else "Could not snap selected label."
        return labels, f"Snapped {updated} label(s) to nearest points."

    return labels, no_update


@callback(
    Output("de-volcano", "figure", allow_duplicate=True),
    Input("de-table-store", "data"),
    Input("de-volcano-style-store", "data"),
    Input("de-volcano-labels-store", "data"),
    Input("de-highlight-labeled-toggle", "value"),
    Input("de-highlight-labeled-color", "value"),
    prevent_initial_call=True,
)
def render_customized_volcano(records, style_data, labels, highlight_toggle, highlight_color):
    if not records:
        return no_update
    highlight_enabled = bool(highlight_toggle and "on" in highlight_toggle)
    return _build_volcano_figure_from_records(
        records,
        style_data,
        labels,
        highlight_enabled=highlight_enabled,
        highlight_color=highlight_color or "#16a34a",
    )


@callback(
    Output("main-tabs", "value"),
    Output("btn-editor-load-de", "n_clicks", allow_duplicate=True),
    Output("btn-editor-load-enrichment", "n_clicks", allow_duplicate=True),
    Input("btn-open-de-editor", "n_clicks"),
    Input("btn-open-enrich-editor", "n_clicks"),
    State("btn-editor-load-de", "n_clicks"),
    State("btn-editor-load-enrichment", "n_clicks"),
    prevent_initial_call=True,
)
def open_editor_from_plot(n_open_de, n_open_enrich, current_de_clicks, current_enrich_clicks):
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    if trigger_id == "btn-open-de-editor" and n_open_de:
        return "tab-editor", int(current_de_clicks or 0) + 1, no_update
    if trigger_id == "btn-open-enrich-editor" and n_open_enrich:
        return "tab-editor", no_update, int(current_enrich_clicks or 0) + 1
    return no_update, no_update, no_update


@callback(
    Output("de-table", "data", allow_duplicate=True),
    Output("de-selection-info", "children"),
    Input("de-volcano", "selectedData"),
    Input("de-volcano", "clickData"),
    State("de-table-store", "data"),
    prevent_initial_call=True,
)
def update_de_table_from_selection(selected_data, click_data, records):
    if not records:
        return no_update, "No DE table available yet."
    n_all = len(records)
    all_rows = _format_de_records_for_table(records)

    if selected_data and selected_data.get("points"):
        filtered, outcome = _de_plotly_selection_rows(selected_data, records)
        n_pts = len(selected_data["points"])
        if outcome == "unresolved":
            return all_rows, (
                f"Volcano selection has {n_pts} point(s) but no rows matched the DE table "
                f"(missing customdata or out of range). Showing all {n_all} rows."
            )
        return _format_de_records_for_table(filtered), f"Showing {len(filtered)} selected rows from volcano selection."

    if click_data and click_data.get("points"):
        filtered, outcome = _de_plotly_selection_rows(click_data, records)
        if outcome == "subset":
            return _format_de_records_for_table(filtered), (
                f"Showing {len(filtered)} row(s) from volcano click. "
                "Use box or lasso select to filter multiple rows."
            )
        return all_rows, (
            "Volcano click did not resolve to a table row. "
            "Use box or lasso selection, or run DE again if the figure was reset."
        )

    return all_rows, f"Showing all {n_all} rows."


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
    Output("editor-source-svg", "value"),
    Output("edited-svg-store", "data", allow_duplicate=True),
    Output("edited-svg-meta-store", "data"),
    Output("editor-dirty-store", "data", allow_duplicate=True),
    Output("editor-source-badge", "children"),
    Output("editor-save-badge", "children"),
    Output("editor-log", "children"),
    Input("btn-editor-load-de", "n_clicks"),
    Input("btn-editor-load-enrichment", "n_clicks"),
    Input("btn-editor-reset", "n_clicks"),
    State("de-table-store", "data"),
    State("de-pval", "value"),
    State("de-log2fc", "value"),
    State("img-enrichment-svg", "src"),
    State("session-id", "data"),
    State("edited-svg-meta-store", "data"),
    State("editor-dirty-store", "data"),
    prevent_initial_call=True,
)
def load_editor_source(
    n_load_de,
    n_load_enrichment,
    n_reset,
    de_records,
    de_pval,
    de_log2fc,
    enrichment_src,
    session_id,
    meta,
    editor_dirty,
):
    trigger_id = (
        callback_context.triggered[0]["prop_id"].split(".")[0]
        if callback_context.triggered
        else ""
    )
    if trigger_id == "btn-editor-load-de":
        if not de_records:
            return no_update, no_update, no_update, False, "Source: none", "Save: idle", "No DE records available yet."
        try:
            plot_key = "de"
            source_svg = volcano_svg_markup_from_records(
                records=de_records,
                pval=float(de_pval or 0.05),
                log2fc=float(de_log2fc or 1.0),
            )
            saved_svg = load_edited_svg_for_session(session_id, plot_key) if session_id else ""
            initial_svg = saved_svg or source_svg
            plot_key, label = _editor_plot_key_label(plot_key)
            next_meta = {
                "plot_key": plot_key,
                "label": label,
                "loaded_at": datetime.now().isoformat(),
                "source_svg": source_svg,
            }
            warning = "Warning: replaced unsaved editor content.\n" if editor_dirty else ""
            save_badge = "Save: loaded saved edit" if saved_svg else "Save: idle"
            return initial_svg, initial_svg, next_meta, False, f"Source: {label}", save_badge, f"{warning}Loaded {label} SVG into editor."
        except Exception as exc:
            return no_update, no_update, no_update, False, "Source: none", "Save: idle", f"Failed to load DE SVG: {exc}"

    if trigger_id == "btn-editor-load-enrichment":
        if not enrichment_src:
            return no_update, no_update, no_update, False, "Source: none", "Save: idle", "No enrichment SVG loaded yet."
        try:
            plot_key = "enrichment"
            source_svg = svg_data_uri_to_markup(enrichment_src)
            saved_svg = load_edited_svg_for_session(session_id, plot_key) if session_id else ""
            initial_svg = saved_svg or source_svg
            plot_key, label = _editor_plot_key_label(plot_key)
            next_meta = {
                "plot_key": plot_key,
                "label": label,
                "loaded_at": datetime.now().isoformat(),
                "source_svg": source_svg,
            }
            warning = "Warning: replaced unsaved editor content.\n" if editor_dirty else ""
            save_badge = "Save: loaded saved edit" if saved_svg else "Save: idle"
            return initial_svg, initial_svg, next_meta, False, f"Source: {label}", save_badge, f"{warning}Loaded {label} SVG into editor."
        except Exception as exc:
            return no_update, no_update, no_update, False, "Source: none", "Save: idle", f"Failed to load enrichment SVG: {exc}"

    if trigger_id == "btn-editor-reset":
        base_svg = (meta or {}).get("source_svg", "")
        label = (meta or {}).get("label", "none")
        if not base_svg:
            return no_update, no_update, no_update, False, f"Source: {label}", "Save: idle", "Nothing to reset. Load an SVG source first."
        return base_svg, base_svg, meta, False, f"Source: {label}", "Save: idle", f"Reset editor to loaded {label} source."

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update


@callback(
    Output("edited-svg-store", "data", allow_duplicate=True),
    Output("editor-dirty-store", "data", allow_duplicate=True),
    Output("editor-dirty-badge", "children"),
    Input("editor-edited-svg", "value"),
    Input("editor-dirty-flag", "value"),
    prevent_initial_call=True,
)
def sync_editor_state(edited_svg_text, dirty_flag):
    if edited_svg_text is None:
        return no_update, no_update, no_update
    if edited_svg_text == "":
        return "", False, "Dirty: no"
    dirty = str(dirty_flag).strip().lower() == "true"
    return edited_svg_text, dirty, f"Dirty: {'yes' if dirty else 'no'}"


@callback(
    Output("editor-save-badge", "children", allow_duplicate=True),
    Output("editor-log", "children", allow_duplicate=True),
    Output("edited-svg-meta-store", "data", allow_duplicate=True),
    Output("editor-dirty-store", "data", allow_duplicate=True),
    Output("editor-dirty-badge", "children", allow_duplicate=True),
    Input("btn-editor-save", "n_clicks"),
    State("session-id", "data"),
    State("edited-svg-store", "data"),
    State("edited-svg-meta-store", "data"),
    prevent_initial_call=True,
)
def save_editor_edits(n_clicks, session_id, edited_svg_text, meta):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update
    if not session_id:
        return "Save: failed", "Missing session id.", meta, True, "Dirty: yes"
    plot_key = ((meta or {}).get("plot_key") or "").strip()
    if not plot_key:
        return "Save: failed", "Load DE or enrichment SVG before saving.", meta, True, "Dirty: yes"
    if not edited_svg_text:
        return "Save: failed", "No edited SVG payload found in editor.", meta, True, "Dirty: yes"
    try:
        save_edited_svg_for_session(session_id, plot_key, edited_svg_text)
        next_meta = dict(meta or {})
        next_meta["saved_at"] = datetime.now().isoformat()
        next_meta["saved_plot_key"] = plot_key
        return "Save: saved", f"Saved edited SVG for {plot_key}.", next_meta, False, "Dirty: no"
    except Exception as exc:
        return "Save: failed", f"Save failed: {exc}", meta, True, "Dirty: yes"


@callback(
    Output("download-edited-svg", "data"),
    Input("btn-download-edited-svg", "n_clicks"),
    State("edited-svg-store", "data"),
    State("edited-svg-meta-store", "data"),
    prevent_initial_call=True,
)
def download_edited_svg(n_clicks, edited_svg_text, meta):
    if not n_clicks or not edited_svg_text:
        return no_update
    plot_key = ((meta or {}).get("plot_key") or "plot").strip() or "plot"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plot_key}_edited_{stamp}.svg"

    def _write_svg(buffer: io.BytesIO):
        buffer.write(edited_svg_text.encode("utf-8"))
        buffer.seek(0)

    return dcc.send_bytes(_write_svg, filename)


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
    State("edited-svg-store", "data"),
    State("edited-svg-meta-store", "data"),
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
    edited_svg_text,
    edited_svg_meta,
):
    if not n_clicks or not session_id:
        return no_update

    summary_df = summary_dataframe(session_id)
    embed_df = embedding_dataframe(session_id)
    de_df = pd.DataFrame(de_records) if de_records else pd.DataFrame()
    enrich_df = enrichment_result_dataframe(session_id, functional_key) if functional_key else pd.DataFrame()
    edited_de_svg = load_edited_svg_for_session(session_id, "de")
    edited_enrichment_svg = load_edited_svg_for_session(session_id, "enrichment")
    active_plot_key = ((edited_svg_meta or {}).get("plot_key") or "").strip()
    if edited_svg_text and active_plot_key == "de":
        edited_de_svg = edited_svg_text
    if edited_svg_text and active_plot_key == "enrichment":
        edited_enrichment_svg = edited_svg_text

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
                "- plots/de_edited.svg: Edited DE volcano SVG (if available).",
                "- plots/enrichment_edited.svg: Edited enrichment SVG (if available).",
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

            if edited_de_svg:
                zf.writestr("plots/de_edited.svg", edited_de_svg)
            if edited_enrichment_svg:
                zf.writestr("plots/enrichment_edited.svg", edited_enrichment_svg)
                data_uri = svg_markup_to_data_uri(edited_enrichment_svg)
                data, _mime = _decode_data_uri(data_uri)
                if data:
                    zf.writestr("plots/enrichment.svg", data)

            if volcano_fig:
                zf.writestr("plots/de_volcano.json", json.dumps(volcano_fig, indent=2))

        buffer.seek(0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_bytes(_write_zip, f"scpviz_bundle_{stamp}.zip")


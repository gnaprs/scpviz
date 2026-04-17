"""Main Dash layout for scpviz app."""

from dash import dash_table, dcc, html


COLORS = {
    "page_bg": "#0b1220",
    "panel_bg": "#0f172a",
    "panel_soft": "#111c33",
    "border": "#24324c",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "brand": "#60a5fa",
    "brand_soft": "#172848",
}

WRAPPER_STYLE = {
    "minHeight": "100vh",
    "padding": "20px",
    "background": "radial-gradient(circle at 0% 0%, #1a2a45 0%, #0b1220 45%), #0b1220",
}

APP_STYLE = {
    "maxWidth": "1240px",
    "margin": "28px auto",
    "padding": "22px 24px",
    "background": "linear-gradient(180deg, #111a2f 0%, #0f172a 100%)",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "16px",
    "boxShadow": "0 18px 42px rgba(0, 0, 0, 0.34)",
    "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
    "color": COLORS["text"],
}

TITLE_STYLE = {
    "margin": "0 0 4px 0",
    "fontSize": "28px",
    "fontWeight": "700",
    "letterSpacing": "-0.02em",
    "color": "#f8fafc",
}

SUBTITLE_STYLE = {
    "margin": "0 0 14px 0",
    "fontSize": "14px",
    "color": COLORS["muted"],
}

TABS_STYLE = {"marginTop": "10px", "borderRadius": "10px", "overflow": "hidden"}
TAB_STYLE = {
    "padding": "10px 14px",
    "fontWeight": "600",
    "fontSize": "14px",
    "border": "none",
    "borderBottom": f"2px solid {COLORS['border']}",
    "backgroundColor": COLORS["panel_soft"],
    "color": "#b7c4d8",
}
TAB_SELECTED_STYLE = {
    "padding": "10px 14px",
    "fontWeight": "700",
    "fontSize": "14px",
    "border": "none",
    "borderBottom": f"2px solid {COLORS['brand']}",
    "backgroundColor": COLORS["panel_bg"],
    "color": COLORS["brand"],
}

TAB_CONTENT_STYLE = {
    "padding": "18px 8px 10px 8px",
    "border": f"1px solid {COLORS['border']}",
    "borderTop": "none",
    "borderRadius": "0 0 12px 12px",
    "backgroundColor": COLORS["panel_bg"],
}

LABEL_STYLE = {
    "display": "block",
    "margin": "0 0 6px 0",
    "fontSize": "13px",
    "fontWeight": "600",
    "color": "#cbd5e1",
}

INPUT_STYLE = {
    "padding": "9px 11px",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "backgroundColor": "#0b1325",
    "color": COLORS["text"],
}

BUTTON_STYLE = {
    "padding": "9px 15px",
    "border": "none",
    "borderRadius": "10px",
    "background": "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
    "color": "#eff6ff",
    "fontWeight": "600",
    "cursor": "pointer",
    "boxShadow": "0 8px 18px rgba(30, 64, 175, 0.35)",
}

BUTTON_SUBTLE_STYLE = {
    "padding": "9px 15px",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "backgroundColor": COLORS["brand_soft"],
    "color": COLORS["brand"],
    "fontWeight": "600",
    "cursor": "pointer",
}

BUNDLE_BUTTON_STYLE = {**BUTTON_SUBTLE_STYLE, "marginBottom": "14px"}

LOG_STYLE = {
    "whiteSpace": "pre-wrap",
    "marginTop": "10px",
    "padding": "10px 12px",
    "backgroundColor": "#0a1428",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "fontSize": "12px",
    "color": "#cbd5e1",
}

SECTION_STYLE = {"marginBottom": "14px"}
H4_STYLE = {"margin": "14px 0 8px 0", "fontSize": "16px", "color": "#dbeafe"}
PLOT_IMG_STYLE = {
    "maxWidth": "100%",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "backgroundColor": "#0b1325",
    "boxShadow": "0 4px 14px rgba(0, 0, 0, 0.35)",
}
LOADING_STYLE = {"marginTop": "10px"}
IFRAME_STYLE = {
    "width": "100%",
    "height": "560px",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "backgroundColor": "#0b1325",
}

HELPER_PANEL_STYLE = {
    "marginTop": "10px",
    "padding": "12px",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "backgroundColor": "#0a1428",
}

HELPER_HINT_STYLE = {"marginTop": "8px", "fontSize": "12px", "color": COLORS["muted"]}
BADGE_STYLE = {
    "display": "inline-block",
    "padding": "4px 8px",
    "marginRight": "8px",
    "borderRadius": "999px",
    "border": f"1px solid {COLORS['border']}",
    "backgroundColor": "#0b1325",
    "fontSize": "12px",
    "color": COLORS["muted"],
}

EDITOR_IFRAME_STYLE = {
    "width": "100%",
    "height": "680px",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "backgroundColor": "#0b1325",
    "marginTop": "10px",
}


def build_layout():
    return html.Div(
        [
            html.Div(
                [
                    dcc.Location(id="url"),
                    dcc.Store(id="session-id", storage_type="session"),
                    dcc.Store(id="de-table-store"),
                    dcc.Store(id="edited-svg-store"),
                    dcc.Store(id="edited-svg-meta-store"),
                    dcc.Store(id="editor-dirty-store", data=False),
                    dcc.Store(
                        id="de-volcano-style-store",
                        data={
                            "pval": 0.05,
                            "log2fc": 1.0,
                            "colors": {"up": "#dc2626", "down": "#2563eb", "ns": "#94a3b8"},
                            "font_family": "Arial",
                            "font_size": 12,
                        },
                    ),
                    dcc.Store(id="de-volcano-labels-store", data=[]),
                    dcc.Download(id="download-import-table"),
                    dcc.Download(id="download-qc-table"),
                    dcc.Download(id="download-prep-table"),
                    dcc.Download(id="download-embed-table"),
                    dcc.Download(id="download-de-table"),
                    dcc.Download(id="download-enrich-table"),
                    dcc.Download(id="download-edited-svg"),
                    dcc.Download(id="download-bundle"),
                    dcc.Textarea(id="editor-source-svg", value="", style={"display": "none"}),
                    dcc.Textarea(id="editor-edited-svg", value="", style={"display": "none"}),
                    dcc.Input(id="editor-dirty-flag", value="false", type="text", style={"display": "none"}),
                    html.H2("scpviz Dash Web App", style=TITLE_STYLE),
                    html.P("Upload files and run the full proteomics workflow.", style=SUBTITLE_STYLE),
                    html.Button("Download all tables + plots (ZIP)", id="btn-download-bundle", n_clicks=0, style=BUNDLE_BUTTON_STYLE),
                    html.A(
                        "Open step-by-step user guide",
                        href="/assets/user_guide.html",
                        target="_blank",
                        style={**BUTTON_SUBTLE_STYLE, "display": "inline-block", "marginLeft": "10px", "marginBottom": "14px", "textDecoration": "none"},
                    ),
                    dcc.Tabs(
                        id="main-tabs",
                        value="tab-import",
                        style=TABS_STYLE,
                        content_style=TAB_CONTENT_STYLE,
                        children=[
                            dcc.Tab(
                                label="1) Import",
                                value="tab-import",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=[
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Source type", style=LABEL_STYLE, title="Choose the import path shown in tutorials: DIA-NN report or Proteome Discoverer exports."),
                                    dcc.RadioItems(
                                        id="source-type",
                                        options=[
                                            {"label": html.Span("Proteome Discoverer (PD)", title="Use for Proteome Discoverer exports."), "value": "pd"},
                                            {"label": html.Span("DIA-NN", title="Use for DIA-NN report exports."), "value": "diann"},
                                        ],
                                        value="pd",
                                        inline=True,
                                        className="source-type-radio-group",
                                        inputClassName="source-type-radio-input",
                                        labelClassName="source-type-radio-label",
                                    ),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("obs_columns (comma-separated)", style=LABEL_STYLE, title="Metadata fields parsed from filenames into .obs/.summary."),
                                    dcc.Input(id="obs-columns", type="text", value="sample,cellline,treatment", style={**INPUT_STYLE, "width": "60%"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Delimiter (optional, mostly for DIA-NN filename parsing)", style=LABEL_STYLE, title="Character used for metadata parsing."),
                                    dcc.Input(id="import-delimiter", type="text", value="", style={**INPUT_STYLE, "width": "180px"}),
                                    html.Button("obs_columns helper", id="btn-open-obs-helper", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Div(
                                        [
                                            html.Div(id="obs-helper-preview", style={"marginBottom": "8px"}),
                                            dash_table.DataTable(
                                                id="obs-helper-rename-table",
                                                columns=[
                                                    {"name": "Detected token", "id": "token"},
                                                    {"name": "obs column name (editable)", "id": "name", "editable": True},
                                                ],
                                                data=[],
                                                editable=True,
                                                style_table={"overflowX": "auto", "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
                                                style_header={"backgroundColor": "#1b2942", "fontWeight": "700", "borderBottom": f"1px solid {COLORS['border']}", "color": "#dbeafe"},
                                                style_cell={"textAlign": "left", "fontSize": "12px", "padding": "8px", "borderBottom": "1px solid #1f2d47", "backgroundColor": "#0f172a", "color": COLORS["text"]},
                                            ),
                                            html.Div(style={"height": "8px"}),
                                            dcc.Input(id="obs-helper-suggest", type="text", value="", style={**INPUT_STYLE, "width": "70%"}, placeholder="Auto-generated obs_columns will appear here"),
                                            html.Button("Apply to obs_columns", id="btn-apply-obs-helper", n_clicks=0, style={**BUTTON_STYLE, "marginLeft": "10px"}),
                                            html.Button("Refresh suggestion", id="btn-refresh-obs-helper", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                            html.Div(
                                                "Suggestion is inferred from uploaded filename patterns and may need manual editing.",
                                                style=HELPER_HINT_STYLE,
                                            ),
                                        ],
                                        id="obs-helper-panel",
                                        style={**HELPER_PANEL_STYLE, "display": "none"},
                                    ),
                                    html.Div(style=SECTION_STYLE),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("PD protein file", style=LABEL_STYLE),
                                                    dcc.Upload(id="upload-prot", children=html.Button("Upload protein file", style=BUTTON_SUBTLE_STYLE), multiple=False),
                                                    html.Div(id="upload-prot-msg"),
                                                ],
                                                style={"display": "inline-block", "marginRight": "30px", "verticalAlign": "top"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("PD peptide file (optional)", style=LABEL_STYLE),
                                                    dcc.Upload(id="upload-pep", children=html.Button("Upload peptide file", style=BUTTON_SUBTLE_STYLE), multiple=False),
                                                    html.Div(id="upload-pep-msg"),
                                                ],
                                                style={"display": "inline-block", "marginRight": "30px", "verticalAlign": "top"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("DIA-NN report file", style=LABEL_STYLE),
                                                    dcc.Upload(id="upload-diann", children=html.Button("Upload DIA-NN report", style=BUTTON_SUBTLE_STYLE), multiple=False),
                                                    html.Div(id="upload-diann-msg"),
                                                ],
                                                style={"display": "inline-block", "verticalAlign": "top"},
                                            ),
                                        ]
                                    ),
                                    html.Div(style=SECTION_STYLE),
                                    html.Button("Import dataset", id="btn-import", n_clicks=0, style=BUTTON_STYLE),
                                    html.Button("Download import table (CSV)", id="btn-download-import-table", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    dcc.Loading(html.Pre(id="import-log", style=LOG_STYLE), type="dot", color=COLORS["brand"], style=LOADING_STYLE),
                                    html.H4("Summary preview", style=H4_STYLE),
                                    dcc.Loading(
                                        dash_table.DataTable(
                                            id="summary-table",
                                            page_size=12,
                                            style_table={"overflowX": "auto", "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
                                            style_header={"backgroundColor": "#1b2942", "fontWeight": "700", "borderBottom": f"1px solid {COLORS['border']}", "color": "#dbeafe"},
                                            style_cell={"textAlign": "left", "fontSize": "12px", "padding": "8px", "borderBottom": "1px solid #1f2d47", "backgroundColor": "#0f172a", "color": COLORS["text"]},
                                        ),
                                        type="dot",
                                        color=COLORS["brand"],
                                        style=LOADING_STYLE,
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="2) QC",
                                value="tab-qc",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=[
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Class columns for grouping (optional)", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="qc-classes",
                                        className="dark-dropdown",
                                        options=[],
                                        value=["cellline", "condition"],
                                        multi=True,
                                        placeholder="Select metadata columns",
                                        style={"width": "60%"},
                                    ),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Summary metric", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="qc-metric",
                                        className="dark-dropdown",
                                        options=[
                                            {"label": "protein_count", "value": "protein_count"},
                                            {"label": "peptide_count", "value": "peptide_count"},
                                            {"label": "protein_quant", "value": "protein_quant"},
                                            {"label": "peptide_quant", "value": "peptide_quant"},
                                        ],
                                        value="protein_count",
                                        clearable=False,
                                        style={"width": "320px", "marginBottom": "10px"},
                                    ),
                                    html.Label("Minimum protein count filter", style=LABEL_STYLE),
                                    dcc.Input(id="min-prot", type="number", value=0, min=0, style={**INPUT_STYLE, "width": "120px"}),
                                    html.Button("Apply filter", id="btn-filter", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Button("Refresh QC plots", id="btn-qc-refresh", n_clicks=0, style={**BUTTON_STYLE, "marginLeft": "10px"}),
                                    html.Button("Download QC table (CSV)", id="btn-download-qc-table", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    dcc.Loading(html.Pre(id="qc-log", style=LOG_STYLE), type="dot", color=COLORS["brand"], style=LOADING_STYLE),
                                    html.H4("QC Summary Table", style=H4_STYLE),
                                    dcc.Loading(
                                        dash_table.DataTable(
                                            id="qc-summary-table",
                                            page_size=12,
                                            style_table={"overflowX": "auto", "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
                                            style_header={"backgroundColor": "#1b2942", "fontWeight": "700", "borderBottom": f"1px solid {COLORS['border']}", "color": "#dbeafe"},
                                            style_cell={"textAlign": "left", "fontSize": "12px", "padding": "8px", "borderBottom": "1px solid #1f2d47", "backgroundColor": "#0f172a", "color": COLORS["text"]},
                                        ),
                                        type="dot",
                                        color=COLORS["brand"],
                                        style=LOADING_STYLE,
                                    ),
                                    html.Div(
                                        [
                                            html.Div([html.H4("Summary", style=H4_STYLE), dcc.Loading(html.Img(id="img-summary", style=PLOT_IMG_STYLE), type="circle", color=COLORS["brand"])], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
                                            html.Div([html.H4("CV", style=H4_STYLE), dcc.Loading(html.Img(id="img-cv", style=PLOT_IMG_STYLE), type="circle", color=COLORS["brand"])], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
                                        ]
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="3) Preprocess",
                                value="tab-preprocess",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=[
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Layer", style=LABEL_STYLE),
                                    dcc.Input(id="prep-layer", type="text", value="X", style={**INPUT_STYLE, "width": "120px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Normalize method", style=LABEL_STYLE),
                                    dcc.Dropdown(id="normalize-method", className="dark-dropdown", options=[{"label": m, "value": m} for m in ["sum", "median", "quantile"]], value="median", clearable=False, style={"width": "220px", "marginBottom": "10px"}),
                                    html.Label("Impute method", style=LABEL_STYLE),
                                    dcc.Dropdown(id="impute-method", className="dark-dropdown", options=[{"label": m, "value": m} for m in ["mean", "median", "min", "knn"]], value="min", clearable=False, style={"width": "220px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Button("Run preprocessing", id="btn-preprocess", n_clicks=0, style=BUTTON_STYLE),
                                    html.Button("Download preprocess table (CSV)", id="btn-download-prep-table", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    dcc.Loading(html.Pre(id="prep-log", style=LOG_STYLE), type="dot", color=COLORS["brand"], style=LOADING_STYLE),
                                ],
                            ),
                            dcc.Tab(
                                label="4) Embeddings + Abundance",
                                value="tab-embed",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=[
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Class columns for coloring (optional)", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="embed-classes",
                                        className="dark-dropdown",
                                        options=[],
                                        value=["cellline", "condition"],
                                        multi=True,
                                        placeholder="Select metadata columns",
                                        style={"width": "60%"},
                                    ),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Layer", style=LABEL_STYLE),
                                    dcc.Input(id="embed-layer", type="text", value="X", style={**INPUT_STYLE, "width": "120px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Abundance genes/proteins (comma-separated)", style=LABEL_STYLE),
                                    dcc.Input(id="abundance-genes", type="text", value="GAPDH,VCP", style={**INPUT_STYLE, "width": "60%"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Button("Compute embeddings + plots", id="btn-embed", n_clicks=0, style=BUTTON_STYLE),
                                    html.Button("Download embedding table (CSV)", id="btn-download-embed-table", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    dcc.Loading(html.Pre(id="embed-log", style=LOG_STYLE), type="dot", color=COLORS["brand"], style=LOADING_STYLE),
                                    html.Div(
                                        [
                                            html.Div([html.H4("PCA", style=H4_STYLE), dcc.Loading(html.Img(id="img-pca", style=PLOT_IMG_STYLE), type="circle", color=COLORS["brand"])], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),
                                            html.Div([html.H4("UMAP", style=H4_STYLE), dcc.Loading(html.Img(id="img-umap", style=PLOT_IMG_STYLE), type="circle", color=COLORS["brand"])], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),
                                            html.Div([html.H4("Abundance", style=H4_STYLE), dcc.Loading(html.Img(id="img-abundance", style=PLOT_IMG_STYLE), type="circle", color=COLORS["brand"])], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),
                                        ]
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="5) Differential Expression",
                                value="tab-de",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=[
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Build group filters with dropdowns", style=LABEL_STYLE),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Group 1 builder", style=LABEL_STYLE),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(id="de-g1-col-1", className="dark-dropdown", options=[], placeholder="Column", style={"width": "32%", "display": "inline-block"}),
                                                            dcc.Dropdown(id="de-g1-val-1", className="dark-dropdown", options=[], placeholder="Value", style={"width": "32%", "display": "inline-block", "marginLeft": "8px"}),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(id="de-g1-col-2", className="dark-dropdown", options=[], placeholder="Column", style={"width": "32%", "display": "inline-block", "marginTop": "8px"}),
                                                            dcc.Dropdown(id="de-g1-val-2", className="dark-dropdown", options=[], placeholder="Value", style={"width": "32%", "display": "inline-block", "marginLeft": "8px"}),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(id="de-g1-col-3", className="dark-dropdown", options=[], placeholder="Column", style={"width": "32%", "display": "inline-block", "marginTop": "8px"}),
                                                            dcc.Dropdown(id="de-g1-val-3", className="dark-dropdown", options=[], placeholder="Value", style={"width": "32%", "display": "inline-block", "marginLeft": "8px"}),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "49%", "display": "inline-block", "verticalAlign": "top"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Group 2 builder", style=LABEL_STYLE),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(id="de-g2-col-1", className="dark-dropdown", options=[], placeholder="Column", style={"width": "32%", "display": "inline-block"}),
                                                            dcc.Dropdown(id="de-g2-val-1", className="dark-dropdown", options=[], placeholder="Value", style={"width": "32%", "display": "inline-block", "marginLeft": "8px"}),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(id="de-g2-col-2", className="dark-dropdown", options=[], placeholder="Column", style={"width": "32%", "display": "inline-block", "marginTop": "8px"}),
                                                            dcc.Dropdown(id="de-g2-val-2", className="dark-dropdown", options=[], placeholder="Value", style={"width": "32%", "display": "inline-block", "marginLeft": "8px"}),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(id="de-g2-col-3", className="dark-dropdown", options=[], placeholder="Column", style={"width": "32%", "display": "inline-block", "marginTop": "8px"}),
                                                            dcc.Dropdown(id="de-g2-val-3", className="dark-dropdown", options=[], placeholder="Value", style={"width": "32%", "display": "inline-block", "marginLeft": "8px"}),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "49%", "display": "inline-block", "verticalAlign": "top"},
                                            ),
                                        ]
                                    ),
                                    html.Button("Apply builder to JSON filters", id="btn-build-de-json", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginTop": "10px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Group 1 filter (JSON object)", style=LABEL_STYLE),
                                    dcc.Textarea(id="de-group1", value='{"cellline":"BE","treatment":"kd"}', style={**INPUT_STYLE, "width": "48%", "height": "80px"}),
                                    html.Label("Group 2 filter (JSON object)", style=LABEL_STYLE),
                                    dcc.Textarea(id="de-group2", value='{"cellline":"BE","treatment":"sc"}', style={**INPUT_STYLE, "width": "48%", "height": "80px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Method", style=LABEL_STYLE),
                                    dcc.Dropdown(id="de-method", className="dark-dropdown", options=[{"label": m, "value": m} for m in ["ttest", "mannwhitneyu", "wilcoxon"]], value="ttest", clearable=False, style={"width": "220px", "marginBottom": "10px"}),
                                    html.Label("Layer", style=LABEL_STYLE),
                                    dcc.Input(id="de-layer", type="text", value="X", style={**INPUT_STYLE, "width": "120px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("p-value threshold", style=LABEL_STYLE),
                                    dcc.Input(id="de-pval", type="number", value=0.05, step=0.001, style={**INPUT_STYLE, "width": "140px"}),
                                    html.Label("log2FC threshold", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "10px", "marginBottom": "0"}),
                                    dcc.Input(id="de-log2fc", type="number", value=1.0, step=0.1, style={**INPUT_STYLE, "width": "140px", "marginLeft": "8px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.H4("Volcano style tools", style=H4_STYLE),
                                    html.Label("Up color", style={**LABEL_STYLE, "display": "inline-block", "marginBottom": "0"}),
                                    dcc.Input(id="de-color-up", type="text", value="#dc2626", style={**INPUT_STYLE, "width": "100px", "marginLeft": "8px"}),
                                    html.Button("Pick", id="btn-pick-de-color-up", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "6px", "padding": "6px 10px"}),
                                    html.Label("Down color", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-color-down", type="text", value="#2563eb", style={**INPUT_STYLE, "width": "100px", "marginLeft": "8px"}),
                                    html.Button("Pick", id="btn-pick-de-color-down", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "6px", "padding": "6px 10px"}),
                                    html.Label("Non-sig color", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-color-ns", type="text", value="#94a3b8", style={**INPUT_STYLE, "width": "100px", "marginLeft": "8px"}),
                                    html.Button("Pick", id="btn-pick-de-color-ns", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "6px", "padding": "6px 10px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Font", style={**LABEL_STYLE, "display": "inline-block", "marginBottom": "0"}),
                                    dcc.Dropdown(
                                        id="de-font-family",
                                        className="dark-dropdown",
                                        options=[
                                            {"label": "Arial", "value": "Arial"},
                                            {"label": "Helvetica", "value": "Helvetica"},
                                            {"label": "Times New Roman", "value": "Times New Roman"},
                                            {"label": "Courier New", "value": "Courier New"},
                                            {"label": "Verdana", "value": "Verdana"},
                                        ],
                                        value="Arial",
                                        clearable=False,
                                        style={"width": "220px", "display": "inline-block", "marginLeft": "8px", "verticalAlign": "middle"},
                                    ),
                                    html.Label("Font size", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-font-size", type="number", value=12, min=8, max=30, step=1, style={**INPUT_STYLE, "width": "100px", "marginLeft": "8px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Label field", style={**LABEL_STYLE, "display": "inline-block", "marginBottom": "0"}),
                                    dcc.Dropdown(
                                        id="de-label-column",
                                        className="dark-dropdown",
                                        options=[],
                                        value="index",
                                        clearable=False,
                                        style={"width": "220px", "display": "inline-block", "marginLeft": "8px", "verticalAlign": "middle"},
                                    ),
                                    html.Button("Apply volcano styling", id="btn-apply-volcano-style", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Button("Add labels from selection/click", id="btn-add-volcano-label", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Button("Add labels by list/cutoff", id="btn-add-volcano-label-rules", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Button("Clear labels", id="btn-clear-volcano-labels", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Label genes/accessions (comma or newline separated)", style=LABEL_STYLE),
                                    dcc.Textarea(
                                        id="de-label-list",
                                        value="",
                                        placeholder="e.g. GAPDH, VCP, P04406",
                                        style={**INPUT_STYLE, "width": "60%", "height": "72px"},
                                    ),
                                    html.Label("Auto-label p-value <= ", style={**LABEL_STYLE, "display": "inline-block", "marginTop": "10px", "marginBottom": "0"}),
                                    dcc.Input(id="de-label-pval-max", type="number", value=None, step=0.001, placeholder="e.g. 0.01", style={**INPUT_STYLE, "width": "130px", "marginLeft": "8px"}),
                                    html.Label("Auto-label |log2FC| >= ", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-label-log2fc-min", type="number", value=None, step=0.1, placeholder="e.g. 1.5", style={**INPUT_STYLE, "width": "130px", "marginLeft": "8px"}),
                                    dcc.Checklist(
                                        id="de-label-exact-match-toggle",
                                        options=[{"label": "Exact list matching", "value": "exact"}],
                                        value=[],
                                        inline=True,
                                        style={"display": "inline-block", "marginLeft": "12px"},
                                    ),
                                    html.Label("Max labels", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-label-max-count", type="number", value=40, min=1, step=1, style={**INPUT_STYLE, "width": "110px", "marginLeft": "8px"}),
                                    dcc.Checklist(
                                        id="de-highlight-labeled-toggle",
                                        options=[{"label": "Highlight labeled dots", "value": "on"}],
                                        value=["on"],
                                        inline=True,
                                        style={"display": "inline-block", "marginLeft": "12px"},
                                    ),
                                    html.Label("Highlight color", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "10px", "marginBottom": "0"}),
                                    dcc.Input(id="de-highlight-labeled-color", type="text", value="#16a34a", style={**INPUT_STYLE, "width": "100px", "marginLeft": "8px"}),
                                    html.Button("Pick", id="btn-pick-de-highlight-color", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "6px", "padding": "6px 10px"}),
                                    html.Div(id="de-label-warning", style={"marginTop": "8px", "color": "#f59e0b", "fontSize": "12px", "fontWeight": "600"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.H4("Label manager", style=H4_STYLE),
                                    html.Label("Select label", style={**LABEL_STYLE, "display": "inline-block", "marginBottom": "0"}),
                                    dcc.Dropdown(
                                        id="de-label-editor-id",
                                        className="dark-dropdown",
                                        options=[],
                                        value=None,
                                        clearable=True,
                                        placeholder="Choose a label to edit",
                                        style={"width": "260px", "display": "inline-block", "marginLeft": "8px", "verticalAlign": "middle"},
                                    ),
                                    html.Label("Text", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-label-editor-text", type="text", value="", style={**INPUT_STYLE, "width": "220px", "marginLeft": "8px"}),
                                    html.Label("X", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-label-editor-x", type="number", value=0, step=0.01, style={**INPUT_STYLE, "width": "120px", "marginLeft": "8px"}),
                                    html.Label("Y", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "12px", "marginBottom": "0"}),
                                    dcc.Input(id="de-label-editor-y", type="number", value=0, step=0.01, style={**INPUT_STYLE, "width": "120px", "marginLeft": "8px"}),
                                    html.Button("Update selected label", id="btn-update-volcano-label", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Button("Delete selected label", id="btn-delete-volcano-label", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Button("Snap labels to nearest points", id="btn-snap-volcano-labels", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Div(id="de-label-manager-summary", style={"marginTop": "8px", "color": COLORS["muted"], "fontSize": "12px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Button("Run DE + volcano", id="btn-de", n_clicks=0, style=BUTTON_STYLE),
                                    html.Button("Download DE table (CSV)", id="btn-download-de-table", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    dcc.Loading(html.Pre(id="de-log", style=LOG_STYLE), type="dot", color=COLORS["brand"], style=LOADING_STYLE),
                                    html.Button("Open SVG editor", id="btn-open-de-editor", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginBottom": "8px"}),
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="de-volcano",
                                            config={
                                                "displaylogo": False,
                                                "editable": True,
                                                "edits": {"annotationPosition": True, "annotationText": True},
                                            },
                                        ),
                                        type="circle",
                                        color=COLORS["brand"],
                                        style=LOADING_STYLE,
                                    ),
                                    html.Div(id="de-selection-info", style={"marginTop": "8px", "color": COLORS["muted"], "fontSize": "12px"}),
                                    dcc.Loading(
                                        dash_table.DataTable(
                                            id="de-table",
                                            page_size=12,
                                            filter_action="native",
                                            sort_action="native",
                                            style_table={"overflowX": "auto", "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "marginTop": "10px"},
                                            style_header={"backgroundColor": "#1b2942", "fontWeight": "700", "borderBottom": f"1px solid {COLORS['border']}", "color": "#dbeafe"},
                                            style_cell={"textAlign": "left", "fontSize": "12px", "padding": "8px", "borderBottom": "1px solid #1f2d47", "backgroundColor": "#0f172a", "color": COLORS["text"]},
                                        ),
                                        type="dot",
                                        color=COLORS["brand"],
                                        style=LOADING_STYLE,
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="6) STRING Enrichment",
                                value="tab-enrich",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=[
                                    html.Div(style=SECTION_STYLE),
                                    html.Button("Refresh DE keys", id="btn-refresh-keys", n_clicks=0, style=BUTTON_SUBTLE_STYLE),
                                    dcc.Dropdown(id="de-key-dropdown", className="dark-dropdown", placeholder="Select DE key", style={"width": "70%", "marginTop": "10px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Species NCBI taxon (optional, default inferred)", style=LABEL_STYLE),
                                    dcc.Input(id="species-id", type="number", value=9606, style={**INPUT_STYLE, "width": "140px"}),
                                    html.Label("Top N", style={**LABEL_STYLE, "display": "inline-block", "marginLeft": "10px", "marginBottom": "0"}),
                                    dcc.Input(id="enrich-topn", type="number", value=150, min=5, step=5, style={**INPUT_STYLE, "width": "120px", "marginLeft": "8px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Button("Run functional enrichment", id="btn-enrich", n_clicks=0, style=BUTTON_STYLE),
                                    dcc.Loading(html.Pre(id="enrich-log", style=LOG_STYLE), type="dot", color=COLORS["brand"], style=LOADING_STYLE),
                                    html.Button("Refresh functional keys", id="btn-refresh-functional", n_clicks=0, style=BUTTON_SUBTLE_STYLE),
                                    dcc.Dropdown(id="functional-key-dropdown", className="dark-dropdown", placeholder="Select functional key", style={"width": "70%", "marginTop": "10px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.Label("Category (optional)", style=LABEL_STYLE),
                                    dcc.Input(id="enrich-category", type="text", value="", style={**INPUT_STYLE, "width": "160px"}),
                                    html.Button("Load STRING SVG", id="btn-load-svg", n_clicks=0, style={**BUTTON_STYLE, "marginLeft": "10px"}),
                                    html.Button("Download enrichment table (CSV)", id="btn-download-enrich-table", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"}),
                                    html.Div(style=SECTION_STYLE),
                                    html.A("Open STRING in new tab", id="string-network-link", href="", target="_blank", style={"color": COLORS["brand"], "fontWeight": "600"}),
                                    html.Div(style=SECTION_STYLE),
                                    dcc.Loading(html.Iframe(id="string-network-iframe", src="", style=IFRAME_STYLE), type="circle", color=COLORS["brand"], style=LOADING_STYLE),
                                    html.Div(style=SECTION_STYLE),
                                    html.Button("Open SVG editor", id="btn-open-enrich-editor", n_clicks=0, style={**BUTTON_SUBTLE_STYLE, "marginBottom": "8px"}),
                                    dcc.Loading(html.Img(id="img-enrichment-svg", style=PLOT_IMG_STYLE), type="circle", color=COLORS["brand"], style=LOADING_STYLE),
                                ],
                            ),
                            dcc.Tab(
                                label="7) Plot Editor",
                                value="tab-editor",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=[
                                    html.Div(style=SECTION_STYLE),
                                    html.P(
                                        "Load DE or enrichment SVG, edit vector elements online, then save/export.",
                                        style=SUBTITLE_STYLE,
                                    ),
                                    html.Button("Load DE SVG", id="btn-editor-load-de", n_clicks=0, style=BUTTON_STYLE),
                                    html.Button(
                                        "Load Enrichment SVG",
                                        id="btn-editor-load-enrichment",
                                        n_clicks=0,
                                        style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"},
                                    ),
                                    html.Button(
                                        "Save Edits",
                                        id="btn-editor-save",
                                        n_clicks=0,
                                        style={**BUTTON_STYLE, "marginLeft": "10px"},
                                    ),
                                    html.Button(
                                        "Reset to Loaded",
                                        id="btn-editor-reset",
                                        n_clicks=0,
                                        style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"},
                                    ),
                                    html.Button(
                                        "Download Edited SVG",
                                        id="btn-download-edited-svg",
                                        n_clicks=0,
                                        style={**BUTTON_SUBTLE_STYLE, "marginLeft": "10px"},
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Div(
                                        [
                                            html.Span("Source: none", id="editor-source-badge", style=BADGE_STYLE),
                                            html.Span("Dirty: no", id="editor-dirty-badge", style=BADGE_STYLE),
                                            html.Span("Save: idle", id="editor-save-badge", style=BADGE_STYLE),
                                        ]
                                    ),
                                    dcc.Loading(
                                        html.Iframe(
                                            id="svg-editor-frame",
                                            srcDoc="",
                                            style=EDITOR_IFRAME_STYLE,
                                        ),
                                        type="circle",
                                        color=COLORS["brand"],
                                        style=LOADING_STYLE,
                                    ),
                                    html.Pre(id="editor-log", style=LOG_STYLE),
                                ],
                            ),
                        ],
                    ),
                ],
                style=APP_STYLE,
            ),
        ],
        style=WRAPPER_STYLE,
    )

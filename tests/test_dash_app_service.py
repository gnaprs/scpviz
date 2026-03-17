import base64
from pathlib import Path

import pandas as pd

from dash_app.services.scpviz_service import (
    import_data_for_session,
    parse_obs_columns,
    plot_summary_image,
    save_upload_contents,
    summary_records,
    volcano_plotly_figure,
)


def _encode_file_as_upload_payload(file_path: Path) -> str:
    content = file_path.read_bytes()
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:text/plain;base64,{encoded}"


def test_parse_obs_columns():
    assert parse_obs_columns("sample, cellline, treatment") == ["sample", "cellline", "treatment"]
    assert parse_obs_columns("") == []


def test_import_pd_short_files_and_summary_preview():
    root = Path(__file__).resolve().parents[1]
    prot_file = root / "dev" / "pd_prot_short.txt"
    pep_file = root / "dev" / "pd_pep_short.txt"
    session_id = "test-dash-pd-short"

    save_upload_contents(
        session_id=session_id,
        upload_key="prot_file",
        contents=_encode_file_as_upload_payload(prot_file),
        filename=prot_file.name,
    )
    save_upload_contents(
        session_id=session_id,
        upload_key="pep_file",
        contents=_encode_file_as_upload_payload(pep_file),
        filename=pep_file.name,
    )

    ok, message = import_data_for_session(
        session_id=session_id,
        source_type="pd",
        obs_columns_text="sample,cellline,treatment",
    )
    assert ok, message

    records, columns = summary_records(session_id=session_id)
    assert records
    assert columns


def test_plot_summary_image_after_import():
    root = Path(__file__).resolve().parents[1]
    prot_file = root / "dev" / "pd_prot_short.txt"
    pep_file = root / "dev" / "pd_pep_short.txt"
    session_id = "test-dash-summary-img"

    save_upload_contents(
        session_id=session_id,
        upload_key="prot_file",
        contents=_encode_file_as_upload_payload(prot_file),
        filename=prot_file.name,
    )
    save_upload_contents(
        session_id=session_id,
        upload_key="pep_file",
        contents=_encode_file_as_upload_payload(pep_file),
        filename=pep_file.name,
    )

    ok, _ = import_data_for_session(
        session_id=session_id,
        source_type="pd",
        obs_columns_text="sample,cellline,treatment",
    )
    assert ok

    src = plot_summary_image(session_id, value="protein_count", classes=["cellline"])
    assert src.startswith("data:image/png;base64,")


def test_plotly_volcano_builder_smoke():
    df = pd.DataFrame(
        {
            "log2fc": [1.5, -1.2, 0.1],
            "p_value": [0.001, 0.02, 0.8],
            "significance": ["upregulated", "downregulated", "not significant"],
            "Genes": ["A", "B", "C"],
        },
        index=["P1", "P2", "P3"],
    )
    fig = volcano_plotly_figure(df, pval=0.05, log2fc=1.0)
    assert fig is not None


import base64
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dash_app.callbacks.main_callbacks import _de_plotly_selection_rows
from dash_app.state.session import set_pdata

from dash_app.services.scpviz_service import (
    _isolate_stdio,
    de_keys,
    embedding_dataframe,
    import_data_for_session,
    load_edited_svg_for_session,
    parse_obs_columns,
    plot_summary_image,
    run_de,
    save_edited_svg_for_session,
    save_upload_contents,
    summary_dataframe,
    summary_records,
    svg_data_uri_to_markup,
    svg_markup_to_data_uri,
    volcano_svg_markup_from_records,
    volcano_plotly_figure,
)


def _encode_file_as_upload_payload(file_path: Path) -> str:
    content = file_path.read_bytes()
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:text/plain;base64,{encoded}"


def test_parse_obs_columns():
    assert parse_obs_columns("sample, cellline, treatment") == ["sample", "cellline", "treatment"]
    assert parse_obs_columns("") == []


def test_isolate_stdio_avoids_cp932_console_unicode_error():
    """Regression: library prints (e.g. U+26A0) must not hit a strict legacy stderr."""

    class StrictLegacyErr:
        """Simulates a cp932 console that cannot encode Unicode warning glyphs."""

        encoding = "cp932"

        def write(self, s: str) -> int:
            if not s:
                return 0
            s.encode(self.encoding)
            return len(s)

        def flush(self) -> None:
            pass

    old_err = sys.stderr
    sys.stderr = StrictLegacyErr()
    try:
        with pytest.raises(UnicodeEncodeError):
            print("\u26a0", file=sys.stderr, flush=True)

        with _isolate_stdio():
            print("\u26a0", file=sys.stderr, flush=True)
            print("\u26a0", flush=True)
    finally:
        sys.stderr = old_err


def test_de_plotly_selection_rows_unresolved_vs_subset():
    records = [{"index": "A"}, {"index": "B"}]
    rows, out = _de_plotly_selection_rows(None, records)
    assert out == "none" and rows is records
    rows, out = _de_plotly_selection_rows({"points": []}, records)
    assert out == "none" and rows is records
    rows, out = _de_plotly_selection_rows({"points": [{"customdata": [0]}]}, records)
    assert out == "subset" and rows == [records[0]]
    rows, out = _de_plotly_selection_rows({"points": [{"customdata": [999]}]}, records)
    assert out == "unresolved" and rows is records


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


def test_svg_data_uri_roundtrip():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text x="5" y="10">hello</text></svg>'
    data_uri = svg_markup_to_data_uri(svg)
    decoded = svg_data_uri_to_markup(data_uri)
    assert "<svg" in decoded
    assert "hello" in decoded


def test_save_and_load_edited_svg_session():
    session_id = "test-svg-edit-session"
    raw_svg = '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script><rect x="1" y="1" width="10" height="10" /></svg>'
    save_edited_svg_for_session(session_id, "enrichment", raw_svg)
    loaded = load_edited_svg_for_session(session_id, "enrichment")
    assert "<svg" in loaded
    assert "<script" not in loaded.lower()
    assert "rect" in loaded


def test_embedding_dataframe_no_collision_when_obs_has_sample_column():
    """Obs may include a 'sample' metadata column; index must not reset as duplicate 'sample'."""
    obs = pd.DataFrame(
        {"sample": ["meta_a", "meta_b"], "cellline": ["c1", "c2"]},
        index=["run1", "run2"],
    )

    class _Prot:
        pass

    class _PData:
        pass

    prot = _Prot()
    prot.obs = obs
    prot.obsm = {"X_pca": np.array([[0.0, 0.0], [1.0, 1.0]])}

    pdata = _PData()
    pdata.prot = prot

    session_id = "test-embed-sample-collision"
    set_pdata(session_id, pdata)

    df = embedding_dataframe(session_id)
    assert len(df) == 2
    assert "sample" in df.columns
    assert df["sample"].tolist() == ["meta_a", "meta_b"]
    # Index became a non-colliding column (sample_1 when 'sample' was taken)
    idx_cols = [c for c in df.columns if c not in ("sample", "cellline", "pca_1", "pca_2")]
    assert idx_cols
    assert set(df[idx_cols[0]].tolist()) == {"run1", "run2"}


def test_summary_dataframe_no_collision_when_summary_has_sample_column():
    summary_df = pd.DataFrame(
        {"sample": ["x", "y"], "protein_count": [10, 20]},
        index=["run1", "run2"],
    )

    class _PData:
        pass

    pdata = _PData()
    pdata.summary = summary_df
    pdata._summary_is_stale = False

    session_id = "test-summary-sample-collision"
    set_pdata(session_id, pdata)

    df = summary_dataframe(session_id)
    assert len(df) == 2
    assert "protein_count" in df.columns
    assert "sample" in df.columns
    assert df["sample"].tolist() == ["x", "y"]
    id_col = [c for c in df.columns if c not in ("sample", "protein_count")][0]
    assert set(df[id_col].tolist()) == {"run1", "run2"}


def test_volcano_svg_markup_from_records():
    records = [
        {"index": "P1", "log2fc": 1.2, "p_value": 0.001, "significance": "upregulated"},
        {"index": "P2", "log2fc": -1.4, "p_value": 0.02, "significance": "downregulated"},
        {"index": "P3", "log2fc": 0.2, "p_value": 0.4, "significance": "not significant"},
    ]
    svg = volcano_svg_markup_from_records(records, pval=0.05, log2fc=1.0)
    assert "<svg" in svg
    assert "</svg>" in svg


def test_run_de_calls_set_pdata_and_de_keys_non_empty():
    """DE mutates pdata.stats; set_pdata ensures Redis-backed sessions see keys on later callbacks."""
    root = Path(__file__).resolve().parents[1]
    prot_file = root / "dev" / "pd_prot_short.txt"
    pep_file = root / "dev" / "pd_pep_short.txt"
    session_id = "test-dash-de-persist"

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

    group1 = '{"cellline": "AS", "treatment": "kd"}'
    group2 = '{"cellline": "AS", "treatment": "sc"}'

    with patch("dash_app.services.scpviz_service.set_pdata") as mock_set_pdata:
        ok_de, msg_de, df = run_de(
            session_id=session_id,
            group1_json=group1,
            group2_json=group2,
            method="ttest",
            layer="X",
            pval=0.05,
            log2fc=1.0,
        )
        assert ok_de, msg_de
        assert df is not None
        mock_set_pdata.assert_called_once()
        assert mock_set_pdata.call_args[0][0] == session_id

    keys = de_keys(session_id)
    assert keys, "Enrichment DE dropdown should list at least one contrast after DE"
    assert any(" vs " in k for k in keys)


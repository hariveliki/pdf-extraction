"""End-to-end pipeline runs against a stand-in model server."""

from __future__ import annotations

import json

import pymupdf
import pytest

from pdfx.config import Config, RenderConfig, parse_page_range
from pdfx.native import PyMuPDFBackend
from pdfx.pipeline import Pipeline, write_outputs
from pdfx.render import render_page
from pdfx.validate import FindingKind, Severity

from .mock_ollama import MockOllama

CLEAN_OUTPUT = """# CTE AG - Measurement Report

Report no. 2026-0731

Ambient temperature: 21.4 C
Reference pressure: 1'013.25 hPa
Sample count: 1'248
Total invoiced: CHF 14'827.60
Deviation: -0.37 %
"""

# Same text, one figure quietly corrupted: 14'827.60 -> 14'287.60.
CORRUPTED_OUTPUT = CLEAN_OUTPUT.replace("14'827.60", "14'287.60")


def _config(port: int, tmp_path, **pipeline_kw) -> Config:
    cfg = Config()
    cfg.ollama.host = "127.0.0.1"
    cfg.ollama.port = port
    cfg.ollama.retry_backoff_s = 0.01
    cfg.pipeline.native_backend = "pymupdf"  # docling is optional
    cfg.output_dir = str(tmp_path / "out")
    for k, v in pipeline_kw.items():
        setattr(cfg.pipeline, k, v)
    return cfg


# ---- routing through the full pipeline ------------------------------------


def test_native_page_never_reaches_the_model(native_pdf, tmp_path):
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        result = Pipeline(_config(srv.port, tmp_path)).run(native_pdf)

    assert srv.requests == [], "a page with a real text layer must not cost inference"
    assert result.native_pages == 1
    assert result.vlm_pages == 0
    assert "CTE AG" in result.pages[0].markdown
    assert result.pages[0].extractor == "pymupdf"


def test_scanned_page_is_read_by_the_model(scanned_pdf, tmp_path):
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        result = Pipeline(_config(srv.port, tmp_path)).run(scanned_pdf)

    assert len(srv.requests) == 1
    assert result.vlm_pages == 1
    assert "Measurement Report" in result.pages[0].markdown
    assert result.pages[0].vlm is not None


def test_blank_page_is_skipped_without_inference(blank_pdf, tmp_path):
    with MockOllama() as srv:
        result = Pipeline(_config(srv.port, tmp_path)).run(blank_pdf)

    assert srv.requests == []
    assert result.skipped_pages == 1
    assert result.pages[0].markdown == ""


def test_mixed_document_routes_each_page_correctly(multi_page_pdf, tmp_path):
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        result = Pipeline(_config(srv.port, tmp_path)).run(multi_page_pdf)

    assert (result.native_pages, result.vlm_pages, result.skipped_pages) == (1, 1, 1)
    assert len(srv.requests) == 1, "only the scanned page should be sent"
    assert [p.page_number for p in result.pages] == [1, 2, 3]


# ---- the point of the whole exercise --------------------------------------


def test_a_quietly_corrupted_figure_is_caught(mixed_pdf, tmp_path):
    """The VLM returns fluent, plausible output with one transposed total.

    The page has a text layer as well as a full-page image, so there is a
    deterministic reading to compare against - and the mismatch must surface.
    """
    with MockOllama(response_text=CORRUPTED_OUTPUT) as srv:
        result = Pipeline(_config(srv.port, tmp_path)).run(mixed_pdf)

    page = result.pages[0]
    assert page.validation is not None and page.validation.checked
    assert page.validation.source == "text_layer"
    kinds = [f.kind for f in page.validation.findings]
    assert FindingKind.TRANSPOSED_DIGITS in kinds
    assert result.worst_severity is Severity.HIGH


def test_clean_vlm_output_on_the_same_page_raises_nothing(mixed_pdf, tmp_path):
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        result = Pipeline(_config(srv.port, tmp_path)).run(mixed_pdf)

    findings = [
        f for p in result.pages if p.validation
        for f in p.validation.findings
        if f.severity is Severity.HIGH
    ]
    assert findings == []


def test_scanned_page_without_a_text_layer_says_why_it_was_not_checked(
    scanned_pdf, tmp_path
):
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        result = Pipeline(_config(srv.port, tmp_path)).run(scanned_pdf)

    validation = result.pages[0].validation
    assert not validation.checked
    assert "tesseract_fallback" in validation.note


# ---- failure handling ------------------------------------------------------


def test_an_unreachable_server_fails_before_any_page_is_processed(scanned_pdf, tmp_path):
    from pdfx.vlm import VLMUnavailable

    cfg = _config(1, tmp_path)  # nothing listening
    with pytest.raises(VLMUnavailable, match="ollama serve"):
        Pipeline(cfg).run(scanned_pdf)


def test_a_failing_page_degrades_instead_of_aborting(mixed_pdf, tmp_path):
    """One bad page must not lose the rest of the document."""
    cfg = _config(0, tmp_path)
    with MockOllama(status=500) as srv:
        cfg.ollama.port = srv.port
        cfg.ollama.retries = 1
        result = Pipeline(cfg).run(mixed_pdf)

    page = result.pages[0]
    assert page.error
    # The page's own text layer is a usable degraded answer.
    assert "CTE AG" in page.markdown
    assert page.extractor == "native_text_fallback"
    assert len(result.failed_pages) == 1


# ---- outputs ---------------------------------------------------------------


def test_markdown_and_json_are_written_with_provenance(multi_page_pdf, tmp_path):
    cfg = _config(0, tmp_path)
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        cfg.ollama.port = srv.port
        result = Pipeline(cfg).run(multi_page_pdf)

    md_path, json_path = write_outputs(result, cfg.output_dir)
    assert md_path.exists() and json_path.exists()

    md = md_path.read_text()
    assert "<!-- page 1 | pymupdf -->" in md
    assert "glm-ocr" in md  # page 2's extractor is recorded inline

    data = json.loads(json_path.read_text())
    assert data["page_count"] == 3
    assert data["summary"]["native_pages"] == 1
    assert data["summary"]["vlm_pages"] == 1
    assert data["config"]["api_mode"] == "generate"
    # Every page explains how it was handled and why.
    for page in data["pages"]:
        assert page["reason"]
        assert page["decision"] in {"parse_native", "parse_vlm", "skip"}


def test_json_is_serialisable_and_flags_severity(mixed_pdf, tmp_path):
    cfg = _config(0, tmp_path)
    with MockOllama(response_text=CORRUPTED_OUTPUT) as srv:
        cfg.ollama.port = srv.port
        result = Pipeline(cfg).run(mixed_pdf)

    payload = json.loads(json.dumps(result.to_json()))
    assert payload["summary"]["worst_severity"] == "high"
    assert "numeric cross-check" in result.to_markdown()


def test_concurrent_workers_produce_the_same_pages(multi_page_pdf, tmp_path):
    """Threaded runs must not interleave pages or share a Document handle."""
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        serial = Pipeline(_config(srv.port, tmp_path, max_workers=1)).run(multi_page_pdf)
        parallel = Pipeline(_config(srv.port, tmp_path, max_workers=4)).run(multi_page_pdf)

    assert [p.page_number for p in serial.pages] == [p.page_number for p in parallel.pages]
    assert [p.markdown for p in serial.pages] == [p.markdown for p in parallel.pages]


# ---- page selection and rendering -----------------------------------------


def test_page_range_selection(multi_page_pdf, tmp_path):
    with MockOllama(response_text=CLEAN_OUTPUT) as srv:
        cfg = _config(srv.port, tmp_path, page_range="2")
        result = Pipeline(cfg).run(multi_page_pdf)

    assert len(result.pages) == 1
    assert result.pages[0].page_number == 2
    assert result.page_count == 3  # the document is still 3 pages


@pytest.mark.parametrize(
    "spec,count,expected",
    [
        (None, 5, [0, 1, 2, 3, 4]),
        ("1", 5, [0]),
        ("2-4", 5, [1, 2, 3]),
        ("1,3,5", 5, [0, 2, 4]),
        ("1-2,4", 5, [0, 1, 3]),
        ("3-99", 5, [2, 3, 4]),  # clamped to the document
    ],
)
def test_page_range_parsing(spec, count, expected):
    assert parse_page_range(spec, count) == expected


@pytest.mark.parametrize("spec", ["9", "4-2", "abc", "1-x"])
def test_bad_page_ranges_are_rejected(spec):
    with pytest.raises(ValueError):
        parse_page_range(spec, 3)


def test_render_respects_the_long_edge_cap(native_pdf):
    doc = pymupdf.open(native_pdf)
    try:
        page = doc.load_page(0)
        big = render_page(page, 0, RenderConfig(dpi=600, max_long_edge_px=800))
        assert max(big.width, big.height) <= 800
        assert big.effective_dpi < 600  # scaled down to fit
        assert big.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

        uncapped = render_page(page, 0, RenderConfig(dpi=72, max_long_edge_px=0))
        assert uncapped.effective_dpi == 72
    finally:
        doc.close()


def test_grayscale_rendering_is_smaller(native_pdf):
    doc = pymupdf.open(native_pdf)
    try:
        page = doc.load_page(0)
        colour = render_page(page, 0, RenderConfig(dpi=100, grayscale=False))
        grey = render_page(page, 0, RenderConfig(dpi=100, grayscale=True))
        assert grey.size_kb <= colour.size_kb
    finally:
        doc.close()


# ---- native backend --------------------------------------------------------


def test_pymupdf_backend_renders_tables_as_markdown(native_pdf):
    doc = pymupdf.open(native_pdf)
    try:
        md = PyMuPDFBackend(doc).page_markdown(0)
    finally:
        doc.close()

    assert "CTE AG" in md
    assert "|" in md, "the ruled table should come out as a Markdown table"
    assert "Channel" in md
    # Table cell contents must not also appear in the prose section.
    assert md.count("4'096") == 1


def test_unknown_backend_preference_is_rejected(native_pdf):
    from pdfx.native import build_native_backend

    doc = pymupdf.open(native_pdf)
    try:
        with pytest.raises(ValueError, match="unknown native_backend"):
            build_native_backend(doc, str(native_pdf), "tesseract")
    finally:
        doc.close()

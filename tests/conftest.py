"""Synthetic PDF fixtures covering each routing branch.

Built with PyMuPDF so the suite needs no checked-in binaries and no network.
"""

from __future__ import annotations

import pymupdf
import pytest

# Deliberately Swiss-formatted figures: the validator has to cope with
# apostrophe thousands separators.
INVOICE_LINES = [
    "CTE AG - Measurement Report",
    "Report no. 2026-0731",
    "",
    "Sensor calibration summary",
    "Ambient temperature: 21.4 C",
    "Reference pressure: 1'013.25 hPa",
    "Sample count: 1'248",
    "Total invoiced: CHF 14'827.60",
    "Deviation: -0.37 %",
]


def _write_text_page(doc: pymupdf.Document, lines: list[str]) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)  # A4 in points
    y = 72.0
    for line in lines:
        if line:
            page.insert_text((72, y), line, fontname="helv", fontsize=11)
        y += 20
    return page


@pytest.fixture
def native_pdf(tmp_path):
    """One page of real embedded text with a small table."""
    path = tmp_path / "native.pdf"
    doc = pymupdf.open()
    page = _write_text_page(doc, INVOICE_LINES)

    # A ruled table so the PyMuPDF table detector has something to find.
    rows = [
        ["Channel", "Reading", "Tolerance"],
        ["CH-1", "4'096", "0.5"],
        ["CH-2", "8'192", "0.5"],
    ]
    x0, y0, cw, rh = 72.0, 300.0, 140.0, 24.0
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            page.insert_text((rect.x0 + 4, rect.y0 + 16), cell,
                             fontname="helv", fontsize=10)

    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path, native_pdf):
    """The same content as a flattened image, with no text layer at all."""
    path = tmp_path / "scanned.pdf"
    src = pymupdf.open(native_pdf)
    pix = src.load_page(0).get_pixmap(dpi=150)
    src.close()

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def blank_pdf(tmp_path):
    """A page with neither text nor images - must not cost a model call."""
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def mixed_pdf(tmp_path, native_pdf):
    """Text layer plus a page-dominating image: the ambiguous case.

    This is the shape that makes the numeric cross-check possible on a page the
    VLM reads - a scan that already carries an OCR text layer, or a text page
    with a large figure. The text layer holds every figure from INVOICE_LINES so
    there is something authoritative to compare the model against.
    """
    path = tmp_path / "mixed.pdf"
    src = pymupdf.open(native_pdf)
    pix = src.load_page(0).get_pixmap(dpi=100)
    src.close()

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Full-page image with text on top: the shape of a searchable scan, where
    # every pixel is a picture and the text layer is someone else's OCR.
    page.insert_image(page.rect, pixmap=pix)
    y = 40.0
    for line in INVOICE_LINES:
        if line:
            page.insert_text((72, y), line, fontname="helv", fontsize=10)
        y += 20
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def multi_page_pdf(tmp_path, native_pdf):
    """native, scanned, blank - one document that exercises all three routes."""
    path = tmp_path / "multi.pdf"
    src = pymupdf.open(native_pdf)
    pix = src.load_page(0).get_pixmap(dpi=150)

    doc = pymupdf.open()
    doc.insert_pdf(src, from_page=0, to_page=0)          # p1 native
    p2 = doc.new_page(width=595, height=842)             # p2 scanned
    p2.insert_image(p2.rect, pixmap=pix)
    doc.new_page(width=595, height=842)                  # p3 blank
    doc.save(path)
    doc.close()
    src.close()
    return path

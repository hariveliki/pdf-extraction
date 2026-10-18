"""Native text-layer extraction backends.

Two implementations behind one interface:

* DoclingBackend - IBM's Docling (MIT). Real layout analysis and TableFormer
  table-structure recognition, so multi-column reading order and merged cells
  come out right. OCR is switched off: scanned pages are the VLM's job, and
  letting Docling silently OCR them would bypass the router.
* PyMuPDFBackend - no model downloads, no torch. Text in natural reading order
  plus PyMuPDF's own table detection rendered as Markdown. This is the fallback
  and the one the test suite exercises.
"""

from __future__ import annotations

import logging
from typing import Protocol

import pymupdf

log = logging.getLogger(__name__)

# PyMuPDF's table finder prints advisory notices straight to stdout, which would
# interleave with the CLI's own report. Route them through logging instead.
try:
    pymupdf.set_messages(pylogging_logger=logging.getLogger("pymupdf"))
except Exception:  # noqa: BLE001 - older PyMuPDF without set_messages
    pass


class NativeBackend(Protocol):
    name: str

    def page_markdown(self, page_index: int) -> str:
        """Return Markdown for a zero-based page index."""
        ...


def docling_available() -> bool:
    try:
        import docling.document_converter  # noqa: F401
    except Exception:
        return False
    return True


class PyMuPDFBackend:
    """Text and tables straight from the PDF's own content stream."""

    name = "pymupdf"

    def __init__(self, doc: pymupdf.Document, detect_tables: bool = True):
        self.doc = doc
        self.detect_tables = detect_tables

    def page_markdown(self, page_index: int) -> str:
        page = self.doc.load_page(page_index)
        tables_md: list[str] = []
        table_rects: list[pymupdf.Rect] = []

        if self.detect_tables:
            try:
                finder = page.find_tables()
                for table in finder.tables:
                    md = table.to_markdown().strip()
                    if md:
                        tables_md.append(md)
                        table_rects.append(pymupdf.Rect(table.bbox))
            except Exception as exc:  # noqa: BLE001
                log.debug("table detection failed on page %d: %s", page_index + 1, exc)

        # Emit the prose with table regions removed, then the tables, so numbers
        # are not duplicated between the two representations.
        body_parts: list[str] = []
        try:
            blocks = page.get_text("blocks")
        except Exception:
            blocks = []

        for b in blocks:
            if len(b) >= 7 and b[6] != 0:
                continue
            text = (b[4] or "").strip() if len(b) > 4 else ""
            if not text:
                continue
            rect = pymupdf.Rect(b[0], b[1], b[2], b[3])
            if any(_mostly_inside(rect, tr) for tr in table_rects):
                continue
            body_parts.append(text)

        chunks = ["\n\n".join(body_parts).strip()] if body_parts else []
        chunks.extend(tables_md)
        return "\n\n".join(c for c in chunks if c).strip()


def _mostly_inside(inner: pymupdf.Rect, outer: pymupdf.Rect, threshold: float = 0.6) -> bool:
    inter = inner & outer
    if not inter or inner.get_area() <= 0:
        return False
    return inter.get_area() / inner.get_area() >= threshold


class DoclingBackend:
    """Docling conversion, lazily run once for the whole document."""

    name = "docling"

    def __init__(self, pdf_path: str, document_timeout_s: float | None = 300.0):
        self.pdf_path = pdf_path
        self.document_timeout_s = document_timeout_s
        self._document = None

    def _convert(self):
        if self._document is not None:
            return self._document

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        # The router already decided which pages are images; Docling must not
        # quietly OCR them behind our back.
        opts.do_ocr = False
        opts.do_table_structure = True
        if self.document_timeout_s:
            opts.document_timeout = self.document_timeout_s

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        result = converter.convert(self.pdf_path)
        self._document = result.document
        return self._document

    def page_markdown(self, page_index: int) -> str:
        doc = self._convert()
        # Docling page numbers are 1-based.
        return (doc.export_to_markdown(page_no=page_index + 1) or "").strip()


def build_native_backend(
    doc: pymupdf.Document, pdf_path: str, preference: str = "auto"
) -> NativeBackend:
    """Pick a backend, honouring config and falling back with a warning."""
    if preference not in {"auto", "docling", "pymupdf"}:
        raise ValueError(
            f"unknown native_backend {preference!r}; expected auto, docling or pymupdf"
        )

    if preference == "pymupdf":
        return PyMuPDFBackend(doc)

    if docling_available():
        return DoclingBackend(pdf_path)

    if preference == "docling":
        raise RuntimeError(
            "native_backend is set to 'docling' but docling is not importable.\n"
            "  Install it with:  pip install 'docling>=2.0'\n"
            "  Or set pipeline.native_backend to 'pymupdf' in config.yaml."
        )

    log.info("docling not installed; using the pymupdf native backend")
    return PyMuPDFBackend(doc)

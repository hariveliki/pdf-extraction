"""Routing decisions - the branch that determines cost and accuracy."""

from __future__ import annotations

import pymupdf

from pdfx.config import RouterConfig
from pdfx.router import Decision, PageKind, garbage_ratio, route_document, route_page


def _route_first(path, cfg=None):
    doc = pymupdf.open(path)
    try:
        return route_page(doc.load_page(0), 0, cfg or RouterConfig())
    finally:
        doc.close()


def test_native_page_parses_directly(native_pdf):
    route = _route_first(native_pdf)
    assert route.decision is Decision.PARSE_NATIVE
    assert route.kind is PageKind.NATIVE_TEXT
    assert route.metrics.char_count > 100
    assert route.metrics.image_count == 0


def test_scanned_page_goes_to_the_vlm(scanned_pdf):
    route = _route_first(scanned_pdf)
    assert route.decision is Decision.PARSE_VLM
    assert route.kind is PageKind.SCANNED_IMAGE
    assert route.metrics.char_count < 60
    assert route.metrics.max_image_area_ratio > 0.9


def test_blank_page_costs_no_model_call(blank_pdf):
    route = _route_first(blank_pdf)
    assert route.decision is Decision.SKIP
    assert route.kind is PageKind.BLANK


def test_mixed_page_prefers_the_vlm_but_keeps_native_text(mixed_pdf):
    route = _route_first(mixed_pdf)
    assert route.decision is Decision.PARSE_VLM
    assert route.kind is PageKind.MIXED
    # The native text is what the numeric cross-check will compare against.
    assert "CTE AG" in route.native_text


def test_mixed_page_can_be_forced_native(mixed_pdf):
    cfg = RouterConfig(vlm_for_mixed_pages=False)
    route = _route_first(mixed_pdf, cfg)
    assert route.decision is Decision.PARSE_NATIVE
    assert route.kind is PageKind.MIXED


def test_document_routing_covers_every_branch(multi_page_pdf):
    doc = pymupdf.open(multi_page_pdf)
    try:
        routes = route_document(doc, RouterConfig())
    finally:
        doc.close()

    assert [r.decision for r in routes] == [
        Decision.PARSE_NATIVE,
        Decision.PARSE_VLM,
        Decision.SKIP,
    ]
    assert [r.page_number for r in routes] == [1, 2, 3]


def test_every_route_carries_a_human_readable_reason(multi_page_pdf):
    doc = pymupdf.open(multi_page_pdf)
    try:
        routes = route_document(doc, RouterConfig())
    finally:
        doc.close()
    for route in routes:
        assert len(route.reason) > 20, route.reason


# ---- garbage detection ----------------------------------------------------


def test_garbage_ratio_ignores_clean_text():
    assert garbage_ratio("Total invoiced: CHF 14'827.60") == 0.0


def test_garbage_ratio_flags_replacement_characters():
    assert garbage_ratio("Tot�l inv��ced") > 0.15


def test_garbage_ratio_counts_cid_runs_as_single_glyphs():
    # Four broken glyphs among four good letters, not forty.
    ratio = garbage_ratio("(cid:12)(cid:13)(cid:14)(cid:15)abcd")
    assert 0.4 < ratio < 0.6


def test_garbage_ratio_of_empty_text_is_zero():
    assert garbage_ratio("") == 0.0


class _BrokenFontPage:
    """A real page whose text extraction yields undecodable glyphs.

    A PDF with a font that has no ToUnicode map is awkward to synthesise -
    PyMuPDF substitutes placeholders for characters the base-14 fonts cannot
    encode, so writing U+FFFD into a page does not produce a broken page. This
    proxy fakes only the text extraction and leaves the geometry and image
    analysis to the real page, which is the part the router reasons over.
    """

    def __init__(self, page: pymupdf.Page, broken_text: str):
        self._page = page
        self._text = broken_text

    @property
    def rect(self):
        return self._page.rect

    def get_text(self, kind="text", *args, **kwargs):
        if kind == "blocks":
            return self._page.get_text("blocks")
        return self._text

    def get_image_info(self, *args, **kwargs):
        return self._page.get_image_info()


def test_corrupt_text_layer_is_rerouted_to_the_vlm(native_pdf):
    """A text layer full of undecodable glyphs must not be trusted.

    Rendering the page and re-reading it is the only way to recover the real
    characters, so this has to route to the VLM even though text is 'present'.
    """
    doc = pymupdf.open(native_pdf)
    try:
        broken = _BrokenFontPage(doc.load_page(0), "(cid:12)" * 80 + "abcdefgh")
        route = route_page(broken, 0, RouterConfig())
    finally:
        doc.close()

    assert route.decision is Decision.PARSE_VLM
    assert route.kind is PageKind.CORRUPT_TEXT_LAYER
    assert "undecodable" in route.reason


def test_a_clean_text_layer_on_the_same_geometry_stays_native(native_pdf):
    """Control for the test above: only the garbage ratio changes the decision."""
    doc = pymupdf.open(native_pdf)
    try:
        proxy = _BrokenFontPage(doc.load_page(0), "Total invoiced: CHF 14'827.60 " * 8)
        route = route_page(proxy, 0, RouterConfig())
    finally:
        doc.close()

    assert route.decision is Decision.PARSE_NATIVE
    assert route.kind is PageKind.NATIVE_TEXT

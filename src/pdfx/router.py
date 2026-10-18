"""Per-page routing: does this page already carry a usable text layer?

This is the cheap decision that saves most of the compute. A page with real
embedded text is parsed directly and exactly; only pages that are genuinely
images get sent to the VLM. The router also catches the nastier middle case -
a text layer that *exists* but is garbage (broken font encodings, bad prior
OCR) - because silently parsing that produces confident nonsense.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pymupdf

from .config import RouterConfig

CID_RE = re.compile(r"\(cid:\d+\)")


class PageKind(str, Enum):
    NATIVE_TEXT = "native_text"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    BLANK = "blank"
    CORRUPT_TEXT_LAYER = "corrupt_text_layer"


class Decision(str, Enum):
    PARSE_NATIVE = "parse_native"
    PARSE_VLM = "parse_vlm"
    SKIP = "skip"


@dataclass
class PageMetrics:
    char_count: int = 0
    word_count: int = 0
    text_area_ratio: float = 0.0
    image_count: int = 0
    max_image_area_ratio: float = 0.0
    garbage_ratio: float = 0.0
    page_width: float = 0.0
    page_height: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "char_count": self.char_count,
            "word_count": self.word_count,
            "text_area_ratio": round(self.text_area_ratio, 5),
            "image_count": self.image_count,
            "max_image_area_ratio": round(self.max_image_area_ratio, 5),
            "garbage_ratio": round(self.garbage_ratio, 5),
            "page_size": [round(self.page_width, 1), round(self.page_height, 1)],
        }


@dataclass
class PageRoute:
    page_index: int
    kind: PageKind
    decision: Decision
    reason: str
    metrics: PageMetrics = field(default_factory=PageMetrics)
    # Native text is kept even when the page is routed to the VLM: it is the
    # deterministic side of the numeric cross-check.
    native_text: str = ""

    @property
    def page_number(self) -> int:
        return self.page_index + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_number,
            "kind": self.kind.value,
            "decision": self.decision.value,
            "reason": self.reason,
            "metrics": self.metrics.to_dict(),
        }


def garbage_ratio(text: str) -> float:
    """Fraction of characters that signal a broken text layer.

    Counts Unicode replacement chars, unassigned/private-use/control code
    points and `(cid:N)` runs, which is what PyMuPDF emits when a font has no
    usable ToUnicode map. Whitespace is excluded from the denominator so that
    sparse pages are not judged unfairly.
    """
    if not text:
        return 0.0

    # Each (cid:N) run is one broken glyph, not N broken characters.
    cid_hits = len(CID_RE.findall(text))
    stripped = CID_RE.sub("", text)

    considered = 0
    bad = 0
    for ch in stripped:
        if ch.isspace():
            continue
        considered += 1
        if ch == "�":
            bad += 1
            continue
        cat = unicodedata.category(ch)
        # Cc control, Cf format, Cn unassigned, Co private use, Cs surrogate.
        if cat in {"Cc", "Cn", "Co", "Cs"}:
            bad += 1

    considered += cid_hits
    bad += cid_hits
    if considered == 0:
        return 0.0
    return bad / considered


def _text_area_ratio(page: pymupdf.Page, page_area: float) -> float:
    """Union-free approximation of the page area covered by text blocks."""
    if page_area <= 0:
        return 0.0
    total = 0.0
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return 0.0
    for b in blocks:
        # (x0, y0, x1, y1, text, block_no, block_type); type 0 == text
        if len(b) >= 7 and b[6] != 0:
            continue
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        total += w * h
    return min(total / page_area, 1.0)


def _image_stats(page: pymupdf.Page, page_area: float) -> tuple[int, float]:
    """Count images on the page and the largest one's area share."""
    count = 0
    largest = 0.0
    try:
        infos = page.get_image_info()
    except Exception:
        infos = []

    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        count += 1
        if page_area <= 0:
            continue
        x0, y0, x1, y1 = bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        largest = max(largest, min(area / page_area, 1.0))

    return count, largest


def measure_page(page: pymupdf.Page) -> tuple[PageMetrics, str]:
    """Collect the metrics the routing decision is made from."""
    rect = page.rect
    page_area = float(rect.width) * float(rect.height)

    try:
        text = page.get_text("text") or ""
    except Exception:
        text = ""

    image_count, max_image_ratio = _image_stats(page, page_area)

    metrics = PageMetrics(
        char_count=len(text.strip()),
        word_count=len(text.split()),
        text_area_ratio=_text_area_ratio(page, page_area),
        image_count=image_count,
        max_image_area_ratio=max_image_ratio,
        garbage_ratio=garbage_ratio(text),
        page_width=float(rect.width),
        page_height=float(rect.height),
    )
    return metrics, text


def route_page(page: pymupdf.Page, page_index: int, cfg: RouterConfig) -> PageRoute:
    """Decide how a single page should be extracted."""
    metrics, text = measure_page(page)

    def make(kind: PageKind, decision: Decision, reason: str) -> PageRoute:
        return PageRoute(
            page_index=page_index,
            kind=kind,
            decision=decision,
            reason=reason,
            metrics=metrics,
            native_text=text,
        )

    # 1. Effectively no text.
    if metrics.char_count < cfg.min_chars:
        if metrics.max_image_area_ratio >= cfg.full_page_image_ratio:
            return make(
                PageKind.SCANNED_IMAGE,
                Decision.PARSE_VLM,
                f"no usable text layer ({metrics.char_count} chars) and an image "
                f"covering {metrics.max_image_area_ratio:.0%} of the page",
            )
        if metrics.image_count > 0:
            return make(
                PageKind.SCANNED_IMAGE,
                Decision.PARSE_VLM,
                f"no usable text layer ({metrics.char_count} chars) with "
                f"{metrics.image_count} image(s) present",
            )
        return make(
            PageKind.BLANK,
            Decision.SKIP,
            "no text and no images - treated as blank, no VLM call made",
        )

    # 2. Text exists but is undecodable. Rendering and re-reading it is the
    #    only way to recover the actual characters.
    if metrics.garbage_ratio > cfg.max_garbage_ratio:
        return make(
            PageKind.CORRUPT_TEXT_LAYER,
            Decision.PARSE_VLM,
            f"text layer present but {metrics.garbage_ratio:.0%} of glyphs are "
            f"undecodable (broken font encoding or bad prior OCR)",
        )

    # 3. Real text plus a page-dominating image: a figure-heavy page, or a scan
    #    that already carries someone else's OCR layer. Ambiguous either way.
    if metrics.max_image_area_ratio >= cfg.full_page_image_ratio:
        if cfg.vlm_for_mixed_pages:
            return make(
                PageKind.MIXED,
                Decision.PARSE_VLM,
                f"{metrics.char_count} chars of text alongside an image covering "
                f"{metrics.max_image_area_ratio:.0%} of the page; VLM reads the "
                f"page, native text retained for cross-checking",
            )
        return make(
            PageKind.MIXED,
            Decision.PARSE_NATIVE,
            f"text layer preferred despite a large image "
            f"({metrics.max_image_area_ratio:.0%} of page)",
        )

    # 4. Thin text over a partial image - not enough native content to trust.
    if metrics.text_area_ratio < cfg.min_text_area_ratio and metrics.image_count > 0:
        return make(
            PageKind.MIXED,
            Decision.PARSE_VLM,
            f"text covers only {metrics.text_area_ratio:.2%} of the page with "
            f"images present; most content is likely pixels",
        )

    return make(
        PageKind.NATIVE_TEXT,
        Decision.PARSE_NATIVE,
        f"{metrics.char_count} chars of clean embedded text covering "
        f"{metrics.text_area_ratio:.1%} of the page",
    )


def route_document(
    doc: pymupdf.Document, cfg: RouterConfig, page_indices: list[int] | None = None
) -> list[PageRoute]:
    """Route every selected page of an open document."""
    indices = page_indices if page_indices is not None else range(doc.page_count)
    return [route_page(doc.load_page(i), i, cfg) for i in indices]

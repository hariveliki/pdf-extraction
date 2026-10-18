"""Page rasterisation for the VLM path.

Kept separate from the VLM client so the pixel decisions (DPI, size cap,
colour) are visible and testable on their own.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import pymupdf

from .config import RenderConfig


@dataclass
class RenderedPage:
    page_index: int
    png_bytes: bytes
    width: int
    height: int
    effective_dpi: float

    @property
    def base64_png(self) -> str:
        return base64.b64encode(self.png_bytes).decode("ascii")

    @property
    def size_kb(self) -> float:
        return len(self.png_bytes) / 1024.0


def render_page(page: pymupdf.Page, page_index: int, cfg: RenderConfig) -> RenderedPage:
    """Rasterise one page to PNG bytes, honouring the long-edge pixel cap.

    Requesting a high DPI on an A0 drawing would otherwise produce an image
    large enough to exhaust memory on the model server, so the requested DPI is
    scaled down until the long edge fits under the cap.
    """
    rect = page.rect
    long_edge_pt = max(float(rect.width), float(rect.height))
    if long_edge_pt <= 0:
        raise ValueError(f"page {page_index + 1} has a degenerate size: {rect}")

    dpi = float(cfg.dpi)
    if cfg.max_long_edge_px > 0:
        # points -> pixels is dpi/72; solve for the dpi that just fits the cap.
        max_dpi = cfg.max_long_edge_px * 72.0 / long_edge_pt
        dpi = min(dpi, max_dpi)

    colorspace = pymupdf.csGRAY if cfg.grayscale else pymupdf.csRGB
    pix = page.get_pixmap(dpi=round(dpi), colorspace=colorspace, alpha=False)

    return RenderedPage(
        page_index=page_index,
        png_bytes=pix.tobytes("png"),
        width=pix.width,
        height=pix.height,
        effective_dpi=dpi,
    )


def png_to_pil(png_bytes: bytes):
    """Decode PNG bytes to a PIL image (used by the Tesseract cross-check)."""
    from PIL import Image

    return Image.open(io.BytesIO(png_bytes))

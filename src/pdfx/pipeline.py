"""Pipeline orchestration: route, extract, cross-check, assemble.

    for each page:
        route  -> native text layer? scan? blank?
        native -> Docling/PyMuPDF, exact, no model call
        scan   -> render -> GLM-OCR -> cross-check numbers deterministically
    assemble -> Markdown with page markers + JSON with per-page provenance
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from . import __version__
from .config import Config, parse_page_range
from .native import build_native_backend
from .render import render_page
from .router import Decision, PageRoute, route_document
from .validate import (
    PageValidation,
    Severity,
    compare_numbers,
    skipped,
    tesseract_text,
    worst_severity,
)
from .vlm import TASK_TABLE, TASK_TEXT, GlmOcrClient, VLMResult

log = logging.getLogger(__name__)


@dataclass
class PageResult:
    route: PageRoute
    markdown: str = ""
    extractor: str = ""
    validation: PageValidation | None = None
    vlm: VLMResult | None = None
    duration_s: float = 0.0
    error: str = ""

    @property
    def page_number(self) -> int:
        return self.route.page_number

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            **self.route.to_dict(),
            "extractor": self.extractor,
            "chars": len(self.markdown),
            "duration_s": round(self.duration_s, 2),
        }
        if self.vlm is not None:
            out["vlm"] = self.vlm.to_dict()
        if self.validation is not None:
            out["validation"] = self.validation.to_dict()
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class DocumentResult:
    pdf_path: str
    page_count: int
    pages: list[PageResult] = field(default_factory=list)
    duration_s: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def native_pages(self) -> int:
        return sum(1 for p in self.pages if p.route.decision is Decision.PARSE_NATIVE)

    @property
    def vlm_pages(self) -> int:
        return sum(1 for p in self.pages if p.route.decision is Decision.PARSE_VLM)

    @property
    def skipped_pages(self) -> int:
        return sum(1 for p in self.pages if p.route.decision is Decision.SKIP)

    @property
    def failed_pages(self) -> list[PageResult]:
        return [p for p in self.pages if p.error]

    @property
    def worst_severity(self) -> Severity:
        return worst_severity(
            [p.validation for p in self.pages if p.validation is not None]
        )

    def to_markdown(self) -> str:
        parts: list[str] = []
        for page in self.pages:
            if not page.markdown.strip():
                continue
            flag = ""
            if page.validation and page.validation.max_severity in {
                Severity.HIGH,
                Severity.MEDIUM,
            }:
                flag = (
                    f" <!-- numeric cross-check: "
                    f"{page.validation.max_severity.value} severity, "
                    f"{len(page.validation.findings)} finding(s) -->"
                )
            parts.append(
                f"<!-- page {page.page_number} | {page.extractor} -->{flag}\n\n"
                f"{page.markdown.strip()}"
            )
        return "\n\n---\n\n".join(parts) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "pdfx_version": __version__,
            "source": self.pdf_path,
            "page_count": self.page_count,
            "pages_processed": len(self.pages),
            "summary": {
                "native_pages": self.native_pages,
                "vlm_pages": self.vlm_pages,
                "skipped_pages": self.skipped_pages,
                "failed_pages": len(self.failed_pages),
                "worst_severity": self.worst_severity.value,
                "duration_s": round(self.duration_s, 2),
            },
            "config": self.config,
            "pages": [p.to_dict() for p in self.pages],
        }


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = GlmOcrClient(cfg.ollama)

    # ---- per-page work ----------------------------------------------------

    def _extract_native(self, route: PageRoute, backend) -> PageResult:
        started = time.monotonic()
        result = PageResult(route=route, extractor=backend.name)
        try:
            result.markdown = backend.page_markdown(route.page_index)
        except Exception as exc:  # noqa: BLE001
            result.error = f"native extraction failed: {exc}"
            log.warning("page %d: %s", route.page_number, result.error)
            # The page's own text is a usable degraded answer.
            result.markdown = route.native_text.strip()
            result.extractor = f"{backend.name}+fallback"

        result.validation = skipped(
            route.page_index,
            "native text layer parsed directly - no model in the loop, nothing to cross-check",
        )
        result.duration_s = time.monotonic() - started
        return result

    def _extract_vlm(self, route: PageRoute, doc: pymupdf.Document) -> PageResult:
        started = time.monotonic()
        result = PageResult(route=route, extractor=f"glm-ocr({self.cfg.ollama.model})")

        page = doc.load_page(route.page_index)
        rendered = render_page(page, route.page_index, self.cfg.render)

        task = TASK_TEXT
        if self.cfg.pipeline.table_prompt_for_table_pages and _looks_tabular(page):
            task = TASK_TABLE
        elif self.cfg.pipeline.default_task in {TASK_TEXT, TASK_TABLE}:
            task = self.cfg.pipeline.default_task

        try:
            vlm_result = self.client.recognise(
                rendered.base64_png, page_index=route.page_index, task=task
            )
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            result.markdown = route.native_text.strip()
            result.extractor = "native_text_fallback" if result.markdown else "failed"
            result.validation = skipped(
                route.page_index, f"not cross-checked: VLM call failed ({exc})"
            )
            result.duration_s = time.monotonic() - started
            log.warning("page %d: %s", route.page_number, exc)
            return result

        result.vlm = vlm_result
        result.markdown = vlm_result.markdown
        result.validation = self._cross_check(route, vlm_result.markdown, rendered.png_bytes)
        result.duration_s = time.monotonic() - started
        return result

    def _cross_check(
        self, route: PageRoute, vlm_markdown: str, png_bytes: bytes
    ) -> PageValidation:
        vcfg = self.cfg.validation
        if not vcfg.enabled:
            return skipped(route.page_index, "numeric cross-check disabled in config")

        native = route.native_text.strip()
        if native:
            return compare_numbers(
                native, vlm_markdown, vcfg, route.page_index, source="text_layer"
            )

        if vcfg.tesseract_fallback:
            ocr = tesseract_text(png_bytes)
            if ocr and ocr.strip():
                return compare_numbers(
                    ocr, vlm_markdown, vcfg, route.page_index, source="tesseract"
                )
            return skipped(
                route.page_index,
                "no text layer and Tesseract produced nothing (or is not installed)",
            )

        return skipped(
            route.page_index,
            "no text layer to compare against; enable validation.tesseract_fallback "
            "for a second deterministic opinion on scanned pages",
        )

    # ---- document ---------------------------------------------------------

    def run(self, pdf_path: str | Path) -> DocumentResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"no such PDF: {pdf_path}")

        started = time.monotonic()
        doc = pymupdf.open(pdf_path)
        try:
            indices = parse_page_range(self.cfg.pipeline.page_range, doc.page_count)
            routes = route_document(doc, self.cfg.router, indices)

            needs_vlm = any(r.decision is Decision.PARSE_VLM for r in routes)
            if needs_vlm:
                # Fail before doing any work if the model server is not ready.
                self.client.health_check()

            backend = build_native_backend(
                doc, str(pdf_path), self.cfg.pipeline.native_backend
            )

            results: list[PageResult] = []

            native_routes = [r for r in routes if r.decision is Decision.PARSE_NATIVE]
            vlm_routes = [r for r in routes if r.decision is Decision.PARSE_VLM]
            skip_routes = [r for r in routes if r.decision is Decision.SKIP]

            for route in native_routes:
                results.append(self._extract_native(route, backend))

            # Ollama serialises per model, so concurrency above 1-2 mostly buys
            # memory pressure. It is configurable because vLLM would benefit.
            workers = max(1, self.cfg.pipeline.max_workers)
            if vlm_routes:
                if workers == 1:
                    for route in vlm_routes:
                        results.append(self._extract_vlm(route, doc))
                else:
                    # Each worker opens its own handle: pymupdf Documents are
                    # not safe to share across threads.
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [
                            pool.submit(self._extract_vlm_isolated, route, pdf_path)
                            for route in vlm_routes
                        ]
                        results.extend(f.result() for f in futures)

            for route in skip_routes:
                results.append(
                    PageResult(
                        route=route,
                        extractor="skipped",
                        validation=skipped(route.page_index, route.reason),
                    )
                )

            results.sort(key=lambda r: r.route.page_index)

            return DocumentResult(
                pdf_path=str(pdf_path),
                page_count=doc.page_count,
                pages=results,
                duration_s=time.monotonic() - started,
                config=self._config_provenance(backend.name),
            )
        finally:
            doc.close()

    def _extract_vlm_isolated(self, route: PageRoute, pdf_path: Path) -> PageResult:
        doc = pymupdf.open(pdf_path)
        try:
            return self._extract_vlm(route, doc)
        finally:
            doc.close()

    def _config_provenance(self, backend_name: str) -> dict[str, Any]:
        return {
            "model": self.cfg.ollama.model,
            "api_mode": self.cfg.ollama.api_mode,
            "endpoint": self.cfg.ollama.base_url,
            "native_backend": backend_name,
            "render_dpi": self.cfg.render.dpi,
            "max_long_edge_px": self.cfg.render.max_long_edge_px,
            "validation_enabled": self.cfg.validation.enabled,
            "number_locale": self.cfg.validation.number_locale,
            "hardware_note": self.cfg.hardware_note,
        }


def _looks_tabular(page: pymupdf.Page) -> bool:
    try:
        return len(page.find_tables().tables) > 0
    except Exception:
        return False


def write_outputs(result: DocumentResult, out_dir: str | Path) -> tuple[Path, Path]:
    """Write <stem>.md and <stem>.json, returning both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.pdf_path).stem

    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"

    md_path.write_text(result.to_markdown(), encoding="utf-8")
    json_path.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return md_path, json_path

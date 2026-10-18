"""Configuration: dataclasses, YAML loading, and hardware-derived defaults.

setup.sh writes a config.yaml with values chosen from detected hardware; this
module is the single place that knows what those knobs mean.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAMES = ("config.yaml", "config.yml")


@dataclass
class OllamaConfig:
    host: str = "127.0.0.1"
    port: int = 11434
    model: str = "glm-ocr:latest"
    # Upstream GLM-OCR guidance: use Ollama's native /api/generate for vision.
    # The OpenAI-compatible /v1/chat/completions path is known to 502 on image
    # requests, so "generate" is the default and "openai" is opt-in.
    api_mode: str = "generate"
    timeout_s: float = 180.0
    retries: int = 3
    retry_backoff_s: float = 2.0
    temperature: float = 0.0
    # Ollama keeps the model resident for this long between pages.
    keep_alive: str = "10m"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class RenderConfig:
    dpi: int = 200
    # Hard cap on the long edge in pixels. GLM-OCR downsamples tokens
    # internally; beyond ~1600px you pay memory for little accuracy.
    max_long_edge_px: int = 1600
    grayscale: bool = False


@dataclass
class RouterConfig:
    """Thresholds for deciding whether a page already has a usable text layer."""

    # Minimum extractable characters for a page to be a native-parse candidate.
    min_chars: int = 60
    # Minimum fraction of the page area covered by text blocks.
    min_text_area_ratio: float = 0.005
    # A single image covering at least this fraction of the page looks like a scan.
    full_page_image_ratio: float = 0.55
    # Above this ratio of undecodable/replacement glyphs, the text layer is junk
    # (broken embedded fonts, bad prior OCR) and we prefer the VLM.
    max_garbage_ratio: float = 0.12
    # Pages with both a real text layer and a large image are ambiguous. When
    # true they go to the VLM but keep their native text for cross-checking.
    vlm_for_mixed_pages: bool = True


@dataclass
class ValidationConfig:
    enabled: bool = True
    # Locales differ on separators; CH uses 1'234.56, DE uses 1.234,56.
    # "auto" infers per token. Set to "ch", "de" or "us" to force.
    number_locale: str = "auto"
    # Ignore numeric differences below this absolute tolerance.
    abs_tolerance: float = 0.0
    # Fail the run (non-zero exit) when a page has high-severity findings.
    fail_on_severity: str = "none"  # none | high | medium
    # Use Tesseract as a second deterministic opinion on pages with no text
    # layer at all. Requires the tesseract binary + pytesseract.
    tesseract_fallback: bool = False


@dataclass
class PipelineConfig:
    # auto -> docling if importable, else pymupdf
    native_backend: str = "auto"  # auto | docling | pymupdf
    # Ollama serialises requests per model; >2 rarely helps and costs memory.
    max_workers: int = 1
    # GLM-OCR task prompt for standard pages.
    default_task: str = "Text Recognition"
    # Re-run pages that look table-heavy with the table-specific prompt.
    table_prompt_for_table_pages: bool = False
    page_range: str | None = None  # e.g. "1-5,9"


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output_dir: str = "out"
    hardware_note: str = ""

    # ---- loading ----------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        """Load config from an explicit path, or the first default name found.

        Missing file is not an error: the dataclass defaults are a working
        configuration for a 16GB Apple Silicon machine.
        """
        data: dict[str, Any] = {}
        resolved: Path | None = None

        if path:
            resolved = Path(path)
            if not resolved.exists():
                raise FileNotFoundError(f"config not found: {resolved}")
        else:
            for name in DEFAULT_CONFIG_NAMES:
                candidate = Path.cwd() / name
                if candidate.exists():
                    resolved = candidate
                    break

        if resolved is not None:
            import yaml

            with resolved.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        def build(dc_type, raw):
            if not raw:
                return dc_type()
            known = {f.name for f in dataclasses.fields(dc_type)}
            unknown = set(raw) - known
            if unknown:
                raise ValueError(
                    f"unknown {dc_type.__name__} keys: {sorted(unknown)}; "
                    f"valid keys are {sorted(known)}"
                )
            return dc_type(**raw)

        top_known = {"ollama", "render", "router", "validation", "pipeline",
                     "output_dir", "hardware_note"}
        unknown_top = set(data) - top_known
        if unknown_top:
            raise ValueError(f"unknown top-level config keys: {sorted(unknown_top)}")

        return cls(
            ollama=build(OllamaConfig, data.get("ollama")),
            render=build(RenderConfig, data.get("render")),
            router=build(RouterConfig, data.get("router")),
            validation=build(ValidationConfig, data.get("validation")),
            pipeline=build(PipelineConfig, data.get("pipeline")),
            output_dir=data.get("output_dir", "out"),
            hardware_note=data.get("hardware_note", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def parse_page_range(spec: str | None, page_count: int) -> list[int]:
    """Parse "1-3,7" into zero-based page indices. None means every page."""
    if not spec:
        return list(range(page_count))

    pages: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, _, hi_s = chunk.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"bad page range chunk: {chunk!r}") from exc
            if lo > hi:
                raise ValueError(f"descending page range: {chunk!r}")
            for p in range(lo, hi + 1):
                pages.add(p - 1)
        else:
            try:
                pages.add(int(chunk) - 1)
            except ValueError as exc:
                raise ValueError(f"bad page number: {chunk!r}") from exc

    out = sorted(p for p in pages if 0 <= p < page_count)
    if not out:
        raise ValueError(
            f"page range {spec!r} selects no pages in a {page_count}-page document"
        )
    return out

"""Command line interface.

    pdfx extract report.pdf
    pdfx extract scans/ --out out/ --config config.yaml
    pdfx route report.pdf          # routing decisions only, no model calls
    pdfx doctor                    # is the local stack actually ready?
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .native import docling_available
from .pipeline import Pipeline, write_outputs
from .router import Decision
from .validate import Severity, severity_at_least
from .vlm import GlmOcrClient, VLMError

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_ERROR = 3
EXIT_UNAVAILABLE = 4


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _collect_pdfs(target: Path) -> list[Path]:
    if target.is_dir():
        found = sorted(p for p in target.rglob("*.pdf") if p.is_file())
        if not found:
            raise SystemExit(f"no PDFs found under {target}")
        return found
    if not target.exists():
        raise SystemExit(f"no such file: {target}")
    return [target]


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if getattr(args, "model", None):
        cfg.ollama.model = args.model
    if getattr(args, "dpi", None):
        cfg.render.dpi = args.dpi
    if getattr(args, "pages", None):
        cfg.pipeline.page_range = args.pages
    if getattr(args, "backend", None):
        cfg.pipeline.native_backend = args.backend
    if getattr(args, "no_validate", False):
        cfg.validation.enabled = False
    if getattr(args, "tesseract", False):
        cfg.validation.tesseract_fallback = True
    if getattr(args, "out", None):
        cfg.output_dir = args.out
    if getattr(args, "workers", None):
        cfg.pipeline.max_workers = args.workers
    return cfg


# ---- commands -------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(Config.load(args.config), args)
    pdfs = _collect_pdfs(Path(args.input))
    pipeline = Pipeline(cfg)

    exit_code = EXIT_OK

    for pdf in pdfs:
        try:
            result = pipeline.run(pdf)
        except VLMError as exc:
            print(f"\n{pdf.name}: {exc}", file=sys.stderr)
            return EXIT_UNAVAILABLE

        md_path, json_path = write_outputs(result, cfg.output_dir)

        print(f"\n{pdf.name}  ({result.page_count} pages, {result.duration_s:.1f}s)")
        print(
            f"  routing:    {result.native_pages} native, "
            f"{result.vlm_pages} via GLM-OCR, {result.skipped_pages} blank"
        )

        checked = [p for p in result.pages if p.validation and p.validation.checked]
        if checked:
            findings = sum(len(p.validation.findings) for p in checked)
            print(
                f"  numerics:   {len(checked)} page(s) cross-checked, "
                f"{findings} finding(s), worst severity "
                f"{result.worst_severity.value}"
            )
            for page in checked:
                for finding in page.validation.findings:
                    if finding.severity in {Severity.HIGH, Severity.MEDIUM}:
                        print(
                            f"    p{page.page_number} [{finding.severity.value}] "
                            f"{finding.kind.value}: {finding.detail}"
                        )

        if result.failed_pages:
            print(f"  errors:     {len(result.failed_pages)} page(s) failed")
            for page in result.failed_pages:
                print(f"    p{page.page_number}: {page.error}")
            exit_code = max(exit_code, EXIT_ERROR)

        print(f"  wrote:      {md_path}")
        print(f"              {json_path}")

        if severity_at_least(result.worst_severity, cfg.validation.fail_on_severity):
            exit_code = max(exit_code, EXIT_VALIDATION)

    return exit_code


def cmd_route(args: argparse.Namespace) -> int:
    """Show routing decisions without touching the model. Cheap and diagnostic."""
    import pymupdf

    from .config import parse_page_range
    from .router import route_document

    cfg = _apply_overrides(Config.load(args.config), args)

    for pdf in _collect_pdfs(Path(args.input)):
        doc = pymupdf.open(pdf)
        try:
            indices = parse_page_range(cfg.pipeline.page_range, doc.page_count)
            routes = route_document(doc, cfg.router, indices)
        finally:
            doc.close()

        print(f"\n{pdf.name}")
        for route in routes:
            marker = {
                Decision.PARSE_NATIVE: "native",
                Decision.PARSE_VLM: "GLM-OCR",
                Decision.SKIP: "skip",
            }[route.decision]
            print(f"  p{route.page_number:<4} {marker:<8} {route.kind.value:<20} {route.reason}")

        vlm_count = sum(1 for r in routes if r.decision is Decision.PARSE_VLM)
        print(f"  -> {vlm_count}/{len(routes)} page(s) would be sent to the model")

    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check every moving part before you trust a batch run."""
    cfg = Config.load(args.config)
    ok = True

    print("pdfx doctor")
    print(f"  version:          {__version__}")
    print(f"  config:           {args.config or 'defaults (no config.yaml found)'}")
    if cfg.hardware_note:
        print(f"  hardware:         {cfg.hardware_note}")

    try:
        import pymupdf

        print(f"  pymupdf:          {pymupdf.__version__}")
    except ImportError:
        print("  pymupdf:          MISSING - pip install pymupdf")
        ok = False

    if docling_available():
        print("  docling:          available")
    else:
        print("  docling:          not installed (pymupdf backend will be used)")
        if cfg.pipeline.native_backend == "docling":
            print("                    but native_backend is set to 'docling' - "
                  "extraction will fail")
            ok = False

    client = GlmOcrClient(cfg.ollama)
    print(f"  endpoint:         {cfg.ollama.base_url} (api_mode={cfg.ollama.api_mode})")
    try:
        models = client.list_models()
        print(f"  ollama:           reachable, {len(models)} model(s) installed")
        client.health_check()
        print(f"  model:            {cfg.ollama.model} installed")
    except VLMError as exc:
        print(f"  ollama/model:     NOT READY\n                    {exc}")
        ok = False

    if cfg.validation.tesseract_fallback:
        from .validate import tesseract_text

        probe = tesseract_text(_blank_png())
        if probe is None:
            print("  tesseract:        requested but unavailable "
                  "(pip install pytesseract + brew install tesseract)")
            ok = False
        else:
            print("  tesseract:        available")

    print(f"\n  {'ready' if ok else 'not ready - see above'}")
    return EXIT_OK if ok else EXIT_UNAVAILABLE


def _blank_png() -> bytes:
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


# ---- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfx",
        description="Self-hosted hybrid PDF extraction: native text layer where "
                    "one exists, local GLM-OCR where it does not, with "
                    "deterministic numeric cross-checking.",
    )
    parser.add_argument("--version", action="version", version=f"pdfx {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--config", help="path to config.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="extract one PDF or a folder of PDFs")
    extract.add_argument("input", help="PDF file or directory")
    extract.add_argument("-o", "--out", help="output directory")
    extract.add_argument("--pages", help='page range, e.g. "1-5,9"')
    extract.add_argument("--model", help="override the Ollama model tag")
    extract.add_argument("--dpi", type=int, help="rasterisation DPI for VLM pages")
    extract.add_argument("--workers", type=int, help="concurrent VLM requests")
    extract.add_argument(
        "--backend", choices=["auto", "docling", "pymupdf"],
        help="native text extraction backend",
    )
    extract.add_argument("--no-validate", action="store_true",
                         help="skip the numeric cross-check")
    extract.add_argument("--tesseract", action="store_true",
                         help="use Tesseract as a second opinion on pages with no text layer")
    extract.set_defaults(func=cmd_extract)

    route = sub.add_parser("route", help="show routing decisions without calling the model")
    route.add_argument("input", help="PDF file or directory")
    route.add_argument("--pages", help='page range, e.g. "1-5,9"')
    route.set_defaults(func=cmd_route)

    doctor = sub.add_parser("doctor", help="check that the local stack is ready")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

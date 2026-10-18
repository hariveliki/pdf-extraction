# pdfx — self-hosted hybrid PDF extraction

Docling + GLM-OCR, entirely on this machine. No document, page image or figure
leaves the box; the only network traffic is package and model downloads at
install time.

```
                    ┌─────────────────────────────────────────┐
  report.pdf  ─────▶│ router: does this page have real text?   │
                    └──────┬──────────────┬─────────────┬──────┘
                     yes   │        no    │      blank  │
                           ▼              ▼             ▼
                  Docling / PyMuPDF   render 200dpi     skip
                  exact, no model      ──▶ GLM-OCR       (free)
                           │           (Ollama, local)
                           │                │
                           │                ▼
                           │      numeric cross-check against
                           │      the deterministic reading
                           ▼                │
                    ┌──────────────────────────────────┐
                    │ out/report.md   + out/report.json│
                    └──────────────────────────────────┘
```

## Install

```bash
./setup.sh
```

The script detects this Mac's chip and memory, picks the model quant, worker
count and render DPI to match, then installs the venv, Docling, Ollama and the
GLM-OCR weights, writes `config.yaml`, runs the tests and health-checks the
stack. `./setup.sh --dry-run` prints the plan and changes nothing;
`--no-docling` skips the torch download (~2–3 GB) and uses the PyMuPDF backend.

What it chooses, by detected hardware:

| Detected               | Model                       | Workers | DPI      |
| ---------------------- | --------------------------- | ------- | -------- |
| Apple Silicon, ≥32 GB | `glm-ocr:latest` (2.2 GB) | 2       | 220      |
| Apple Silicon, ≥16 GB | `glm-ocr:latest`          | 2       | 200      |
| Apple Silicon, 8 GB    | `glm-ocr:q8_0` (1.6 GB)   | 1       | 180      |
| Intel / CPU-only       | `glm-ocr:q8_0`            | 1       | 150–180 |

## Use

```bash
source .venv/bin/activate

pdfx route   report.pdf     # what would be sent to the model — no inference, instant
pdfx extract report.pdf     # writes out/report.md and out/report.json
pdfx extract scans/ --out results/
pdfx doctor                 # is Ollama up, is the model pulled, is Docling importable
```

`pdfx route` first is worth the habit. It tells you how much of a document is
actually going to cost inference before you spend it:

```
p1    native   native_text     178 chars of clean embedded text covering 2.7% of the page
p2    GLM-OCR  scanned_image   no usable text layer (0 chars) and an image covering 100% of the page
p3    skip     blank           no text and no images - treated as blank, no VLM call made
p4    GLM-OCR  mixed           178 chars of text alongside an image covering 100% of the page;
                               VLM reads the page, native text retained for cross-checking
-> 2/4 page(s) would be sent to the model
```

Exit codes: `0` clean, `2` validation findings at or above
`validation.fail_on_severity`, `3` a page failed, `4` the model server or model
is not available.

## The part that matters: quiet failures

Rule-based parsers fail loudly — you get garbled text and you know it. VLM
parsers fail *quietly*, producing fluent, well-formatted output with two digits
swapped in a total. On measurement reports and invoices that is the expensive
failure, and no amount of benchmark score protects you from it.

So every page the model reads is also read deterministically — from the page's
own text layer where it has one, or Tesseract if you enable
`validation.tesseract_fallback` — and the numbers from both readings are
compared:

```
  numerics:   1 page(s) cross-checked, 1 finding(s), worst severity high
    p4 [high] transposed_digits: same digits in a different order: "14'827.60" vs "14'287.60"
```

Findings are classified rather than lumped together, because the kinds mean
different things:

| Kind                     | What it means                                                 |
| ------------------------ | ------------------------------------------------------------- |
| `transposed_digits`    | Same digits, different order. Almost always a real misread.   |
| `substituted_digit`    | Exactly one digit differs — the classic 8/3, 6/5 confusion.  |
| `missing_from_vlm`     | A figure the deterministic reading has and the model dropped. |
| `not_in_deterministic` | A figure the model produced that nothing else supports.       |

Severity is graded by how authoritative the deterministic side is: disagreement
with a page's own text layer is `high`, disagreement with Tesseract output is
`medium`, and single-digit noise (page numbers, list markers) is `low`.
Reformatting is not a finding — `14'827.60` and `14827.60` match.

Number parsing is locale-aware without needing to be told the locale. `1'234.56`
(CH), `1.234,56` (DE) and `1,234.56` (US) all read correctly, and genuinely
ambiguous tokens like `1,234` carry both interpretations and match if either
agrees. Set `validation.number_locale` to `ch`, `de` or `us` to collapse that.

## Routing rules

A page goes to the model when, and only when, parsing it directly would be
worse:

- **no usable text layer** (< `min_chars`) with images present → scan
- **text layer present but undecodable** (> `max_garbage_ratio` of glyphs are
  `(cid:N)`, replacement or unassigned code points) → broken embedded font or
  bad prior OCR; rendering and re-reading is the only way to recover it
- **text plus a page-dominating image** (≥ `full_page_image_ratio`) → the model
  reads the page, the native text is kept for cross-checking
- **no text and no images** → blank, skipped, costs nothing

Everything else is parsed exactly from the content stream. Docling's TableFormer
handles table structure and multi-column reading order; if Docling is not
installed, PyMuPDF's own table detection is used instead. Docling runs with
`do_ocr=False` on purpose — letting it quietly OCR scanned pages would bypass
the router and the cross-check.

## Why `/api/generate` and not the OpenAI-compatible endpoint

Ollama's `/v1/chat/completions` shim returns 502 on vision requests; upstream
GLM-OCR recommends the native `/api/generate` path, and that is the default here
(`ollama.api_mode: generate`). GLM-OCR is also prompted with a bare task prefix
(`Text Recognition:`, `Table Recognition:`, `Figure Recognition:`) rather than a
free-form instruction. The OpenAI path is implemented and tested anyway, so
pointing `api_mode: openai` at vLLM or SGLang later needs no code change.

## Configuration

`setup.sh` generates `config.yaml`; `config.example.yaml` documents every knob.
Unknown keys are rejected at load time with the list of valid ones, so a typo
fails immediately instead of silently doing nothing. CLI flags
(`--dpi`, `--model`, `--pages`, `--backend`, `--workers`, `--no-validate`,
`--tesseract`) override the file per run.

## Layout

```
src/pdfx/
  router.py     per-page text-layer detection and routing
  render.py     rasterisation with a hard long-edge pixel cap
  native.py     Docling and PyMuPDF backends behind one interface
  vlm.py        GLM-OCR client (retries, health check, both API shapes)
  validate.py   locale-aware number parsing and the cross-check
  pipeline.py   orchestration, per-page provenance, Markdown/JSON output
  cli.py        extract / route / doctor
tests/          85 tests, no network, no model — synthetic PDFs + a stand-in Ollama
```

## Tests

```bash
make test
```

85 tests, no network and no model required. Synthetic PDFs cover each routing
branch (native, scanned, blank, mixed, corrupt text layer) and a stand-in Ollama
covers the VLM path, including simulated 502s and retry behaviour. The pipeline
tests assert the load-bearing behaviours directly: a page with a real text layer
must never cost inference, a blank page must never cost inference, and a
deliberately transposed total in the model's output must surface as a `high`
severity finding.

## Notes and limits

- **Scanned pages with no text layer cannot be cross-checked** unless you enable
  `validation.tesseract_fallback` (`pip install 'pdfx[tesseract]'` and
  `brew install tesseract`). Without it those pages are marked
  `checked: false` with the reason recorded, rather than being quietly presented
  as verified. Read the JSON, not just the Markdown.
- **A page that fails degrades rather than aborting.** If the model errors after
  its retry budget, that page falls back to its native text and is listed under
  `failed_pages`; the rest of the document still completes.
- **Concurrency above 2 is not useful with Ollama**, which serialises requests
  per model. The knob exists for vLLM.
- **First Docling run downloads models from Hugging Face** (~2–3 GB, layout and
  TableFormer). After that it is fully offline.
- Docling is MIT, GLM-OCR weights are open — check the model licence against
  your intended use before putting customer documents through it commercially.

## Sources

- [GLM-OCR on Ollama](https://ollama.com/library/glm-ocr) — model tags and sizes
- [GLM-OCR Ollama deployment guide](https://github.com/zai-org/GLM-OCR/blob/main/examples/ollama-deploy/README.md) — the `/api/generate` recommendation
- [GLM-OCR](https://github.com/zai-org/GLM-OCR) — 0.9B, 94.62 on OmniDocBench V1.5
- [Docling](https://github.com/docling-project/docling) — pipeline and table structure

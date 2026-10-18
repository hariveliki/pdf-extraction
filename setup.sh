#!/usr/bin/env bash
# Set up the self-hosted PDF extraction stack on this Mac.
#
#   ./setup.sh              full install
#   ./setup.sh --no-docling skip the torch/HF download (~2-3 GB)
#   ./setup.sh --dry-run    detect hardware and print the plan, change nothing
#
# Nothing here leaves the machine except package downloads.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

WITH_DOCLING=1
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-docling) WITH_DOCLING=0 ;;
    --dry-run)    DRY_RUN=1 ;;
    -h|--help)    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. hardware

say "Detecting hardware"

OS="$(uname -s)"
ARCH="$(uname -m)"
MEM_GB=0
CHIP="unknown"
CORES=0

if [ "$OS" = "Darwin" ]; then
  MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
  CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || echo unknown)"
  CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 0)"
elif [ "$OS" = "Linux" ]; then
  MEM_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  MEM_GB=$(( MEM_KB / 1024 / 1024 ))
  # ARM /proc/cpuinfo has no "model name" line, so fall back through the
  # alternatives before giving up.
  CHIP="$(awk -F': ' '/model name|^Model|^Hardware/{print $2; exit}' /proc/cpuinfo 2>/dev/null)"
  [ -n "$CHIP" ] || CHIP="$(cat /sys/firmware/devicetree/base/model 2>/dev/null | tr -d '\0')"
  [ -n "$CHIP" ] || CHIP="unknown ($ARCH)"
  CORES="$(nproc 2>/dev/null || echo 0)"
else
  warn "unrecognised OS '$OS'; falling back to conservative defaults"
fi

APPLE_SILICON=0
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  APPLE_SILICON=1
fi

echo "     os:      $OS ($ARCH)"
echo "     chip:    $CHIP"
echo "     cores:   $CORES"
echo "     memory:  ${MEM_GB} GB"

# --------------------------------------------------- 2. defaults from hardware
#
# GLM-OCR is 0.9B. glm-ocr:latest is a 2.2 GB download, q8_0 is 1.6 GB. Memory
# headroom decides which one and how many pages we dare run at once.

if [ "$APPLE_SILICON" -eq 1 ] && [ "$MEM_GB" -ge 32 ]; then
  MODEL="glm-ocr:latest"; WORKERS=2; DPI=220; MAXPX=1800
  TIER="Apple Silicon, ${MEM_GB} GB - plenty of headroom"
elif [ "$APPLE_SILICON" -eq 1 ] && [ "$MEM_GB" -ge 16 ]; then
  MODEL="glm-ocr:latest"; WORKERS=2; DPI=200; MAXPX=1600
  TIER="Apple Silicon, ${MEM_GB} GB - comfortable"
elif [ "$APPLE_SILICON" -eq 1 ]; then
  MODEL="glm-ocr:q8_0"; WORKERS=1; DPI=180; MAXPX=1400
  TIER="Apple Silicon, ${MEM_GB} GB - tight, using the q8_0 quant serially"
elif [ "$MEM_GB" -ge 16 ]; then
  MODEL="glm-ocr:q8_0"; WORKERS=1; DPI=180; MAXPX=1400
  TIER="CPU-only inference, ${MEM_GB} GB - expect seconds per scanned page"
else
  MODEL="glm-ocr:q8_0"; WORKERS=1; DPI=150; MAXPX=1200
  TIER="CPU-only inference, ${MEM_GB} GB - conservative settings"
fi

if [ "$WITH_DOCLING" -eq 1 ]; then
  BACKEND="auto"
else
  BACKEND="pymupdf"
fi

say "Chosen defaults"
echo "     tier:    $TIER"
echo "     model:   $MODEL"
echo "     workers: $WORKERS"
echo "     dpi:     $DPI (long edge capped at ${MAXPX}px)"
echo "     native:  $BACKEND backend"

if [ "$DRY_RUN" -eq 1 ]; then
  say "Dry run - stopping before any changes"
  exit 0
fi

# ------------------------------------------------------------- 3. python venv

command -v python3 >/dev/null 2>&1 || die "python3 not found. Install it with: brew install python@3.12"

PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)')"
[ "$PY_OK" = "1" ] || die "python 3.10+ required, found $PY_VER"

say "Creating virtualenv (.venv, python $PY_VER)"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip

EXTRAS="dev"
if [ "$WITH_DOCLING" -eq 1 ]; then
  say "Installing pdfx with Docling (this pulls torch, a few GB - be patient)"
  EXTRAS="dev,docling"
else
  say "Installing pdfx (core only)"
fi
python -m pip install --quiet -e ".[${EXTRAS}]"
echo "     installed: $(python -c 'import pdfx; print("pdfx", pdfx.__version__)')"

# ----------------------------------------------------------------- 4. ollama

if command -v ollama >/dev/null 2>&1; then
  say "Ollama already installed ($(ollama --version 2>&1 | head -1))"
elif [ "$OS" = "Darwin" ]; then
  if command -v brew >/dev/null 2>&1; then
    say "Installing Ollama via Homebrew"
    brew install ollama
  else
    die "Ollama is not installed and Homebrew is not available.
       Download the macOS app from https://ollama.com/download, then re-run this script.
       (The curl|sh installer is for Linux only - do not use it on macOS.)"
  fi
else
  die "Ollama is not installed. See https://ollama.com/download"
fi

OLLAMA_URL="http://127.0.0.1:11434"

ollama_up() { curl -fsS --max-time 3 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

if ollama_up; then
  say "Ollama is already serving on $OLLAMA_URL"
else
  say "Starting ollama serve in the background (log: .ollama-serve.log)"
  nohup ollama serve > .ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do
    sleep 1
    if ollama_up; then break; fi
  done
  ollama_up || die "ollama did not come up within 30s. Check .ollama-serve.log"
  echo "     up"
fi

# ------------------------------------------------------------- 5. pull model

if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
  say "$MODEL already present"
else
  say "Pulling $MODEL"
  ollama pull "$MODEL"
fi

# ------------------------------------------------------------- 6. write config

CONFIG_PATH="config.yaml"
if [ -f "$CONFIG_PATH" ]; then
  BACKUP="config.yaml.bak.$(date +%Y%m%d%H%M%S)"
  warn "$CONFIG_PATH exists; backing it up to $BACKUP"
  cp "$CONFIG_PATH" "$BACKUP"
fi

say "Writing $CONFIG_PATH"
cat > "$CONFIG_PATH" <<YAML
# Generated by setup.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ')
# Detected: $TIER
hardware_note: "$TIER"

ollama:
  host: 127.0.0.1
  port: 11434
  model: $MODEL
  # Ollama's OpenAI-compatible endpoint 502s on vision requests; upstream
  # GLM-OCR recommends the native /api/generate path.
  api_mode: generate
  timeout_s: 180.0
  retries: 3
  temperature: 0.0
  keep_alive: 10m

render:
  dpi: $DPI
  max_long_edge_px: $MAXPX
  grayscale: false

router:
  min_chars: 60
  min_text_area_ratio: 0.005
  full_page_image_ratio: 0.55
  max_garbage_ratio: 0.12
  vlm_for_mixed_pages: true

validation:
  enabled: true
  # auto keeps 1'234.56 / 1.234,56 / 1,234.56 all readable. Force with ch/de/us.
  number_locale: auto
  abs_tolerance: 0.0
  fail_on_severity: none
  tesseract_fallback: false

pipeline:
  native_backend: $BACKEND
  max_workers: $WORKERS
  default_task: Text Recognition
  table_prompt_for_table_pages: false

output_dir: out
YAML

# --------------------------------------------------------------- 7. verify

say "Running the test suite"
python -m pytest -q || die "tests failed - stopping before declaring success"

say "Checking the stack end to end"
python -m pdfx.cli doctor -c "$CONFIG_PATH"

cat <<EOF

$(printf '\033[1mReady.\033[0m') Activate the venv and extract something:

    source .venv/bin/activate
    pdfx route  yourfile.pdf      # what would be sent to the model, no inference
    pdfx extract yourfile.pdf     # writes out/yourfile.md and out/yourfile.json
    pdfx doctor                   # re-check the stack any time

EOF

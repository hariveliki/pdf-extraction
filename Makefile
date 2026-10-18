.PHONY: help setup setup-lite test doctor route extract serve clean

VENV    := .venv
PY      := $(VENV)/bin/python
PDFX    := $(VENV)/bin/pdfx
PDF     ?=
OUT     ?= out

help:
	@echo "make setup       install everything (venv, deps, Docling, Ollama, glm-ocr model)"
	@echo "make setup-lite  same but skip Docling (no torch download)"
	@echo "make doctor      check the local stack is ready"
	@echo "make test        run the test suite"
	@echo "make serve       start ollama serve in the background"
	@echo "make route  PDF=file.pdf   show routing decisions, no inference"
	@echo "make extract PDF=file.pdf  extract to $(OUT)/"
	@echo "make clean       remove venv, caches and outputs"

setup:
	./setup.sh

setup-lite:
	./setup.sh --no-docling

test: $(VENV)
	$(PY) -m pytest -q

doctor: $(VENV)
	$(PDFX) doctor

serve:
	@curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
		&& echo "ollama already serving" \
		|| (nohup ollama serve > .ollama-serve.log 2>&1 & echo "started ollama serve (.ollama-serve.log)")

route: $(VENV) require-pdf
	$(PDFX) route "$(PDF)"

extract: $(VENV) require-pdf
	$(PDFX) extract "$(PDF)" --out "$(OUT)"

require-pdf:
	@test -n "$(PDF)" || (echo "usage: make $(MAKECMDGOALS) PDF=path/to/file.pdf" >&2; exit 1)

$(VENV):
	@echo "no virtualenv yet - run 'make setup' first" >&2; exit 1

clean:
	rm -rf $(VENV) $(OUT) .pytest_cache .ollama-serve.log
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +

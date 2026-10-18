"""GLM-OCR client for a locally running Ollama.

Uses Ollama's native `/api/generate` endpoint rather than the
OpenAI-compatible `/v1/chat/completions` one. That is upstream GLM-OCR's own
recommendation: Ollama's OpenAI compatibility layer is unreliable for vision
requests and returns 502s on images. The OpenAI path is implemented anyway so
the pipeline can point at vLLM or SGLang later without a rewrite.

Only stdlib HTTP is used, so the core install stays at three dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import OllamaConfig

# GLM-OCR is prompted with a task prefix rather than free-form instructions.
TASK_TEXT = "Text Recognition"
TASK_TABLE = "Table Recognition"
TASK_FIGURE = "Figure Recognition"
VALID_TASKS = {TASK_TEXT, TASK_TABLE, TASK_FIGURE}


class VLMError(RuntimeError):
    """Raised when the model server cannot be reached or refuses the request."""


class VLMUnavailable(VLMError):
    """Raised when the server is not listening or the model is not installed."""


@dataclass
class VLMResult:
    page_index: int
    markdown: str
    task: str
    model: str
    latency_s: float
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def eval_tokens(self) -> int | None:
        return self.raw.get("eval_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "latency_s": round(self.latency_s, 2),
            "attempts": self.attempts,
            "output_tokens": self.eval_tokens,
        }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise VLMError(f"{url} returned HTTP {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise VLMUnavailable(f"cannot reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise VLMError(f"{url} timed out after {timeout:.0f}s") from exc
    except json.JSONDecodeError as exc:
        raise VLMError(f"{url} returned a non-JSON body") from exc


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise VLMUnavailable(f"cannot reach {url}: {exc.reason}") from exc
    except Exception as exc:  # noqa: BLE001 - health check must not mask errors
        raise VLMError(f"health check against {url} failed: {exc}") from exc


class GlmOcrClient:
    """Thin, retrying client around one local model server."""

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    # ---- health -----------------------------------------------------------

    def list_models(self) -> list[str]:
        data = _get_json(f"{self.cfg.base_url}/api/tags", timeout=10.0)
        return [m.get("name", "") for m in data.get("models", [])]

    def health_check(self) -> None:
        """Fail fast with an actionable message before processing any pages."""
        try:
            models = self.list_models()
        except VLMUnavailable as exc:
            raise VLMUnavailable(
                f"{exc}\n"
                f"  Ollama does not appear to be running on {self.cfg.base_url}.\n"
                f"  Start it with:  ollama serve"
            ) from exc

        wanted = self.cfg.model
        # "glm-ocr" should match an installed "glm-ocr:latest".
        base = wanted.split(":")[0]
        if not any(m == wanted or m.split(":")[0] == base for m in models):
            raise VLMUnavailable(
                f"model {wanted!r} is not installed in Ollama.\n"
                f"  Installed: {', '.join(models) or '(none)'}\n"
                f"  Install it with:  ollama pull {wanted}"
            )

    # ---- inference --------------------------------------------------------

    def _payload_generate(self, image_b64: str, task: str) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            # GLM-OCR expects a bare task prefix, not a chatty instruction.
            "prompt": f"{task}:",
            "images": [image_b64],
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": self.cfg.temperature},
        }

    def _payload_openai(self, image_b64: str, task: str) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{task}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
        }

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self.cfg.api_mode == "openai":
            choices = data.get("choices") or []
            if not choices:
                raise VLMError("OpenAI-compatible response contained no choices")
            return (choices[0].get("message") or {}).get("content") or ""
        if "response" not in data:
            raise VLMError(
                f"Ollama response missing 'response' field; got keys "
                f"{sorted(data)[:8]}"
            )
        return data["response"] or ""

    def recognise(
        self, image_b64: str, page_index: int = 0, task: str = TASK_TEXT
    ) -> VLMResult:
        """Send one page image to the model and return its Markdown."""
        if task not in VALID_TASKS:
            raise ValueError(f"unknown GLM-OCR task {task!r}; expected one of {sorted(VALID_TASKS)}")

        if self.cfg.api_mode == "openai":
            url = f"{self.cfg.base_url}/v1/chat/completions"
            payload = self._payload_openai(image_b64, task)
        elif self.cfg.api_mode == "generate":
            url = f"{self.cfg.base_url}/api/generate"
            payload = self._payload_generate(image_b64, task)
        else:
            raise ValueError(
                f"unknown api_mode {self.cfg.api_mode!r}; expected 'generate' or 'openai'"
            )

        last_exc: Exception | None = None
        started = time.monotonic()

        for attempt in range(1, max(1, self.cfg.retries) + 1):
            try:
                data = _post_json(url, payload, timeout=self.cfg.timeout_s)
                text = self._extract_text(data).strip()
                return VLMResult(
                    page_index=page_index,
                    markdown=text,
                    task=task,
                    model=self.cfg.model,
                    latency_s=time.monotonic() - started,
                    attempts=attempt,
                    raw={k: v for k, v in data.items() if k != "response"},
                )
            except VLMUnavailable:
                # A refused connection will not fix itself by retrying hard;
                # surface it immediately with the actionable message.
                raise
            except VLMError as exc:
                last_exc = exc
                if attempt < self.cfg.retries:
                    time.sleep(self.cfg.retry_backoff_s * attempt)

        raise VLMError(
            f"page {page_index + 1}: {self.cfg.retries} attempt(s) against {url} "
            f"failed; last error: {last_exc}"
        )

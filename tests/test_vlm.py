"""VLM client behaviour against a stand-in Ollama."""

from __future__ import annotations

import base64

import pytest

from pdfx.config import OllamaConfig
from pdfx.vlm import TASK_TABLE, GlmOcrClient, VLMError, VLMUnavailable

from .mock_ollama import MockOllama

IMAGE_B64 = base64.b64encode(b"not-really-a-png").decode()


def _cfg(port: int, **kw) -> OllamaConfig:
    return OllamaConfig(host="127.0.0.1", port=port, retry_backoff_s=0.01, **kw)


def test_generate_endpoint_returns_markdown():
    with MockOllama(response_text="# Heading\n\nTotal: 1'248") as srv:
        client = GlmOcrClient(_cfg(srv.port))
        result = client.recognise(IMAGE_B64, page_index=3)

    assert result.markdown == "# Heading\n\nTotal: 1'248"
    assert result.page_index == 3
    assert result.attempts == 1
    assert result.latency_s >= 0


def test_request_uses_the_native_endpoint_and_task_prefix():
    """Upstream GLM-OCR expects a bare task prefix, not a chatty instruction."""
    with MockOllama() as srv:
        GlmOcrClient(_cfg(srv.port)).recognise(IMAGE_B64, task=TASK_TABLE)
        sent = srv.requests[0]

    assert sent["prompt"] == "Table Recognition:"
    assert sent["images"] == [IMAGE_B64]
    assert sent["stream"] is False
    assert sent["options"]["temperature"] == 0.0


def test_openai_mode_is_supported_for_vllm_later():
    with MockOllama(response_text="from openai path") as srv:
        client = GlmOcrClient(_cfg(srv.port, api_mode="openai"))
        result = client.recognise(IMAGE_B64)

    assert result.markdown == "from openai path"
    content = srv.requests[0]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_transient_502s_are_retried():
    """Ollama's vision path is known to 502; two failures then success."""
    with MockOllama(response_text="recovered", fail_times=2) as srv:
        client = GlmOcrClient(_cfg(srv.port, retries=3))
        result = client.recognise(IMAGE_B64)

    assert result.markdown == "recovered"
    assert result.attempts == 3


def test_persistent_failure_raises_after_the_retry_budget():
    with MockOllama(status=500) as srv:
        client = GlmOcrClient(_cfg(srv.port, retries=2))
        with pytest.raises(VLMError, match="2 attempt"):
            client.recognise(IMAGE_B64)


def test_a_dead_server_fails_fast_with_instructions():
    client = GlmOcrClient(_cfg(1))  # nothing listening on port 1
    with pytest.raises(VLMUnavailable, match="ollama serve"):
        client.health_check()


def test_health_check_names_the_missing_model():
    with MockOllama(model="llama3:latest") as srv:
        client = GlmOcrClient(_cfg(srv.port, model="glm-ocr:latest"))
        with pytest.raises(VLMUnavailable, match="ollama pull glm-ocr:latest"):
            client.health_check()


def test_health_check_accepts_an_untagged_model_name():
    with MockOllama(model="glm-ocr:latest") as srv:
        GlmOcrClient(_cfg(srv.port, model="glm-ocr")).health_check()


def test_unknown_task_is_rejected_before_any_request():
    with MockOllama() as srv:
        client = GlmOcrClient(_cfg(srv.port))
        with pytest.raises(ValueError, match="unknown GLM-OCR task"):
            client.recognise(IMAGE_B64, task="Vibes Recognition")
        assert srv.requests == []


def test_unknown_api_mode_is_rejected():
    with MockOllama() as srv:
        client = GlmOcrClient(_cfg(srv.port, api_mode="grpc"))
        with pytest.raises(ValueError, match="unknown api_mode"):
            client.recognise(IMAGE_B64)

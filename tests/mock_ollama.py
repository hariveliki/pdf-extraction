"""A stand-in Ollama for tests.

Serves /api/tags and /api/generate on a free port. The canned reply is
configurable so the suite can simulate the failure mode this pipeline exists to
catch: fluent output with a corrupted figure.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable


class MockOllama:
    def __init__(
        self,
        response_text: str = "mock output",
        model: str = "glm-ocr:latest",
        status: int = 200,
        fail_times: int = 0,
        responder: Callable[[dict], str] | None = None,
    ):
        self.response_text = response_text
        self.model = model
        self.status = status
        self.fail_times = fail_times
        self.responder = responder
        self.requests: list[dict] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        assert self._server is not None, "server not started"
        return self._server.server_address[1]

    def __enter__(self) -> "MockOllama":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence test output
                pass

            def _send(self, code: int, payload: dict):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": outer.model}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(payload)

                if outer.fail_times > 0:
                    outer.fail_times -= 1
                    self._send(502, {"error": "simulated bad gateway"})
                    return

                if outer.status != 200:
                    self._send(outer.status, {"error": "simulated failure"})
                    return

                text = (
                    outer.responder(payload) if outer.responder else outer.response_text
                )

                if self.path == "/api/generate":
                    self._send(200, {
                        "model": payload.get("model"),
                        "response": text,
                        "done": True,
                        "eval_count": len(text.split()),
                    })
                elif self.path == "/v1/chat/completions":
                    self._send(200, {
                        "choices": [{"message": {"role": "assistant", "content": text}}]
                    })
                else:
                    self._send(404, {"error": "not found"})

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        return False

"""Tests for the OpenAI-compatible provider layer (hermetic: local HTTP server)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dse.config import ModelConfig, provider_models
from dse.providers import OpenAICompatibleLLM, make_provider


def _serve(status: int = 200, payload: dict | None = None):
    captured: dict = {}
    payload = payload or {
        "choices": [{"message": {"content": "ANSWER: ok"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            captured["path"] = self.path
            captured["body"] = json.loads(body) if body else None
            captured["auth"] = self.headers.get("Authorization")
            if status != 200:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # silence the server
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, captured


def test_openai_compatible_llm_roundtrip():
    server, captured = _serve()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        models = provider_models(cheap_model="deepseek-chat", expensive_model="deepseek-reasoner")
        llm = OpenAICompatibleLLM(models, base, api_key="test-key")
        out = llm.complete([{"role": "user", "content": "hi"}], model="cheap")
        assert out.text == "ANSWER: ok"
        assert out.tokens_in == 12 and out.tokens_out == 3
        assert out.latency_s >= 0.0
        # payload sent to the wire must use the provider model, not the tier key
        assert captured["path"] == "/chat/completions"
        assert captured["body"]["model"] == "deepseek-chat"
        assert captured["body"]["temperature"] == 0.0
        assert captured["auth"] == "Bearer test-key"
    finally:
        server.shutdown()


def test_missing_provider_model_raises():
    models = {"cheap": ModelConfig(name="cheap")}  # no provider_model set
    llm = OpenAICompatibleLLM(models, "http://127.0.0.1:1")
    with pytest.raises(ValueError, match="provider_model"):
        llm.complete([{"role": "user", "content": "hi"}], model="cheap")


def test_http_error_is_surfaced():
    server, _ = _serve(status=404)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        models = provider_models()
        llm = OpenAICompatibleLLM(models, base)
        with pytest.raises(RuntimeError, match="HTTP 404"):
            llm.complete([{"role": "user", "content": "hi"}], model="cheap")
    finally:
        server.shutdown()


def test_malformed_provider_response_raises():
    server, _ = _serve(payload={"unexpected": True})
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        models = provider_models()
        llm = OpenAICompatibleLLM(models, base)
        with pytest.raises(RuntimeError, match="unexpected provider response"):
            llm.complete([{"role": "user", "content": "hi"}], model="cheap")
    finally:
        server.shutdown()


def test_make_provider_mock_returns_none():
    assert make_provider("mock") is None


def _sequence_server(payloads):
    """Threaded server that replays a list of payloads, one per request."""
    counter = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = payloads[min(counter["n"], len(payloads) - 1)]
            counter["n"] += 1
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # silence the server
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, counter


def _blank(usage=None):
    return {
        "choices": [{"message": {"content": ""}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 0},
    }


def test_empty_completion_is_retried_until_nonempty():
    """DeepSeek V4 Flash blank-completion quirk: an empty reply is retried
    automatically, and the non-empty completion is returned."""
    from dse.providers import _EMPTY_COMPLETION_RETRIES

    server, counter = _sequence_server([
        _blank(),
        {"choices": [{"message": {"content": "ANSWER: ok"}}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ])
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        models = provider_models()
        llm = OpenAICompatibleLLM(models, base)
        out = llm.complete([{"role": "user", "content": "hi"}], model="cheap")
        assert out.text == "ANSWER: ok"
        assert counter["n"] == 2  # one blank attempt was retried
    finally:
        server.shutdown()


def test_empty_completion_gives_up_gracefully():
    """If every attempt returns blank, complete() returns the empty completion
    (no exception) so callers keep their own fallback behaviour."""
    from dse.providers import _EMPTY_COMPLETION_RETRIES

    server, counter = _sequence_server([_blank()])
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        models = provider_models()
        llm = OpenAICompatibleLLM(models, base)
        out = llm.complete([{"role": "user", "content": "hi"}], model="cheap")
        assert out.text == ""
        assert counter["n"] == _EMPTY_COMPLETION_RETRIES
        assert llm.empty_completions == _EMPTY_COMPLETION_RETRIES
    finally:
        server.shutdown()


def test_reasoning_content_fallback_is_counted():
    """Empty content with non-empty reasoning_content: we retry with an
    escalated budget so the model can finish and write a real answer; if every
    retry stays blank the reasoning is salvaged and the fallback is counted."""
    from dse.providers import _EMPTY_COMPLETION_RETRIES

    payload = {
        "choices": [{"message": {
            "content": "",
            "reasoning_content": "Let me think: the answer is 42.",
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    server, _ = _serve(payload=payload)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        models = provider_models()
        llm = OpenAICompatibleLLM(models, base)
        out = llm.complete([{"role": "user", "content": "hi"}], model="cheap")
        assert out.text == "Let me think: the answer is 42."
        assert llm.fallback_count == 1
        # each empty attempt is counted; the reasoning is salvaged only at the end
        assert llm.empty_completions == _EMPTY_COMPLETION_RETRIES
    finally:
        server.shutdown()


def test_make_provider_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider("not-a-provider")


def test_make_provider_env_config(monkeypatch):
    monkeypatch.setenv("DSE_MODEL_CHEAP", "custom-cheap")
    monkeypatch.setenv("DSE_MODEL_EXPENSIVE", "custom-exp")
    monkeypatch.setenv("DSE_PROVIDER_URL", "http://localhost:12345")
    llm = make_provider("ollama")
    assert llm._provider_model("cheap") == "custom-cheap"
    assert llm._provider_model("expensive") == "custom-exp"
    assert llm._base_url == "http://localhost:12345"

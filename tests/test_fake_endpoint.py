"""Tests against a fake OpenAI-compatible HTTP endpoint.

Unlike test_forward_pass.py / test_backward_pass.py (which monkeypatch
call_llm directly), these tests spin up a real local HTTP server that speaks
the OpenAI chat-completions wire format and point the Nebius provider at it.
This exercises the actual HTTP client, JSON request/response handling, and
env-var wiring in src/processing.py without needing real API keys or network
access.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from src import processing


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    response_fn = staticmethod(lambda request_data: "{}")
    requests_received: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        request_data = json.loads(body) if body else {}
        type(self).requests_received.append(request_data)

        content = type(self).response_fn(request_data)
        payload = {
            "id": "fake-completion-1",
            "object": "chat.completion",
            "model": request_data.get("model", "fake-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence default request logging


@pytest.fixture
def fake_llm_endpoint(monkeypatch):
    """Start a local fake OpenAI-compatible server and point LLM_PROVIDER at it."""
    handler_cls = type(
        "FakeOpenAIHandler",
        (_FakeOpenAIHandler,),
        {"response_fn": staticmethod(lambda request_data: "{}"), "requests_received": []},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1/"

    # Ensure requests to the local fake server never get routed through the
    # outbound HTTPS proxy configured for this environment.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    monkeypatch.setenv("LLM_PROVIDER", "nebius")
    monkeypatch.setenv("NEBIUS_API_KEY", "fake-test-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", base_url)

    def set_response(fn):
        handler_cls.response_fn = staticmethod(fn)

    ns = SimpleNamespace(
        base_url=base_url,
        set_response=set_response,
        requests_received=handler_cls.requests_received,
    )
    try:
        yield ns
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class TestFakeEndpointCallLlm:
    def test_call_llm_round_trips_through_fake_endpoint(self, fake_llm_endpoint):
        fake_llm_endpoint.set_response(lambda request_data: "Hello from the fake endpoint")

        result = processing.call_llm("Say hello", model="fake-model")

        assert result == "Hello from the fake endpoint"
        assert len(fake_llm_endpoint.requests_received) == 1
        assert fake_llm_endpoint.requests_received[0]["model"] == "fake-model"

    def test_call_llm_sends_prompt_content_to_fake_endpoint(self, fake_llm_endpoint):
        fake_llm_endpoint.set_response(lambda request_data: "ack")

        processing.call_llm("What is the diagnosis for I21.0?", model="fake-model")

        sent_messages = fake_llm_endpoint.requests_received[0]["messages"]
        assert any("I21.0" in m.get("content", "") for m in sent_messages)


class TestFakeEndpointGenerateFromCodes:
    def test_generate_from_codes_works_end_to_end_against_fake_endpoint(
        self, fake_llm_endpoint, sample_patient
    ):
        code = "I21.0"

        def respond(request_data):
            return json.dumps(
                {
                    "admission_type": "emergency",
                    "chief_complaint": "Central crushing chest pain",
                    "working_diagnosis": f"{code} - STEMI confirmed on ECG",
                    "estimated_los_days": 5,
                }
            )

        fake_llm_endpoint.set_response(respond)

        admission = processing.generate_from_codes(
            diagnostic_codes=[code],
            procedure_codes=[],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
            model="fake-model",
        )

        # Backward check: the fake endpoint's response is correctly parsed
        # and the diagnosis carried through into the admission record.
        assert admission["diagnostic_codes"] == [code]
        assert code in admission["working_diagnosis"]

        # Forward check: the code was actually part of the outbound request.
        sent_prompt = fake_llm_endpoint.requests_received[0]["messages"][-1]["content"]
        assert code in sent_prompt

    def test_generate_from_codes_works_end_to_end_for_opcs4(self, fake_llm_endpoint, sample_patient):
        code = "K40.1"

        def respond(request_data):
            return json.dumps(
                {
                    "admission_type": "elective",
                    "planned_procedure": "Coronary artery bypass grafting using saphenous vein graft",
                    "indication": f"Triple vessel disease - {code} planned",
                    "estimated_los_days": 8,
                }
            )

        fake_llm_endpoint.set_response(respond)

        admission = processing.generate_from_codes(
            diagnostic_codes=[],
            procedure_codes=[code],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
            model="fake-model",
        )

        assert admission["procedure_codes"] == [code]
        assert code in admission["indication"]

        sent_prompt = fake_llm_endpoint.requests_received[0]["messages"][-1]["content"]
        assert code in sent_prompt

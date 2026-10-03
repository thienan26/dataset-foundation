from __future__ import annotations

import json

import pytest

from multicare_data.common import PipelineError, Project
from multicare_data.llm_review import Gemini
from multicare_data.provider_check import check_gemini


def make_project(tmp_path, max_attempts=1):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "review.yaml").write_text(
        f"reviewer_a_model: gemini-3.8-flash\nmax_attempts: {max_attempts}\ntimeout_seconds: 1\n"
        "requests_per_minute: 10\nmax_output_tokens: 128\nprompt_version: test_v1\n",
        encoding="utf-8",
    )
    return Project(tmp_path)


def test_connectivity_check_sends_only_generic_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"modelVersion": "test-version", "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": '{"status":"ok"}'}]}}
            ]}).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["headers"] = request.headers
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("multicare_data.provider_check.urllib.request.urlopen", fake_urlopen)
    result = check_gemini(make_project(tmp_path))

    assert result["status"] == "ok"
    assert result["dataset_data_sent"] is False
    assert "case" not in json.dumps(captured["body"]).lower()
    assert "x-goog-api-key" in {key.lower() for key in captured["headers"]}
    assert "gemini-3.8-flash:generateContent" in captured["url"]


def test_connectivity_check_requires_key_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def unexpected_call(*args, **kwargs):
        pytest.fail("network must not be called without an API key")

    monkeypatch.setattr("multicare_data.provider_check.urllib.request.urlopen", unexpected_call)
    with pytest.raises(PipelineError, match="GEMINI_API_KEY is missing"):
        check_gemini(make_project(tmp_path))


def test_plain_text_connectivity_check_omits_json_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"modelVersion": "lite-test", "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": "GEMINI_OK"}]}}
            ]}).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("multicare_data.provider_check.urllib.request.urlopen", fake_urlopen)
    result = check_gemini(make_project(tmp_path), "gemini-3.5-flash-lite", plain_text=True)

    assert result["probe_mode"] == "plain_text"
    assert result["probe_text"] == "GEMINI_OK"
    assert result["dataset_data_sent"] is False
    assert "responseJsonSchema" not in captured["body"]["generationConfig"]
    assert "responseMimeType" not in captured["body"]["generationConfig"]
    assert "Reply exactly with: GEMINI_OK" in captured["body"]["contents"][0]["parts"][0]["text"]
    assert "gemini-3.5-flash-lite:generateContent" in captured["url"]


def test_gemini_keeps_raw_response_and_resumes_from_cache_without_duplicate_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    raw = {"modelVersion": "gemini-3.8-flash-test", "usageMetadata": {"totalTokenCount": 10},
           "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"status":"ok"}'}]}}]}
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(raw).encode()

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr("multicare_data.llm_review.urllib.request.urlopen", fake_urlopen)
    schema = {"type": "object", "properties": {"status": {"type": "string"}},
              "required": ["status"], "additionalProperties": False}
    project = make_project(tmp_path)

    first = Gemini(project).run("label_a", {"text": "generic test"}, schema, "gemini-3.8-flash")
    second = Gemini(project).run("label_a", {"text": "generic test"}, schema, "gemini-3.8-flash")
    audit = (tmp_path / "data" / "reviews" / "gemini_raw.jsonl").read_text(encoding="utf-8").splitlines()

    assert calls == 1
    assert first["request_sha256"] == second["request_sha256"]
    assert len(audit) == 1
    assert json.loads(audit[0])["raw_response"] == raw


def test_gemini_429_stops_after_first_call_with_accurate_attempt_count(tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = 0

    def rate_limited(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(request.full_url, 429, "rate limit", {}, None)

    monkeypatch.setattr("multicare_data.llm_review.urllib.request.urlopen", rate_limited)
    client = Gemini(make_project(tmp_path, max_attempts=3))
    schema = {"type": "object", "properties": {"status": {"type": "string"}},
              "required": ["status"], "additionalProperties": False}

    with pytest.raises(PipelineError, match="Gemini HTTP 429") as exc_info:
        client.run("label_a", {"text": "generic test"}, schema, "gemini-3.6-flash")

    assert calls == 1
    assert exc_info.value.attempt_count == 1

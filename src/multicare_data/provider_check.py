from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from dotenv import load_dotenv

from .common import PipelineError, Project


def check_gemini(project: Project, model: str | None = None, plain_text: bool = False) -> dict:
    """Send a generic connectivity probe; never reads or sends dataset rows or images."""
    cfg = project.config("review")
    load_dotenv(project.path(".env"), override=False)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise PipelineError("GEMINI_API_KEY is missing. Set it in the project .env or current process environment.")

    model = model or cfg["reviewer_a_model"]
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
        raise PipelineError("Invalid Gemini model ID")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    if plain_text:
        body = {"contents": [{"role": "user", "parts": [{
            "text": "Reply exactly with: GEMINI_OK"
        }]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 32}}
    else:
        schema = {"type": "OBJECT", "properties": {"status": {"type": "STRING"}},
                  "required": ["status"]}
        body = {"contents": [{"role": "user", "parts": [{
            "text": 'Connectivity test only. Reply as JSON: {"status":"ok"}. No medical data is included.'
        }]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 256,
                                  "thinkingConfig": {"thinkingLevel": "low"},
                                      "responseMimeType": "application/json", "responseJsonSchema": schema}}

    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        # A diagnostic probe is one request only; never amplify 429s or hang for the
        # much longer clinical-review timeout configured for the main pipeline.
        with urllib.request.urlopen(request, timeout=min(cfg["timeout_seconds"], 45)) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        hints = {401: "API key invalid", 403: "key/API permission denied", 404: "model not found",
                 429: "quota or rate limit", 503: "Google service temporarily overloaded"}
        hint = hints.get(exc.code, "check API access and network")
        raise PipelineError(f"Gemini connectivity check failed: HTTP {exc.code} ({hint}).") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if isinstance(exc, urllib.error.URLError):
            raise PipelineError("Gemini could not be reached; check this machine's HTTPS proxy or network.") from None
        raise PipelineError(f"Gemini connectivity check failed: {type(exc).__name__}.") from None

    candidates = raw.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        finish = candidates[0].get("finishReason", "NO_CANDIDATE") if candidates else "NO_CANDIDATE"
        raise PipelineError(f"Gemini connectivity probe did not finish (finishReason={finish}).")
    text = "".join(part.get("text", "") for part in candidates[0]["content"].get("parts", []))
    if plain_text and text.strip() != "GEMINI_OK":
        raise PipelineError("Gemini plain-text probe returned an unexpected response.")
    if not plain_text:
        try:
            valid_json = json.loads(text).get("status") == "ok"
        except json.JSONDecodeError:
            valid_json = False
        if not valid_json:
            raise PipelineError("Gemini connectivity probe returned an unexpected response.")
    return {"status": "ok", "model": model,
            "model_version": raw.get("modelVersion", model),
            "usage": raw.get("usageMetadata", {}), "dataset_data_sent": False,
            "probe_mode": "plain_text" if plain_text else "structured_json",
            **({"probe_text": text.strip()} if plain_text else {})}

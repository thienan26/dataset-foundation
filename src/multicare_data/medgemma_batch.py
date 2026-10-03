"""Finite per-case retries, checkpoints and explicit human routing for failed outputs."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import jsonschema
from dotenv import load_dotenv

from .common import PipelineError, canonical, file_hash, now, read_json, write_json
from .medgemma import (
    apply_results,
    cache_path,
    failure_path,
    grounded_result,
    jobs,
    prompt_text,
    recover_failed_reviews,
    save_review,
)

POLICY = "medgemma_batch_v002"


def terminal_failure(project, job):
    path = failure_path(project, job["request_sha256"])
    if not path.exists():
        return None
    saved = read_json(path)
    if (saved.get("terminal") and saved.get("processing_policy") == POLICY
            and all(saved.get(k) == v for k, v in job.items())):
        return saved
    return None


def infer(project, job, base):
    content = [{"type": "text", "text": prompt_text(job)}]
    if job["kind"] == "image":
        from .image_qc import review_image_bytes
        path = project.path(job["image_path"])
        if file_hash(path) != job["image_sha256"]:
            raise PipelineError("Source image changed since QC")
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                        + base64.b64encode(review_image_bytes(path)).decode("ascii")}})
    body = {"model": job["model"], "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": job["max_output_tokens"],
            "response_format": {"type": "json_object", "schema": job["schema"]}}
    headers = {"Content-Type": "application/json"}
    if os.getenv("MEDGEMMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["MEDGEMMA_API_KEY"]
    cfg, attempts = project.config("medgemma"), []
    error = ""
    for attempt in range(cfg.get("max_attempts", 3)):
        try:
            request = urllib.request.Request(base + "/chat/completions", canonical(body).encode(), headers)
            with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                raw = json.load(response)
            attempts.append({"messages": body["messages"], "response": raw})
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise PipelineError("Incomplete model response")
            result, alignment = grounded_result(job, json.loads(choice["message"]["content"]))
            save_review(project, job, result, raw, attempts, alignment)
            return "valid"
        except urllib.error.HTTPError as exc:
            message = exc.read(4096).decode("utf-8", errors="replace")
            if exc.code in {400, 422} and "context" in message.lower():
                error = f"HTTP {exc.code}: input exceeds model context"
                attempts.append({"messages": body["messages"], "error": error})
                break
            raise PipelineError(f"MedGemma HTTP {exc.code}; server/access error, pass stopped") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            error = f"Transport failed: {type(exc).__name__}"
            attempts.append({"messages": body["messages"], "error": error})
            if attempt + 1 == cfg.get("max_attempts", 3):
                write_json(failure_path(project, job["request_sha256"]),
                           {**job, "terminal": False, "processing_policy": POLICY,
                            "attempts": attempts, "error": error, "failed_at": now()})
                raise PipelineError("MedGemma server became unavailable; checkpoints preserved") from None
            time.sleep(min(attempt + 1, 3))
            continue
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, jsonschema.ValidationError, PipelineError) as exc:
            error = str(exc)[:1000]
        body = {**body, "messages": [{"role": "user", "content": content + [{
            "type": "text", "text": "Previous validation failed: " + error[:500]
             + ". Return a SHORT exact evidence quote (one or two sentences), preserving source characters. "
               "Accept requires confirmed, current, exactly one target and no other confirmed infectious disease. "
               "If you cannot support acceptance, return uncertain. Recheck every field for consistency."}]}]}
    write_json(failure_path(project, job["request_sha256"]),
               {**job, "terminal": True, "processing_policy": POLICY, "attempts": attempts,
                "error": error, "failed_at": now()})
    return "validation_failed"


def run(project, limit=25):
    load_dotenv(project.path(".env"), override=False)
    base = os.getenv("MEDGEMMA_BASE_URL", "").rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
        raise PipelineError("Set MEDGEMMA_BASE_URL to HTTPS or a local HTTP server")
    all_jobs = jobs(project)
    recovered = recover_failed_reviews(project, all_jobs)
    summary = apply_results(project, all_jobs)
    valid, failed = sum(summary["completed"].values()), sum(summary["failed"].values())
    attempted, started, error = 0, time.monotonic(), None
    checkpoint = max(1, project.config("medgemma").get("checkpoint_every", 5))
    write_json(project.path("reports/medgemma_runtime.json"),
               {"status": "RUNNING", "phase": "starting", "total_jobs": len(all_jobs),
                "completed_total": valid, "failed_total": failed, "processed_total": valid + failed,
                "pending_total": len(all_jobs) - valid - failed, "updated_at": now()})
    try:
        for job in all_jobs:
            if cache_path(project, job["request_sha256"]).exists() or terminal_failure(project, job):
                continue
            if attempted >= limit:
                break
            attempted += 1
            write_json(project.path("reports/medgemma_runtime.json"),
                       {"status": "RUNNING", "phase": "inference", "in_progress_case_id": job["case_id"],
                        "total_jobs": len(all_jobs), "completed_total": valid, "failed_total": failed,
                        "processed_total": valid + failed, "pending_total": len(all_jobs) - valid - failed,
                        "updated_at": now()})
            status = infer(project, job, base)
            valid += status == "valid"
            failed += status != "valid"
            write_json(project.path("reports/medgemma_runtime.json"),
                       {"status": "RUNNING", "total_jobs": len(all_jobs), "completed_total": valid,
                        "failed_total": failed, "processed_total": valid + failed,
                        "pending_total": len(all_jobs) - valid - failed, "last_case_id": job["case_id"],
                        "elapsed_seconds": round(time.monotonic() - started, 1), "updated_at": now()})
            print(f"MedGemma {valid + failed}/{len(all_jobs)}: {job['case_id']} {status}", flush=True)
            if attempted % checkpoint == 0:
                apply_results(project, all_jobs)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = apply_results(project, all_jobs)
        write_json(project.path("reports/medgemma_runtime.json"),
                   {"status": "PROCESS_FAILED" if error else "PASS_COMPLETE" if summary["pass_complete"] else "BATCH_COMPLETE",
                    "error": error, "attempted_this_run": attempted, "recovered_reviews": recovered,
                    **summary})
    return {"attempted_this_run": attempted, "recovered_reviews": recovered, **summary}

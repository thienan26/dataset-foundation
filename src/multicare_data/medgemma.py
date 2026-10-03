"""Source-bound MedGemma batch exchange and resumable endpoint inference."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import jsonschema
from dotenv import load_dotenv

from .common import PipelineError, canonical, digest, file_hash, now, read_json, write_json
from .evidence import unique_quote_span
from .llm_review import SYSTEM, label_payload
from .review_schema import IMAGE_REVIEW, LABEL_REVIEW, TEXT_REVIEW
from .sample import single_targets

SCHEMAS = {"label": LABEL_REVIEW, "text": TEXT_REVIEW, "image": IMAGE_REVIEW}


def jobs(project):
    from .image_qc import image_payload
    from .text_qc import text_payload

    cfg = project.config("medgemma")
    cases = {r["case_id"]: r for r in single_targets(project)}
    if cfg.get("scope", "human_conflicts") == "human_conflicts":
        unresolved = {r["case_id"] for r in project.rows("data/reviews/human_requirements.parquet", optional=True)
                      if r.get("consensus") == "NEEDS_ADJUDICATION"}
        if not unresolved:
            unresolved = {r["case_id"] for r in project.rows("data/reviews/adjudication.parquet", optional=True)
                          if r.get("reason") == "HUMAN_ADJUDICATION_REQUIRED"}
        resolved_humans = {r["case_id"] for r in project.rows("data/reviews/human_label_adjudications.parquet", optional=True)
                           if r.get("human_decision") in {"accept", "reject"}}
        cases = {cid: case for cid, case in cases.items() if cid in unresolved and cid not in resolved_humans}
    texts = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet")}
    result = []

    def add(kind, case, payload, image=None):
        item = {"kind": kind, "case_id": case["case_id"],
                "candidate_disease": case["candidate_disease"],
                "raw_text_sha256": case["raw_text_sha256"], "model": cfg["model"],
                "prompt_version": cfg["prompt_version"], "payload": payload,
                "schema": SCHEMAS[kind], "system": SYSTEM,
                "max_output_tokens": cfg["max_output_tokens"], "local_dtype": cfg.get("local_dtype", "bfloat16")}
        item["runtime"] = cfg.get("runtime", "transformers")
        if cfg.get("runtime") == "gguf":
            item["quantized_model"] = {"repo": cfg["gguf_repo"], "filename": cfg["gguf_filename"]}
        if image:
            item.update(image_id=image["image_id"], image_path=image["path"],
                        image_sha256=image["image_sha256"])
        item["request_sha256"] = digest(canonical(item))
        result.append(item)

    for cid, case in cases.items():
        add("label", case, label_payload(case))
        if cfg.get("scope", "human_conflicts") == "human_conflicts":
            continue
        text = texts.get(cid)
        if text and text["deterministic_pass"]:
            add("text", case, text_payload(case, text))
    for image in project.rows("data/qc/image_qc.parquet"):
        case = cases.get(image["case_id"])
        if cfg.get("scope", "human_conflicts") != "human_conflicts" and case and image["decode_pass"]:
            add("image", case, image_payload(case, image), image)
    return result


def cache_path(project, key):
    return project.path(f"data/reviews/medgemma_cache/{key}.json")


def validate_result(job, result):
    jsonschema.validate(result, SCHEMAS[job["kind"]])
    if (job["kind"] == "label" and result["decision"] == "accept"
            and (result["diagnosis_status"] != "confirmed" or not result["disease_is_current"]
                or result["multiple_target_diseases"] or result["other_target_diseases"]
                or unique_quote_span(job["payload"]["raw_case_text"], result["evidence_quote"]) is None)):
        raise PipelineError("MedGemma acceptance requires unique literal evidence and one confirmed current target")
    if job["kind"] == "text":
        span = result["leaking_span"]
        if result["has_direct_leakage"] and (not span or span not in job["payload"]["clinical_text"]):
            raise PipelineError("MedGemma leakage span must be a literal clinical-text substring")


def export_jobs(project, output="data/reviews/medgemma_input.jsonl"):
    items = jobs(project)
    path = project.path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for job in items:
            stream.write(canonical(job) + "\n")
    temp.replace(path)
    return {"jobs": len(items), "by_kind": dict(Counter(j["kind"] for j in items)), "output": str(path)}


def import_results(project, input_path):
    path = Path(input_path)
    if not path.is_absolute():
        path = project.path(input_path)
    expected = {j["request_sha256"]: j for j in jobs(project)}
    staged = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        job = expected.get(row.get("request_sha256"))
        if not job or row.get("model") != job["model"]:
            raise PipelineError(f"Unknown/stale request or wrong model on line {line_number}")
        validate_result(job, row["result"])
        key = job["request_sha256"]
        if key in staged and staged[key]["result"] != row["result"]:
            raise PipelineError(f"Conflicting duplicate result on line {line_number}")
        saved = {**job, "result": row["result"], "reviewed_at": row.get("reviewed_at", now()),
                 "model_version": row.get("model_version", row["model"]), "imported_at": now()}
        existing = cache_path(project, key)
        if existing.exists() and read_json(existing)["result"] != row["result"]:
            raise PipelineError(f"Conflicting immutable MedGemma result: {key}")
        staged[key] = saved
    for key, saved in staged.items():
        path = cache_path(project, key)
        if not path.exists():
            write_json(path, saved, immutable=True)
    return apply_results(project)


def run(project, limit=25):
    load_dotenv(project.path(".env"), override=False)
    base = os.getenv("MEDGEMMA_BASE_URL", "").rstrip("/")
    if not base:
        raise PipelineError("Set MEDGEMMA_BASE_URL for a MedGemma chat-completions server, or use medgemma-import")
    if not base.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise PipelineError("MedGemma endpoint must use HTTPS or a local HTTP server")
    completed = 0
    all_jobs = jobs(project)
    pending_count = sum(not cache_path(project, j["request_sha256"]).exists() for j in all_jobs)
    started = time.monotonic()
    for job in all_jobs:
        cache = cache_path(project, job["request_sha256"])
        if cache.exists():
            continue
        if completed >= limit:
            break
        content = [{"type": "text", "text": SYSTEM + "\n" + canonical(job["payload"])
                    + "\nReturn only JSON matching this schema: " + canonical(job["schema"])}]
        if job["kind"] == "image":
            from .image_qc import review_image_bytes
            path = project.path(job["image_path"])
            if file_hash(path) != job["image_sha256"]:
                raise PipelineError("Source image changed since QC; rerun image-qc")
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                           + base64.b64encode(review_image_bytes(path)).decode("ascii")}})
        body = {"model": job["model"], "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": job["max_output_tokens"],
                "response_format": {"type": "json_object", "schema": job["schema"]}}
        headers = {"Content-Type": "application/json"}
        if os.getenv("MEDGEMMA_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["MEDGEMMA_API_KEY"]
        attempts = []
        cfg = project.config("medgemma")
        for attempt in range(cfg.get("max_attempts", 3)):
            request = urllib.request.Request(base + "/chat/completions", canonical(body).encode(), headers)
            try:
                with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                    raw = json.load(response)
                attempts.append({"messages": body["messages"], "response": raw})
                choice = raw["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise PipelineError("MedGemma response incomplete; no review accepted")
                result = json.loads(choice["message"]["content"])
                validate_result(job, result)
                break
            except urllib.error.HTTPError as exc:
                raise PipelineError(f"MedGemma HTTP {exc.code}; check endpoint/model access") from None
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
                    json.JSONDecodeError, jsonschema.ValidationError, PipelineError) as exc:
                write_json(project.path(f"data/reviews/medgemma_failures/{job['request_sha256']}.json"),
                           {"request_sha256": job["request_sha256"], "case_id": job["case_id"],
                            "attempts": attempts, "error": str(exc), "failed_at": now()})
                if attempt + 1 == cfg.get("max_attempts", 3):
                    raise PipelineError(f"MedGemma response failed validation for {job['case_id']}; audit saved") from None
                body = {**body, "messages": [{"role": "user", "content": content + [
                    {"type": "text", "text": "A previous attempt failed validation: " + str(exc)[:500]
                     + ". Recheck the source. If the diagnosis cannot be supported by a unique literal quote, "
                       "return uncertain rather than accept. Return JSON matching the schema."}]}]}
        write_json(cache, {**job, "result": result, "reviewed_at": now(),
                          "model_version": raw.get("model", job["model"]), "raw_response": raw,
                          "attempts": attempts,
                          "inference_artifact": read_json(project.path("models/medgemma_local.lock.json"))
                          if job.get("runtime") == "gguf" and project.path("models/medgemma_local.lock.json").exists()
                          else None}, immutable=True)
        completed += 1
        write_json(project.path("reports/medgemma_runtime.json"),
                   {"status": "RUNNING", "scope": "human_conflicts" if len(all_jobs) <= 1170 else "configured",
                    "total_jobs": len(all_jobs), "completed_total": len(all_jobs) - pending_count + completed,
                    "pending_total": pending_count - completed, "last_case_id": job["case_id"],
                    "elapsed_seconds": round(time.monotonic() - started, 1), "updated_at": now()})
        print(f"MedGemma: {completed}/{limit} {job['kind']} {job['case_id']}", flush=True)
    summary = {"new_reviews": completed, **apply_results(project)}
    write_json(project.path("reports/medgemma_runtime.json"),
               {"status": "AI_COMPLETE" if summary["ai_complete"] else "BATCH_COMPLETE", **summary})
    return summary


def apply_results(project):
    from .image_qc import selection_key

    texts = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet")}
    images = {r["image_id"]: r for r in project.rows("data/qc/image_qc.parquet")}
    labels, counts = [], Counter()
    pending = Counter()
    for job in jobs(project):
        path = cache_path(project, job["request_sha256"])
        if not path.exists():
            pending[job["kind"]] += 1
            if job["kind"] != "label":
                row = texts[job["case_id"]] if job["kind"] == "text" else images[job["image_id"]]
                if row.get("ai_provider") == "medgemma":
                    row.update(ai_request_sha256=None, ai_reason=None)
                    if job["kind"] == "text":
                        row.update(text_usable=False, has_direct_leakage=None, leaking_span=None,
                                   usable_prediagnosis_context=None, reason="TEXT_AI_REVIEW_REQUIRED")
                    else:
                        row.update(image_usable=False, image_relevance="unclear", visible_diagnosis_text=None,
                                   case_image_consistent=None, reason="IMAGE_AI_REVIEW_REQUIRED")
            continue
        saved = read_json(path)
        if any(saved.get(k) != job[k] for k in job):
            raise PipelineError("MedGemma cache identity mismatch")
        result = saved["result"]
        validate_result(job, result)
        counts[job["kind"]] += 1
        if job["kind"] == "label":
            span = unique_quote_span(job["payload"]["raw_case_text"], result["evidence_quote"])
            labels.append({**result, **{k: job[k] for k in
                          ("case_id", "candidate_disease", "raw_text_sha256", "request_sha256", "model", "prompt_version")},
                           "evidence_valid": span is not None,
                           "evidence_start": span[0] if span else None, "evidence_end": span[1] if span else None,
                           "reviewed_at": saved["reviewed_at"]})
            continue
        row = texts[job["case_id"]] if job["kind"] == "text" else images[job["image_id"]]
        row.update({k: v for k, v in result.items() if k != "reason"})
        row.update(ai_reason=result["reason"], ai_request_sha256=job["request_sha256"],
                   ai_provider="medgemma", ai_model=job["model"])
        if job["kind"] == "text":
            row["text_usable"] = bool(row["deterministic_pass"] and not result["has_direct_leakage"]
                                      and result["usable_prediagnosis_context"])
            row["reason"] = "PASS" if row["text_usable"] else "TEXT_AI_REJECTED"
        else:
            if file_hash(project.path(row["path"])) != job["image_sha256"]:
                raise PipelineError("Source image changed since QC")
            row["image_usable"] = bool(row["decode_pass"] and result["image_relevance"] in {"diagnostic", "supportive"}
                                       and not result["visible_diagnosis_text"] and result["case_image_consistent"])
            row["reason"] = "PASS" if row["image_usable"] else "IMAGE_AI_REJECTED"
    project.write("data/reviews/medgemma_labels.parquet", labels)
    project.write("data/qc/text_qc.parquet", list(texts.values()))
    project.write("data/qc/image_qc.parquet", list(images.values()))
    selected = {}
    for image in sorted([r for r in images.values() if r["image_usable"]],
                        key=lambda r: selection_key(r, project.config("qc"))):
        selected.setdefault(image["case_id"], image)
    project.write("data/qc/selected_images.parquet", list(selected.values()))
    summary = {"completed": dict(counts), "pending": dict(pending),
               "ai_complete": not pending, "selected_cases": len(selected), "updated_at": now()}
    write_json(project.path("reports/medgemma_progress.json"), summary)
    return summary


def run_local(project, limit=25):
    """Load official weights once; persist each validated response before the next job."""
    if project.config("medgemma").get("runtime") == "gguf":
        return run_gguf(project, limit)
    load_dotenv(project.path(".env"), override=False)
    try:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    except ImportError:
        raise PipelineError("Install local inference dependencies: python -m pip install '.[medgemma]'") from None
    cfg = project.config("medgemma")
    cache_dir = project.path("models/huggingface")
    os.environ["HF_HOME"] = str(cache_dir)
    pending = [job for job in jobs(project) if not cache_path(project, job["request_sha256"]).exists()][:limit]
    if not pending:
        return {"new_reviews": 0, **apply_results(project)}
    token = os.getenv("HF_TOKEN")
    kwargs = {"cache_dir": str(cache_dir), "token": token}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_json(project.path("reports/medgemma_runtime.json"),
               {"status": "LOADING_MODEL", "backend": "local", "device": device,
                "planned_this_run": len(pending), "updated_at": now()})
    print(f"Loading {cfg['model']} on {device}; weights cached under {cache_dir}", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(cfg["model"], **kwargs)
        dtype_name = cfg.get("local_dtype", "bfloat16")
        if dtype_name not in {"bfloat16", "float32"}:
            raise PipelineError("local_dtype must be bfloat16 or float32")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg["model"], device_map=device, torch_dtype=getattr(torch, dtype_name),
            **kwargs).eval()
    except OSError:
        write_json(project.path("reports/medgemma_runtime.json"),
                   {"status": "MODEL_LOAD_FAILED", "updated_at": now()})
        raise PipelineError("Cannot load MedGemma weights. Accept model access on Hugging Face and set HF_TOKEN in .env.") from None
    completed = 0
    started = time.monotonic()
    for job in pending:
        content = [{"type": "text", "text": SYSTEM + "\n" + canonical(job["payload"])
                    + "\nReturn only JSON matching this schema: " + canonical(job["schema"])}]
        if job["kind"] == "image":
            from PIL import Image, ImageOps
            image_path = project.path(job["image_path"])
            if file_hash(image_path) != job["image_sha256"]:
                raise PipelineError("Source image changed since QC")
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((3072, 3072))
            content.insert(0, {"type": "image", "image": image})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=getattr(torch, dtype_name))
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=job["max_output_tokens"], do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = processor.decode(generated, skip_special_tokens=True).strip()
        # Some instruction models wrap otherwise valid JSON in a Markdown fence.
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        try:
            result = json.loads(text)
            validate_result(job, result)
        except (json.JSONDecodeError, jsonschema.ValidationError, PipelineError) as exc:
            write_json(project.path(f"data/reviews/medgemma_failures/{job['request_sha256']}.json"),
                       {"request_sha256": job["request_sha256"], "raw_output": text,
                        "error": str(exc), "failed_at": now()})
            write_json(project.path("reports/medgemma_runtime.json"),
                       {"status": "INVALID_MODEL_RESPONSE", "case_id": job["case_id"],
                        "completed_this_run": completed, "updated_at": now()})
            raise PipelineError(f"Invalid MedGemma response for {job['case_id']}; saved for inspection, no review accepted") from None
        write_json(cache_path(project, job["request_sha256"]),
                   {**job, "result": result, "reviewed_at": now(), "model_version": cfg["model"],
                    "weights_revision": getattr(model.config, "_commit_hash", None), "raw_output": text}, immutable=True)
        completed += 1
        write_json(project.path("reports/medgemma_runtime.json"),
                   {"status": "RUNNING", "backend": "local", "device": device, "dtype": dtype_name,
                    "completed_this_run": completed, "planned_this_run": len(pending),
                    "last_kind": job["kind"], "last_case_id": job["case_id"],
                    "elapsed_seconds": round(time.monotonic() - started, 1), "updated_at": now()})
        print(f"MedGemma local: {completed}/{len(pending)} {job['kind']} {job['case_id']}", flush=True)
    summary = {"new_reviews": completed, **apply_results(project)}
    write_json(project.path("reports/medgemma_runtime.json"),
               {"status": "AI_COMPLETE" if summary["ai_complete"] else "BATCH_COMPLETE",
                "backend": "local", "device": device, "dtype": dtype_name,
                "completed_this_run": completed, "updated_at": now()})
    return summary


def run_gguf(project, limit=25):
    import subprocess

    cfg = project.config("medgemma")
    lock_path = project.path("models/medgemma_local.lock.json")
    if not lock_path.exists():
        raise PipelineError("Run python scripts/setup_medgemma_local.py to install the local Q4 runtime")
    lock = read_json(lock_path)
    if lock["gguf_repo"] != cfg["gguf_repo"] or lock["filename"] != cfg["gguf_filename"]:
        raise PipelineError("Local model lock does not match the configured MedGemma")
    model = project.path(lock["model_path"])
    if file_hash(model) != lock["model_sha256"]:
        raise PipelineError("Local GGUF model checksum mismatch")
    binaries = list(project.path("models/llama_cpp").rglob("llama-server.exe"))
    if len(binaries) != 1:
        raise PipelineError("Expected one local llama-server.exe")
    port = cfg["local_port"]
    base = f"http://127.0.0.1:{port}/v1"
    log_path = project.path(".tmp/medgemma_server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([str(binaries[0]), "--model", str(model), "--alias", cfg["model"],
                                    "--host", "127.0.0.1", "--port", str(port), "--ctx-size", "16384",
                                    "--parallel", "1", "--n-gpu-layers", "0", "--threads", "6", "--jinja"],
                                   stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        previous_base = os.environ.get("MEDGEMMA_BASE_URL")
        try:
            ready = False
            for _ in range(120):
                if process.poll() is not None:
                    raise PipelineError("Local llama.cpp server exited; inspect .tmp/medgemma_server.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        ready = response.status == 200
                except (urllib.error.URLError, TimeoutError):
                    pass
                if ready:
                    break
                time.sleep(1)
            if not ready:
                raise PipelineError("Local MedGemma server did not become ready")
            os.environ["MEDGEMMA_BASE_URL"] = base
            result = run(project, limit)
            write_json(project.path("reports/medgemma_local_runtime.json"), {"backend": "llama.cpp",
                       "model_artifact": lock, "result": result, "updated_at": now()})
            return result
        finally:
            if previous_base is None:
                os.environ.pop("MEDGEMMA_BASE_URL", None)
            else:
                os.environ["MEDGEMMA_BASE_URL"] = previous_base
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

from __future__ import annotations

import re
from collections import Counter

from .common import Project, digest
from .discover import Matcher
from .evidence import sentence_spans
from .llm_review import Gemini, review_key
from .review_schema import TEXT_REVIEW
from .sample import single_targets

POST_DIAGNOSIS = re.compile(
    r"\b(?:diagnos\w*|confirmed|positive for|treated|treatment|therapy|therapeutic|"
    r"administered|prescribed|discharged|follow.up|recovered|recovery|died|"
    r"antibiotic\w*|antiviral\w*|antifungal\w*|antituberc\w*)\b", re.IGNORECASE)


def clean_text(row: dict, matcher: Matcher, minimum_words: int, title_words: int = 8) -> dict:
    text = row["raw_case_text"]
    cutoff = row["evidence_start"]
    for start, end in sentence_spans(text):
        if start >= cutoff:
            break
        # Retain a source prefix only; no generated or paraphrased clinical input.
        if matcher.find(text[start:end]) or POST_DIAGNOSIS.search(text[start:end]):
            cutoff = start
            break
    clinical = text[:cutoff].strip()
    hits = matcher.find(clinical)
    title_tokens = re.findall(r"\w+", row.get("title", "").lower())
    clinical_tokens = " ".join(re.findall(r"\w+", clinical.lower()))
    title_leak = any(" ".join(title_tokens[i:i + title_words]) in clinical_tokens
                     for i in range(max(0, len(title_tokens) - title_words + 1)))
    reason = "PASS"
    if len(clinical.split()) < minimum_words:
        reason = "TEXT_TOO_SHORT"
    if hits or (row["diagnosis_evidence"].strip() and row["diagnosis_evidence"].strip() in clinical) or title_leak:
        reason = "TEXT_LABEL_LEAKAGE"
    return {"case_id": row["case_id"], "raw_text_sha256": row["raw_text_sha256"],
            "clinical_text": clinical, "clinical_text_sha256": digest(clinical),
            "clinical_source_end": cutoff, "deterministic_pass": reason == "PASS", "reason": reason}


def text_payload(row, qc):
    return {"task": "Audit model input for direct diagnosis leakage, including pathogens, abbreviations, revealing treatments and post-diagnosis outcomes. Do not predict the disease.",
            "target_disease": row["disease"], "clinical_text": qc["clinical_text"],
            "instructions": "has_direct_leakage must include explicit diagnostic terminology or pathogen names revealing the target. leaking_span must be a literal input substring if leakage is found. Assess whether useful prediagnosis clinical context remains."}


def text_qc(project: Project, use_ai: bool = False, limit: int = 25) -> dict:
    terms = [r for r in project.rows("data/catalog/disease_catalog.parquet") if r["in_scope"]]
    matcher = Matcher(terms)
    cfg, review_cfg = project.config("qc"), project.config("review")
    previous = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet", optional=True)}
    rows, requests = [], 0
    client = Gemini(project) if use_ai else None
    for row in single_targets(project):
        qc = clean_text(row, matcher, cfg["minimum_clinical_words"], cfg["title_fragment_words"])
        payload = text_payload(row, qc)
        key = review_key("text_qc", payload, review_cfg["qc_model"], review_cfg)
        old = previous.get(row["case_id"], {})
        result = None
        if old.get("ai_request_sha256") == key:
            result = {k: old[k] for k in ["has_direct_leakage", "leaking_span", "usable_prediagnosis_context", "ai_reason"]}
        elif client and qc["deterministic_pass"] and requests < limit:
            saved = client.run("text_qc", payload, TEXT_REVIEW, review_cfg["qc_model"])
            result = dict(saved["result"])
            result["ai_reason"] = result.pop("reason")
            requests += 1
        qc.update({"ai_request_sha256": key if result else None,
                   "has_direct_leakage": result["has_direct_leakage"] if result else None,
                   "leaking_span": result["leaking_span"] if result else None,
                   "usable_prediagnosis_context": result["usable_prediagnosis_context"] if result else None,
                   "ai_reason": result["ai_reason"] if result else None})
        qc["text_usable"] = qc["deterministic_pass"] and bool(result) and not result["has_direct_leakage"] and result["usable_prediagnosis_context"]
        if qc["deterministic_pass"] and not qc["text_usable"]:
            qc["reason"] = "TEXT_AI_REVIEW_REQUIRED" if not result else "TEXT_AI_REJECTED"
        rows.append(qc)
    project.write("data/qc/text_qc.parquet", rows)
    return {"new_ai_reviews": requests, "counts": dict(Counter(r["reason"] for r in rows))}


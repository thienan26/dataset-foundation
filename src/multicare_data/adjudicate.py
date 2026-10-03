from __future__ import annotations

import json
from collections import Counter

import jsonschema

from .common import PipelineError, Project, canonical, digest, now
from .evidence import unique_quote_span
from .llm_review import accepts, current_reviews, needs_b
from .review_schema import HUMAN_REVIEW
from .sample import single_targets


def human_records(project: Project) -> dict:
    rows = project.rows("data/reviews/human_reviews.parquet", optional=True)
    result = {}
    for row in sorted(rows, key=lambda r: r["imported_at"]):
        result[row["case_id"]] = row
    return result


def valid_human(row, human):
    return (human and human["raw_text_sha256"] == row["raw_text_sha256"]
            and human["candidate_disease"] == row["candidate_disease"])


def human_accepts(row, human):
    return (valid_human(row, human) and human["human_decision"] == "accept"
            and human["human_status"] == "confirmed" and human["human_evidence_valid"]
            and not human["multiple_target_diseases"]
            and not human["text_leakage_found"] and not human["image_label_leakage"]
            and human["image_relevance"] in {"diagnostic", "supportive"}
            and unique_quote_span(row["raw_case_text"], human["evidence_quote"]) is not None)


def human_label_accepts(row, human):
    return (valid_human(row, human) and human["human_decision"] == "accept"
            and human["human_status"] == "confirmed" and human["human_evidence_valid"]
            and not human["multiple_target_diseases"]
            and unique_quote_span(row["raw_case_text"], human["evidence_quote"]) is not None)


def import_human(project: Project, path) -> dict:
    cases = {r["case_id"]: r for r in single_targets(project)}
    texts = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet")}
    images = {r["case_id"]: r for r in project.rows("data/qc/selected_images.parquet")}
    requirements = {r["case_id"]: r for r in project.rows("data/reviews/human_requirements.parquet")}
    existing = project.rows("data/reviews/human_reviews.parquet", optional=True)
    ids = {r["record_sha256"] for r in existing}
    additions = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        context_fields = {"raw_case_text", "proposed_evidence", "clinical_text", "image_path",
                          "image_relevance_proposal", "image_caption_for_context"}
        allowed = set(HUMAN_REVIEW["properties"]) | context_fields
        if set(row) - allowed:
            raise PipelineError(f"Unexpected fields in human review: {sorted(set(row) - allowed)}")
        annotation = {k: v for k, v in row.items() if k in HUMAN_REVIEW["properties"]}
        jsonschema.validate(annotation, HUMAN_REVIEW, format_checker=jsonschema.FormatChecker())
        row = annotation
        case = cases.get(row["case_id"])
        if not case or not valid_human(case, row) or row["case_id"] not in requirements:
            raise PipelineError(f"Human review has wrong case, label or source hash: {row['case_id']}")
        text, image, required = texts.get(row["case_id"]), images.get(row["case_id"]), requirements[row["case_id"]]
        if (not text or row["clinical_text_sha256"] != text["clinical_text_sha256"] or not image
                or row["selected_image_id"] != image["image_id"]
                or row["image_sha256"] != image["image_sha256"] or not required["required"]):
            raise PipelineError(f"Human review does not match the selected input snapshot: {row['case_id']}")
        if row["human_decision"] == "accept" and not human_accepts(case, row):
            raise PipelineError("Human acceptance requires one confirmed target and unique exact evidence")
        identity = digest(canonical(row))
        if identity not in ids:
            additions.append({**row, "record_sha256": identity, "imported_at": now()})
            ids.add(identity)
    project.write("data/reviews/human_reviews.parquet", existing + additions)
    return {"imported": len(additions), "total": len(existing) + len(additions)}


def prepare_human(project: Project, export_path=None) -> dict:
    from .common import digest
    # Pre-split disagreements can still be sent to human adjudication; regular
    # development/test audits are queued once grouped split assignments exist.
    split = {r["case_id"]: r for r in project.rows("data/splits/assignments.parquet", optional=True)}
    prior_requirements = {r["case_id"]: r for r in
                          project.rows("data/reviews/human_requirements.parquet", optional=True)}
    candidates = project.rows("data/qc/feasible_candidates.parquet")
    small_all = len(candidates) <= project.config("release")["human_review_all_below"]
    cfg = project.config("release")
    humans = human_records(project)
    requirements, queue = [], []
    originals = {r["case_id"]: r for r in single_targets(project)}
    image_map = {r["case_id"]: r for r in project.rows("data/qc/selected_images.parquet")}
    text_map = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet")}
    adjudication = {r["case_id"]: r for r in project.rows("data/reviews/adjudication.parquet")}
    manual_adjudication_ids = set()
    # Both pipeline-generated disagreements and imported A/B consensus conflicts
    # enter the human queue once deterministic text QC and image selection pass.
    for cid, review in adjudication.items():
        if (review["reason"] == "HUMAN_ADJUDICATION_REQUIRED"
                and cid in originals and cid in image_map and text_map.get(cid, {}).get("deterministic_pass")):
            manual_adjudication_ids.add(cid)
    manual_adjudication_ids.update(
        cid for cid, requirement in prior_requirements.items()
        if requirement.get("consensus") == "NEEDS_ADJUDICATION"
        and cid in originals and cid in image_map and text_map.get(cid, {}).get("deterministic_pass")
    )
    candidate_by_id = {row["case_id"]: row for row in candidates}
    for cid in manual_adjudication_ids:
        if cid in candidate_by_id:
            candidate_by_id[cid]["manual_adjudication_required"] = True
            continue
        original, text = originals[cid], text_map[cid]
        candidate_by_id[cid] = {**original, **text, "candidate_disease": original["candidate_disease"],
                                "manual_adjudication_required": True,
                                "partition": "adjudication_queue", "fold": None}
    candidates = list(candidate_by_id.values())
    for item in candidates:
        assignment = split.get(item["case_id"])
        if not assignment and item.get("manual_adjudication_required"):
            assignment = {"partition": "adjudication_queue", "fold": None}
        if not assignment:
            continue
        cid = item["case_id"]
        sample_fraction = int(digest(cid + ":human-v001")[:8], 16) / 0x100000000
        consensus_required = prior_requirements.get(cid, {}).get("consensus") == "NEEDS_ADJUDICATION"
        audit_required = (small_all or assignment["partition"] == "frozen_test"
                          or sample_fraction < cfg["human_development_fraction"])
        required = item.get("manual_adjudication_required", False) or consensus_required or audit_required
        existing = humans.get(cid)
        if required and not existing:
            original = originals[cid]
            image = image_map[cid]
            text = text_map[cid]
            queue.append({"case_id": cid, "candidate_disease": item["candidate_disease"],
                          "raw_text_sha256": original["raw_text_sha256"],
                          "raw_case_text": original["raw_case_text"],
                          "proposed_evidence": original["diagnosis_evidence"],
                          "clinical_text": text["clinical_text"],
                          "clinical_text_sha256": text["clinical_text_sha256"],
                          "selected_image_id": image["image_id"], "image_path": image["path"],
                          "image_sha256": image["image_sha256"], "image_relevance_proposal": image["image_relevance"],
                          "image_caption_for_context": image["caption"],
                          "human_status": "", "human_decision": "", "human_evidence_valid": False,
                          "evidence_quote": "", "multiple_target_diseases": False,
                          "text_leakage_found": False, "image_relevance": "unclear",
                          "image_label_leakage": False, "reason_code": "",
                          "reviewer_id": "", "reviewed_at": ""})
        prior = prior_requirements.get(cid, {})
        reasons = []
        if consensus_required:
            reasons.append("CONSENSUS_NEEDS_ADJUDICATION")
        if item.get("manual_adjudication_required") and not consensus_required:
            reasons.append("HUMAN_ADJUDICATION_REQUIRED")
        if audit_required:
            reasons.append("RELEASE_AUDIT")
        requirements.append({**prior, "case_id": cid, "partition": assignment["partition"],
                             "fold": assignment["fold"], "required": required,
                             "consensus_required": consensus_required,
                             "release_audit_required": audit_required,
                             "requirement_reason": ";".join(reasons) or prior.get("requirement_reason", "")})

    # Preserve the complete imported consensus register, including cases that
    # have not passed text/image QC or do not yet have split assignments.
    requirement_ids = {row["case_id"] for row in requirements}
    for cid, prior in prior_requirements.items():
        if cid in requirement_ids:
            continue
        consensus_required = prior.get("consensus") == "NEEDS_ADJUDICATION"
        requirements.append({**prior,
                             "consensus_required": consensus_required,
                             "release_audit_required": False,
                             "required": bool(prior.get("required")) or consensus_required,
                             "requirement_reason": ("CONSENSUS_NEEDS_ADJUDICATION" if consensus_required
                                                    else prior.get("requirement_reason", ""))})
    project.write("data/reviews/human_requirements.parquet", requirements)
    project.write("data/reviews/human_queue.parquet", queue)
    if export_path:
        import json
        export_path.parent.mkdir(parents=True, exist_ok=True)
        temp = export_path.with_suffix(export_path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            for row in queue:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(export_path)
    return {"required_human_reviews": len(requirements) - sum(not r["required"] for r in requirements),
            "awaiting_human_reviews": len(queue), "export": str(export_path) if export_path else None}


def adjudicate(project: Project) -> dict:
    samples = single_targets(project)
    a, b = current_reviews(project, "a", samples), current_reviews(project, "b", samples)
    humans, cfg = human_records(project), project.config("review")
    third_reviews = {}
    if project.path("configs/medgemma.yaml").exists():
        from .medgemma import jobs
        third_keys = {r["case_id"]: r["request_sha256"] for r in jobs(project) if r["kind"] == "label"}
        third_reviews = {r["case_id"]: r for r in
                         project.rows("data/reviews/medgemma_labels.parquet", optional=True)
                         if r.get("request_sha256") == third_keys.get(r["case_id"])}
    label_humans = {}
    for item in sorted(project.rows("data/reviews/human_label_adjudications.parquet", optional=True),
                       key=lambda row: row.get("imported_at", "")):
        label_humans[item["case_id"]] = item
    rows = []
    for case in samples:
        cid = case["case_id"]
        ra, rb, human, label_human = a.get(cid), b.get(cid), humans.get(cid), label_humans.get(cid)
        need_b = needs_b(case, ra, cfg)
        status, reason = "pending", "REVIEW_A_REQUIRED"
        if ra:
            use_b = need_b or rb is not None
            if use_b and not rb:
                reason = "REVIEW_B_REQUIRED"
            elif ra["decision"] == "reject" and rb and rb["decision"] == "reject":
                status, reason = "rejected", "REVIEWER_CONSENSUS_REJECTED"
            elif accepts(ra, case) and (not use_b or accepts(rb, case)):
                status, reason = "verified_silver", "AUTOMATED_CONSENSUS"
            else:
                reason = "HUMAN_ADJUDICATION_REQUIRED"
        third = third_reviews.get(cid)
        if third and status == "verified_silver" and not accepts(third, case):
            status, reason = "pending", "HUMAN_ADJUDICATION_REQUIRED"
        if valid_human(case, label_human):
            if label_human["human_decision"] == "reject":
                status, reason = "rejected", "HUMAN_LABEL_ADJUDICATED_REJECTED"
            elif human_label_accepts(case, label_human):
                status, reason = "human_reviewed", "HUMAN_LABEL_ADJUDICATED"
            else:
                status, reason = "pending", "HUMAN_ADJUDICATION_REQUIRED"
        if valid_human(case, human):
            if human["human_decision"] == "reject":
                status, reason = "rejected", "HUMAN_REJECTED"
            elif ra and (not need_b or rb) and human_accepts(case, human):
                status, reason = "human_reviewed", "HUMAN_ACCEPTED"
        final_human = human if valid_human(case, human) else label_human
        rows.append({"case_id": cid, "candidate_disease": case["candidate_disease"],
                     "review_status": status, "reason": reason, "needs_b": need_b,
                     "llm_a_decision": ra["decision"] if ra else "pending",
                     "llm_b_decision": rb["decision"] if rb else "not_required" if not need_b else "pending",
                     "human_decision": final_human["human_decision"] if valid_human(case, final_human) else "pending",
                     "verified": status in {"verified_silver", "human_reviewed"}})
    project.write("data/reviews/adjudication.parquet", rows)
    project.write("data/reviews/human_queue.parquet", [r for r in rows if r["reason"] == "HUMAN_ADJUDICATION_REQUIRED"])
    return dict(Counter(r["reason"] for r in rows))

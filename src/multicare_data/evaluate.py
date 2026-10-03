from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .common import PipelineError, Project, digest, file_hash, write_json
from .evidence import exact_evidence
from .sample import single_targets


def evaluate(project: Project, frozen=False) -> dict:
    cfg = project.config("source")
    try:
        from .acquire import verify_source
        verify_source(project, require_images=True)
        source_ok = True
    except PipelineError:
        source_ok = False
    manifest = project.rows("data/final/manifest.parquet" if frozen else "data/qc/feasible_candidates.parquet", optional=frozen)
    cases = {r["case_id"]: r for r in single_targets(project)}
    labels = {r["case_id"]: r for r in project.rows("data/screening/labels.parquet")}
    texts = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet", optional=True)}
    selected = {r["case_id"]: r for r in project.rows("data/qc/selected_images.parquet", optional=True)}
    group_rows = {r["case_id"]: r for r in project.rows("data/groups/groups.parquet", optional=True)}
    review = {r["case_id"]: r for r in project.rows("data/reviews/adjudication.parquet", optional=True)}
    humans = {r["case_id"]: r for r in project.rows("data/reviews/human_reviews.parquet", optional=True)}
    human_requirements = {r["case_id"]: r for r in project.rows("data/reviews/human_requirements.parquet", optional=True)}
    splits = {r["case_id"]: r for r in project.rows("data/splits/assignments.parquet", optional=True)}
    errors = Counter()
    if not source_ok:
        errors["SOURCE_VERIFICATION_FAILED"] += 1
    if not manifest:
        errors["NO_FEASIBLE_RELEASE_CASES"] += 1
    for row in manifest:
        cid = row["case_id"]
        sample, label = cases.get(cid), labels.get(cid)
        if not sample or not label:
            errors["MISSING_SOURCE_CASE_OR_LABEL"] += 1
            continue
        if label["confirmed_target_count"] != 1 or label["diagnosis_status"] != "confirmed":
            errors["SINGLE_CONFIRMED_LABEL_FAILURE"] += 1
        if not exact_evidence(sample["raw_case_text"], label["diagnosis_evidence"],
                              label["evidence_start"], label["evidence_end"]):
            errors["EVIDENCE_SPAN_INVALID"] += 1
        if not row.get("text_usable") or not row.get("clinical_text"):
            errors["TEXT_NOT_USABLE"] += 1
        if digest(row["clinical_text"]) != row.get("clinical_text_sha256"):
            errors["CLINICAL_TEXT_HASH_MISMATCH"] += 1
        chosen = selected.get(cid)
        if not chosen or not chosen.get("image_usable"):
            errors["IMAGE_QC_FAILED"] += 1
        elif (not Path(project.path(chosen["path"])).is_file()
              or file_hash(project.path(chosen["path"])) != chosen["image_sha256"]):
            errors["IMAGE_HASH_MISMATCH"] += 1
        if group_rows.get(cid, {}).get("duplicate_conflict"):
            errors["DUPLICATE_CONFLICT"] += 1
        if row.get("source_license") in {None, "UNKNOWN", "NO-CC CODE", ""}:
            errors["LICENSE_UNKNOWN"] += 1
        if chosen and chosen.get("license") in {None, "UNKNOWN", "NO-CC CODE", ""}:
            errors["IMAGE_LICENSE_UNKNOWN"] += 1
        if not review.get(cid, {}).get("verified"):
            errors["REVIEW_PENDING"] += 1
        human = humans.get(cid)
        required = human_requirements.get(cid, {}).get("required", False)
        if required and not human:
            errors["HUMAN_REVIEW_MISSING"] += 1
        elif required and (not human.get("human_evidence_valid") or human.get("human_decision") != "accept"
              or human.get("human_status") != "confirmed" or human.get("multiple_target_diseases")
              or human.get("text_leakage_found") or human.get("image_label_leakage")
              or human.get("image_relevance") not in {"diagnostic", "supportive"}):
            errors["HUMAN_AUDIT_FAILED"] += 1
        split = splits.get(cid)
        if not split:
            errors["SPLIT_MISSING"] += 1
        elif split.get("partition") == "frozen_test" and not required:
            errors["FROZEN_TEST_NOT_HUMAN_REVIEWED"] += 1
    if frozen:
        partitions = defaultdict(list)
        for row in manifest:
            partitions[row["partition"]].append(row)
        dev = partitions["development"]
        test = partitions["frozen_test"]
        for field in ("case_id", "article_id", "split_group", "raw_text_sha256", "clinical_text_sha256", "image_sha256", "pixel_sha256"):
            if {r.get(field) for r in dev} & {r.get(field) for r in test}:
                errors[f"SPLIT_{field.upper()}_OVERLAP"] += 1
        labels_in_test = {r["diagnosis_label"] for r in test}
        labels_in_dev = {r["diagnosis_label"] for r in dev}
        if labels_in_test != labels_in_dev:
            errors["SPLIT_CLASS_COVERAGE"] += 1
        for fold in sorted({r.get("fold") for r in dev if r.get("fold") is not None}):
            fold_cases = [r for r in dev if r["fold"] == fold]
            if not fold_cases or {r["diagnosis_label"] for r in fold_cases} != labels_in_dev:
                errors["SPLIT_FOLD_CLASS_COVERAGE"] += 1
        for i in sorted({r.get("fold") for r in dev if r.get("fold") is not None}):
            for j in sorted({r.get("fold") for r in dev if r.get("fold") is not None and r.get("fold") > i}):
                left = [r for r in dev if r["fold"] == i]
                right = [r for r in dev if r["fold"] == j]
                for field in ("split_group", "article_id", "raw_text_sha256", "clinical_text_sha256", "image_sha256", "pixel_sha256"):
                    if {r.get(field) for r in left} & {r.get(field) for r in right}:
                        errors[f"FOLD_{field.upper()}_OVERLAP"] += 1
    result = {"status": "PASS" if manifest and source_ok and not errors else "FAIL",
              "source_version": cfg["version"], "source_verified": source_ok,
              "samples": len(manifest), "failed_checks": dict(errors),
              "evidence_exact_match_rate": 1.0 if manifest and not errors["EVIDENCE_SPAN_INVALID"] else 0.0}
    write_json(project.path("reports/release_audit.json"), result)
    if not frozen:
        quarantine = []
        scope_rows = {r["case_id"]: r for r in project.rows("data/screening/scope.parquet", optional=True)}
        for row in project.rows("data/screening/routing.parquet"):
            if row["confirmed_target_count"] == 0:
                scope = scope_rows.get(row["case_id"], {})
                code = row["reason"] if row["reason"] != "NO_CONFIRMED_TARGET" else "NO_CONFIRMED_TARGET"
                quarantine.append({**row, "out_of_scope_concepts": scope.get("out_of_scope_concepts", []),
                                   "reason_code": code})
            elif row["confirmed_target_count"] > 1:
                quarantine.append({**row, "reason_code": "MULTI_TARGET_DIAGNOSIS"})
        included = {r["case_id"] for r in manifest}
        for cid, sample in cases.items():
            if cid in included:
                continue
            reasons = []
            text, image = texts.get(cid), selected.get(cid)
            human = humans.get(cid)
            review_row = review.get(cid)
            group = group_rows.get(cid)
            if sample["source_license"] in {None, "UNKNOWN", "NO-CC CODE", ""}:
                reasons.append("LICENSE_UNKNOWN")
            if not text or not text.get("text_usable"):
                reasons.append(text.get("reason", "TEXT_QC_PENDING") if text else "TEXT_QC_PENDING")
            if not image:
                reasons.append("IMAGE_MISSING_OR_QC_PENDING")
            if not review_row or not review_row.get("verified"):
                reasons.append("HUMAN_ADJUDICATION_REQUIRED" if review_row and
                               review_row.get("reason") == "HUMAN_ADJUDICATION_REQUIRED"
                               else "LABEL_REVIEW_PENDING_OR_REJECTED")
            if group and group.get("duplicate_conflict"):
                reasons.append("DUPLICATE_CONFLICT")
            if human and human.get("human_decision") == "reject":
                reasons.append("HUMAN_REJECTED")
            for reason in sorted(set(reasons or ["INSUFFICIENT_CLASS_SUPPORT"])):
                quarantine.append({"case_id": cid, "article_id": sample["article_id"],
                                   "candidate_disease": sample["candidate_disease"], "reason_code": reason})
        # The original case-level scope table includes candidates of zero confirmed diagnoses,
        # out-of-scope-only mentions and cases with no disease-search hits.
        quarantined_ids = {r.get("case_id") for r in quarantine}
        for scope in project.rows("data/screening/scope.parquet", optional=True):
            if (scope["scope_status"] != "in_scope_candidate"
                    and scope["case_id"] not in quarantined_ids):
                quarantine.append({**scope, "reason_code": scope["scope_status"]})
                quarantined_ids.add(scope["case_id"])
        project.write("data/qc/quarantine.parquet", quarantine)
    from .metrics import report_metrics
    report_metrics(project, result)
    return result

from __future__ import annotations

import json
from collections import Counter

import jsonschema

from .common import PipelineError, Project, canonical, digest, now, read_json
from .evidence import unique_quote_span
from .sample import single_targets


def _latest_by_case(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in sorted(rows, key=lambda item: item.get("imported_at", "")):
        result[row["case_id"]] = row
    return result


def _review_summary(row: dict) -> dict:
    fields = ("decision", "diagnosis_status", "disease_is_current", "evidence_quote", "evidence_valid",
              "evidence_start", "evidence_end", "multiple_target_diseases", "other_target_diseases",
              "confidence", "reason_codes", "reason", "source_decision")
    return {field: row.get(field) for field in fields}


def _editable_fields(row: dict | None) -> dict:
    if row:
        return {"human_decision": row["human_decision"], "human_status": row["human_status"],
                "evidence_quote": row["evidence_quote"],
                "multiple_target_diseases": row["multiple_target_diseases"],
                "reason_code": row["reason_code"], "reviewer_id": row["reviewer_id"],
                "reviewed_at": row["reviewed_at"]}
    return {"human_decision": None, "human_status": None, "evidence_quote": "",
            "multiple_target_diseases": None, "reason_code": "", "reviewer_id": "", "reviewed_at": ""}


def export_human_adjudication(project: Project, output: str = "data/reviews/human_adjudication_input.jsonl") -> dict:
    requirements = {row["case_id"]: row for row in project.rows("data/reviews/human_requirements.parquet")}
    unresolved = {cid for cid, row in requirements.items() if row.get("consensus") == "NEEDS_ADJUDICATION"}
    if not unresolved:
        raise PipelineError("No NEEDS_ADJUDICATION cases are present in human_requirements.parquet.")

    samples = {row["case_id"]: row for row in single_targets(project)}
    reviewer_a = {row["case_id"]: row for row in project.rows("data/reviews/reviewer_a.parquet")}
    reviewer_b = {row["case_id"]: row for row in project.rows("data/reviews/reviewer_b.parquet")}
    medgemma = {row["case_id"]: row for row in project.rows("data/reviews/medgemma_labels.parquet", optional=True)}
    prior = _latest_by_case(project.rows("data/reviews/human_label_adjudications.parquet", optional=True))
    rows = []
    for case_id in sorted(unresolved):
        case, a, b = samples.get(case_id), reviewer_a.get(case_id), reviewer_b.get(case_id)
        if not case or not a or not b:
            raise PipelineError(f"Cannot export complete adjudication context for {case_id}.")
        requirement = requirements[case_id]
        if (case["candidate_disease"] != requirement["candidate_disease"]
                or case["raw_text_sha256"] != requirement["raw_text_sha256"]
                or a["raw_text_sha256"] != case["raw_text_sha256"]
                or b["raw_text_sha256"] != case["raw_text_sha256"]):
            raise PipelineError(f"Adjudication source hash or target mismatch for {case_id}.")
        rows.append({
            "case_id": case_id,
            "article_id": case["article_id"],
            "candidate_disease": case["candidate_disease"],
            "disease": case["disease"],
            "raw_text_sha256": case["raw_text_sha256"],
            "raw_case_text": case["raw_case_text"],
            "rule_review": {"decision": "accept", "diagnosis_status": case["diagnosis_status"],
                            "evidence_quote": case["diagnosis_evidence"],
                            "evidence_start": case["evidence_start"], "evidence_end": case["evidence_end"]},
            "reviewer_a": _review_summary(a),
            "reviewer_b": _review_summary(b),
            "consensus": "NEEDS_ADJUDICATION",
            "human_adjudication": _editable_fields(prior.get(case_id)),
        })
        third = medgemma.get(case_id)
        if (third and third.get("raw_text_sha256") == case["raw_text_sha256"]
                and third.get("candidate_disease") == case["candidate_disease"]):
            rows[-1]["medgemma_review"] = _review_summary(third)

    path = project.path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    decision_pairs = Counter(f"{row['reviewer_a']['decision']} / {row['reviewer_b']['decision']}" for row in rows)
    return {"path": str(path), "rows": len(rows),
            "already_adjudicated": sum(bool(prior.get(cid)) for cid in unresolved),
            "reviewer_decision_pairs": dict(decision_pairs)}


def import_human_adjudication(project: Project, input_path: str) -> dict:
    path = project.path(input_path)
    schema = read_json(project.path("schemas/human_adjudication.json"))
    requirements = {row["case_id"]: row for row in project.rows("data/reviews/human_requirements.parquet")}
    cases = {row["case_id"]: row for row in single_targets(project)}
    reviewer_a = {row["case_id"]: row for row in project.rows("data/reviews/reviewer_a.parquet")}
    reviewer_b = {row["case_id"]: row for row in project.rows("data/reviews/reviewer_b.parquet")}
    medgemma = {row["case_id"]: row for row in project.rows("data/reviews/medgemma_labels.parquet", optional=True)}
    existing = project.rows("data/reviews/human_label_adjudications.parquet", optional=True)
    identities = {row["record_sha256"] for row in existing}
    additions, seen, completed = [], set(), set()
    expected_fields = {"case_id", "article_id", "candidate_disease", "disease", "raw_text_sha256",
                       "raw_case_text", "rule_review", "reviewer_a", "reviewer_b", "consensus",
                       "human_adjudication"}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) not in (expected_fields, expected_fields | {"medgemma_review"}):
            raise PipelineError(f"Adjudication queue fields changed on line {line_number}.")
        case_id = row["case_id"]
        if case_id in seen:
            raise PipelineError(f"Duplicate case in adjudication file: {case_id}")
        seen.add(case_id)
        case, requirement = cases.get(case_id), requirements.get(case_id)
        if (not case or not requirement or requirement.get("consensus") != "NEEDS_ADJUDICATION"
                or row["consensus"] != "NEEDS_ADJUDICATION"
                or row["candidate_disease"] != case["candidate_disease"]
                or row["raw_text_sha256"] != case["raw_text_sha256"]
                or row["raw_case_text"] != case["raw_case_text"]
                or row["article_id"] != case["article_id"]
                or row["disease"] != case["disease"]):
            raise PipelineError(f"Stale or mismatched adjudication source for {case_id}.")
        expected_rule = {"decision": "accept", "diagnosis_status": case["diagnosis_status"],
                         "evidence_quote": case["diagnosis_evidence"],
                         "evidence_start": case["evidence_start"], "evidence_end": case["evidence_end"]}
        if row["rule_review"] != expected_rule:
            raise PipelineError(f"Rule evidence context changed for {case_id}.")
        if not reviewer_a.get(case_id) or row["reviewer_a"] != _review_summary(reviewer_a[case_id]):
            raise PipelineError(f"Reviewer A context changed for {case_id}.")
        if not reviewer_b.get(case_id) or row["reviewer_b"] != _review_summary(reviewer_b[case_id]):
            raise PipelineError(f"Reviewer B context changed for {case_id}.")
        if "medgemma_review" in row:
            third = medgemma.get(case_id)
            if (not third or third.get("raw_text_sha256") != case["raw_text_sha256"]
                    or third.get("candidate_disease") != case["candidate_disease"]
                    or row["medgemma_review"] != _review_summary(third)):
                raise PipelineError(f"MedGemma context changed for {case_id}.")

        annotation = row["human_adjudication"]
        blank_annotation = _editable_fields(None)
        if annotation == blank_annotation:
            continue
        if not isinstance(annotation, dict) or not annotation.get("human_decision"):
            raise PipelineError(f"Incomplete human adjudication fields for {case_id}.")
        jsonschema.validate(annotation, schema, format_checker=jsonschema.FormatChecker())
        span = unique_quote_span(case["raw_case_text"], annotation["evidence_quote"])
        if annotation["human_decision"] == "accept" and (
                annotation["human_status"] != "confirmed" or annotation["multiple_target_diseases"] or not span):
            raise PipelineError(f"Accept requires one confirmed diagnosis and a unique exact quote: {case_id}.")
        record = {"case_id": case_id, "candidate_disease": case["candidate_disease"],
                  "raw_text_sha256": case["raw_text_sha256"], "human_decision": annotation["human_decision"],
                  "human_status": annotation["human_status"], "evidence_quote": annotation["evidence_quote"],
                  "evidence_start": span[0] if span else None, "evidence_end": span[1] if span else None,
                  "human_evidence_valid": span is not None,
                  "multiple_target_diseases": annotation["multiple_target_diseases"],
                  "reason_code": annotation["reason_code"], "reviewer_id": annotation["reviewer_id"],
                  "reviewed_at": annotation["reviewed_at"]}
        identity = digest(canonical(record))
        completed.add(case_id)
        if identity not in identities:
            additions.append({**record, "record_sha256": identity, "imported_at": now()})
            identities.add(identity)

    if not completed:
        raise PipelineError("The human adjudication file contains no completed records.")
    project.write("data/reviews/human_label_adjudications.parquet", existing + additions)
    return {"imported": len(additions), "total_records": len(existing) + len(additions),
            "unique_cases_reviewed": len(completed)}

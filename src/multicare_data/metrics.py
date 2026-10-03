from __future__ import annotations

from collections import Counter
from math import sqrt

from .common import Project, read_json, write_json


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> dict:
    if trials <= 0:
        return {"lower_95": None, "upper_95": None}
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    radius = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return {"lower_95": round(center - radius, 6), "upper_95": round(center + radius, 6)}


def _kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    n = len(left)
    labels = set(left) | set(right)
    po = sum(a == b for a, b in zip(left, right, strict=True)) / n
    pe = sum((left.count(label) / n) * (right.count(label) / n) for label in labels)
    if pe == 1:
        return 1.0 if po == 1 else None
    return round((po - pe) / (1 - pe), 6)


def report_metrics(project: Project, audit: dict) -> dict:
    labels = {(r["case_id"], r["candidate_disease"]): r
              for r in project.rows("data/screening/labels.parquet")}
    samples = {r["case_id"]: r for r in project.rows("data/qc/feasible_candidates.parquet", optional=True)}
    a = {r["case_id"]: r for r in project.rows("data/reviews/reviewer_a.parquet", optional=True)}
    b = {r["case_id"]: r for r in project.rows("data/reviews/reviewer_b.parquet", optional=True)}
    humans = {r["case_id"]: r for r in project.rows("data/reviews/human_reviews.parquet", optional=True)}
    adjudication = {r["case_id"]: r for r in project.rows("data/reviews/adjudication.parquet", optional=True)}
    auto_agree_a, auto_agree_b, human_acceptances, human_trial_count = [], [], 0, 0
    for cid, review_a in a.items():
        rule = labels.get((cid, review_a["candidate_disease"]))
        if rule:
            auto_agree_a.append(rule["diagnosis_status"] == review_a["diagnosis_status"])
    for cid in a.keys() & b.keys():
        if a[cid]["candidate_disease"] == b[cid]["candidate_disease"]:
            auto_agree_b.append(a[cid]["diagnosis_status"] == b[cid]["diagnosis_status"])
    for cid, human in humans.items():
        if adjudication.get(cid, {}).get("review_status") == "verified_silver":
            human_trial_count += 1
            human_acceptances += bool(human["human_decision"] == "accept"
                                      and human["human_status"] == "confirmed"
                                      and human["human_evidence_valid"])
    text = project.rows("data/qc/text_qc.parquet", optional=True)
    image = project.rows("data/qc/image_qc.parquet", optional=True)
    group = project.rows("data/groups/groups.parquet", optional=True)
    conflicts = project.rows("data/qc/duplicate_conflicts.parquet", optional=True)
    routes = project.rows("data/screening/routing.parquet", optional=True)
    splits = project.rows("data/splits/assignments.parquet", optional=True)
    text_reasons = Counter(row.get("reason", "unknown") for row in text)
    image_relevance = Counter(row.get("image_relevance", "unreviewed") for row in image)
    route_counts = Counter(row.get("reason", "unknown") for row in routes)
    class_feasibility = project.rows("reports/disease_feasibility.parquet", optional=True)
    source_lock = read_json(project.path("data/raw/source_lock.json")) if project.path("data/raw/source_lock.json").exists() else {}
    split_counts = Counter((row.get("partition"), row.get("fold")) for row in splits)
    human_report = {"automated_consensus_human_audited": human_trial_count,
                    "consensus_human_acceptance_rate": (round(human_acceptances / human_trial_count, 6)
                                                         if human_trial_count else None),
                    **wilson(human_acceptances, human_trial_count),
                    "interpretation": "Acceptance among human-audited automated-consensus labels; report only, not population clinical accuracy."}
    result = {
        "status": audit["status"],
        "provenance": {"source_verified": audit["source_verified"],
                       "source_version": audit["source_version"],
                       "source_file_count": len(source_lock.get("files", [])),
                       "source_lock_complete": bool(source_lock.get("complete"))},
        "structural": {"single_target_cases": sum(r["confirmed_target_count"] == 1 for r in routes),
                       "eligible_release_cases": len(samples),
                       "confirmed_target_cases": sum(r["confirmed_target_count"] == 1 for r in routes)},
        "label_evidence": {"exact_match_rate": audit["evidence_exact_match_rate"],
                           "rule_reviewer_a_status_agreement": round(sum(auto_agree_a) / len(auto_agree_a), 6)
                           if auto_agree_a else None,
                           "reviewer_a_b_status_agreement": round(sum(auto_agree_b) / len(auto_agree_b), 6)
                           if auto_agree_b else None,
                           "reviewer_a_b_overlap_cases": len(auto_agree_b)},
        "human_audit": human_report,
        "text_quality": {"case_count": len(text), "eligible": sum(r.get("text_usable", False) for r in text),
                         "reason_counts": dict(text_reasons)},
        "image_quality": {"candidate_image_count": len(image),
                           "decode_pass": sum(r.get("decode_pass", False) for r in image),
                           "relevance_counts": dict(image_relevance),
                           "selected_cases": len(project.rows("data/qc/selected_images.parquet", optional=True))},
        "duplicates": {"independent_components": len({r["split_group"] for r in group}),
                       "conflict_groups": len(conflicts)},
        "class_coverage": {"class_count": sum(r.get("status") == "in_v001" for r in class_feasibility),
                           "classes": class_feasibility},
        "split": {"partition_fold_counts": {f"{partition or 'unknown'}:fold_{fold}": count
                                             for (partition, fold), count in sorted(split_counts.items(), key=str)},
                  "sample_count": len(splits), "integrity_failures": {
                      k: v for k, v in audit["failed_checks"].items()
                      if k.startswith(("SPLIT_", "FOLD_"))}},
        "screening": {"routing_counts": dict(route_counts)},
        "failed_checks": audit["failed_checks"],
    }
    write_json(project.path("reports/dataset_evaluation.json"), result)
    return result

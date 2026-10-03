from __future__ import annotations

from collections import Counter, defaultdict

from .common import PipelineError, canonical, now, write_json
from .sample import single_targets


def prepare(project, allow_pending=False):
    from .adjudicate import adjudicate, prepare_human
    from .grouping import feasibility, group
    from .medgemma import apply_results

    progress = apply_results(project)
    adjudicate(project)
    group(project)
    support = feasibility(project)
    if support["feasible_classes"]:
        from .release import freeze_taxonomy
        from .split import grouped_split
        freeze_taxonomy(project)
        grouped_split(project)
    queue = prepare_human(project, project.path("data/reviews/human_input.jsonl"))
    from .human_adjudication import export_human_adjudication
    label_queue = export_human_adjudication(project)
    cases = single_targets(project)
    a = {r["case_id"]: r for r in project.rows("data/reviews/reviewer_a.parquet")}
    b = {r["case_id"]: r for r in project.rows("data/reviews/reviewer_b.parquet")}
    mg = {r["case_id"]: r for r in project.rows("data/reviews/medgemma_labels.parquet", optional=True)}
    text = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet")}
    selected = {r["case_id"]: r for r in project.rows("data/qc/selected_images.parquet")}
    images = defaultdict(list)
    for row in project.rows("data/qc/image_qc.parquet"):
        images[row["case_id"]].append(row)
    decisions = {r["case_id"]: r for r in project.rows("data/reviews/adjudication.parquet")}
    groups = {r["case_id"]: r for r in project.rows("data/groups/groups.parquet")}
    rows = []
    for case in cases:
        cid = case["case_id"]
        review = decisions[cid]
        blocks = []
        needs_third = review["reason"] == "HUMAN_ADJUDICATION_REQUIRED"
        if needs_third and cid not in mg:
            blocks.append("MEDGEMMA_LABEL_PENDING")
        elif cid in mg and (mg[cid]["decision"] != a.get(cid, {}).get("decision")
                            or mg[cid]["decision"] != b.get(cid, {}).get("decision")):
            blocks.append("MEDGEMMA_REVIEWER_DISAGREEMENT")
        if not text.get(cid, {}).get("text_usable"):
            blocks.append(text.get(cid, {}).get("reason", "TEXT_QC_MISSING"))
        if cid not in selected:
            blocks.append("NO_USABLE_REVIEWED_IMAGE")
        if not review["verified"]:
            blocks.append(review["reason"])
        if groups[cid]["duplicate_conflict"]:
            blocks.append("DUPLICATE_CONFLICT")
        rows.append({"case_id": cid, "candidate_disease": case["candidate_disease"],
                     "disease": case["disease"], "raw_text_sha256": case["raw_text_sha256"],
                     "raw_case_text": case["raw_case_text"], "proposed_evidence": case["diagnosis_evidence"],
                     "reviewer_a_decision": a.get(cid, {}).get("decision"),
                     "reviewer_b_decision": b.get(cid, {}).get("decision"),
                     "medgemma_decision": mg.get(cid, {}).get("decision", "pending" if needs_third else "not_required"),
                     "medgemma_confidence": mg.get(cid, {}).get("confidence"),
                     "medgemma_evidence_quote": mg.get(cid, {}).get("evidence_quote"),
                     "medgemma_reason": mg.get(cid, {}).get("reason"),
                     "reviewer_a_json": canonical(a.get(cid)), "reviewer_b_json": canonical(b.get(cid)),
                     "medgemma_json": canonical(mg.get(cid)), "text_qc_json": canonical(text.get(cid)),
                     "images_qc_json": canonical(images[cid]), "selected_image_json": canonical(selected.get(cid)),
                     "adjudication_json": canonical(review), "split_group": groups[cid]["split_group"],
                     "label_status": "ACCEPTED_AB_CONSENSUS" if review["verified"] else
                                     "REJECTED_AB_CONSENSUS" if review["review_status"] == "rejected" else
                                     "MEDGEMMA_REVIEWED_AWAITING_HUMAN" if cid in mg else "MEDGEMMA_PENDING",
                     "medgemma_required": needs_third, "requires_human_label_review": needs_third,
                     "blocking_reasons": blocks, "human_review_completed": False})
    final = progress["ai_complete"]
    accepted_ab = [r for r in rows if r["label_status"] == "ACCEPTED_AB_CONSENSUS"]
    project.write("data/reviews/final_ab_accepted.parquet", accepted_ab)
    name = "final_review" if final else "pre_human_review_draft"
    project.write(f"data/reviews/{name}.parquet", rows)
    path = project.path(f"data/reviews/{name}.jsonl")
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    temp.replace(path)
    summary = {"status": "AI_COMPLETE_AWAITING_HUMAN" if final else "BLOCKED_MEDGEMMA_PENDING",
               "cases": len(rows), "medgemma": progress, "feasibility": support, "human_queue": queue,
               "label_human_queue": label_queue,
               "accepted_ab_cases": len(accepted_ab), "label_status_counts": dict(Counter(r["label_status"] for r in rows)),
               "blocking_reasons": dict(Counter(reason for row in rows for reason in row["blocking_reasons"])),
               "review_file": str(path), "updated_at": now()}
    write_json(project.path("reports/pre_human_review.json"), summary)
    if not final and not allow_pending:
        raise PipelineError("MedGemma reviews are incomplete; draft saved, final review was not emitted")
    return summary

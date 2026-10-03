from .common import PipelineError, Project


def single_targets(project: Project) -> list[dict]:
    routes = [r for r in project.rows("data/screening/routing.parquet")
              if r["confirmed_target_count"] == 1]
    targets = {r["case_id"]: r["confirmed_target_diseases"][0] for r in routes}
    target_pairs = {(case_id, target) for case_id, target in targets.items()}
    labels = {(r["case_id"], r["candidate_disease"]): r
              for r in project.iter_rows("data/screening/labels.parquet",
                                         ["case_id", "candidate_disease", "disease", "diagnosis_status",
                                          "diagnosis_evidence", "evidence_start", "evidence_end"])
              if (r["case_id"], r["candidate_disease"]) in target_pairs}
    cases = {r["case_id"]: r for r in project.iter_rows(
        "data/normalized/cases.parquet",
        ["case_id", "article_id", "raw_case_text", "raw_text_sha256", "age", "gender", "patient_link_id"],
        batch_size=128) if r["case_id"] in targets}
    article_ids = {r["article_id"] for r in cases.values()}
    articles = {r["article_id"]: r for r in project.iter_rows(
        "data/normalized/articles.parquet", ["article_id", "license", "source_url", "title"])
        if r["article_id"] in article_ids}
    rows = []
    for route in routes:
        cid = route["case_id"]
        target = targets[cid]
        case = cases.get(cid)
        if not case:
            raise PipelineError(f"Single-target route has no normalized case: {cid}")
        lab = labels.get((cid, target))
        if not lab:
            raise PipelineError(f"Single-target route has no matching label: {cid} / {target}")
        article = articles.get(case["article_id"])
        if not article:
            raise PipelineError(f"Normalized case has no article metadata: {cid}")
        rows.append({**case, "candidate_disease": target, "diagnosis_label": target,
                     "disease": lab["disease"], "diagnosis_status": lab["diagnosis_status"],
                     "diagnosis_evidence": lab["diagnosis_evidence"], "evidence_start": lab["evidence_start"],
                     "evidence_end": lab["evidence_end"], "confirmed_target_count": 1,
                     "source_license": article["license"], "source_url": article["source_url"],
                     "title": article["title"], "rule_decision": "accept"})
    return rows


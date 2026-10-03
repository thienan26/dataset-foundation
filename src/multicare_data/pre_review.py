from __future__ import annotations

from collections import defaultdict

from .common import PipelineError, Project, canonical, digest
from .llm_review import label_payload
from .sample import single_targets


def export_pre_ai_review(project: Project, output: str = "data/reviews/pre_ai_review.parquet") -> dict:
    """Materialize one auditable row per single-target case before Gemini review."""
    samples = single_targets(project)
    if not samples:
        raise PipelineError("No single-target cases are available for the pre-AI review export.")

    case_ids = {row["case_id"] for row in samples}
    candidates_by_case: dict[str, list[dict]] = defaultdict(list)
    candidate_by_pair: dict[tuple[str, str], dict] = {}
    for row in project.iter_rows("data/candidates/candidates.parquet"):
        if row["case_id"] not in case_ids:
            continue
        candidates_by_case[row["case_id"]].append(row)
        candidate_by_pair[(row["case_id"], row["candidate_disease"])] = row

    label_by_pair: dict[tuple[str, str], dict] = {}
    for row in project.iter_rows("data/screening/labels.parquet"):
        if row["case_id"] not in case_ids:
            continue
        label_by_pair[(row["case_id"], row["candidate_disease"])] = row

    route_columns = ["case_id", "confirmed_target_diseases", "confirmed_target_count", "reason", "release_eligible"]
    routes = {row["case_id"]: row for row in project.iter_rows("data/screening/routing.parquet", route_columns)
              if row["case_id"] in case_ids}
    scope_columns = ["case_id", "in_scope_concepts", "out_of_scope_concepts", "scope_status"]
    scopes = {row["case_id"]: row for row in project.iter_rows("data/screening/scope.parquet", scope_columns)
              if row["case_id"] in case_ids}

    article_ids = {row["article_id"] for row in samples}
    article_columns = ["article_id", "title", "pmcid", "pmid", "doi", "year", "license", "source_url",
                       "keywords", "mesh_terms", "publication_metadata_json"]
    articles = {row["article_id"]: row for row in project.iter_rows("data/normalized/articles.parquet", article_columns)
                if row["article_id"] in article_ids}

    review_cfg = project.config("review")
    output_rows = []
    for sample in samples:
        case_id = sample["case_id"]
        pair = (case_id, sample["candidate_disease"])
        candidate = candidate_by_pair.get(pair)
        label = label_by_pair.get(pair)
        route = routes.get(case_id)
        scope = scopes.get(case_id)
        article = articles.get(sample["article_id"])
        if not all((candidate, label, route, scope, article)):
            raise PipelineError(f"Pre-AI export has a missing joined artifact for case {case_id}.")

        payload = label_payload(sample)
        all_hypotheses = []
        for hypothesis in sorted(candidates_by_case.get(case_id, []), key=lambda row: row["candidate_disease"]):
            hypothesis_label = label_by_pair.get((case_id, hypothesis["candidate_disease"]))
            all_hypotheses.append({"candidate": hypothesis, "rule_label": hypothesis_label})

        output_rows.append({
            "case_id": sample["case_id"],
            "article_id": sample["article_id"],
            "raw_case_text": sample["raw_case_text"],
            "raw_text_sha256": sample["raw_text_sha256"],
            "age": sample["age"],
            "gender": sample["gender"],
            "patient_link_id": sample["patient_link_id"],
            "target_concept_id": sample["candidate_disease"],
            "target_disease": sample["disease"],
            "candidate_matched_aliases": candidate["matched_aliases"],
            "candidate_match_locations_json": canonical(candidate["match_locations"]),
            "candidate_reason": candidate["candidate_reason"],
            "rule_diagnosis_status": label["diagnosis_status"],
            "rule_diagnosis_evidence": label["diagnosis_evidence"],
            "rule_evidence_start": label["evidence_start"],
            "rule_evidence_end": label["evidence_end"],
            "rule_reason": label["reason"],
            "rule_version": label["rule_version"],
            "confirmed_target_diseases": route["confirmed_target_diseases"],
            "confirmed_target_count": route["confirmed_target_count"],
            "routing_reason": route["reason"],
            "release_eligible_by_rule": route["release_eligible"],
            "scope_status": scope["scope_status"],
            "in_scope_concepts": scope["in_scope_concepts"],
            "out_of_scope_concepts": scope["out_of_scope_concepts"],
            "article_title": article["title"],
            "article_pmcid": article["pmcid"],
            "article_pmid": article["pmid"],
            "article_doi": article["doi"],
            "article_year": article["year"],
            "article_license": article["license"],
            "article_source_url": article["source_url"],
            "article_keywords": article["keywords"],
            "article_mesh_terms": article["mesh_terms"],
            "article_publication_metadata_json": article["publication_metadata_json"],
            "all_candidate_hypotheses_json": canonical(all_hypotheses),
            "reviewer_a_model": review_cfg["reviewer_a_model"],
            "prompt_version": review_cfg["prompt_version"],
            "gemini_label_payload_json": canonical(payload),
            "gemini_label_payload_sha256": digest(canonical(payload)),
        })

    if len(output_rows) != len(case_ids) or len({row["case_id"] for row in output_rows}) != len(output_rows):
        raise PipelineError("Pre-AI export did not preserve one unique row per single-target case.")
    project.write(output, output_rows)
    return {"path": str(project.path(output)), "rows": len(output_rows),
            "columns": len(output_rows[0]), "gemini_payload_only_column": "gemini_label_payload_json"}

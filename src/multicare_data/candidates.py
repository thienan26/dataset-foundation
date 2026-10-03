from collections import defaultdict

from .common import Project
from .discover import Matcher


def build_candidates(project: Project) -> dict:
    catalog = project.rows("data/catalog/disease_catalog.parquet")
    terms = [r for r in catalog if r["in_scope"]]
    out_terms = [r for r in catalog if not r["in_scope"]]
    lookup = {r["concept_id"]: r["disease"] for r in terms}
    out_lookup = {r["concept_id"]: r["disease"] for r in out_terms}
    matcher = Matcher(terms)
    out_matcher = Matcher(out_terms)
    articles = {r["article_id"]: r for r in project.rows("data/normalized/articles.parquet")}
    metadata_hits = {}
    for aid, a in articles.items():
        text = "\n".join([a["title"]] + a["keywords"] + a["mesh_terms"])
        metadata_hits[aid] = [{**m, "field": "article_metadata"} for m in matcher.find(text)]
    out_metadata_hits = {}
    for aid, a in articles.items():
        text = "\n".join([a["title"]] + a["keywords"] + a["mesh_terms"])
        out_metadata_hits[aid] = [{**m, "field": "article_metadata"} for m in out_matcher.find(text)]
    rows, scope_rows, out_of_scope = [], [], []
    for case in project.rows("data/normalized/cases.parquet"):
        matches = [{**m, "field": "raw_case_text"} for m in matcher.find(case["raw_case_text"])]
        matches += metadata_hits[case["article_id"]]
        other_hits = [{**m, "field": "raw_case_text"} for m in out_matcher.find(case["raw_case_text"])]
        other_hits += out_metadata_hits[case["article_id"]]
        groups = defaultdict(list)
        for m in matches:
            groups[m["concept_id"]].append(m)
        for cid in sorted(groups):
            rows.append({"case_id": case["case_id"], "article_id": case["article_id"],
                         "candidate_disease": cid, "disease": lookup[cid],
                         "matched_aliases": sorted({m["alias"] for m in groups[cid]}),
                         "match_locations": groups[cid], "candidate_reason": "ONTOLOGY_ALIAS_OR_METADATA"})
        other_groups = defaultdict(list)
        for match in other_hits:
            other_groups[match["concept_id"]].append(match)
        for cid, locations in sorted(other_groups.items()):
            out_of_scope.append({"case_id": case["case_id"], "article_id": case["article_id"],
                                 "candidate_disease": cid, "disease": out_lookup[cid],
                                 "matched_aliases": sorted({m["alias"] for m in locations}),
                                 "match_locations": locations, "candidate_reason": "OUT_OF_SCOPE_ONTOLOGY_ALIAS"})
        scope_rows.append({"case_id": case["case_id"], "article_id": case["article_id"],
                           "in_scope_concepts": sorted(groups),
                           "out_of_scope_concepts": sorted(other_groups),
                           "scope_status": "in_scope_candidate" if groups else
                                          "OUT_OF_SCOPE" if other_groups else "NO_DISEASE_MENTION_CANDIDATE"})
    project.write("data/candidates/candidates.parquet", rows)
    project.write("data/candidates/out_of_scope_candidates.parquet", out_of_scope)
    project.write("data/screening/scope.parquet", scope_rows)
    return {"hypotheses": len(rows), "candidate_cases": len({r["case_id"] for r in rows}),
            "out_of_scope_hypotheses": len(out_of_scope),
            "out_of_scope_cases": sum(bool(r["out_of_scope_concepts"]) for r in scope_rows)}


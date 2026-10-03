from __future__ import annotations

import re
from collections import Counter, defaultdict

from .common import Project
from .evidence import sentence_spans


def mention_status(sentence: str, alias_start: int, alias_end: int) -> str:
    """Conservative local assertions. Clinical semantics still require independent review."""
    before = sentence[:alias_start].lower()
    after = sentence[alias_end:].lower()
    # Clause boundaries reduce attributing a neighbour diagnosis to this disease.
    before = re.split(r"[;:]|\b(?:but|whereas|however)\b", before)[-1][-100:]
    after = re.split(r"[;:]|\b(?:but|whereas|however)\b", after)[0][:100]
    local = before + " <disease> " + after
    if re.search(r"\b(?:family history|mother|father|sister|brother)\b", before):
        return "historical"
    if re.search(r"\b(?:history of|previous|prior|past|resolved)\b", before):
        return "historical"
    if (re.search(r"\b(?:no evidence of|negative for|ruled out|excluded|without)\s*(?:\w+\s+){0,3}$", before)
            or re.match(r"\s*(?:was |is |had been )?(?:ruled out|excluded|negative|not confirmed)", after)
            or re.search(r"\b(?:not|never)\s+(?:\w+\s+){0,2}(?:confirmed|diagnosed)", before)):
        return "ruled_out"
    if re.search(r"\b(?:differential|versus|vs\.?|considered)\b", local):
        return "differential"
    if re.search(r"\b(?:probable|likely|presumed)\b", local):
        return "probable"
    if re.search(r"\b(?:suspect\w*|possible|potential|may|might|could|cannot exclude)\b", local):
        return "suspected"
    if (re.search(r"\b(?:confirmed|diagnosed with|diagnosis (?:of|was)|positive for|consistent with a diagnosis of)\s+(?:\w+\s+){0,3}$", before)
            or re.match(r"\s*(?:was |is |has been )?(?:confirmed|diagnosed|established)\b", after)):
        return "confirmed"
    return "unclear"


def diagnose(text: str, matches: list[dict]) -> dict:
    evidence = []
    spans = sentence_spans(text)
    for match in matches:
        if match["field"] != "raw_case_text":
            continue
        for start, end in spans:
            if start <= match["start"] < match["end"] <= end:
                sentence = text[start:end]
                status = mention_status(sentence, match["start"] - start, match["end"] - start)
                evidence.append({"diagnosis_status": status, "diagnosis_evidence": sentence,
                                 "evidence_start": start, "evidence_end": end})
                break
    confirmed = [r for r in evidence if r["diagnosis_status"] == "confirmed"]
    negative = [r for r in evidence if r["diagnosis_status"] == "ruled_out"]
    if confirmed and negative:
        return {**confirmed[0], "diagnosis_status": "unclear", "reason": "CONTRADICTORY_ASSERTIONS"}
    ranked = {s: i for i, s in enumerate(["confirmed", "probable", "suspected", "differential",
                                         "ruled_out", "historical", "unclear"])}
    chosen = min(evidence, key=lambda r: ranked[r["diagnosis_status"]]) if evidence else {
        "diagnosis_status": "unclear", "diagnosis_evidence": "", "evidence_start": None, "evidence_end": None}
    return {**chosen, "reason": "EXPLICIT_ASSERTION" if chosen["diagnosis_status"] == "confirmed" else "NOT_CONFIRMED"}


def most_specific(ids: list[str], terms: dict) -> list[str]:
    ancestors = {a for cid in ids for a in terms[cid]["ancestors"]}
    return sorted(set(ids) - ancestors)


def label(project: Project) -> dict:
    cases = {r["case_id"]: r for r in project.rows("data/normalized/cases.parquet")}
    terms = {r["concept_id"]: r for r in project.rows("data/catalog/disease_catalog.parquet")}
    output, by_case = [], defaultdict(list)
    for candidate in project.rows("data/candidates/candidates.parquet"):
        row = {**candidate, **diagnose(cases[candidate["case_id"]]["raw_case_text"], candidate["match_locations"]),
               "rule_version": project.config("labeling")["version"]}
        output.append(row)
        by_case[row["case_id"]].append(row)
    routing = []
    scope_by_case = {r["case_id"]: r for r in project.rows("data/screening/scope.parquet")}
    for cid, case in cases.items():
        ids = [r["candidate_disease"] for r in by_case[cid] if r["diagnosis_status"] == "confirmed"]
        if project.config("labeling")["collapse_ontology_ancestors"]:
            ids = most_specific(ids, terms)
        reason = "SINGLE_TARGET" if len(ids) == 1 else "MULTI_TARGET_DIAGNOSIS" if len(ids) > 1 else \
                 "OUT_OF_SCOPE" if scope_by_case[cid]["scope_status"] == "OUT_OF_SCOPE" else \
                 "NO_CANDIDATE_DISEASE_MENTION" if scope_by_case[cid]["scope_status"] == "NO_DISEASE_MENTION_CANDIDATE" else \
                 "NO_CONFIRMED_TARGET"
        routing.append({"case_id": cid, "article_id": case["article_id"], "confirmed_target_diseases": ids,
                        "confirmed_target_count": len(ids), "reason": reason,
                        "release_eligible": len(ids) == 1})
    project.write("data/screening/labels.parquet", output)
    project.write("data/screening/routing.parquet", routing)
    return dict(Counter(r["reason"] for r in routing))

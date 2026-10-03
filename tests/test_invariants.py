
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from multicare_data.adjudicate import adjudicate, prepare_human
from multicare_data.common import PipelineError, Project, safe_path
from multicare_data.discover import Matcher, parse_obo
from multicare_data.evidence import exact_evidence, sentence_spans, unique_quote_span
from multicare_data.grouping import UnionFind
from multicare_data.human_adjudication import export_human_adjudication, import_human_adjudication
from multicare_data.labeling import diagnose, mention_status
from multicare_data.llm_review import current_reviews
from multicare_data.metrics import wilson
from multicare_data.review_schema import HUMAN_REVIEW
from multicare_data.sample import single_targets
from multicare_data.split import grouped_split


def test_project_rejects_path_traversal(tmp_path):
    with pytest.raises(PipelineError):
        safe_path(tmp_path, "../outside.txt")
    with pytest.raises(PipelineError):
        safe_path(tmp_path, "C:/outside.txt")


def test_single_target_selection_streams_source_tables_and_joins_label_evidence(tmp_path):
    root = tmp_path / "data"
    for folder in ("normalized", "screening"):
        (root / folder).mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"case_id": "case-1", "article_id": "article-1", "raw_case_text": "confirmed case text",
         "raw_text_sha256": "raw-hash", "age": 32.0, "gender": "unknown", "patient_link_id": None},
        {"case_id": "case-2", "article_id": "article-2", "raw_case_text": "not included",
         "raw_text_sha256": "other-hash", "age": 44.0, "gender": "unknown", "patient_link_id": None},
    ]), root / "normalized" / "cases.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"article_id": "article-1", "license": "CC-BY", "source_url": "https://example.org/1",
         "title": "Case one"},
        {"article_id": "article-2", "license": "CC-BY", "source_url": "https://example.org/2",
         "title": "Case two"},
    ]), root / "normalized" / "articles.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"case_id": "case-1", "candidate_disease": "DOID:1", "disease": "Disease one",
         "diagnosis_status": "confirmed", "diagnosis_evidence": "confirmed case text",
         "evidence_start": 0, "evidence_end": 19},
    ]), root / "screening" / "labels.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"case_id": "case-1", "confirmed_target_count": 1, "confirmed_target_diseases": ["DOID:1"]},
        {"case_id": "case-2", "confirmed_target_count": 0, "confirmed_target_diseases": []},
    ]), root / "screening" / "routing.parquet")

    rows = single_targets(Project(tmp_path))

    assert len(rows) == 1
    assert rows[0]["case_id"] == "case-1"
    assert rows[0]["disease"] == "Disease one"
    assert rows[0]["diagnosis_evidence"] == "confirmed case text"
    assert rows[0]["source_license"] == "CC-BY"


def test_ontology_scope_and_alias_match_keep_source_offsets():
    obo = '''[Term]\nid: DOID:0050117\nname: disease by infectious agent\n\n[Term]\nid: DOID:1\nname: viral disease\nis_a: DOID:0050117 ! parent\nsynonym: "virus infection" EXACT []\n\n[Term]\nid: DOID:2\nname: fracture\n'''
    terms = parse_obo(obo, {"infectious_roots": ["DOID:0050117"],
                            "excluded_concepts": ["DOID:0050117"],
                            "minimum_alias_length": 3, "extra_aliases": {}, "tropical_concepts": []})
    viral = next(row for row in terms if row["concept_id"] == "DOID:1")
    fracture = next(row for row in terms if row["concept_id"] == "DOID:2")
    assert viral["in_scope"] is True
    assert fracture["in_scope"] is False
    match = Matcher([viral]).find("α virus infection β")[0]
    assert "virus infection" == "α virus infection β"[match["start"]:match["end"]]
    assert Matcher([viral]).find("myvirus infection") == []


def test_evidence_is_exact_and_ambiguous_repeat_is_not_auto_accepted():
    text = "The PCR confirmed dengue. Dengue was not ruled out."
    spans = sentence_spans(text)
    sentence = text[slice(*spans[0])]
    start = text.index("dengue")
    match = {"field": "raw_case_text", "start": start, "end": start + 6}
    result = diagnose(text, [match])
    assert result["diagnosis_status"] == "confirmed"
    assert exact_evidence(text, result["diagnosis_evidence"], result["evidence_start"], result["evidence_end"])
    assert unique_quote_span("TB. TB.", "TB.") is None
    assert unique_quote_span(sentence, "PCR confirmed dengue") is not None


@pytest.mark.parametrize(("sentence", "alias", "status"), [
    ("PCR confirmed malaria.", "malaria", "confirmed"),
    ("Malaria was suspected.", "Malaria", "suspected"),
    ("Malaria was ruled out.", "Malaria", "ruled_out"),
    ("History of malaria.", "malaria", "historical"),
])
def test_status_rules_abstain_or_separate_negation_history(sentence, alias, status):
    start = sentence.lower().index(alias.lower())
    assert mention_status(sentence, start, start + len(alias)) == status


def test_union_find_connects_transitively():
    items = UnionFind(["a", "b", "c"])
    items.union("a", "b")
    items.union("b", "c")
    assert items.find("a") == items.find("c")


def test_wilson_interval_reports_uncertainty_without_inventing_a_point_estimate():
    assert wilson(9, 10)["lower_95"] < 0.9 < wilson(9, 10)["upper_95"]
    assert wilson(0, 0) == {"lower_95": None, "upper_95": None}


def test_group_split_keeps_groups_separate_and_all_classes_in_folds(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "split.yaml").write_text(yaml.safe_dump({
        "seed": 42, "test_fraction": 0.2, "development_folds": 5,
        "minimum_groups_per_class": 30, "preferred_groups_per_class": 50,
        "minimum_cases_per_class": 30}), encoding="utf-8")
    rows = []
    for label in ["DOID:1", "DOID:2"]:
        for index in range(35):
            cid = f"{label}-{index}"
            rows.append({"case_id": cid, "article_id": cid, "split_group": cid,
                         "candidate_disease": label, "raw_text_sha256": f"raw-{cid}",
                         "clinical_text_sha256": f"clean-{cid}", "image_sha256": f"image-{cid}",
                         "pixel_sha256": f"pixels-{cid}"})
    p = tmp_path / "data" / "qc" / "feasible_candidates.parquet"
    p.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), p)
    result = grouped_split(Project(tmp_path))
    assert result["samples"] == len(rows)
    assigned = Project(tmp_path).rows("data/splits/assignments.parquet")
    assert {r["diagnosis_label"] for r in assigned if r["partition"] == "frozen_test"} == {"DOID:1", "DOID:2"}
    assert {r["diagnosis_label"] for r in assigned if r["partition"] == "development"} == {"DOID:1", "DOID:2"}
    assert all({r["diagnosis_label"] for r in assigned if r["fold"] == i} == {"DOID:1", "DOID:2"}
               for i in range(5))
    assert HUMAN_REVIEW["properties"]["reviewer_id"]["minLength"] == 1


def test_human_export_preserves_imported_consensus_and_queues_qc_ready_conflict(tmp_path):
    project = Project(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "release.yaml").write_text(yaml.safe_dump({
        "human_review_all_below": 500, "human_development_fraction": 0.1,
    }), encoding="utf-8")
    source_rows = {
        "data/normalized/cases.parquet": [{
            "case_id": "case-1", "article_id": "article-1", "raw_case_text": "confirmed case text",
            "raw_text_sha256": "raw-hash", "age": 32.0, "gender": "unknown", "patient_link_id": None,
        }],
        "data/normalized/articles.parquet": [{
            "article_id": "article-1", "license": "CC-BY", "source_url": "https://example.org/1",
            "title": "Case one",
        }],
        "data/screening/routing.parquet": [{
            "case_id": "case-1", "confirmed_target_count": 1, "confirmed_target_diseases": ["DOID:1"],
        }],
        "data/screening/labels.parquet": [{
            "case_id": "case-1", "candidate_disease": "DOID:1", "disease": "Disease one",
            "diagnosis_status": "confirmed", "diagnosis_evidence": "confirmed case text",
            "evidence_start": 0, "evidence_end": 19,
        }],
        "data/qc/feasible_candidates.parquet": [],
        "data/qc/text_qc.parquet": [{
            "case_id": "case-1", "deterministic_pass": True, "clinical_text": "clinical case text",
            "clinical_text_sha256": "clinical-hash",
        }],
        "data/qc/selected_images.parquet": [{
            "case_id": "case-1", "image_id": "image-1", "path": "image.jpg", "image_sha256": "image-hash",
            "image_relevance": "supportive", "caption": "Clinical image",
        }],
        "data/reviews/adjudication.parquet": [],
        "data/reviews/human_requirements.parquet": [{
            "case_id": "case-1", "article_id": "article-1", "candidate_disease": "DOID:1",
            "target_disease": "Disease one", "raw_text_sha256": "raw-hash", "partition": None,
            "fold": None, "required": True, "requirement_priority": "high",
            "ai_decision": "human_review", "ai_diagnosis_status": "confirmed",
            "ai_confidence": "medium", "ai_reason_codes_json": "[]", "review_engine": "test",
            "review_method": "test", "review_version": "test", "reviewed_at": "2026-01-01T00:00:00Z",
            "requirement_source": "consensus_ab_v001.parquet", "requirement_stage": "consensus_ab_v001",
            "consensus": "NEEDS_ADJUDICATION", "requirement_reason": "NEEDS_ADJUDICATION",
        }],
    }
    for path, rows in source_rows.items():
        project.write(path, rows, columns=["case_id"] if not rows else None)

    result = prepare_human(project)

    requirements = project.rows("data/reviews/human_requirements.parquet")
    queue = project.rows("data/reviews/human_queue.parquet")
    assert result["required_human_reviews"] == 1
    assert len(requirements) == 1
    assert requirements[0]["consensus"] == "NEEDS_ADJUDICATION"
    assert requirements[0]["consensus_required"] is True
    assert requirements[0]["partition"] == "adjudication_queue"
    assert queue[0]["case_id"] == "case-1"


def test_current_reviews_accepts_source_bound_chatgpt_and_claude_imports(tmp_path):
    project = Project(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "review.yaml").write_text(yaml.safe_dump({
        "reviewer_a_model": "gemini-old", "reviewer_b_model": "gemini-old", "prompt_version": "review_v002",
    }), encoding="utf-8")
    sample = {"case_id": "case-1", "candidate_disease": "DOID:1", "disease": "Disease one",
              "raw_text_sha256": "raw-hash", "raw_case_text": "confirmed flu"}
    common = {"case_id": "case-1", "candidate_disease": "DOID:1", "raw_text_sha256": "raw-hash",
              "decision": "accept", "diagnosis_status": "confirmed", "disease_is_current": True,
              "confidence": "high", "evidence_quote": "confirmed flu", "evidence_start": 0, "evidence_end": 13,
              "evidence_valid": True, "multiple_target_diseases": False, "other_target_diseases": []}
    project.write("data/reviews/reviewer_a.parquet", [{
        **common, "model": "chatgpt", "prompt_version": "chatgpt_review_v003",
    }])
    project.write("data/reviews/reviewer_b.parquet", [{
        **common, "model": "claude", "external_review_source": "reviewer_b_claude_full.parquet",
    }])

    assert current_reviews(project, "a", [sample])["case-1"]["model"] == "chatgpt"
    assert current_reviews(project, "b", [sample])["case-1"]["model"] == "claude"


@pytest.mark.parametrize("medgemma_status", [None, "valid", "validation_failed"])
def test_human_label_adjudication_export_import_is_source_bound_and_idempotent(tmp_path, medgemma_status):
    project = Project(tmp_path)
    cases, articles, routes, labels, requirements, review_a, review_b = [], [], [], [], [], [], []
    specifications = [
        ("case-1", "article-1", "DOID:1", "malaria", "Patient had PCR confirmed malaria.",
         "PCR confirmed malaria", "raw-hash-1", "confirmed"),
        ("case-2", "article-2", "DOID:2", "dengue", "Dengue was suspected in this patient.",
         "Dengue was suspected", "raw-hash-2", "suspected"),
    ]
    for cid, aid, doid, disease, text, quote, text_hash, status in specifications:
        start = text.index(quote)
        common = {"case_id": cid, "article_id": aid, "candidate_disease": doid, "disease": disease,
                  "raw_text_sha256": text_hash, "evidence_start": start,
                  "evidence_end": start + len(quote), "evidence_quote": quote}
        cases.append({"case_id": cid, "article_id": aid, "raw_case_text": text,
                      "raw_text_sha256": text_hash, "age": 28.0, "gender": "unknown",
                      "patient_link_id": None})
        articles.append({"article_id": aid, "license": "CC-BY", "source_url": f"https://example.org/{cid}",
                         "title": f"Case {cid}"})
        routes.append({"case_id": cid, "confirmed_target_count": 1, "confirmed_target_diseases": [doid]})
        labels.append({**common, "diagnosis_status": status, "diagnosis_evidence": quote})
        requirements.append({"case_id": cid, "candidate_disease": doid, "raw_text_sha256": text_hash,
                             "consensus": "NEEDS_ADJUDICATION"})
        review_a.append({**common, "decision": "accept", "diagnosis_status": "confirmed",
                         "disease_is_current": True, "evidence_valid": True,
                         "multiple_target_diseases": False, "model": "chatgpt",
                         "prompt_version": "chatgpt_review_v003"})
        review_b.append({**common, "decision": "human_review", "diagnosis_status": "unclear",
                         "disease_is_current": False, "evidence_valid": False,
                         "evidence_quote": "", "evidence_start": None, "evidence_end": None,
                         "multiple_target_diseases": False, "source_decision": "uncertain",
                         "external_review_source": "reviewer_b_claude_full.parquet"})
    project.write("data/normalized/cases.parquet", cases)
    project.write("data/normalized/articles.parquet", articles)
    project.write("data/screening/routing.parquet", routes)
    project.write("data/screening/labels.parquet", labels)
    project.write("data/reviews/human_requirements.parquet", requirements)
    project.write("data/reviews/reviewer_a.parquet", review_a)
    project.write("data/reviews/reviewer_b.parquet", review_b)
    if medgemma_status:
        third = {**review_a[0], "processing_status": medgemma_status, "reported_decision": "accept",
                 "model_result_json": '{"decision":"accept"}', "evidence_alignment_json": "null",
                 "validation_error": None}
        if medgemma_status == "validation_failed":
            third.update(decision=None, evidence_valid=False, evidence_quote="",
                         validation_error="Model quote is not a source span")
        project.write("data/reviews/medgemma_labels.parquet", [third])
    schema_path = Path(__file__).parents[1] / "schemas" / "human_adjudication.json"
    (tmp_path / "schemas").mkdir()
    shutil.copyfile(schema_path, tmp_path / "schemas" / schema_path.name)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "review.yaml").write_text(yaml.safe_dump({
        "reviewer_a_model": "chatgpt", "reviewer_b_model": "claude", "random_b_fraction": 0,
    }), encoding="utf-8")

    exported = export_human_adjudication(project)
    queue_path = project.path("data/reviews/human_adjudication_input.jsonl")
    rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    if medgemma_status:
        assert row["medgemma_review"]["processing_status"] == medgemma_status
        assert row["medgemma_review"]["reported_decision"] == "accept"
        assert row["medgemma_review"]["model_result_json"] == '{"decision":"accept"}'
        if medgemma_status == "validation_failed":
            assert row["medgemma_review"]["decision"] is None
            assert row["medgemma_review"]["validation_error"]
    quote = row["rule_review"]["evidence_quote"]
    start = row["rule_review"]["evidence_start"]
    row["human_adjudication"] = {
        "human_decision": "accept", "human_status": "confirmed", "evidence_quote": quote,
        "multiple_target_diseases": False, "reason_code": "CLEAR_CONFIRMATION",
        "reviewer_id": "reviewer-test", "reviewed_at": "2026-10-03T12:30:00Z",
    }
    queue_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")

    imported = import_human_adjudication(project, "data/reviews/human_adjudication_input.jsonl")
    record = project.rows("data/reviews/human_label_adjudications.parquet")[0]
    repeated = import_human_adjudication(project, "data/reviews/human_adjudication_input.jsonl")
    adjudication_counts = adjudicate(project)
    decisions = {row["case_id"]: row for row in project.rows("data/reviews/adjudication.parquet")}

    assert exported["rows"] == 2
    assert imported["imported"] == 1
    assert imported["unique_cases_reviewed"] == 1
    assert repeated["imported"] == 0
    assert record["human_evidence_valid"] is True
    assert (record["evidence_start"], record["evidence_end"]) == (start, start + len(quote))
    assert adjudication_counts["HUMAN_ADJUDICATION_REQUIRED"] == 1
    assert decisions["case-1"]["reason"] == "HUMAN_LABEL_ADJUDICATED"
    assert decisions["case-1"]["verified"] is True

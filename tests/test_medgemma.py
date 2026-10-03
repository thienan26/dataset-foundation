import json

import pytest

from multicare_data.common import PipelineError, Project, read_json
from multicare_data.medgemma import cache_path, import_results, validate_result


def label_job():
    return {"kind": "label", "request_sha256": "a" * 64, "model": "google/medgemma-1.5-4b-it",
            "payload": {"raw_case_text": "Tuberculosis was confirmed."}}


def accepted():
    return {"diagnosis_status": "confirmed", "disease_is_current": True,
            "evidence_quote": "Tuberculosis was confirmed.", "multiple_target_diseases": False,
            "other_target_diseases": [], "decision": "accept", "confidence": "high",
            "reason_codes": ["EXPLICIT_DIAGNOSIS"], "reason": "Explicit confirmation."}


@pytest.mark.parametrize("change", [
    {"evidence_quote": "Invented quote"}, {"diagnosis_status": "suspected"},
    {"multiple_target_diseases": True}, {"disease_is_current": False},
])
def test_reject_invalid_acceptance(change):
    with pytest.raises(PipelineError):
        validate_result(label_job(), {**accepted(), **change})


def test_leakage_must_be_literal():
    with pytest.raises(PipelineError):
        validate_result({"kind": "text", "payload": {"clinical_text": "Fever and cough."}},
                        {"has_direct_leakage": True, "leaking_span": "tuberculosis",
                         "usable_prediagnosis_context": True, "reason": "Leakage"})


def test_import_validates_entire_batch_before_writing(tmp_path, monkeypatch):
    job = label_job()
    monkeypatch.setattr("multicare_data.medgemma.jobs", lambda _: [job])
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"request_sha256": job["request_sha256"], "model": job["model"],
                               "result": accepted()}) + "\n" + json.dumps(
                                   {"request_sha256": "stale", "model": job["model"], "result": accepted()}),
                    encoding="utf-8")
    with pytest.raises(PipelineError):
        import_results(Project(tmp_path), str(path))
    assert not cache_path(Project(tmp_path), job["request_sha256"]).exists()


def test_import_idempotent_and_conflicts_preserved(tmp_path, monkeypatch):
    job = label_job()
    project = Project(tmp_path)
    monkeypatch.setattr("multicare_data.medgemma.jobs", lambda _: [job])
    monkeypatch.setattr("multicare_data.medgemma.apply_results", lambda _: {"applied": True})
    path = tmp_path / "results.jsonl"
    row = {"request_sha256": job["request_sha256"], "model": job["model"], "result": accepted()}
    path.write_text(json.dumps(row), encoding="utf-8")
    import_results(project, str(path))
    saved = read_json(cache_path(project, job["request_sha256"]))
    import_results(project, str(path))
    assert read_json(cache_path(project, job["request_sha256"])) == saved
    row["result"]["reason"] = "Different response"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(PipelineError):
        import_results(project, str(path))
    assert read_json(cache_path(project, job["request_sha256"])) == saved


def test_third_reviewer_conflict_survives_adjudication(tmp_path, monkeypatch):
    from multicare_data.adjudicate import adjudicate

    project = Project(tmp_path)
    project.path("configs").mkdir()
    project.path("configs/review.yaml").write_text("random_b_fraction: 0.1", encoding="utf-8")
    project.path("configs/medgemma.yaml").touch()
    case = {"case_id": "case1", "candidate_disease": "DOID:1", "raw_text_sha256": "source"}
    review = {**accepted(), **case, "evidence_valid": True}
    project.write("data/reviews/medgemma_labels.parquet", [{**review, "decision": "uncertain",
                                                          "request_sha256": "current"}])
    monkeypatch.setattr("multicare_data.adjudicate.single_targets", lambda _: [case])
    monkeypatch.setattr("multicare_data.adjudicate.current_reviews", lambda *_: {"case1": review})
    monkeypatch.setattr("multicare_data.medgemma.jobs", lambda _: [
        {"case_id": "case1", "kind": "label", "request_sha256": "current"}])
    adjudicate(project)
    result = project.rows("data/reviews/adjudication.parquet")[0]
    assert result["verified"] is False
    assert result["reason"] == "HUMAN_ADJUDICATION_REQUIRED"


def test_changed_prompt_invalidates_previous_medgemma_qc(tmp_path, monkeypatch):
    from multicare_data.medgemma import apply_results

    project = Project(tmp_path)
    project.write("data/qc/text_qc.parquet", [{"case_id": "case1", "text_usable": True,
                                             "ai_provider": "medgemma", "ai_request_sha256": "old"}])
    project.write("data/qc/image_qc.parquet", [{"case_id": "case1", "image_id": "image1",
                                              "image_usable": True, "ai_provider": "medgemma",
                                              "ai_request_sha256": "old"}])
    monkeypatch.setattr("multicare_data.medgemma.jobs", lambda _: [
        {"case_id": "case1", "kind": "text", "request_sha256": "new_text"},
        {"case_id": "case1", "image_id": "image1", "kind": "image", "request_sha256": "new_image"}])
    result = apply_results(project)
    assert result["ai_complete"] is False
    assert result["selected_cases"] == 0
    assert project.rows("data/qc/text_qc.parquet")[0]["text_usable"] is False
    assert project.rows("data/qc/image_qc.parquet")[0]["image_usable"] is False


def test_current_scope_only_reviews_unresolved_conflicts(tmp_path, monkeypatch):
    from multicare_data.medgemma import jobs

    project = Project(tmp_path)
    project.path("configs").mkdir()
    project.path("configs/medgemma.yaml").write_text(
        "model: medgemma\nprompt_version: test\nmax_output_tokens: 256\nscope: human_conflicts\n",
        encoding="utf-8")
    base = {"article_id": "article", "candidate_disease": "DOID:1", "raw_text_sha256": "source",
            "raw_case_text": "Confirmed tuberculosis.", "disease": "tuberculosis"}
    monkeypatch.setattr("multicare_data.medgemma.single_targets", lambda _: [
        {**base, "case_id": "agreed"}, {**base, "case_id": "conflict"}, {**base, "case_id": "resolved"}])
    project.write("data/reviews/human_requirements.parquet", [
        {"case_id": "agreed", "consensus": "VERIFIED"},
        {"case_id": "conflict", "consensus": "NEEDS_ADJUDICATION"},
        {"case_id": "resolved", "consensus": "NEEDS_ADJUDICATION"}])
    project.write("data/reviews/human_label_adjudications.parquet", [
        {"case_id": "resolved", "human_decision": "accept"}])
    project.write("data/qc/text_qc.parquet", [{"case_id": "conflict", "deterministic_pass": True}])
    project.write("data/qc/image_qc.parquet", [{"case_id": "conflict", "decode_pass": True}])
    result = jobs(project)
    assert [(r["case_id"], r["kind"]) for r in result] == [("conflict", "label")]

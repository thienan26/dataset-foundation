import io
import json
import urllib.error

import pytest

from multicare_data.common import PipelineError, Project, read_json, write_json
from multicare_data.evidence import align_whitespace_quote
from multicare_data.medgemma import (
    cache_path,
    failure_path,
    grounded_result,
    prompt_text,
    recover_failed_reviews,
)
from multicare_data.medgemma_batch import run
from multicare_data.review_schema import LABEL_REVIEW


def job(case_id, source="Tuberculosis was confirmed."):
    return {"kind": "label", "case_id": case_id, "candidate_disease": "DOID:1", "raw_text_sha256": "source",
            "request_sha256": case_id, "model": "medgemma", "prompt_version": "test",
            "payload": {"case_id": case_id, "raw_case_text": source}, "schema": LABEL_REVIEW,
            "max_output_tokens": 256}


def accepted(quote="Tuberculosis was confirmed."):
    return {"diagnosis_status": "confirmed", "disease_is_current": True, "evidence_quote": quote,
            "multiple_target_diseases": False, "other_target_diseases": [], "decision": "accept",
            "confidence": "high", "reason_codes": ["CONFIRMED"], "reason": "Confirmed by source."}


def response(result):
    return {"model": "medgemma", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\\n", " \\n\t ", "\t  "])
def test_whitespace_alignment_preserves_exact_source(separator):
    source = "Before. The test" + separator + "confirmed tuberculosis. After."
    span = align_whitespace_quote(source, "The test confirmed tuberculosis.")
    assert span is not None
    assert source[span[0]:span[1]] == "The test" + separator + "confirmed tuberculosis."


@pytest.mark.parametrize("source,quote", [
    ("Confirmed\nTB. Confirmed TB.", "Confirmed TB."),
    ("Confirmed: TB.", "Confirmed TB."),
    ("Confirmed\nTB.", "confirmed TB."),
    ("Confirmed TB.", "Diagnosed TB."),
])
def test_alignment_rejects_ambiguous_or_changed_words(source, quote):
    assert align_whitespace_quote(source, quote) is None


def test_grounding_keeps_original_model_quote_in_audit():
    original = accepted("The test confirmed tuberculosis.")
    result, audit = grounded_result(job("case", "The test\\nconfirmed tuberculosis."), original)
    assert original["evidence_quote"] == "The test confirmed tuberculosis."
    assert result["evidence_quote"] == "The test\\nconfirmed tuberculosis."
    assert audit["model_evidence_quote"] == original["evidence_quote"]


def make_project(tmp_path):
    project = Project(tmp_path)
    project.path("configs").mkdir()
    project.path("configs/medgemma.yaml").write_text(
        "timeout_seconds: 1\nmax_attempts: 3\ncheckpoint_every: 1\n", encoding="utf-8")
    project.write("data/qc/text_qc.parquet", [])
    project.write("data/qc/image_qc.parquet", [])
    return project


def test_recover_existing_failure_without_new_inference(tmp_path):
    project = make_project(tmp_path)
    item = job("case", "The test\nconfirmed tuberculosis.")
    raw = response(accepted("The test confirmed tuberculosis."))
    audit = {"request_sha256": "case", "case_id": "case", "attempts": [
        {"messages": [{"content": [{"text": prompt_text(item)}]}], "response": raw}]}
    write_json(failure_path(project, "case"), audit)
    assert recover_failed_reviews(project, [item]) == 1
    saved = read_json(cache_path(project, "case"))
    assert saved["result"]["evidence_quote"] == "The test\nconfirmed tuberculosis."
    assert saved["raw_response"] == raw
    assert recover_failed_reviews(project, [item]) == 0


def test_recovery_rejects_wrong_source_prompt(tmp_path):
    project = make_project(tmp_path)
    item = job("case")
    write_json(failure_path(project, "case"), {"request_sha256": "case", "case_id": "case", "attempts": [
        {"messages": [{"content": [{"text": "different source"}]}], "response": response(accepted())}]})
    assert recover_failed_reviews(project, [item]) == 0


@pytest.mark.parametrize("limit,processed,pending", [(1, 1, 1), (2, 2, 0)])
def test_failed_case_does_not_abort_later_cases_or_count_as_accept(tmp_path, monkeypatch, limit, processed, pending):
    project = make_project(tmp_path)
    items = [job("case_bad"), job("case_good")]
    monkeypatch.setattr("multicare_data.medgemma_batch.jobs", lambda _: items)
    monkeypatch.setenv("MEDGEMMA_BASE_URL", "http://127.0.0.1:8087/v1")
    calls = []

    def urlopen(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        result = accepted("An invented source quote.") if "case_bad" in json.dumps(body) else accepted()
        return io.BytesIO(json.dumps(response(result)).encode())

    monkeypatch.setattr("multicare_data.medgemma_batch.urllib.request.urlopen", urlopen)
    result = run(project, limit)
    assert result["processed_cases"] == processed
    assert sum(result["pending"].values()) == pending
    assert result["failed"] == {"label": 1}
    assert result["decision_counts"] == ({"accept": 1} if limit == 2 else {})
    assert result["ai_complete"] is False
    assert result["pass_complete"] is (pending == 0)
    rows = project.rows("data/reviews/medgemma_labels.parquet")
    assert rows[0]["decision"] is None
    assert rows[0]["reported_decision"] == "accept"
    assert rows[0]["processing_status"] == "validation_failed"
    assert len(calls) == (4 if limit == 2 else 3)
    if limit == 2:
        monkeypatch.setattr("multicare_data.medgemma_batch.urllib.request.urlopen",
                            lambda *_args, **_kwargs: pytest.fail("Resume must not repeat completed cases"))
        assert run(project, 2)["attempted_this_run"] == 0


def test_server_access_error_keeps_unattempted_cases_pending(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    monkeypatch.setattr("multicare_data.medgemma_batch.jobs", lambda _: [job("case")])
    monkeypatch.setenv("MEDGEMMA_BASE_URL", "http://127.0.0.1:8087/v1")

    def unavailable(*_args, **_kwargs):
        raise urllib.error.HTTPError("http://127.0.0.1", 401, "unauthorized", {}, io.BytesIO(b"error"))

    monkeypatch.setattr("multicare_data.medgemma_batch.urllib.request.urlopen", unavailable)
    with pytest.raises(PipelineError, match="HTTP 401"):
        run(project, 1)
    summary = read_json(project.path("reports/medgemma_progress.json"))
    assert summary["pending"] == {"label": 1}
    assert summary["failed"] == {}


@pytest.mark.parametrize("raw", [[], {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}])
def test_malformed_response_envelope_is_routed_to_human(tmp_path, monkeypatch, raw):
    project = make_project(tmp_path)
    monkeypatch.setattr("multicare_data.medgemma_batch.jobs", lambda _: [job("case")])
    monkeypatch.setenv("MEDGEMMA_BASE_URL", "http://127.0.0.1:8087/v1")
    monkeypatch.setattr("multicare_data.medgemma_batch.urllib.request.urlopen",
                        lambda *_args, **_kwargs: io.BytesIO(json.dumps(raw).encode()))
    summary = run(project, 1)
    assert summary["pass_complete"] is True
    assert summary["failed"] == {"label": 1}
    row = project.rows("data/reviews/medgemma_labels.parquet")[0]
    assert row["decision"] is None
    assert row["reported_decision"] is None
    assert row["processing_status"] == "validation_failed"


@pytest.mark.parametrize("outcome", ["pending", "validation_failed"])
def test_final_human_handoff_requires_complete_pass_and_preserves_ab_consensus(tmp_path, monkeypatch, outcome):
    from multicare_data.final_review import prepare

    project = make_project(tmp_path)
    cases = [{"case_id": cid, "candidate_disease": "DOID:1", "disease": "tuberculosis",
              "raw_text_sha256": "source", "raw_case_text": "Tuberculosis was confirmed.",
              "diagnosis_evidence": "Tuberculosis was confirmed."}
             for cid in ("agreed_accept", "agreed_reject", "conflict")]
    monkeypatch.setattr("multicare_data.final_review.single_targets", lambda _: cases)
    monkeypatch.setattr("multicare_data.adjudicate.adjudicate", lambda _: None)
    monkeypatch.setattr("multicare_data.grouping.group", lambda _: None)
    monkeypatch.setattr("multicare_data.grouping.feasibility", lambda _: {"feasible_classes": []})
    monkeypatch.setattr("multicare_data.adjudicate.prepare_human", lambda *_: {"rows": 0})
    monkeypatch.setattr("multicare_data.human_adjudication.export_human_adjudication", lambda _: {"rows": 1})
    complete = outcome == "validation_failed"
    monkeypatch.setattr("multicare_data.medgemma.apply_results",
                        lambda _: {"pass_complete": complete, "ai_complete": False})
    for reviewer, decision in (("a", "accept"), ("b", "reject")):
        project.write(f"data/reviews/reviewer_{reviewer}.parquet", [
            {"case_id": "agreed_accept", "decision": "accept"},
            {"case_id": "agreed_reject", "decision": "reject"},
            {"case_id": "conflict", "decision": decision}])
    project.write("data/qc/selected_images.parquet", [])
    project.write("data/groups/groups.parquet", [
        {"case_id": case["case_id"], "split_group": case["case_id"], "duplicate_conflict": False} for case in cases])
    project.write("data/reviews/adjudication.parquet", [
        {"case_id": "agreed_accept", "verified": True, "review_status": "verified", "reason": "AB_AGREEMENT"},
        {"case_id": "agreed_reject", "verified": False, "review_status": "rejected", "reason": "AB_REJECTED"},
        {"case_id": "conflict", "verified": False, "review_status": "unresolved",
         "reason": "HUMAN_ADJUDICATION_REQUIRED"}])
    if complete:
        project.write("data/reviews/medgemma_labels.parquet", [
            {"case_id": "conflict", "decision": None, "processing_status": outcome,
             "validation_error": "Invalid exact evidence"}])
        summary = prepare(project)
        assert summary["status"] == "MEDGEMMA_PASS_COMPLETE_AWAITING_HUMAN"
    else:
        with pytest.raises(PipelineError, match="pass is incomplete"):
            prepare(project)
    assert project.path("data/reviews/final_review.jsonl").exists() is complete
    accepted_rows = project.rows("data/reviews/final_ab_accepted.parquet")
    assert [row["case_id"] for row in accepted_rows] == ["agreed_accept"]
    name = "final_review" if complete else "pre_human_review_draft"
    rows = {row["case_id"]: row for row in project.rows(f"data/reviews/{name}.parquet")}
    assert rows["agreed_reject"]["label_status"] == "REJECTED_AB_CONSENSUS"
    assert all(row["human_review_completed"] is False for row in rows.values())
    assert rows["conflict"]["requires_human_label_review"] is True
    assert rows["conflict"]["medgemma_decision"] == (None if complete else "pending")
    if complete:
        assert rows["conflict"]["label_status"] == "MEDGEMMA_FAILED_AWAITING_HUMAN"
        assert "MEDGEMMA_VALIDATION_FAILED" in rows["conflict"]["blocking_reasons"]

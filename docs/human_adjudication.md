# Human label adjudication

This workflow resolves the current Reviewer A/B label conflicts. It is separate from `human-export`: it adjudicates the diagnosis label from the source case narrative and does not certify text leakage or image quality.

Export the unresolved set locally:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m multicare_data.cli human-adjudication-export
```

The command writes `data/reviews/human_adjudication_input.jsonl`, one row for each case whose current consensus is `NEEDS_ADJUDICATION`. Each row includes the original case text and source hash, rule-engine evidence, and Reviewer A/B decisions and evidence. Edit only the nested `human_adjudication` fields.

For every case, read the source text and assess whether the proposed disease is explicitly confirmed for the current patient. Treat the rule and AI outputs as evidence to compare with the source, not as votes. Use:

- `human_decision`: `accept` only for one current, explicitly confirmed target diagnosis; `reject` when the source does not support that target; `uncertain` when the case remains ambiguous.
- `human_status`: one of `confirmed`, `probable`, `suspected`, `differential`, `ruled_out`, `historical`, or `unclear`.
- `evidence_quote`: the shortest complete literal source span supporting the decision. For `accept`, it must occur exactly once in `raw_case_text` and support a confirmed diagnosis.
- `multiple_target_diseases`: true only when the case confirms multiple in-scope target diseases for the current patient.
- `reason_code`: a short code such as `CLEAR_CONFIRMATION`, `HISTORICAL_MENTION`, `DIFFERENTIAL_ONLY`, `NO_PATIENT_DIAGNOSIS`, `MULTIPLE_TARGETS`, or `AMBIGUOUS_SOURCE`.
- `reviewer_id`: a stable reviewer pseudonym; `reviewed_at`: an ISO 8601 timestamp in UTC, such as `2026-10-03T12:30:00Z`.

Import the completed queue and refresh adjudication:

```powershell
python -m multicare_data.cli human-adjudication-import data/reviews/human_adjudication_input.jsonl
python -m multicare_data.cli adjudicate
```

You can work in batches: leave untouched rows as exported, fill only the cases reviewed in that batch, then run `human-adjudication-import`. The importer skips untouched rows, validates each completed row, and ignores exact duplicate imports. Re-run `human-adjudication-export` to refresh the queue while preserving imported decisions.

The importer checks the case ID, target disease, raw-text hash, A/B context and exact evidence. Rejected or uncertain cases remain outside the eligible pool. Accepted label adjudications can resolve the label disagreement, but they do not satisfy the later release audit: text leakage and image relevance still need their own review, and the predeclared human release-review queue must still be completed after feasible cases and splits exist.

The broader `human-import` command is for that later release audit. It expects cleaned-text and selected-image hashes, so it cannot import this label-only queue.

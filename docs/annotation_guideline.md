# Annotation guideline

## Disease mention vs diagnosis

Search hits are candidate hypotheses, never labels. Determine whether the disease belongs to the current case patient and is explicitly confirmed in the case narrative. Require the shortest complete literal sentence or clause that supports the diagnosis; store exact zero-based `[start, end)` Python string offsets.

Use the controlled statuses `confirmed`, `probable`, `suspected`, `differential`, `ruled_out`, `historical`, and `unclear`. Negative, historical, family-history, speculative, and contradictory evidence cannot be accepted. Multiple current confirmed in-scope diseases route the complete case to quarantine.

Automated language rules deliberately abstain unless confirmation terms are explicit. LLM output does not repair or paraphrase an evidence span. Invalid or repeated quotes are manually adjudicated from the source.

## Text and image assessment

Retain source text only through the point before diagnosis evidence and remove direct aliases, diagnostic claims, revealing titles and obvious postdiagnosis treatment/outcome statements. Confirm that useful prediagnosis context remains and direct diagnosis leakage is absent.

Each candidate image must be opened and assessed in context. Relevance must be diagnostic or supportive, the case/image match must be credible and no visible label, caption overlay or report text may directly name the disease/pathogen. Post-treatment, irrelevant, ambiguous or unopenable images are excluded. Choose one eligible image per case using the configured relevance, modality, metadata, resolution and image-id order.

## Human review JSONL

Review exports contain the frozen case-text hash, candidate concept ID, case text, proposed literal evidence, cleaned text/hash, selected image/checksum, and review fields. Preserve the original source and do not edit normalized source data. Each exported line uses `schemas/human_review.json`; `human import` checks the source hash, case and disease, accepted exact evidence, image selection and text/image audit before appending an immutable review record.

An accept response means all checks pass. Reject or uncertain cases remain outside the release. Record a stable reviewer pseudonym and UTC timestamp; never include a patient's identity.


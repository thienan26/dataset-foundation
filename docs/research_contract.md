# Research contract: dataset_v001

## Objective

Curate a reproducible, evidence-backed, single-label multimodal dataset of confirmed infectious-disease cases from the exact MultiCaRe 3.0.1 source release. Taxonomy selection follows observed case, image, and independent-group support; it is not a preset number of popular diseases.

## Unit, inclusion and exclusion

- Unit of observation: one MultiCaRe patient case. An article is provenance and a grouping key, never a sample.
- A release row pairs one confirmed diagnosis with prediagnosis clinical text and one selected relevant image.
- Candidate diseases are discovered using the pinned Human Disease Ontology. Explicit tropical-disease membership is recorded only when supported by a curated annotation or source-controlled concept; unknown is not silently treated as true.
- A mention in an article title, abstract, keyword, or case narrative is a search hypothesis. Only a current, explicit diagnosis supported by source-exact evidence may qualify.
- Zero confirmed in-scope diagnoses is excluded from the release and preserved in screening/quarantine outputs. Two or more confirmed target diagnoses are preserved in quarantine as `MULTI_TARGET_DIAGNOSIS`.
- Ambiguity, contradiction, absent/invalid evidence, unresolved AI disagreement, license uncertainty, text leakage, or image failure cannot be promoted to a release label.

## Input, evidence and leakage

- Raw MultiCaRe files and their lock are immutable after acquisition. Every source and release artifact is SHA-256 fingerprinted; publisher checksums are also verified on acquisition.
- `raw_case_text`, literal diagnosis evidence and `clinical_text` remain separate. Evidence offsets use Python Unicode code-point indexing and must satisfy `raw_case_text[start:end] == diagnosis_evidence`.
- Model text is a deterministic source prefix ending before diagnosis evidence. Disease aliases, postdiagnosis material, title, abstract, and image caption are not model text. Deterministic and AI leakage checks must pass.
- A single, deterministically selected image must decode, be judged diagnostic/supportive for this case, contain no direct diagnosis text and retain its own source license and hashes.
- Reviewer A sees the candidate disease and case text, but no rule conclusion. Reviewer B checks uncertain/disagreeing cases and a predeclared random audit sample. AI is annotation support, not ground truth. Human review resolves disagreements and audits the required subset.

## Independence, taxonomy and freeze

- Related cases are unioned before split by source article, exact raw/clinical text, exact selected pixels/image, and detectable near-duplicate image. Conflicting labels within an exact text/pixel group fail the release gate.
- Disease feasibility reports counts for confirmed candidates, accepted text/image cases and independent groups. Classes below the predeclared 30-group project threshold do not enter the main taxonomy. Fifty groups is preferred. These are project design thresholds, not clinical/statistical laws.
- The release uses approximately 80% development and 20% frozen test, grouped by connected component. Five group-disjoint folds are stored inside development. Every class must remain represented in every partition; otherwise no release is created.
- The release gate requires complete source provenance/licensing, exact evidence, one target label, completed required reviews, usable leakage-free text, one reviewed decodeable image, no unresolved duplicate conflicts and zero overlap across partitions.
- Frozen output is write-once at `data/releases/dataset_v001`. Any subsequent protocol/taxonomy change requires a new version.

## External review and data handling

Gemini label requests send one target disease and its case narrative, without the rule decision, title or image; image review is a separate QC request. Set `GEMINI_API_KEY` only in an ignored local `.env` or process environment. Review is rate-limited and resumable. `data/reviews/gemini_input.parquet` snapshots requests, `gemini_raw.jsonl` preserves raw successful responses, `gemini_reviews.parquet` stores normalized results and `gemini_failures.parquet` records failed requests. Evidence quotes are verified and offsets computed from source text in Python. Cached replies are bound to prompt, model, source content and checksum. Google's [current pricing page](https://ai.google.dev/gemini-api/docs/pricing) states free-tier content may be used to improve its products; the study team must confirm this data use is approved before sending source material. Do not commit credentials.

## Source citation and rights

Nievas Offidani, M. *MultiCaRe: An open-source clinical case dataset for medical image classification and multimodal AI applications*, version 3.0.1. Zenodo record 20416562. <https://doi.org/10.5281/zenodo.20416562>. The record describes the dataset-wide CC BY-NC-SA 4.0 license and per-article/per-image license variation. Cite and preserve each article/image license; unknown source rights are not cleared by the aggregate license.


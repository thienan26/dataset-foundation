# MultiCaRe infectious dataset foundation

This repository builds an auditable, single-label, multimodal infectious-disease dataset from the exact MultiCaRe 3.0.1 release. It preserves the research contract in [`docs/research_contract.md`](docs/research_contract.md): one case plus one reviewed image, literal diagnosis evidence, separate prediagnosis model text, human audit, duplicate-aware splits and an immutable release gate.

The source is pinned to [Zenodo record 20416562](https://doi.org/10.5281/zenodo.20416562), version 3.0.1, published 27 May 2026. The download archives are several gigabytes, so the complete source, extracted images, Python environment and temporary download files all stay under the selected project root on drive D. The source is ignored by Git.

## Windows setup

Open PowerShell in this repository. Python 3.11 or newer is required. Run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:TEMP = Join-Path (Get-Location) '.tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -m pip install '.[dev]'
```

If this computer lacks space in its system temporary directory, the commands above direct package-install scratch files to this D-drive repository.

## Acquire the source

Start with the smaller normalized tables:

```powershell
python -m multicare_data.cli resolve-latest
python -m multicare_data.cli acquire --tables-only
python -m multicare_data.cli verify-source
python -m multicare_data.cli normalize
python -m multicare_data.cli ontology
python -m multicare_data.cli discover
python -m multicare_data.cli build-candidates
python -m multicare_data.cli label
```

`resolve-latest` reports the latest Zenodo record. `acquire` still downloads the exact audited pin and will never silently move to a newer release. To obtain the original medical images, download and check all nine pinned PMC archives, then safely extract them:

```powershell
python -m multicare_data.cli acquire
python -m multicare_data.cli verify-source --require-images
python -m multicare_data.cli extract-images
```

The download resumes from `.part` files and checks publisher size/MD5 plus a locally recorded SHA-256. Do not edit `data/raw/multicare` or its lock.

## Reviews, QC and freeze

The requested completion path adds **local MedGemma only for the 1,170 A/B
conflicts**, before human adjudication. The 1,441 A/B accepted cases enter the
final accepted-label table directly; 476 agreed rejections remain rejected. See
[`docs/medgemma_scope.md`](docs/medgemma_scope.md) for the current local Q4 workflow.
Run `medgemma-run --backend local --limit 1` to validate a first response, then
`medgemma-run --backend local --limit 10000` to finish the pass, and
`prepare-human-review` to generate `data/reviews/final_review.jsonl` and Parquet.
The final bundle is emitted only after all MedGemma jobs complete. While waiting
for model access, `prepare-human-review --allow-pending` produces a clearly named
diagnostic draft. The offline workflow below describes the prior A/B baseline.

The active review workflow is offline. Reviewer A is the validated ChatGPT full pass in `data/reviews/reviewer_a.parquet`; Reviewer B is the imported Claude pass in `data/reviews/reviewer_b.parquet`. These tables cover all 3,087 source cases, so adjudication reads them locally and does not need `GEMINI_API_KEY` or network access. Avoid `--with-ai` QC options in this offline workflow. The code retains optional Gemini integrations for other runs, but they are not part of the current review pass.

To combine the pre-review artifacts into one local table, run `python -m multicare_data.cli export-pre-ai-review`. It writes one Parquet row per single-target case to `data/reviews/pre_ai_review.parquet`, joining normalized case/article fields with candidate hypotheses, rule evidence, scope and routing. The `gemini_label_payload_json` column is the blinded payload prepared for label review. Other columns include rule conclusions and article metadata, so do not send the whole table to Gemini.

The latest review bundle includes `reviewer_a_chatgpt_full_validation.json`, `reviewer_b_claude_full.parquet`, `reviewer_b_claude_full_validation.json`, `consensus_ab_v002.parquet` and `review_b_summary.json`. `human_requirements.parquet` keeps one row per case and marks `required=true` for `NEEDS_ADJUDICATION` consensus rows: 1,441 are verified, 476 rejected and 1,170 need human adjudication. Reviewer B's `uncertain` decisions are normalized to `human_review` for the pipeline. Of its 3,087 cases, 1,424 were manual snippet reviews and 1,663 were assigned by cue-based rules, so this remains a heuristic pass rather than clinician review. The prior B artifact and previous A-only register were removed; `human-export` preserves the current consensus register and only queues unresolved cases after deterministic text QC and image selection pass.

Historical Gemini audit files remain under `data/reviews/gemini_*`; old Reviewer A rows and caches were removed while Reviewer B's audit history was retained. The active operational tables are `reviewer_a.parquet` and `reviewer_b.parquet`. Both imported review sets are source-bound by case ID, concept ID, text hash and validated evidence offsets. The consensus path uses both reviews and sends conflicts to human adjudication. The current MedGemma pass reviews only those label conflicts and preserves A/B agreements; its validated answers are included as context for human adjudication.

Refresh the offline consensus and human adjudication queue with:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m multicare_data.cli adjudicate
```

To adjudicate the current 1,170 A/B conflicts before image and text QC finish, use the separate label-only workflow in [`docs/human_adjudication.md`](docs/human_adjudication.md). Export the source-bound queue with `human-adjudication-export`, fill its `human_adjudication` fields, then import it with `human-adjudication-import`. This resolves label conflicts only; the later release audit still requires text/image QC and the full `human-export` review.

(`adjudicate` is also refreshed by `evaluate`; if you need an intermediate queue, `evaluate` runs all non-destructive checks and emits it.) For this offline workflow, run deterministic QC without AI options:

```powershell
python -m multicare_data.cli text-qc
python -m multicare_data.cli image-qc
```

Run the QC commands again to continue. Cache keys include the exact source/input hash, reviewer model and prompt version. Without `--with-ai`, deterministic checks still run and unreviewed items stay ineligible.

```powershell
python -m multicare_data.cli deduplicate
python -m multicare_data.cli feasibility
python -m multicare_data.cli freeze-taxonomy
python -m multicare_data.cli split
python -m multicare_data.cli human-export
```

Review `data/reviews/human_input.jsonl` in line-delimited JSON form without changing the frozen IDs or hashes. The queue includes every frozen-test sample, a fixed 10% development sample, and every sample when the feasible set is below 500. Exported rows are input snapshots; check each image in its original resolution. Import the completed records from a separate JSONL file, then run:

```powershell
python -m multicare_data.cli human-import path\to\completed_reviews.jsonl
python -m multicare_data.cli feasibility
python -m multicare_data.cli split
python -m multicare_data.cli human-export
python -m multicare_data.cli adjudicate
python -m multicare_data.cli evaluate
python -m multicare_data.cli freeze --version dataset_v001
python -m multicare_data.cli verify-release dataset_v001
```

The gate reports failures; it does not weaken requirements to produce a release. If no class reaches the configured independent-group threshold, no `dataset_v001` will be emitted.

## CLI overview

Run `python -m multicare_data.cli --help` or `python -m multicare_data.cli <command> --help` from the project root with the virtual environment active. Pipeline files live under `src/multicare_data/`; project decisions live under `configs/`; controlled schemas and annotation policies live under `schemas/` and `docs/`. There is no model-training, RAG, API-server or deployment code in this repository.

## License and citation

The source record states CC BY-NC-SA 4.0 overall, with varying per-article and per-image licenses. The curated release carries a per-row license ledger; retain attribution and check commercial-use restrictions for every input. Cite MultiCaRe version 3.0.1 using the DOI in [`docs/research_contract.md`](docs/research_contract.md).

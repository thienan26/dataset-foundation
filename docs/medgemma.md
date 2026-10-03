# Local MedGemma before human review

**Current scope:** follow [medgemma_scope.md](medgemma_scope.md). MedGemma reviews
only unresolved A/B label conflicts using the Q4 llama.cpp runtime. The broader
Transformers/text/image workflow below is retained for optional future runs
(`runtime: transformers`, `scope: full`), not the current requested pass.

The model is `google/medgemma-1.5-4b-it`. Accept access on its Hugging Face
model page, then set `HF_TOKEN` in the project's `.env`. Never commit this token.
Weights are cached in `models/huggingface` on the project drive.
Local inference uses CUDA when available, otherwise CPU. Weights use bfloat16
to reduce memory use; `local_dtype: float32` is available in `configs/medgemma.yaml`.
CPU inference
for thousands of reviews can take a long time and needs enough RAM for 4B weights.
After setup, `scripts/run_medgemma_to_human.ps1` runs the remaining local pass
and prepares the final human review artifacts in one command.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $PWD 'src'
$env:TEMP = Join-Path $PWD '.tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip install '.[medgemma]'
.\.venv\Scripts\python.exe -m multicare_data.cli medgemma-run --backend local --limit 1
# Resume the full pass after the first response validates.
.\.venv\Scripts\python.exe -m multicare_data.cli medgemma-run --backend local --limit 10000
.\.venv\Scripts\python.exe -m multicare_data.cli prepare-human-review
```

The MedGemma pass covers labels, deterministic-pass text, and decodable images.
Each validated response is saved immediately in an immutable source-bound cache.
Reruns resume without repeating completed requests. Under the current GGUF
workflow, invalid outputs are recorded under `data/reviews/medgemma_failures`
and routed to human review without aborting subsequent cases. The optional
Transformers workflow still stops on invalid output.
Prompts and output schemas are exported with `medgemma-export`.
Alternatively run those jobs elsewhere and import JSONL lines containing
`request_sha256`, `model`, and `result` with `medgemma-import path/to/results.jsonl`.
Imports validate the whole batch before writing new cache entries.

`prepare-human-review` refreshes QC, grouping, feasibility, split (when feasible),
and the existing human queue. It writes `final_review.parquet` and
`final_review.jsonl` only after every MedGemma job is complete. The final bundle
contains all single-target cases, including exclusions and unresolved conflicts,
with all three reviewers, text QC, every image QC result, and source hashes.
MedGemma is an additional reviewer: it does not replace ChatGPT/Claude or human
decisions. A MedGemma disagreement with automated acceptance goes to human
adjudication. Cases without usable text/images stay excluded.

`prepare-human-review --allow-pending` emits `pre_human_review_draft.*` and a
status report while model access or responses are pending. This is a diagnostic
draft, not a final review or release. Complete human annotations in
`human_input.jsonl` using the existing human-import schema. Resolve label-only
conflicts using the documented human-adjudication workflow. Dataset freeze
remains gated on completed human review and the existing release checks.

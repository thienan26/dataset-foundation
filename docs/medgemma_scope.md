# Current requested review scope

Reviewer A and B already cover 3,087 source cases. Preserve their agreements:
1,441 accepted cases enter `data/reviews/final_ab_accepted.parquet`; 476 rejected
cases retain their rejection. MedGemma reviews only the remaining 1,170
`NEEDS_ADJUDICATION` cases. It does not repeat agreed cases or run text/image QC
under this scope.

The configured local runtime uses llama.cpp with Unsloth's Q4_K_M conversion of
MedGemma 1.5 4B. The model and runtime downloads are checksummed in
`models/medgemma_local.lock.json`; prompts bind the quantized model name.
Weights and binaries stay under `models/` on the project drive.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe scripts/setup_medgemma_local.py
.\scripts\run_medgemma_to_human.ps1
```

The final review bundle includes all 3,087 cases. A/B agreements retain their
decision. Disputed cases carry the independent MedGemma result and still await
human adjudication; this model pass never fabricates human annotations.
`human_adjudication_input.jsonl` includes the MedGemma context once available.
Each case has at most three output-validation attempts. Whitespace-only quote
differences can be mapped to a unique untouched source span; the original model
quote and mapping are retained in the raw audit. Changes to words, punctuation,
case, or ambiguous quotes are not repaired.

One malformed response no longer aborts the pass. Exhausted cases are recorded
with `processing_status: validation_failed`, a null validated `decision`, the
model's `reported_decision`, and the error/audit for human review. Failed cases
are never counted as accept. `pass_complete` means all scoped jobs have a
validated result or an explicit failed outcome; `ai_complete` also requires no
failed outcomes. The final review bundle can include failed outcomes awaiting
human adjudication. Transport/server outages remain pending and trigger a
bounded restart from checkpoints.

`medgemma_labels.parquet` and the readable `medgemma_labels.jsonl` are refreshed
every five processed cases and at shutdown. `reports/medgemma_runtime.json`
tracks live progress; `reports/medgemma_progress.json` gives decision counts.
Validated decision counts exclude failed outcomes. During execution, the live
runtime can be up to four cases ahead of the published review files.
On Windows the active local model process prevents automatic system sleep while
it runs, then releases that request. This does not change the power plan.

After the pass finishes, the wrapper automatically prepares these files:

- `data/reviews/final_review.jsonl` and `.parquet`: all 3,087 cases, including
  agreements, MedGemma outcomes, and unresolved cases.
- `data/reviews/human_adjudication_input.jsonl`: the 1,170 disputed cases with
  A/B and MedGemma context. Fill only `human_adjudication` for cases reviewed.
- `data/reviews/final_ab_accepted.parquet`: the 1,441 already accepted A/B cases.

Human can review in batches and accept the disputed cases they want to retain;
untouched or uncertain cases remain excluded. Follow
[human_adjudication.md](human_adjudication.md) to import those decisions.
If execution is interrupted, rerun `scripts/run_medgemma_to_human.ps1`; validated
caches and explicit terminal failures are skipped. Keep the computer powered on
until `reports/medgemma_runtime.json` reports `PASS_COMPLETE`.

The separate multimodal release still requires text/image QC, dataset support,
and its prescribed human audit. Agreement here completes the label review step,
not those separate release checks.

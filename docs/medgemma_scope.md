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
The separate multimodal release still requires text/image QC, dataset support,
and its prescribed human audit. Agreement here completes the label review step,
not those separate release checks.

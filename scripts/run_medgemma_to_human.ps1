param([int]$Limit = 10000)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:HF_HOME = Join-Path $projectRoot 'models/huggingface'
$env:TEMP = Join-Path $projectRoot '.tmp'
$env:TMP = $env:TEMP
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if ($Limit -le 0) { throw 'Limit must be positive.' }
try {
    & $python -m multicare_data.cli medgemma-run --backend local --limit $Limit
    if ($LASTEXITCODE -ne 0) { throw 'MedGemma stopped; inspect its error before continuing.' }
    & $python -m multicare_data.cli prepare-human-review
    if ($LASTEXITCODE -ne 0) { throw 'Final review is not ready; inspect reports/pre_human_review.json.' }
} catch {
    @{ status = 'PROCESS_FAILED'; error = $_.Exception.Message; updated_at = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'reports/medgemma_runtime.json') -Encoding UTF8
    throw
}

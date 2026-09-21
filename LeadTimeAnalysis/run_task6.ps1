$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Project venv not found: $Python"
    exit 1
}

& $Python (Join-Path $PSScriptRoot "task6_continuous_preprint_lead_time.py") @args

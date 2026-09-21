$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Project venv not found: $Python"
    exit 1
}

& $Python (Join-Path $PSScriptRoot "task7_institutional_tier_robustness.py") @args

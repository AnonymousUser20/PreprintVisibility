$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Project venv not found: $Python"
    exit 1
}

Push-Location $PSScriptRoot
try {
    & $Python ".\task5_rating_ols.py" @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python ".\task5_confidence_ols.py" @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

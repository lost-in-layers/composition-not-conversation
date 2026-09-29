# One-time local setup for the MuLAn human-review tool.
# Creates a Python venv, installs backend requirements, installs frontend
# dependencies and builds the static frontend. Localhost-only; nothing is
# deployed or uploaded anywhere.
$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "== MuLAn human-review setup ==" -ForegroundColor Cyan
Write-Host "App root: $AppRoot"

# --- Python backend ---------------------------------------------------------
$Venv = Join-Path $AppRoot ".venv"
if (-not (Test-Path $Venv)) {
    Write-Host "Creating Python venv at $Venv"
    python -m venv $Venv
}
$Pip = Join-Path $Venv "Scripts\pip.exe"
Write-Host "Installing backend requirements..."
& $Pip install --quiet -r (Join-Path $AppRoot "backend\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# --- Frontend ----------------------------------------------------------------
$Frontend = Join-Path $AppRoot "frontend"
Write-Host "Installing frontend dependencies (npm install)..."
Push-Location $Frontend
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    Write-Host "Building frontend (npm run build)..."
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start the tool with:  powershell -ExecutionPolicy Bypass -File `"$AppRoot\run.ps1`""

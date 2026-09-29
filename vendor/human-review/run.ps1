# Start the local MuLAn human-review app (127.0.0.1 only) and open the browser.
#   powershell -ExecutionPolicy Bypass -File vendor\human-review\run.ps1 [-Port 8791]
param(
    [int]$Port = 8791
)
$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$Python = Join-Path $AppRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Python venv not found. Run setup.ps1 first:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$AppRoot\setup.ps1`""
    exit 1
}
if (-not (Test-Path (Join-Path $AppRoot "frontend\dist\index.html"))) {
    Write-Host "Frontend not built. Run setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

$Url = "http://127.0.0.1:$Port"
Write-Host "Starting MuLAn human review at $Url (Ctrl+C to stop)" -ForegroundColor Cyan

# Open the default browser shortly after the server starts listening.
Start-Job -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $r = Invoke-WebRequest -Uri "$u/api/health" -UseBasicParsing -TimeoutSec 1
            if ($r.StatusCode -eq 200) { Start-Process $u; break }
        } catch { }
    }
} -ArgumentList $Url | Out-Null

Push-Location (Join-Path $AppRoot "backend")
try {
    & $Python -m hr_backend.main --port $Port
}
finally {
    Pop-Location
}

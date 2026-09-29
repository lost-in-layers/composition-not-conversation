# Start the separate single-reviewer category-only UI on 127.0.0.1.
param([int]$Port = 8792)
$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path (Split-Path -Parent $AppRoot) "human-review\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Human-review Python environment not found. Run vendor\human-review\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}
$Url = "http://127.0.0.1:$Port"
Write-Host "Starting MuLAn category review at $Url (Ctrl+C to stop)" -ForegroundColor Cyan
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
& $Python "$AppRoot\app.py" --port $Port

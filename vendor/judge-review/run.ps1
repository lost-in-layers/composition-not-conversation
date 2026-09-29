# Start the judge-validation review UI on 127.0.0.1.
param([int]$Port = 8794)
$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path (Split-Path -Parent $AppRoot) "human-review\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Human-review Python environment not found. Run vendor\human-review\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}
$Sample = Join-Path (Split-Path -Parent $AppRoot) "conv-smi\human_validation\sample.json"
if (-not (Test-Path $Sample)) {
    Write-Host "No sample found at $Sample." -ForegroundColor Yellow
    Write-Host "Run this first, from vendor\conv-smi:" -ForegroundColor Yellow
    Write-Host "  python sample_judge_validation.py --n 400" -ForegroundColor Yellow
    exit 1
}
$Url = "http://127.0.0.1:$Port"
Write-Host "Starting Layered-VQA judge validation at $Url (Ctrl+C to stop)" -ForegroundColor Cyan
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

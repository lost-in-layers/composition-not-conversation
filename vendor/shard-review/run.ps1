# Start the local shard/evidence review UI on 127.0.0.1.
param([int]$Port = 8793)
$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SharedPython = Join-Path (Split-Path -Parent $AppRoot) "human-review\.venv\Scripts\python.exe"
$Python = if (Test-Path $SharedPython) { $SharedPython } else { "python" }
$Url = "http://127.0.0.1:$Port"
Write-Host "Starting Layered-VQA shard review at $Url (Ctrl+C to stop)" -ForegroundColor Cyan
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

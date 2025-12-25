$ErrorActionPreference = 'Stop'

function Test-Health {
  try {
    Invoke-WebRequest -Uri "http://localhost:3000/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
    return $true
  } catch {
    return $false
  }
}

$serverProcess = $null
$startedServer = $false

if (-not (Test-Health)) {
  Write-Host "Starting server with node server.js..."
  $serverProcess = Start-Process -FilePath "node" -ArgumentList "server.js" -WorkingDirectory $PSScriptRoot -PassThru
  $startedServer = $true

  $ready = $false
  for ($i = 0; $i -lt 15; $i++) {
    if (Test-Health) { $ready = $true; break }
    Start-Sleep -Seconds 1
  }
  if (-not $ready) {
    Write-Host "Server not ready; continuing smoke check."
  }
} else {
  Write-Host "Server already running; skipping start."
}

try {
  npm run check:js
  npm run smoke:server
} finally {
  if ($startedServer -and $serverProcess) {
    try {
      $proc = Get-Process -Id $serverProcess.Id -ErrorAction SilentlyContinue
      if ($proc) { Stop-Process -Id $serverProcess.Id -Force }
    } catch { }
  }
}

npm run smoke:ollama

[CmdletBinding()]
param(
    # Optional: override python executable (defaults to .venv\Scripts\python.exe if present, else python on PATH)
    [string]$Python = "",
    # Optional: path to a training data file to copy to ai_training/language_model/data/train.txt
    [string]$DataFile = "",
    # Optional: JSON file with config overrides for language_model.CONFIG
    [string]$ConfigPath = "",
    # Optional: raw JSON string with config overrides (takes precedence over ConfigPath)
    [string]$ConfigJson = "",
    # Skip using .venv even if it exists
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Warn { param($Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }

$Root = Split-Path -Parent $PSCommandPath
Set-Location $Root

# Resolve python path
$venvPython = Join-Path $Root ".venv/Scripts/python.exe"
if (-not $SkipVenv -and (Test-Path $venvPython)) {
    $Python = $venvPython
} elseif (-not $Python) {
    $Python = "python"
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found at '$Python'. Set -Python to a valid interpreter."
}

# Ensure training data location exists and optionally copy provided data file
$trainPath = Join-Path $Root "ai_training/language_model/data/train.txt"
$trainDir = Split-Path $trainPath -Parent
if (-not (Test-Path $trainDir)) {
    New-Item -ItemType Directory -Path $trainDir -Force | Out-Null
}

if ($DataFile) {
    if (-not (Test-Path $DataFile)) {
        Write-Error "DataFile '$DataFile' not found."
    }
    Write-Info "Copying training data from '$DataFile' to '$trainPath'"
    Copy-Item -Path $DataFile -Destination $trainPath -Force
} elseif (-not (Test-Path $trainPath)) {
    Write-Warn "No training data found. language_model.py will generate a sample at $trainPath"
}

# Load config overrides
$configRaw = "{}"
if ($ConfigJson) {
    $configRaw = $ConfigJson
} elseif ($ConfigPath) {
    if (-not (Test-Path $ConfigPath)) {
        Write-Error "ConfigPath '$ConfigPath' not found."
    }
    $configRaw = Get-Content -Path $ConfigPath -Raw
}

# Validate JSON if provided
try {
    $null = $configRaw | ConvertFrom-Json
} catch {
    Write-Error "Config overrides are not valid JSON: $($_.Exception.Message)"
}

$configB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($configRaw))

Write-Info "Running language_model training..."
Write-Info "Python: $Python"
Write-Info "Config overrides: $configRaw"

$argsList = @("training_runner.py", "--module", "language_model")
if ($configRaw.Trim() -ne "{}") {
    $argsList += @("--config-b64", $configB64)
}

& $Python @argsList

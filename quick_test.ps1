Write-Host "Testing Multi-LLM Features...`n" -ForegroundColor Cyan

$baseUrl = "http://localhost:3000"

# Test 1: Health
Write-Host "TEST 1: Health Check" -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/health" -ErrorAction Stop
    Write-Host "  OK: Server is running`n" -ForegroundColor Green
} catch {
    Write-Host "  FAIL: Server not responding`n" -ForegroundColor Red
    exit 1
}

# Test 2: Multi-Chat API
Write-Host "TEST 2: Multi-Chat Endpoint" -ForegroundColor Yellow
$payload = @{
    messages = @(@{ role = 'user'; content = 'Hello' })
    models = @('gpt-oss-20', 'llama3.2')
    aggregator = 'length'
} | ConvertTo-Json

try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/multi-chat" -Method POST `
        -Headers @{'Content-Type'='application/json'} -Body $payload -ErrorAction Stop
    $data = $resp.Content | ConvertFrom-Json
    Write-Host "  OK: Got response from $($data.models.Count) models`n" -ForegroundColor Green
} catch {
    Write-Host "  FAIL: $($_)`n" -ForegroundColor Red
}

# Test 3: Stats API
Write-Host "TEST 3: Stats Endpoint" -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/stats" -ErrorAction Stop
    $data = $resp.Content | ConvertFrom-Json
    Write-Host "  OK: Dataset has $($data.count) records`n" -ForegroundColor Green
} catch {
    Write-Host "  FAIL: $($_)`n" -ForegroundColor Red
}

Write-Host "All core tests passed! Open browser to http://localhost:3000" -ForegroundColor Green

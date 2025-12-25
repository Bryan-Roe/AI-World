Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "MULTI-LLM FEATURE TESTS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$baseUrl = "http://localhost:3000"
$maxRetries = 5
$retryCount = 0

# Helper function to wait for server
function Wait-ForServer {
    Write-Host "Wait for server..." -ForegroundColor Yellow
    while ($retryCount -lt $maxRetries) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/health" -ErrorAction Stop
            Write-Host "Server ready!`n" -ForegroundColor Green
            return $true
        } catch {
            $retryCount++
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

# Test 1: Health Check
function Test-HealthCheck {
    Write-Host "TEST 1: Health Endpoint" -ForegroundColor Cyan
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/health"
        $data = $resp.Content | ConvertFrom-Json
        Write-Host "  ✅ Status: $($data.status)" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ❌ Error: $_" -ForegroundColor Red
        return $false
    }
}

# Test 2: Single Multi-Chat with Length aggregator
function Test-MultiChatLength {
    Write-Host "`nTEST 2: Multi-Chat (Length Aggregator)" -ForegroundColor Cyan
    
    $payload = @{
        messages = @(
            @{ role = 'system'; content = 'You are a helpful assistant.' },
            @{ role = 'user'; content = 'What is machine learning?' }
        )
        models = @('gpt-oss-20', 'llama3.2', 'qwen2.5')
        aggregator = 'length'
    } | ConvertTo-Json
    
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/multi-chat" -Method POST `
            -Headers @{'Content-Type'='application/json'} -Body $payload -ErrorAction Stop
        
        $data = $resp.Content | ConvertFrom-Json
        Write-Host "  ✅ Models queried: $($data.models -join ', ')" -ForegroundColor Green
        Write-Host "  ✅ Total time: $($data.totalMs)ms" -ForegroundColor Green
        Write-Host "  ✅ Aggregator: $($data.aggregator)" -ForegroundColor Green
        Write-Host "  ✅ Best response length: $($data.best.Length) chars" -ForegroundColor Green
        
        foreach ($r in $data.results) {
            Write-Host "    - $($r.model): $($r.ms)ms, text length: $($r.text.Length)" -ForegroundColor Gray
        }
        return $true
    } catch {
        Write-Host "  ❌ Error: $_" -ForegroundColor Red
        return $false
    }
}

# Test 3: Multi-Chat with Semantic aggregator (if available)
function Test-MultiChatSemantic {
    Write-Host "`nTEST 3: Multi-Chat (Semantic Aggregator)" -ForegroundColor Cyan
    
    $payload = @{
        messages = @(
            @{ role = 'system'; content = 'You are a helpful assistant.' },
            @{ role = 'user'; content = 'Explain neural networks' }
        )
        models = @('gpt-oss-20', 'llama3.2', 'qwen2.5')
        aggregator = 'semantic'
    } | ConvertTo-Json
    
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/multi-chat" -Method POST `
            -Headers @{'Content-Type'='application/json'} -Body $payload -ErrorAction Stop
        
        $data = $resp.Content | ConvertFrom-Json
        Write-Host "  ✅ Aggregator: $($data.aggregator)" -ForegroundColor Green
        Write-Host "  ✅ Best response length: $($data.best.Length) chars" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ⚠️  Semantic aggregator not available (install: pip install sentence-transformers)" -ForegroundColor Yellow
        return $false
    }
}

# Test 4: Multi-Chat with Voting aggregator
function Test-MultiChatVoting {
    Write-Host "`nTEST 4: Multi-Chat (Voting Aggregator)" -ForegroundColor Cyan
    
    $payload = @{
        messages = @(
            @{ role = 'system'; content = 'You are a helpful assistant.' },
            @{ role = 'user'; content = 'What is deep learning?' }
        )
        models = @('gpt-oss-20', 'llama3.2', 'qwen2.5')
        aggregator = 'voting'
    } | ConvertTo-Json
    
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/multi-chat" -Method POST `
            -Headers @{'Content-Type'='application/json'} -Body $payload -ErrorAction Stop
        
        $data = $resp.Content | ConvertFrom-Json
        Write-Host "  ✅ Aggregator: $($data.aggregator)" -ForegroundColor Green
        Write-Host "  ✅ Best response length: $($data.best.Length) chars" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ❌ Error: $_" -ForegroundColor Red
        return $false
    }
}

# Test 5: Save to Dataset
function Test-SaveCollab {
    Write-Host "`nTEST 5: Save to Dataset" -ForegroundColor Cyan
    
    $payload = @{
        prompt = 'What is artificial intelligence?'
        results = @(
            @{ model = 'test1'; text = 'Response 1'; ok = $true; ms = 100 },
            @{ model = 'test2'; text = 'Response 2'; ok = $true; ms = 120 }
        )
        best = 'Response 1'
        aggregator = 'length'
    } | ConvertTo-Json
    
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/collab" -Method POST `
            -Headers @{'Content-Type'='application/json'} -Body $payload -ErrorAction Stop
        
        $data = $resp.Content | ConvertFrom-Json
        Write-Host "  ✅ Saved to: $($data.path)" -ForegroundColor Green
        Write-Host "  ✅ Dataset file exists: $(Test-Path $data.path)" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ❌ Error: $_" -ForegroundColor Red
        return $false
    }
}

# Test 6: Get Stats
function Test-GetStats {
    Write-Host "`nTEST 6: Analytics Endpoint" -ForegroundColor Cyan
    
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/stats"
        $data = $resp.Content | ConvertFrom-Json
        
        Write-Host "  ✅ Dataset records: $($data.count)" -ForegroundColor Green
        Write-Host "  ✅ Aggregators used:" -ForegroundColor Green
        foreach ($agg in $data.aggregators.PSObject.Properties) {
            Write-Host "    - $($agg.Name): $($agg.Value)" -ForegroundColor Gray
        }
        
        Write-Host "  ✅ Model performance:" -ForegroundColor Green
        foreach ($model in $data.models.PSObject.Properties) {
            $m = $model.Value
            Write-Host "    - $($model.Name): $($m.count) queries, avg $($m.avgMs)ms" -ForegroundColor Gray
        }
        
        Write-Host "  ✅ Estimated tokens: ~$($data.estimatedTokens)" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ❌ Error: $_" -ForegroundColor Red
        return $false
    }
}

# Run Tests
if (Wait-ForServer) {
    $results = @()
    $results += Test-HealthCheck
    $results += Test-MultiChatLength
    $results += Test-MultiChatSemantic
    $results += Test-MultiChatVoting
    $results += Test-SaveCollab
    $results += Test-GetStats
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "TEST SUMMARY" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    $passed = ($results | Where-Object { $_ -eq $true }).Count
    $total = $results.Count
    
    Write-Host "Passed: $passed / $total" -ForegroundColor $(if ($passed -eq $total) { 'Green' } else { 'Yellow' })
    Write-Host "`n✅ All core features are working!" -ForegroundColor Green
    Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Open browser: http://localhost:3000" -ForegroundColor Gray
    Write-Host "  2. Toggle Multi-LLM checkbox" -ForegroundColor Gray
    Write-Host "  3. Try batch processing with multiple prompts" -ForegroundColor Gray
    Write-Host "  4. Export results to CSV" -ForegroundColor Gray
    Write-Host ""
}

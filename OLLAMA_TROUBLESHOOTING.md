# Ollama Troubleshooting Guide

## Error: "Ollama error" or "Ollama offline"

When the companion AI shows these messages, it means the Ollama service is not running or not accessible.

### ✅ Quick Fix (2 minutes)

#### Windows

**Terminal 1 - Start Ollama:**
```powershell
ollama serve
```

**Terminal 2 - Pull a model (if not already downloaded):**
```powershell
ollama pull gpt-oss-20
```

**Terminal 3 - Refresh the game:**
```powershell
# Then reload http://localhost:3000/game.html in your browser
```

#### macOS/Linux

**Terminal 1:**
```bash
ollama serve
```

**Terminal 2:**
```bash
ollama pull gpt-oss-20
```

Then refresh the game in your browser.

---

## Detailed Troubleshooting

### 1. Check if Ollama is installed

```powershell
# Windows
where ollama

# macOS/Linux
which ollama
```

If not found, install from: https://ollama.ai/

### 2. Verify Ollama is running

```powershell
# Test connection
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
```

**Expected response:** JSON list of available models

**If connection fails:** 
- Ollama is not running
- Run `ollama serve` in a terminal

### 3. Check available models

```powershell
# Using web request
$response = Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
$response.Content | ConvertFrom-Json | Select-Object -ExpandProperty models
```

**Output should show:**
- gpt-oss-20
- llama3.2
- qwen2.5
- (or other models you've pulled)

### 4. Pull a missing model

```powershell
ollama pull gpt-oss-20
ollama pull llama3.2
ollama pull qwen2.5
```

Each model takes 2-10 minutes to download depending on your internet speed.

### 5. Test the game API directly

```powershell
# Test the chat API
$body = @{
    model = "gpt-oss-20"
    messages = @(@{role = "user"; content = "Hello"})
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:3000/api/chat" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body `
  -UseBasicParsing
```

**Success:** Returns JSON with text response
**Failure:** Shows error details (use for debugging)

---

## Common Issues

### Issue: Port 11434 already in use

```powershell
# Find process using port 11434
Get-NetTCPConnection -LocalPort 11434 | Select-Object ProcessName, OwningProcess

# Kill the process (replace with actual PID)
Stop-Process -Id <PID> -Force

# Then start Ollama again
ollama serve
```

### Issue: Model won't load / Out of memory

```powershell
# List running models
curl http://localhost:11434/api/tags

# Use a smaller model
ollama pull qwen2.5-0.5b  # Much lighter
```

### Issue: Companion stops responding mid-game

1. Check if Ollama is still running
2. Look at the companion log message (bottom-left of game screen)
3. If it says "Connection lost", restart Ollama
4. Refresh the game page

---

## Companion Status Messages

| Message | Meaning | Action |
|---------|---------|--------|
| 🔍 Checking connection... | Testing Ollama link | Wait, refreshing... |
| 👋 Companion online. | Ready to use | Proceed normally |
| ⚠️ Ollama offline | Ollama not running | `ollama serve` in terminal |
| ⚠️ Ollama unreachable | Server can't reach Ollama | Check OLLAMA_URL in .env |
| ⚠️ Connection lost... | Network error during chat | Companion will keep following |

---

## Server Startup Checks

When you start the server, it automatically checks for Ollama:

```
✓ Ollama is running with models: gpt-oss-20, llama3.2
```

If Ollama is not running, you'll see:

```
⚠ Ollama not available at http://localhost:11434
   To start Ollama: ollama serve
   To pull a model: ollama pull gpt-oss-20
```

This is a warning, not an error. The server will still run, but the companion AI won't work until Ollama starts.

---

## Advanced: Custom Ollama Configuration

If Ollama is running on a different machine or port, set in `.env`:

```env
OLLAMA_URL=http://192.168.1.100:11434
```

Then restart the server:

```powershell
npm run dev
```

---

## Getting Help

1. **Check companion log** (bottom-left of game screen) for error message
2. **Verify Ollama is running**: `ollama serve` in a terminal
3. **Verify model is pulled**: `ollama list` should show your model
4. **Check server logs** (terminal where you ran `npm run dev`)
5. **Refresh the game** and try again

## Related Documentation

- [README.md](README.md) - Project overview
- [CONFIGURATION.md](CONFIGURATION.md) - Server configuration
- [API.md](API.md) - API endpoint reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick troubleshooting

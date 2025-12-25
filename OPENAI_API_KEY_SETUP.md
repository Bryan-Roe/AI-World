# OPENAI_API_KEY Setup Guide

## What This Error Means

When you select a **cloud model** (like `gpt-4o` or `gpt-4o-mini`), the system needs an OpenAI API key to authenticate.

**Local models** (like `gpt-oss-20`, `llama3.2`) don't need an API key - they run on your machine via Ollama.

---

## Quick Solution

### Option A: Use Local Models (No Key Required) ✅

**Just use these models - they're free and local:**
- `gpt-oss-20` (default)
- `llama3.2`
- `qwen2.5`

**No setup needed!**

---

### Option B: Add OpenAI API Key (To Use GPT-4/GPT-4o)

#### Step 1: Get Your API Key

1. Visit: https://platform.openai.com/api/keys
2. Sign up or log in with your OpenAI account
3. Click "Create new secret key"
4. Copy the key (looks like: `sk-proj-...`)
5. ⚠️ Save it somewhere safe - you won't see it again!

#### Step 2: Add Key to `.env`

**Method 1: Edit .env File Directly**

Open `.env` in your editor and update:

```bash
# Before:
# OPENAI_API_KEY=sk-your-actual-key-here

# After:
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

**Method 2: Command Line**

```bash
# Windows PowerShell
$content = Get-Content .env
$content = $content -replace '# OPENAI_API_KEY', 'OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE'
Set-Content .env $content
```

#### Step 3: Restart Server

```bash
# Kill existing server (if running)
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force

# Start fresh
npm run dev
```

#### Step 4: Test It

```bash
# In UI: Select a gpt-* model
# Or test via API:
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Understanding the System

### Local Models (Ollama) ✅
- **Cost:** Free
- **Setup:** Requires Ollama installed
- **Models:** gpt-oss-20, llama3.2, qwen2.5, etc.
- **Speed:** Fast (on your machine)
- **Privacy:** Completely local

```bash
# Use by selecting in UI or via API:
"model": "gpt-oss-20"       # No API key needed
"model": "llama3.2"          # No API key needed
```

### Cloud Models (OpenAI) 💳
- **Cost:** Pay per request
- **Setup:** Requires OPENAI_API_KEY
- **Models:** gpt-4o, gpt-4o-mini, gpt-4-turbo
- **Speed:** Fast (OpenAI servers)
- **Privacy:** Sent to OpenAI

```bash
# Use by setting OPENAI_API_KEY:
"model": "gpt-4o"           # Requires API key
"model": "gpt-4o-mini"      # Requires API key
```

---

## Pricing Info

**OpenAI API Usage (approximate):**

| Model | Input | Output | Typical Cost |
|-------|-------|--------|---------|
| gpt-4o-mini | $0.15/1M tokens | $0.60/1M tokens | $0.01-0.05/req |
| gpt-4o | $5.00/1M tokens | $15.00/1M tokens | $0.10-0.50/req |
| gpt-4-turbo | $10.00/1M tokens | $30.00/1M tokens | $0.20+/req |

**Ollama (Local - Free):**
- No cost at all! 🎉
- Just needs a computer to run on

---

## Setting API Key Securely

### ✅ DO:
- Store key in `.env` file
- Add `.env` to `.gitignore` (never commit!)
- Rotate keys regularly
- Use least-privilege API keys

### ❌ DON'T:
- Hardcode key in source code
- Share key in repositories
- Commit `.env` file to git
- Use key in frontend code

**Verify .env is in .gitignore:**

```bash
cat .gitignore | grep .env
# Should output: .env
```

---

## Troubleshooting

### "OPENAI_API_KEY not set" Error
```bash
# Check if key is set:
echo $env:OPENAI_API_KEY    # PowerShell
echo $OPENAI_API_KEY        # Bash

# Should output your key, not blank!
```

### "Invalid API key" Error
- Check key is correct (starts with `sk-`)
- Verify it's active (not revoked)
- Restart server after adding key
- No extra spaces in .env

### "Insufficient credits" Error
- Check OpenAI account balance
- Add payment method if needed
- Note: Free trial may have expired

### "Rate limit exceeded" Error
- Slow down requests
- Upgrade OpenAI plan
- Use local Ollama for free testing

---

## Recommended Setup

### For Development (Local & Free):
```bash
# Use Ollama models
OPENAI_API_KEY=                    # Leave empty
OLLAMA_URL=http://localhost:11434
```

### For Production (Reliable):
```bash
# Use OpenAI
OPENAI_API_KEY=sk-proj-...        # Add your key
OLLAMA_URL=http://localhost:11434 # Optional backup
```

### For Testing Multiple Providers:
```bash
# Use both
OPENAI_API_KEY=sk-proj-...        # For cloud models
OLLAMA_URL=http://localhost:11434 # For local models
```

---

## Testing Models

### Test Local Model
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

### Test OpenAI Model
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

---

## Next Steps

**If using Local Models (Recommended for now):**
1. Keep OPENAI_API_KEY empty
2. Start server: `npm run dev`
3. Select `gpt-oss-20` in UI
4. Chat away! No costs.

**If adding OpenAI:**
1. Get API key from openai.com
2. Add to `.env`: `OPENAI_API_KEY=sk-proj-...`
3. Restart server
4. Select `gpt-4o-mini` model
5. Enjoy! Pay per request.

---

**Questions?** Check these docs:
- [API.md](API.md) - API endpoints
- [CONFIGURATION.md](CONFIGURATION.md) - Full config guide
- [AUTH_API.md](AUTH_API.md) - Authentication

**Ready to go!** 🚀

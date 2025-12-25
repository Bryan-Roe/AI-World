# OPENAI_API_KEY Error - Resolution Guide

## Problem

You saw this error:
```
Error: OPENAI_API_KEY not set for cloud model
```

This happens when trying to use a **cloud model** (like `gpt-4o-mini`) without setting the OpenAI API key.

---

## Solution Overview

You have **2 main options:**

### ✅ **Option 1: Use Local Models (Recommended - FREE)**
- No API key needed
- No costs
- Runs on your machine
- Start immediately

### ⚠️ **Option 2: Add OpenAI API Key (Requires Payment)**
- Need OpenAI account
- API key from openai.com
- Pay per request
- Better models available

---

## Quick Setup (Choose One)

### Choice A: Use Local Models NOW ⚡

**1. Keep OPENAI_API_KEY empty** (it already is)

**2. In your UI, select one of these models:**
- `gpt-oss-20` (recommended)
- `llama3.2`
- `qwen2.5`

**3. Start chatting!**
```bash
npm run dev
open http://localhost:3000
```

**That's it!** No API key needed. 🎉

---

### Choice B: Add OpenAI API Key ⚡

**1. Get an API key:**
   - Go to: https://platform.openai.com/api/keys
   - Sign up or log in
   - Click "Create new secret key"
   - Copy the key (looks like: `sk-proj-xyz...`)

**2. Add to `.env` file:**

Edit `.env` and change this line:
```bash
# Before:
# OPENAI_API_KEY=sk-your-actual-key-here

# After:
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

**3. Restart server:**
```bash
# Kill running server (if any)
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force

# Restart
npm run dev
```

**4. Select a cloud model:**
- `gpt-4o` (best, most expensive)
- `gpt-4o-mini` (good, cheaper)

**5. Start chatting!**

---

## Comparison: Local vs Cloud Models

| Feature | Local (Ollama) | Cloud (OpenAI) |
|---------|---|---|
| **Cost** | Free | $0.01-0.50/request |
| **Speed** | Fast (local) | Fast (cloud) |
| **Quality** | Good | Excellent |
| **Privacy** | 100% private | Sent to OpenAI |
| **Setup** | Ollama only | API key + payment |
| **Reliability** | Local machine | 99.9% uptime |
| **Models** | gpt-oss-20, llama3.2 | gpt-4o, gpt-4o-mini |

**Recommendation for beginners:** Start with **local models** (Option A)

---

## What Changed in Your Code

I improved the error message to be more helpful. Now when you try to use a cloud model without an API key, you see:

**Before:**
```
Error: OPENAI_API_KEY not set for cloud model
```

**After:**
```json
{
  "error": "OpenAI API key not configured",
  "suggestion": "Add OPENAI_API_KEY to .env file or use a local model like gpt-oss-20",
  "guide": "See OPENAI_API_KEY_SETUP.md for instructions"
}
```

Much better! 😊

---

## How the System Works

```
Your Request
    ↓
Is it a gpt-* model? (like gpt-4o)
    ├─ YES → Need OPENAI_API_KEY
    │         ├─ Have key? → Send to OpenAI
    │         └─ No key? → Show helpful error
    │
    └─ NO → Use local Ollama
            ├─ Ollama running? → Send to Ollama
            └─ Not running? → Explain how to start

Response comes back to you ✓
```

---

## Testing Models

### Test Local Model (No API Key Needed)
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20",
    "messages": [{"role": "user", "content": "What is AI?"}]
  }'
```

**Works immediately!** ✓

---

### Test Cloud Model (With API Key)
```bash
# First, add OPENAI_API_KEY to .env
# Then:
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is AI?"}]
  }'
```

**Works after setting API key!** ✓

---

## Common Issues & Fixes

### Issue: "OpenAI API key not configured"
**Fix:** Add `OPENAI_API_KEY=sk-proj-...` to `.env` and restart server

### Issue: "Invalid API key"
**Fix:** 
- Check key starts with `sk-`
- Verify it's from openai.com (not elsewhere)
- No extra spaces in `.env`
- Restart server

### Issue: "Insufficient credits"
**Fix:** Add payment method to OpenAI account at openai.com/account/billing

### Issue: "Rate limit exceeded"
**Fix:** 
- Slow down requests
- Upgrade OpenAI plan
- Use local models for testing

---

## Environment Configuration

### Current Status
```bash
.env file:
  OPENAI_API_KEY=          # Empty (uses local models)
  PORT=3000                # Server port
  OLLAMA_URL=http://localhost:11434  # Local Ollama
```

### To Enable OpenAI:
```bash
.env file:
  OPENAI_API_KEY=sk-proj-abc123def456  # Your API key
  PORT=3000                             # Server port
  OLLAMA_URL=http://localhost:11434     # Local Ollama (backup)
```

### To Verify Configuration:
```bash
# Check if key is set (PowerShell)
echo $env:OPENAI_API_KEY

# Should output your key, or be blank if not set
```

---

## Security Notes

✅ **Good Practices:**
- Store key in `.env` (not in code)
- Add `.env` to `.gitignore`
- Never commit `.env` to git
- Rotate keys periodically
- Use least-privilege permissions

❌ **Bad Practices:**
- Hardcoding key in source files
- Sharing key in repos
- Committing `.env` to git
- Exposing key in logs
- Using key in frontend code

---

## Next Steps

### If You Want Local Models NOW:
1. ✓ OPENAI_API_KEY is already empty
2. ✓ Server is running
3. ✓ Select `gpt-oss-20` in UI
4. **You're done!** Start chatting

### If You Want to Add OpenAI:
1. Get API key from openai.com
2. Add to `.env`
3. Restart server
4. Select `gpt-4o-mini` in UI
5. Start chatting (with $ costs)

### If You Have Questions:
- Read: [OPENAI_API_KEY_SETUP.md](OPENAI_API_KEY_SETUP.md) - Detailed setup guide
- Read: [CONFIGURATION.md](CONFIGURATION.md) - All config options
- Read: [API.md](API.md) - API endpoint reference

---

## Summary

| What | Status | Action |
|------|--------|--------|
| **Using Local Models** | ✅ Ready Now | Just select `gpt-oss-20` |
| **Using Cloud Models** | ⚠️ Needs Setup | Get API key, add to `.env` |
| **Error Message** | ✅ Improved | Now shows helpful suggestion |
| **System** | ✅ Working | Both local & cloud ready |

---

## Quick Reference

```bash
# Local models (FREE)
"model": "gpt-oss-20"     # No key needed
"model": "llama3.2"        # No key needed
"model": "qwen2.5"         # No key needed

# Cloud models (PAID)
"model": "gpt-4o"          # Needs OPENAI_API_KEY
"model": "gpt-4o-mini"     # Needs OPENAI_API_KEY
```

**Choose what works for you!** 🚀

---

*Last updated: December 22, 2025*

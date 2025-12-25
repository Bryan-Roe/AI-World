# Automation Tools Reference

Complete guide to all automation tools and scripts in your project.

## 🎯 Quick Navigation

| Tool | Purpose | Platform | Launch |
|------|---------|----------|--------|
| **Interactive CLI Menu** | Master control panel | Win/Mac/Linux | `cli-menu.bat` or `./cli-menu.sh` |
| **Setup Validator** | Environment verification | Win/Mac/Linux | `setup-validator.bat` or `./setup-validator.sh` |
| **Quick Start Script** | Server launcher | Win/Mac/Linux | `start.bat` or `./start.sh` |
| **Model Puller** | Download Ollama models | Win/Mac/Linux | `pull-models.bat` or `./pull-models.sh` |
| **Health Checker** | Service monitoring | Win/Mac/Linux | `health-check.bat` or `./health-check.sh` |
| **API Examples** | Curl/fetch samples | Unix/Mac/Linux | `./API_EXAMPLES.sh` |
| **LLM Training Runner** | One-click language model fine-tune | Windows | `./train-language-model.ps1` |

---

## 1. Interactive CLI Menu (★ START HERE)

**Purpose:** Master automation control panel  
**Type:** Interactive Node.js application  
**Size:** 750+ lines, ~20 KB

### Launch
```bash
# Windows
cli-menu.bat

# macOS / Linux
chmod +x cli-menu.sh
./cli-menu.sh

# Direct
node cli-menu.js
```

### Features (6 categories)
- ✓ Setup & Configuration (4 tools)
- ✓ Server Management (5 tools)
- ✓ API Testing (5 tools + interactive chat!)
- ✓ Model Management (4 tools)
- ✓ Deployment (4 guides)
- ✓ Documentation (5 sections)

### What It Does
```
┌─────────────────────────────────────────┐
│  Interactive CLI Menu                   │
├─────────────────────────────────────────┤
│  1. Setup & Configuration               │
│  2. Server Management                   │
│  3. API Testing                         │
│  4. Model Management                    │
│  5. Deployment                          │
│  6. Documentation                       │
│  0. Exit                                │
└─────────────────────────────────────────┘
```

### Documentation
See [CLI_MENU_GUIDE.md](CLI_MENU_GUIDE.md)

---

## 2. Setup Validator

**Purpose:** Verify environment is configured correctly  
**Type:** Bash (Unix/macOS) or Batch (Windows)  
**Runtime:** 30 seconds

### Launch
```bash
# Windows
setup-validator.bat

# macOS / Linux
chmod +x setup-validator.sh
./setup-validator.sh
```

### Checks (6 categories)
1. **Node.js & npm** - Version detection
2. **Project files** - Verify essential files
3. **Dependencies** - Check node_modules
4. **Ollama** - Check if installed & running
5. **OpenAI API key** - Validate if set
6. **Port availability** - Port 3000 free?

### Output Example
```
✓ Node.js v20.10.0
✓ npm 10.2.3
✓ Found: server.js
⚠ Ollama not running (install from https://ollama.ai)
✓ Port 3000 available

Summary:
Passed: 5
Warnings: 1
Failed: 0
```

### When to Use
- First-time setup (check everything)
- After cloning repo
- Before deployment
- Debugging issues

---

## 3. Quick Start Script

**Purpose:** Start server with dependency checks  
**Type:** Bash or Batch launcher  
**Does:** Checks deps → Starts server → Provides URL

### Launch
```bash
# Windows
start.bat

# macOS / Linux
chmod +x start.sh
./start.sh
```

### What It Does
1. Checks if Node.js installed
2. Installs dependencies if needed (npm install)
3. Starts server (npm run dev)
4. Displays server URL: http://localhost:3000

### Output
```
========================================
 Starting LLM Chat Server
========================================

Checking dependencies...
Installing Express, node-fetch, etc...
Dependencies installed ✓

Starting server...
Server running on http://localhost:3000
Open in browser now!
```

### When to Use
- Quick server launch
- One-command startup
- Dependency verification

---

## 4. Model Puller

**Purpose:** Download Ollama models easily  
**Type:** Bash or Batch launcher  
**Models:** gpt-oss-20, llama3.2, qwen2.5, etc.

### Launch
```bash
# Windows (interactive)
pull-models.bat

# macOS / Linux (interactive)
chmod +x pull-models.sh
./pull-models.sh

# Or direct
ollama pull gpt-oss-20
```

### What It Does
- Lists available models
- Prompts for selection
- Downloads & installs
- Verifies installation

### Example
```
Available Ollama Models:
1. gpt-oss-20 (5.5 GB)
2. llama3.2 (3.8 GB)
3. qwen2.5 (4.2 GB)
4. mistral (7.2 GB)

Select model to pull: 1
Downloading gpt-oss-20...
✓ Model installed successfully
```

### When to Use
- First time setting up models
- Adding new models
- Replacing old models

---

## 5. Health Checker

**Purpose:** Monitor server & Ollama status  
**Type:** Bash or Batch monitor  
**Checks:** Server, Ollama, API connectivity

### Launch
```bash
# Windows (loop mode)
health-check.bat

# macOS / Linux (loop mode)
chmod +x health-check.sh
./health-check.sh
```

### Checks
```
Server Status:  ✓ Running on port 3000
Ollama Status:  ✓ Running on localhost:11434
API Health:     ✓ /health endpoint responding
Models:         ✓ 2 models installed
Port 3000:      ✓ Available (expected: running)
API Key:        ⚠ OPENAI_API_KEY not set (optional)
```

### Output
- Color-coded results (green ✓, yellow ⚠, red ✗)
- Detailed status for each service
- Suggestions for fixing issues

### When to Use
- Verify server is healthy
- Monitor before deployment
- Troubleshoot connection issues
- Keep running during development

---

## 6. API Examples

**Purpose:** Copy-paste curl & fetch examples  
**Type:** Shell script with code samples  
**Formats:** curl, JavaScript fetch, JSON

### Launch
```bash
# macOS / Linux
chmod +x API_EXAMPLES.sh
./API_EXAMPLES.sh

# Windows PowerShell
type API_EXAMPLES.sh
# Copy examples manually
```

### Examples Included
```bash
# Test health endpoint
curl http://localhost:3000/health

# Basic chat
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20","messages":[{"role":"user","content":"Hello"}]}'

# Streaming chat
curl -N -X POST http://localhost:3000/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '...'

# JavaScript fetch
fetch('http://localhost:3000/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({...})
})
```

### When to Use
- Testing endpoints manually
- Integration examples
- Documentation reference
- Copy-paste quick tests

---

## Recommended Workflow

### 1. First-Time Setup (5 minutes)
```bash
# Step 1: Verify environment
setup-validator.bat  (or ./setup-validator.sh)

# Step 2: Start server
start.bat  (or ./start.sh)

# Step 3: Download model
pull-models.bat  (or ./pull-models.sh)

# Step 4: Verify status
health-check.bat  (or ./health-check.sh)

# Step 5: Open browser
http://localhost:3000
```

### 2. Daily Development
```bash
# Launch master menu (everything in one place)
cli-menu.bat  (or ./cli-menu.sh)

# Then use menu to:
# → Check environment (Menu 1 → 1)
# → Manage server (Menu 2)
# → Test APIs (Menu 3)
# → Manage models (Menu 4)
```

### 3. Debugging
```bash
# Run validator
setup-validator.bat

# Launch CLI menu → API Testing → Interactive Chat
cli-menu.bat

# Or run specific health check
health-check.bat
```

### 4. Deployment
```bash
# Check production checklist
cli-menu.bat  (Menu 5 → 2)

# Or follow deployment guide
cli-menu.bat  (Menu 5 → 1)
```

---

## Tool Comparison

| Task | Tool | Time |
|------|------|------|
| Verify setup | setup-validator | 30 sec |
| Start server | start.bat | 5 sec |
| Download model | pull-models.bat | 5-30 min |
| Check health | health-check.bat | 5 sec |
| Run test | cli-menu.bat → Menu 3 | 1 min |
| Interactive chat | cli-menu.bat → Menu 3 → 5 | Variable |
| Copy API example | API_EXAMPLES.sh | 1 min |
| Full control | cli-menu.bat | Variable |

---

## Multi-Platform Support

All scripts work on:

| Platform | CLI Menu | Setup Validator | Start Server | Models | Health Check | API Examples |
|----------|----------|-----------------|--------------|--------|--------------|--------------|
| Windows | ✓ | ✓ | ✓ | ✓ | ✓ | (PowerShell) |
| macOS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Linux | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Docker | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## File Locations

```
your-project/
├── cli-menu.js              ← Main menu code
├── cli-menu.bat             ← Windows launcher
├── cli-menu.sh              ← Unix launcher
├── setup-validator.bat      ← Windows validator
├── setup-validator.sh       ← Unix validator
├── start.bat                ← Windows start
├── start.sh                 ← Unix start
├── pull-models.bat          ← Windows models
├── pull-models.sh           ← Unix models
├── health-check.bat         ← Windows health
├── health-check.sh          ← Unix health
├── API_EXAMPLES.sh          ← API samples
├── CLI_MENU_GUIDE.md        ← Menu guide
├── CLI_MENU_SUMMARY.md      ← Summary (new!)
└── AUTOMATION_TOOLS.md      ← This file
```

---

## Troubleshooting

### "Command not found"
```bash
# Make script executable first
chmod +x cli-menu.sh
chmod +x setup-validator.sh
# etc.
```

### Script exits immediately
```bash
# Run directly to see error
node cli-menu.js
```

### "Node.js not installed"
- Install from https://nodejs.org
- Requires v18+

### No color output (Windows)
- Use PowerShell, not cmd.exe
- Or enable ANSI in legacy console

### Ollama connection failed
- Start Ollama: `ollama serve`
- Check: `curl http://localhost:11434/api/tags`

---

## 7. LLM Training Runner (PowerShell)

**Purpose:** One-click fine-tuning for the enhanced language model  
**Type:** PowerShell wrapper around `training_runner.py` and `language_model.py`  
**Launch:** `./train-language-model.ps1`

### Options
- `-DataFile <path>`: Copy a text/JSONL corpus into `ai_training/language_model/data/train.txt` before training.
- `-ConfigPath <path>` or `-ConfigJson '<json>'`: Override `language_model.CONFIG` (epochs, learning rate, base_model, etc.).
- `-Python <path>`: Custom interpreter; defaults to `.venv/Scripts/python.exe` when present.
- `-SkipVenv`: Force PATH python even if `.venv` exists.

### Example Runs
```powershell
# vanilla run (uses existing train.txt or auto-creates sample)
./train-language-model.ps1

# override epochs and learning rate
./train-language-model.ps1 -ConfigJson '{"epochs":5,"learning_rate":1e-4}'

# point at custom data file
./train-language-model.ps1 -DataFile C:\data\my_corpus.txt
```

Outputs land in `ai_training/language_model/models/run_*` with `metrics.json` alongside the best model.

---

## Next Steps

1. **Start with CLI Menu:**
   ```bash
   cli-menu.bat  # (or ./cli-menu.sh)
   ```

2. **Run environment check:**
   - Menu 1 → Option 1

3. **Test interactive chat:**
   - Menu 3 → Option 5

4. **Deploy to production:**
   - Menu 5 → Option 2

---

## Summary

✅ **7 automated tools** covering:
- Environment verification
- Server management
- API testing
- Model management
- Deployment guides
- Quick reference

✅ **Multi-platform**: Windows, macOS, Linux  
✅ **User-friendly**: Color output, helpful guidance  
✅ **Comprehensive**: 30+ functions, 1000+ lines total  

**Everything you need for project automation!** 🚀

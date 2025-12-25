# Local Development Setup - Complete Guide

## ✅ Status: READY FOR LOCAL DEVELOPMENT

### Current Environment
- **Node.js:** ✅ Dependencies installed (96 packages)
- **Python:** ✅ Version 3.11.9
- **Ollama:** ✅ Running on http://localhost:11434
- **Server:** Ready to start on port 3000

### Available Models
- `gpt-oss:20b` (using "gpt-oss-20" in chat)
- `phi4:latest`

---

## 🚀 Quick Start (3 Steps)

### Step 1: Ensure Ollama is Running
```bash
# On another terminal, start Ollama if not already running
ollama serve
```

**Check Ollama status:**
```bash
curl http://localhost:11434/api/tags
```

### Step 2: Start the Express Server
```bash
npm run dev
```

Expected output:
```
Server running on http://localhost:3000
Using OLLAMA_URL: http://localhost:11434
OPENAI_API_KEY not set; will prefer local Ollama if available
```

### Step 3: Open in Browser
```
http://localhost:3000
```

---

## 🎯 Local Development Workflow

### Development Server (Terminal 1)
```bash
npm run dev
```
- Starts Express on port 3000
- Serves frontend from `/public`
- Enables morgan logging for all requests
- Hot reload supported

### Testing (Terminal 2)
```bash
# Test health endpoint
curl http://localhost:3000/health

# Test single chat
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

# Test multi-LLM
curl -X POST http://localhost:3000/api/multi-chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"What is AI?"}],
    "models":["gpt-oss-20","phi4"],
    "aggregator":"length"
  }'
```

### Python AI Training (Terminal 3)
```bash
# Check dataset stats
python analyze_dataset.py

# Auto-trigger distillation (monitors at 50 records)
python auto_distill.py

# Run batch data collection
python collect_collab_data.py
```

---

## 📋 Configuration

### .env Settings (Already Configured)
```env
PORT=3000
OLLAMA_URL=http://localhost:11434
# OPENAI_API_KEY=sk-... (optional, for cloud models)
```

### Models Configuration (in app.js)
```javascript
const MODELS = ['gpt-oss-20', 'phi4', 'llama3.2', 'qwen2.5'];
const DEFAULT_AGGREGATOR = 'length';
```

---

## 🔍 Testing Multi-LLM Features

### 1. Single Multi-LLM Query
**Open http://localhost:3000**
1. Toggle "Multi-LLM" checkbox
2. Enter models: `gpt-oss-20,phi4`
3. Select aggregator: "length"
4. Type prompt: "What is machine learning?"
5. Click Send
6. Compare responses and best selection

### 2. Batch Collection (5 prompts)
1. Click Batch icon (📄)
2. Paste prompts (one per line)
3. Select aggregator
4. Click "Run Batch"
5. Click "Export CSV"
6. Click "Save to Dataset"

### 3. Analytics Dashboard
1. Click Stats icon (📊)
2. View:
   - Record count
   - Aggregator distribution
   - Model performance
   - Token estimates

---

## 🐍 Python Setup (Optional for Training)

### Install Python Dependencies
```bash
# Core dependencies (already in requirements)
pip install torch torchvision transformers huggingface-hub sentence-transformers

# Or run setup script
python ai_training_setup.py
```

### Available Python Scripts
| Script | Purpose |
|--------|---------|
| `semantic_rank.py` | Compute similarity scores |
| `analyze_dataset.py` | Dataset analysis & insights |
| `auto_distill.py` | Auto-trigger training at 50 records |
| `multi_llm_distill.py` | Fine-tune student model |
| `collect_collab_data.py` | Batch collection from API |

---

## 📊 Data Flow

### 1. Chat Interface
```
User Input
    ↓
Single Chat: /api/chat → Ollama → Response
    ↓
Multi-LLM: /api/multi-chat → Multiple Models → Aggregation → Best Response
    ↓
Save: /api/collab → JSONL Dataset (ai_training/language_model/data/collab.jsonl)
```

### 2. Training Pipeline
```
JSONL Dataset (50+ records)
    ↓
auto_distill.py (monitors)
    ↓
multi_llm_distill.py (fine-tunes distilgpt2)
    ↓
Trained Model: ai_training/language_model/models/distill_student/
```

### 3. Analytics
```
/api/stats endpoint
    ↓
Reads collab.jsonl
    ↓
Returns: {count, aggregators, models, estimatedTokens}
```

---

## 🛠️ Common Commands

### Server Management
```bash
# Start development server
npm run dev

# Run tests
node test.js

# Check health
curl http://localhost:3000/health
```

### Data Management
```bash
# Analyze current dataset
python analyze_dataset.py

# Collection prompts (if using batch collection script)
python collect_collab_data.py

# Monitor for auto-distillation trigger
python auto_distill.py
```

### Model Management
```bash
# Pull new Ollama model
ollama pull llama3.2

# See available models
ollama list
```

---

## 🐛 Troubleshooting

### Ollama Not Connecting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

### Port 3000 Already in Use
```bash
# Find and kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or change PORT in .env
PORT=3001
```

### Python Dependencies Missing
```bash
# Install all dependencies
python ai_training_setup.py

# Or manually
pip install torch transformers sentence-transformers huggingface-hub
```

### Models Not Available
```bash
# Check available models
ollama list

# Pull a model
ollama pull phi4
ollama pull gpt-oss-20b
```

---

## ✨ What You Can Do Now

✅ **Chat with local LLM** - Single model inference via /api/chat
✅ **Multi-LLM orchestration** - Query multiple models with 6 aggregation strategies
✅ **Batch processing** - Collect 20-50 prompts at once
✅ **Analytics** - Real-time dataset stats and model performance
✅ **Auto-training** - Automatic distillation when reaching 50 records
✅ **CSV export** - Download batch results as spreadsheet

---

## 📚 Next Steps

1. **Start server:** `npm run dev`
2. **Open browser:** http://localhost:3000
3. **Test multi-LLM:** Toggle and send a query
4. **Collect data:** Use batch mode to gather 50+ prompts
5. **Train student:** auto_distill.py triggers automatically
6. **Deploy:** Use trained model for faster inference

---

## 🎓 Learning Resources

- See [MULTILLM_TESTING_COMPLETE.md](MULTILLM_TESTING_COMPLETE.md) for detailed testing guide
- See [API.md](API.md) for endpoint documentation
- See [README.md](README.md) for project overview

---

**Status: Everything is configured and ready to use locally! 🚀**

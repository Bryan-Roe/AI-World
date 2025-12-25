# 🚀 LOCAL SETUP - QUICK REFERENCE

## ✅ Everything is Running!

**Server:** http://localhost:3000 ✅
**Ollama:** http://localhost:11434 ✅
**Models:** gpt-oss:20b, phi4 ✅

---

## 📖 Testing Guide (5 Minutes)

### Test 1: Single Chat (1 min)
```
Browser: http://localhost:3000
1. Type: "What is AI?"
2. Click Send
3. See response from gpt-oss:20b
```

### Test 2: Multi-LLM Comparison (2 min)
```
1. Toggle "Multi-LLM" checkbox
2. Keep models: gpt-oss-20,phi4
3. Type: "Explain machine learning"
4. Click Send
5. See:
   - 2 separate responses
   - "Best" selection at top
   - Timing for each model
```

### Test 3: Try Different Aggregators (2 min)
```
1. Change Aggregator dropdown:
   - length (longest response)
   - keyword+length (overlap + length)
   - rank-by-model (priority order)
   - meta-llm (gpt-oss-20 judges)
   - voting (consensus)
2. Notice different "best" for each
```

### Test 4: Batch Collection
```
Click 📄 (Batch icon):
1. Paste 5 prompts:
   - What is quantum computing?
   - Explain neural networks
   - Define deep learning
   - What is NLP?
   - Describe transfer learning

2. Select aggregator: "meta-llm"
3. Click "Run Batch"
4. When done:
   - Click "Export CSV"
   - Click "Save All to Dataset"
```

### Test 5: View Analytics
```
Click 📊 (Stats icon):
1. See dataset statistics
2. Record count
3. Aggregator distribution
4. Model performance
5. Token estimates
```

---

## 💻 Terminal Commands

### Start/Stop Server
```bash
# Start server (already running)
npm run dev

# Stop server
Ctrl+C (in server terminal)

# Restart if needed
Ctrl+C, then npm run dev
```

### Check Dataset
```bash
# View dataset statistics
python analyze_dataset.py

# Check collected prompts
ls ai_training/language_model/data/collab.jsonl
```

### Monitor Auto-Training
```bash
# Run in separate terminal to watch for 50+ records
python auto_distill.py

# This will auto-trigger fine-tuning when threshold hit
```

### Run Full Analysis
```bash
python analyze_dataset.py
```

---

## 📊 What Happens When You Collect Data

```
Batch Mode (UI)
  ↓ Sends prompts to /api/multi-chat
  ↓ Gets responses from all models
  ↓ Aggregates "best" response
  ↓ Saves to collab.jsonl
  ↓
After 50 records:
  ↓
auto_distill.py triggers:
  ↓ Reads 50+ prompts
  ↓ Fine-tunes distilgpt2
  ↓ Saves trained model
  ↓
Trained Student Model:
  ├─ Faster than querying 2 models
  ├─ Learned from ensemble responses
  └─ Ready for inference
```

---

## 🎯 Success Metrics

- [ ] Server running on :3000
- [ ] Ollama connected on :11434
- [ ] Single query responds
- [ ] Multi-LLM toggle works
- [ ] All 5 aggregators produce different results
- [ ] Batch mode processes 5 prompts
- [ ] CSV export works
- [ ] Analytics dashboard shows data
- [ ] Collected 20+ prompts
- [ ] Reached 50 record threshold
- [ ] Auto-training triggered
- [ ] Student model saved

---

## 🐛 Quick Troubleshooting

**Server not responding?**
```powershell
# Check if running
curl http://localhost:3000/health

# Kill and restart
Get-Process node | Stop-Process -Force
npm run dev
```

**Ollama not connecting?**
```powershell
# Check status
curl http://localhost:11434/api/tags

# Start if needed
ollama serve
```

**Port 3000 in use?**
```powershell
# Check what's using it
netstat -ano | findstr :3000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

---

## 📚 See Also

- [LOCAL_SETUP.md](LOCAL_SETUP.md) - Detailed setup guide
- [MULTILLM_TESTING_COMPLETE.md](MULTILLM_TESTING_COMPLETE.md) - Full testing guide
- [API.md](API.md) - API endpoint reference
- [README.md](README.md) - Project overview

---

## ✨ You're All Set!

**Browser:** Open http://localhost:3000
**Start testing:** Toggle Multi-LLM and send your first query!


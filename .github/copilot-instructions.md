# AI Training & LLM Chat Application - Copilot Instructions

## Architecture Overview

This is a hybrid Node.js/Python project combining:
1. **Web LLM Chat** (Node.js/Express): Real-time chat with local Ollama and OpenAI fallback, user auth, message streaming
2. **AI Training Suite** (Python): PyTorch LLM fine-tuning, image classification, RL game AI, custom neural networks
3. **3D Game World** (Three.js): Browser-based 3D environment trainable via Python game_ai.py
4. **CLI Automation** (Node.js): Interactive menu (cli-menu.js) for environment setup, model management, health checks

### System Flow
- **User Browser** ↔ `server.js` (Express, ES modules, morgan logging, rate limiting)
- **server.js** ↔ **Ollama** (http://localhost:11434, local model inference)
- **server.js** ↔ **OpenAI API** (fallback for cloud models; requires OPENAI_API_KEY)
- **AI Training Scripts** → `ai_training/*/models/` (checkpoint persistence)
- **Training Data** ← `ai_training/*/data/` (auto-discovered, format-flexible)

## Startup & Development Workflow

### Local LLM Chat (Recommended)
```bash
# Terminal 1: Start Ollama (pull models first)
ollama pull gpt-oss-20 llama3.2 qwen2.5
ollama serve

# Terminal 2: Run Express server
npm install          # One-time
npm run dev          # http://localhost:3000
```

### Quick Server Validation
```powershell
# Verify server is running
Invoke-WebRequest http://localhost:3000/health

# Check Ollama connection
Invoke-WebRequest http://localhost:11434/api/tags
```

### Python AI Training (Optional)
```bash
# One-time: Create directories + install PyTorch
python ai_training_setup.py

# Fine-tune language model on custom text data
python language_model.py  # Reads from ai_training/language_model/data/*.txt or *.jsonl

# Train image classifier on labeled images
python image_classifier.py

# Train RL agent for 3D world
python game_ai.py

# Build custom architecture
python custom_nn.py
```

## Critical Project Patterns

### 1. Express API Design (`server.js`)
**Every endpoint follows strict validation pattern:**
```javascript
// POST /api/chat expects: {messages: [{role, content}], model}
if (!Array.isArray(messages) || messages.length === 0) {
  return res.status(400).json({ error: 'messages array required' });
}

// History trimming: keep system message + last 12 dialog messages
const systemMsg = messages.find(m => m.role === 'system');
const dialog = messages.filter(m => m.role !== 'system');
const trimmed = systemMsg ? [systemMsg, ...dialog.slice(-12)] : dialog.slice(-12);
```

**Model routing logic:**
- Defaults to Ollama (`gpt-oss-20`) if model is local or unspecified
- Automatically routes to OpenAI if model name matches cloud variants (gpt-4o, gpt-4o-mini)
- Returns `{text: reply, raw: data}` for consistency
- Uses 429 rate limiting: 30 requests/minute per IP

**Streaming endpoints:**
- `/api/chat-stream` — HTTP Transfer-Encoding: chunked
- `/api/chat-sse` — Server-Sent Events format
- Both handle message trimming identically

### 2. Python Training Module Structure
**ALL training files follow this invariant pattern:**

```python
# 1. TOP: CONFIG dict (ONLY place to tune hyperparameters)
CONFIG = {
    "data_dir": "ai_training/language_model/data",
    "model_dir": "ai_training/language_model/models",
    "batch_size": 4,
    "epochs": 3,
    "learning_rate": 5e-5,
    "use_lora": True,  # Efficient fine-tuning
}

# 2. Custom Dataset or Model class (extends nn.Module/Dataset)
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        # Auto-detect: .json, .jsonl, or plain text
        if file_path.endswith('.jsonl'):
            # Parse line-delimited JSON
        else:
            # Read plain text, chunk by words
    
# 3. DataLoader creation (uses torch.utils.data)
train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'])

# 4. Training loop with checkpoint saving
checkpoint_path = os.path.join(CONFIG['model_dir'], f'model_{timestamp}.pt')
torch.save(model.state_dict(), checkpoint_path)

# 5. Inference/evaluation section
# Load checkpoint and test on new text

# 6. Entry point
if __name__ == '__main__':
    # main()
```

**Data format conventions:**
- `language_model`: Accepts `.txt` or `.jsonl` with `{"text": "..."}` per line
- `image_classifier`: Expects `data/train/{class_name}/{image.jpg}`
- `game_ai`: Generates synthetic trajectories; no data dir needed
- All relative paths use `os.path.join()` with project root

### 3. Frontend Message Flow (`public/app.js`)
```javascript
// Global state: message history with system prompt
let messages = [
  { role: 'system', content: 'You are a helpful assistant.' }
];

// Trimming mirrors server: max 12 non-system messages
const MAX_HISTORY = 12;
function getTrimmedMessages() {
  const sysMsg = messages.find(m => m.role === 'system');
  const dialog = messages.filter(m => m.role !== 'system');
  return sysMsg ? [sysMsg, ...dialog.slice(-MAX_HISTORY)] : dialog.slice(-MAX_HISTORY);
}

// POST to /api/chat-stream with user message appended
// Client receives streaming response, appends to DOM as chunks arrive
// formatContent() converts markdown code blocks and inline backticks
```

### 4. Authentication Pattern (`auth.js`)
- JWT tokens with bcryptjs password hashing
- Endpoints: `/auth/register`, `/auth/login`, `/auth/refresh`
- Demo user: `username: demo, password: demo`
- Test users auto-created on server startup via `setupTestUsers()`

## Directory Structure & Ownership

```
.
├── server.js                    # Express app, /api/chat endpoint
├── public/
│   ├── app.js                   # Chat UI logic, DOM manipulation
│   ├── game.js                  # Three.js 3D scene, WebGL rendering
│   ├── index.html               # Chat interface
│   └── game.html                # 3D world interface
│
├── ai_training/                 # Created by ai_training_setup.py
│   ├── language_model/
│   │   ├── data/train.txt       # Input: line-delimited JSON or plain text
│   │   └── models/              # Output: checkpoint folders
│   ├── image_classifier/
│   │   ├── data/train/{class}/  # Input: organized by class
│   │   └── models/              # Output: saved models
│   ├── game_ai/
│   │   └── models/              # Output: trained RL agents
│   └── custom_nn/
│       └── models/              # Output: trained networks
│
├── package.json                 # Node.js dependencies (Express, morgan, dotenv, node-fetch)
└── .env                         # Runtime config (OPENAI_API_KEY optional, OLLAMA_URL)
```

## Important Conventions

### Environment & Dependencies
- **Node.js**: Uses ES modules (`"type": "module"` in package.json)
- **Python**: PyTorch 2.x with CUDA 12.1, optional LoRA for efficient fine-tuning
- **Ollama models**: `gpt-oss-20`, `llama3.2`, `qwen2.5` (add to `index.html` select)
- **GPU assumed**: All training scripts use CUDA if available; CPU fallback implicit

### Data Paths
- **Absolute vs relative**: All Python scripts use `os.path.join(BASE_DIR, config_path)` for portability
- **train/val/test splits**: Handled by PyTorch's `random_split()` or torchvision's `ImageFolder`
- **Checkpoint naming**: `{model_name}_{timestamp}.pt` or similar in `model_dir`

### Error Handling
- **Express**: Returns `{error: string, details: string}` on 4xx/5xx
- **Python training**: Assumes clean data; limited validation (add checks before training)
- **Frontend**: Shows typing indicator; disables send button during loading

### Testing & Debugging
- **server.js uses morgan('dev')**: HTTP request/response logging
- **Frontend formatContent()**: Handles \`\`\`code\`\`\` and \`inline\` markdown rendering
- **Python logging**: Add `print()` statements; model training prints epoch/loss
- **Browser DevTools**: Check Network tab for /api/chat payloads; Console for formatContent() issues

## Common Modification Points

### Adding a New LLM Model
1. Update `index.html` `<select id="model">` with new option
2. If cloud model: Ensure OPENAI_API_KEY is set; server auto-routes to OpenAI Responses API
3. If local model: Run `ollama pull modelname` and test with Ollama endpoint

### Training on Custom Data
1. **Language model**: Drop JSON/text files in `ai_training/language_model/data/`; script auto-discovers
2. **Image classifier**: Create class folders in `ai_training/image_classifier/data/train/`; ImageFolder auto-loads
3. **Game AI**: Modify state/action spaces in `CONFIG` (world_size, state_dim, action_dim)

### Modifying Model Architecture
- Edit `CONFIG` (preferred): `hidden_layers`, `activation`, `dropout`, `batch_norm`
- Edit Model class only if CONFIG doesn't expose needed parameters
- Ensure input/output dims match training data

### CLI Automation Tools (`cli-menu.js`)
- Used for interactive environment setup, server management, and health monitoring
- Cross-platform: Windows (cli-menu.bat), Unix/Linux (cli-menu.sh)
- Provides menu-driven interface to: validate environment, manage Ollama, start/stop server, run training
- Each menu option spawns child processes; check exit codes for error handling

## Red Flags / Anti-Patterns

- ❌ Hardcoding paths (use `os.path.join()` with BASE_DIR)
- ❌ Storing API keys in client code (always proxy through server.js)
- ❌ Training without GPU check (silent CPU fallback is slow; add warning)
- ❌ Modifying system message mid-chat (breaks assistant context)
- ❌ Saving entire model state instead of checkpoints (use torch.save for weights only)
- ❌ Running multiple Python scripts simultaneously (CUDA memory conflicts; use subprocess or separate terminals)

## Key Files to Reference

- **Chat logic**: [public/app.js](public/app.js) — Message handling, UI updates, /api/chat calls
- **Server routing**: [server.js](server.js#L28) — POST /api/chat endpoint, Ollama vs OpenAI logic
- **Training template**: [language_model.py](language_model.py#L28) — CONFIG + training loop pattern
- **3D rendering**: [public/game.js](public/game.js#L1) — Three.js setup, animation loop
- **Setup script**: [ai_training_setup.py](ai_training_setup.py#L13) — Directory structure, dependency installation

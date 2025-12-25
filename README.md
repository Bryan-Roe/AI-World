# AI World 🌍✨

[![Node.js Version](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen)](https://nodejs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security](https://img.shields.io/badge/vulnerabilities-0-success)](https://www.npmjs.com/package/web-llm-chat)

A modern web application for chatting with LLMs via a Node/Express server. Supports local Ollama models (free, fast) and OpenAI cloud models as fallback, plus a stunning 3D interactive world powered by Three.js.

## ✨ Features

### 🤖 AI & LLM Integration
- **Dual-Transport Streaming**: Choose between Fetch streams or Server-Sent Events (SSE) for real-time responses  
- **Local + Cloud**: Run local models via Ollama or use OpenAI (gpt-4o, gpt-4o-mini)  
- **Multi-Model Comparison**: Query multiple models in parallel and compare responses  
- **Message History Management**: Automatic history capping (system + last 12 messages) for efficiency  
- **Browser-based WebLLM**: Run models directly in browser with WebGPU support

### 🎮 3D Interactive World
- **Three.js Rendering**: Stunning 3D graphics and environment
- **Procedural Generation**: Infinite world with dynamic chunk loading
- **AI Companions**: Spawn AI-powered NPCs with personalities
- **Interactive Objects**: Collect, craft, and interact with world objects
- **Weather & Biomes**: Dynamic weather systems and diverse biomes
- **Day/Night Cycle**: Realistic time progression

### 🎨 Developer Experience
- **Beautiful UI**: Dark-mode chat interface with markdown rendering  
- **Full API**: RESTful endpoints for streaming and non-streaming requests  
- **Docker Support**: Containerized setup with docker-compose for easy deployment  
- **Authentication**: JWT-based auth with bcrypt password hashing
- **Rate Limiting**: Built-in protection against abuse

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)  

## Quick Start

### Option 1: Windows (Batch Script)
```batch
start.bat
```
This will check for Node.js and Ollama, install dependencies, and start the server.

### Option 2: Unix/Linux/macOS (Shell Script)
```bash
chmod +x start.sh
./start.sh
```

### Option 3: Manual Setup
```bash
# 1. Install dependencies
npm install

# 2. Start Ollama in a separate terminal (required for local models)
ollama serve

# 3. Pull models (in another terminal)
ollama pull gpt-oss-20
ollama pull llama3.2
ollama pull qwen2.5

# 4. Run the server
npm run dev
```

**Open your browser to http://localhost:3000**

## Configuration

### Environment Variables
Create a `.env` file (see `.env.example`):

```env
# OpenAI API Key (optional; required for cloud models)
# OPENAI_API_KEY=sk-your-actual-key

# Server port
PORT=3000

# Ollama endpoint
OLLAMA_URL=http://localhost:11434
```

### Model Selection

**Local Models (Free, Fast):**
- `gpt-oss-20` — Default, solid general purpose
- `llama3.2` — Meta's Llama, good for reasoning
- `qwen2.5` — Alibaba's model, multilingual

**Cloud Models (Requires OpenAI API key):**
- `gpt-4o-mini` — Fast, cost-effective
- `gpt-4o` — Most capable

## Usage

### Web Interface
1. Select a model from the dropdown (local models require Ollama running)
2. Toggle "Stream" for incremental responses
3. Choose transport: "Fetch" (default) or "SSE"
4. Type your message and press Ctrl+Enter or click Send

### API

See [API.md](API.md) for full documentation.

**Quick Examples:**

Non-streaming:
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Hello!"}],
    "model": "gpt-oss-20"
  }'
```

JavaScript streaming:
```javascript
const res = await fetch('/api/chat-stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: 'Tell me a joke' }],
    model: 'llama3.2'
  })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  console.log(decoder.decode(value, { stream: true }));
}
```

## Docker Deployment

```bash
# Start Ollama + Web app (docker-compose)
docker-compose up -d

# Open http://localhost:3000
```

This will:
- Start an Ollama container on port 11434
- Start the chat server on port 3000
- Persist Ollama models in a volume

To pull models inside Docker:
```bash
docker exec ollama-local ollama pull gpt-oss-20
docker exec ollama-local ollama pull llama3.2
docker exec ollama-local ollama pull qwen2.5
```

## API Endpoints

- **GET** `/health` — Health check
- **POST** `/api/chat` — Non-streaming chat
- **POST** `/api/chat-stream` — Streaming (fetch)
- **GET** `/api/chat-sse?payload=<base64>` — Streaming (SSE)
- **POST** `/api/multi-chat` — Multi-model comparison

See [API.md](API.md) for detailed specs and examples.

## Performance Notes

### Local Models (Ollama)
- **Pros**: Free, private, instant startup
- **Cons**: Slower inference, less capable than cloud models
- **Best for**: Local development, privacy-first, no API costs

### Cloud Models (OpenAI)
- **Pros**: Faster, more capable, handles complex reasoning
- **Cons**: API costs, requires internet, rate limits
- **Best for**: Production, quality responses, complex tasks

### Optimization Tips
1. Use local `gpt-oss-20` for quick feedback during development
2. Switch to `gpt-4o-mini` for production with cost control
3. Enable streaming for better UX with longer responses
4. Use SSE for more reliable connections (EventSource retry logic)
5. Leverage `/api/multi-chat` to compare model outputs

## Troubleshooting

### "Ollama connection refused"
Ensure Ollama is running: `ollama serve` in a separate terminal

### "OPENAI_API_KEY not set"
Set your key in `.env`:
```env
OPENAI_API_KEY=sk-your-actual-key
```

### Server won't start (Port 3000 in use)
```bash
# Find and kill the process
lsof -i :3000
kill -9 <PID>

# Or use a different port
PORT=3001 npm run dev
```

### Models not appearing in Ollama
Pull them explicitly:
```bash
ollama pull gpt-oss-20
ollama pull llama3.2
ollama pull qwen2.5
```

## Notes

- Messages are capped at system + last 12 dialog entries to manage API limits and improve response speed.
- Rate limited to 30 requests/minute per IP to prevent abuse.
- All API keys are processed server-side; never exposed to the client.
- CORS is enabled for development (remove in production if needed).

## Project Structure

```
├── server.js                 # Express server with API endpoints
├── public/
│   ├── app.js               # Chat UI logic with multi-LLM and batch support
│   ├── game.js              # 3D world game logic
│   ├── index.html           # Chat interface with batch mode
│   └── game.html            # 3D world interface
├── semantic_rank.py         # Semantic similarity aggregator
├── collect_collab_data.py   # Multi-LLM dataset collector
├── multi_llm_distill.py     # Student model distillation trainer
├── ai_training/             # Training data and model outputs
│   └── language_model/
│       ├── data/            # collab.jsonl and train.txt
│       └── models/          # Saved distilled models
├── API.md                   # API documentation
├── API_EXAMPLES.sh          # curl examples
├── start.bat / start.sh     # Quick start scripts
├── pull-models.bat/.sh      # Model download helper
├── docker-compose.yml       # Docker setup
├── Dockerfile               # Container image
└── .env                     # Configuration (local)
```

## 🛠 Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run start        # Start production server
npm run test         # Run tests
npm run lint         # Check code syntax
npm run validate     # Run lint + tests
npm run smoke        # Run smoke tests (server + Ollama)
npm run audit:check  # Security vulnerability check
npm run audit:fix    # Auto-fix security issues
npm run clean        # Remove node_modules and lockfile
npm run reinstall    # Clean reinstall dependencies
```

### Code Quality Checks

Before committing:
```bash
npm run validate     # Runs linting and tests
npm run audit:check  # Checks for vulnerabilities
npm run smoke        # Tests server + Ollama integration
```

### Testing

Run syntax checks:
```bash
npm run check:js
```

Test server health:
```bash
npm run dev
# In another terminal:
curl http://localhost:3000/health
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Commit message conventions
- Pull request process
- Areas for contribution

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and test: `npm run validate`
4. Commit with conventional commits: `git commit -m "feat: add new feature"`
5. Push and open a Pull Request

## 📚 Additional Documentation

- [API.md](API.md) - Complete API reference
- [AUTH_API.md](AUTH_API.md) - Authentication endpoints
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [SECURITY.md](SECURITY.md) - Security policy
- [CHANGELOG.md](CHANGELOG.md) - Version history

## License

MIT

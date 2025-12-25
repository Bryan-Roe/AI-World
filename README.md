# Web LLM Chat

A modern web application for chatting with LLMs via a Node/Express server. Supports local Ollama models (free, fast) and OpenAI cloud models as fallback.

## Features

✨ **Dual-Transport Streaming**: Choose between Fetch streams or Server-Sent Events (SSE) for real-time responses  
🚀 **Local + Cloud**: Run local models via Ollama or use OpenAI (gpt-4o, gpt-4o-mini)  
🔄 **Multi-Model Comparison**: Query multiple models in parallel and compare responses  
💾 **Message History Management**: Automatic history capping (system + last 12 messages) for efficiency  
🎨 **Beautiful UI**: Dark-mode chat interface with markdown rendering  
📝 **Full API**: RESTful endpoints for streaming and non-streaming requests  
🐳 **Docker Support**: Containerized setup with docker-compose for easy deployment  

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
│   ├── game.js              # 3D world (optional)
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

## License

MIT

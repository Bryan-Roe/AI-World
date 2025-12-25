# Web LLM Chat - Configuration Guide

## Environment Setup

### Required
- **Node.js** 18+ ([https://nodejs.org](https://nodejs.org))
- **Ollama** (for local models) — [https://ollama.com](https://ollama.com)

### Optional
- **OpenAI API Key** (for cloud models gpt-4o, gpt-4o-mini)
- **Docker** (for containerized deployment)

## File Configuration

### .env (Local Configuration)
```env
# OpenAI API Key (optional)
# OPENAI_API_KEY=sk-xxx

# Server port (default: 3000)
PORT=3000

# Ollama endpoint (default: http://localhost:11434)
OLLAMA_URL=http://localhost:11434
```

See `.env.example` for defaults.

### Optional TOML Configuration
You can centralize non-secret runtime defaults in a TOML file. Env vars still take precedence.

Location (Windows): `c:\Users\Bryan\.codex\config.toml`

Keys used by the server:
- `server.port` — fallback port if `PORT` is not set
- `logging.morgan_format` — request log format (e.g., `dev`, `combined`)
- `limits.max_payload_mb` — JSON body limit (MB)
- `server.rate_limit_per_minute` — per-IP rate limit
- `server.trust_proxy` — enables proxy-aware IP handling
- `ollama.url` — Ollama endpoint (fallback if `OLLAMA_URL` is not set)
- `ollama.default_model` — default local model (e.g., `gpt-oss-20`)
- `openai.model` — default cloud model (e.g., `gpt-4o-mini`)
- `openai.cloud_models` — array of model names routed to OpenAI (e.g., `["gpt-4o-mini", "gpt-4o"]`)
- `chat.max_history` — server-side message history cap
- `sse.heartbeat_interval_ms` — SSE keep-alive interval (default: 25000)
- `sse.retry_ms` — SSE client retry delay on disconnect (default: 3000)

Notes:
- Secrets must remain in `.env` (e.g., `OPENAI_API_KEY`, `JWT_SECRET`).
- If the TOML is malformed, the server logs a warning and continues with env/defaults.

## Model Configuration

### Local Models (Ollama)

**Default Models:**
- `gpt-oss-20` — General purpose, smaller, fast
- `llama3.2` — Meta's LLM, good reasoning
- `qwen2.5` — Alibaba's model, multilingual

**Pull Models:**
```bash
ollama pull gpt-oss-20
ollama pull llama3.2
ollama pull qwen2.5
```

**Add Custom Model:**
1. Pull the model: `ollama pull <modelname>`
2. Add option in `public/index.html`:
```html
<option value="modelname">modelname (local)</option>
```

### Cloud Models (OpenAI)

**Set API Key in `.env`:**
```env
OPENAI_API_KEY=sk-your-key-here
```

**Available Models:**
- `gpt-4o-mini` — Fast, cost-effective
- `gpt-4o` — Most capable

## Server Configuration

### Ports
- Default: `PORT=3000`
- Custom: `PORT=8080 npm run dev`

### Ollama Connection
- Default: `OLLAMA_URL=http://localhost:11434`
- Custom host: `OLLAMA_URL=http://192.168.1.100:11434 npm run dev`

### Rate Limiting
Located in `server.js` (line ~40):
```javascript
const RATE_LIMIT = 30;      // requests per window
const RATE_WINDOW = 60000;  // 1 minute in ms
```

Change values and restart server to apply.

### History Capping
All endpoints cap to: **System message + last 12 dialog messages**

To change:
- `public/app.js` line ~13: `const MAX_HISTORY = 12;`
- `server.js` `/api/chat` line ~60: `const MAX_HISTORY = 12;`
- `server.js` `/api/chat-stream` line ~110: `const MAX_HISTORY = 12;`

## Docker Configuration

### docker-compose.yml

**Environment Variables:**
```yaml
environment:
  - OLLAMA_URL=http://ollama:11434  # Service name (not localhost)
  - OPENAI_API_KEY=${OPENAI_API_KEY}  # Passed from host .env
  - PORT=3000
```

**Port Mapping:**
- Ollama: `11434:11434` (internal to container network)
- Web app: `3000:3000` (accessible from host)

**Volume Mounts:**
- `ollama_data` — Persists models between restarts
- `./public` — Hot-reload for static files
- `./server.js` — Hot-reload for server code

### Rebuild After Changes
```bash
docker-compose down
docker-compose up -d --build
```

## Client Configuration

### Browser Settings

**Chat Interface** (`public/index.html`):
- Model selector dropdown (add/remove models)
- Stream toggle (default: off)
- Transport selector (Fetch vs SSE)

**UI Customization** (CSS in `index.html`):
```css
:root { 
  --bg: #0f172a;           /* Background */
  --panel: #111827;        /* Panel background */
  --text: #e5e7eb;         /* Text color */
  --accent: #3b82f6;       /* Button/link color */
  --user-bg: #1e3a5f;      /* User message bg */
  --assistant-bg: #1e293b; /* Assistant message bg */
}
```

### JavaScript Settings (`public/app.js`):
```javascript
const MAX_HISTORY = 12;     // Messages to keep in history
const formatContent(...);   // Markdown rendering function
```

## API Configuration

### Endpoints
All endpoints validate:
- `messages` array (required)
- `model` string (default: `gpt-oss-20`)

### Routing Logic
```javascript
if (model.startsWith('gpt-')) {
  // Use OpenAI (requires OPENAI_API_KEY)
} else {
  // Use Ollama (requires running Ollama)
}
```

### Response Format
- Non-streaming: `{ text: string, raw: object }`
- Streaming: Plain text chunks or SSE events

## Security Configuration

### API Key Protection
- Keys must be in `.env` (server-side only)
- Never expose keys in client code
- Keys are not logged or returned to client

### CORS
- Default: Allow all origins (`*`)
- Production: Restrict to specific origins
```javascript
// In server.js, change:
res.header('Access-Control-Allow-Origin', 'https://example.com');
```

### Rate Limiting
- Default: 30 requests/minute per IP
- Can be adjusted in server configuration

## Troubleshooting Configuration

### Wrong Model Endpoint
**Symptom:** "Ollama error" for cloud models or API errors for local

**Fix:** Model routing depends on name:
- Cloud: Must start with `gpt-` or be in cloudModels set
- Local: Any other model name

**Check:** Edit `server.js` cloudModels set (line ~65):
```javascript
const cloudModels = new Set(['gpt-4o-mini', 'gpt-4o', ...]);
```

### Port Already in Use
**Symptom:** "EADDRINUSE: address already in use :::3000"

**Fix:**
```bash
# Change port
PORT=3001 npm run dev

# Or kill process using port 3000
lsof -i :3000 | grep node | awk '{print $2}' | xargs kill -9
```

### Models Not Found
**Symptom:** No models in Ollama

**Fix:**
```bash
ollama serve        # Terminal 1
ollama pull gpt-oss-20  # Terminal 2
```

### High Latency
**Symptom:** Slow responses

**Optimize:**
1. Use local model (`gpt-oss-20`) vs cloud
2. Reduce history (smaller `MAX_HISTORY`)
3. Ensure Ollama has GPU acceleration
4. Check network latency to OpenAI

## Production Configuration

### Environment Variables
```bash
export NODE_ENV=production
export PORT=8080
export OPENAI_API_KEY=sk-prod-key
export OLLAMA_URL=http://prod-ollama:11434
npm start
```

### Disable CORS (if behind reverse proxy)
Comment out in `server.js`:
```javascript
// app.use(cors middleware);
```

### Use Process Manager
```bash
npm install -g pm2
pm2 start server.js --name "llm-chat"
pm2 save
pm2 startup
```

### SSL/TLS (with reverse proxy)
Use nginx/Apache in front:
```nginx
server {
  listen 443 ssl;
  location / {
    proxy_pass http://localhost:3000;
  }
}
```

## Monitoring & Logs

### Server Logs
```bash
# Real-time logs
npm run dev

# Background with logging
npm start > logs.txt 2>&1 &
```

### Health Check
```bash
# Manual
curl http://localhost:3000/health

# Script (included)
./health-check.sh  # Unix
health-check.bat   # Windows
```

### Error Logs
Check `server.js` console output for:
- Connection errors
- Validation errors
- Rate limit hits
- API failures

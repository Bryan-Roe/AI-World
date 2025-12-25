# Project Summary - Web LLM Chat

## Overview
Complete, production-ready LLM chat application combining local Ollama models with OpenAI cloud fallback, featuring streaming, multi-model orchestration, and full documentation.

## What Was Built

### Core Application
- **Express.js Server** (`server.js`)
  - 5 REST API endpoints
  - Automatic Ollama ↔ OpenAI routing
  - Streaming (Fetch + SSE)
  - Multi-model comparison
  - CORS + rate limiting (30 req/min)
  - Message history capping (system + 12 messages)

- **Web UI** (`public/index.html`, `public/app.js`)
  - Real-time chat interface
  - Model selector (local + cloud)
  - Stream toggle + transport choice (Fetch/SSE)
  - Markdown rendering
  - Typing indicators

### API Endpoints
1. **GET** `/health` — Server health check
2. **POST** `/api/chat` — Non-streaming (standard request/response)
3. **POST** `/api/chat-stream` — Streaming via HTTP text chunks
4. **GET** `/api/chat-sse` — Streaming via Server-Sent Events
5. **POST** `/api/multi-chat` — Query multiple models in parallel

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Quick start, features, usage, troubleshooting |
| `API.md` | Complete API reference with examples |
| `DEVELOPMENT.md` | Testing, debugging, extension guide |
| `CONFIGURATION.md` | Environment, models, security configuration |

### Scripts & Automation
| File | Purpose |
|------|---------|
| `start.bat` / `start.sh` | One-click startup with dependency checks |
| `pull-models.bat` / `pull-models.sh` | Download Ollama models |
| `health-check.bat` / `health-check.sh` | Monitor server + Ollama status |
| `API_EXAMPLES.sh` | Copy-paste cURL/fetch examples |

### Infrastructure
| File | Purpose |
|------|---------|
| `Dockerfile` | Alpine Node.js container image |
| `docker-compose.yml` | Ollama + web app orchestration |
| `.env` / `.env.example` | Configuration templates |
| `package.json` | Dependencies + metadata |

## Key Features

✅ **Streaming**: Choose between Fetch streams or EventSource-based SSE  
✅ **Smart Routing**: Automatically route to Ollama or OpenAI based on model name  
✅ **Local First**: Run free, offline models via Ollama  
✅ **Cloud Fallback**: Use OpenAI (gpt-4o, gpt-4o-mini) when needed  
✅ **History Management**: Auto-cap to 12 messages to manage API costs  
✅ **Multi-Model**: Query 3+ models in parallel and compare responses  
✅ **Production Ready**: CORS, rate limiting, error handling, monitoring  
✅ **Docker Support**: Single command deployment with docker-compose  
✅ **Full Documentation**: API specs, configuration, development guide  

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Browser                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ public/index.html + public/app.js                    │   │
│  │ - Chat UI, message history, model selector           │   │
│  │ - Streaming support (Fetch + SSE)                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP(S)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Express Server                          │
│                    (server.js - Node.js)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Middleware:                                          │   │
│  │ - Morgan logging                                     │   │
│  │ - CORS (allow all origins)                           │   │
│  │ - Rate limiting (30 req/min/IP)                      │   │
│  │ - JSON parsing (10MB limit)                          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ API Endpoints:                                       │   │
│  │ - /health                                            │   │
│  │ - /api/chat                                          │   │
│  │ - /api/chat-stream                                   │   │
│  │ - /api/chat-sse                                      │   │
│  │ - /api/multi-chat                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────┬──────────────────────────────────┬──────────────┘
             │                                  │
             ▼                                  ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │     Ollama       │           │   OpenAI Responses   │
    │   (local)        │           │      (cloud)         │
    │                  │           │                      │
    │ gpt-oss-20       │           │  gpt-4o-mini         │
    │ llama3.2         │           │  gpt-4o              │
    │ qwen2.5          │           │                      │
    └──────────────────┘           └──────────────────────┘
```

## Technology Stack

**Backend**
- Node.js 18+
- Express 4.x
- Morgan (HTTP logging)
- node-fetch (HTTP client)
- dotenv (configuration)

**Frontend**
- HTML5
- CSS3 (custom dark theme)
- Vanilla JavaScript (no frameworks)

**Infrastructure**
- Docker + docker-compose
- Ollama (local LLM runtime)
- OpenAI API (cloud fallback)

## File Structure

```
.
├── server.js                    # Express server (main)
├── package.json                 # Node.js dependencies
├── .env                         # Config (local, git-ignored)
├── .env.example                 # Config template
│
├── public/
│   ├── index.html              # Chat UI
│   ├── app.js                  # Chat logic
│   ├── game.js                 # 3D world (optional)
│   └── game.html               # 3D interface (optional)
│
├── docker-compose.yml          # Ollama + server orchestration
├── Dockerfile                  # Server container image
│
├── Documentation/
│   ├── README.md               # Quick start & overview
│   ├── API.md                  # API reference
│   ├── CONFIGURATION.md        # Configuration guide
│   └── DEVELOPMENT.md          # Testing & extension guide
│
├── Scripts/
│   ├── start.bat               # Windows launcher
│   ├── start.sh                # Unix launcher
│   ├── pull-models.bat         # Model downloader (Windows)
│   ├── pull-models.sh          # Model downloader (Unix)
│   ├── health-check.bat        # Status monitor (Windows)
│   ├── health-check.sh         # Status monitor (Unix)
│   └── API_EXAMPLES.sh         # Copy-paste examples
```

## Getting Started

### Quickest Path (Windows)
```batch
start.bat
```

### Quickest Path (Unix/Mac)
```bash
chmod +x start.sh
./start.sh
```

### Docker
```bash
docker-compose up -d
ollama pull gpt-oss-20  # In another terminal
```

Open http://localhost:3000

## Configuration Checklist

- [ ] Install Node.js 18+
- [ ] Install Ollama from https://ollama.com
- [ ] Run `ollama serve` in terminal
- [ ] Run `start.bat` or `./start.sh`
- [ ] Open http://localhost:3000
- [ ] Optional: Set `OPENAI_API_KEY` in `.env` for cloud models

## API Quick Reference

### Non-Streaming
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"model":"gpt-oss-20"}'
```

### Streaming (Fetch)
```bash
curl -X POST http://localhost:3000/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a joke"}],"model":"llama3.2"}'
```

### Streaming (SSE) — Browser
```javascript
const json = JSON.stringify({messages:[{role:'user',content:'Hello'}],model:'gpt-oss-20'});
const payload = btoa(unescape(encodeURIComponent(json)));
const es = new EventSource(`/api/chat-sse?payload=${encodeURIComponent(payload)}`);
es.onmessage = e => console.log(e.data);
es.addEventListener('done', () => es.close());
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | (none) | OpenAI API key for cloud models |
| `PORT` | 3000 | Server port |
| `OLLAMA_URL` | http://localhost:11434 | Ollama endpoint |

## Rate Limiting

- **Limit**: 30 requests/minute per IP
- **Response**: 429 Too Many Requests
- **Configurable**: Edit `server.js` lines ~40-60

## Message History

All endpoints cap history to:
- 1 system message (if present)
- Last 12 dialog messages (user/assistant)

**Why?** Reduces API costs, improves performance, manages context window limits.

**To change:**
Edit `MAX_HISTORY` constant in `server.js` and `public/app.js`

## Security Notes

✅ API keys stored server-side only (never sent to client)  
✅ CORS enabled for development (restrict in production)  
✅ Rate limiting prevents abuse  
✅ Input validation on all endpoints  
✅ No logging of sensitive data  

**Production**: Change `Access-Control-Allow-Origin` in `server.js` to specific domain.

## Monitoring

### Health Check
```bash
./health-check.sh     # Unix
health-check.bat      # Windows
```

### Server Logs
```bash
npm run dev          # Real-time logs
tail -f logs.txt     # Persistent logs
```

### Docker Logs
```bash
docker-compose logs -f web-llm-chat
docker-compose logs -f ollama
```

## Next Steps

### Extend
- Add authentication/multi-user
- Persist chat history to database
- Add voice input/output
- Custom system prompts per conversation
- Model fine-tuning pipeline

### Deploy
- Railway, Heroku, DigitalOcean (Node.js)
- AWS ECS, GKE, AKS (Docker)
- Hugging Face Spaces (optional)

### Optimize
- Implement request/response caching
- Add connection pooling
- Use Redis for session storage
- Compress responses

## Support Files

All documentation is markdown (`.md`) and can be:
- Read in any text editor
- Viewed on GitHub
- Converted to PDF/HTML with Pandoc
- Referenced in IDE's markdown preview

---

**Status**: ✅ Production-Ready  
**Last Updated**: December 22, 2025  
**License**: MIT  

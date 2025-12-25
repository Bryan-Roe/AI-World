# Quick Reference Card

## Commands

### Start Server
```bash
npm run dev          # Development
npm start            # Production
PORT=8080 npm run dev # Custom port
```

### Ollama
```bash
ollama serve                    # Start Ollama
ollama pull gpt-oss-20          # Download model
ollama list                     # Show models
ollama run gpt-oss-20 "hello"   # Test model
```

### Docker
```bash
docker-compose up -d            # Start services
docker-compose down             # Stop services
docker-compose logs -f          # View logs
docker-compose ps               # Show status
```

### Health Checks
```bash
curl http://localhost:3000/health         # Server
curl http://localhost:11434/api/tags      # Ollama
./health-check.sh                         # Both
```

### Models
```bash
./pull-models.sh                # Auto pull all (Unix)
pull-models.bat                 # Auto pull all (Windows)
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Server status |
| POST | `/api/chat` | Standard request/response |
| POST | `/api/chat-stream` | Stream text chunks |
| GET | `/api/chat-sse?payload=...` | Stream via EventSource |
| POST | `/api/multi-chat` | Multi-model comparison |

---

## Model Names

### Local (Free)
- `gpt-oss-20` (default)
- `llama3.2`
- `qwen2.5`

### Cloud (Paid)
- `gpt-4o`
- `gpt-4o-mini`

---

## Environment Variables

```env
OPENAI_API_KEY=sk-...           # Cloud API key
PORT=3000                       # Server port
OLLAMA_URL=http://localhost:11434  # Ollama endpoint
```

---

## Request Template (JSON)

```json
{
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Question?" }
  ],
  "model": "gpt-oss-20"
}
```

---

## Response Template

```json
{
  "text": "Response text...",
  "raw": { "model": "...", "..." }
}
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| Connection refused | Ollama not running | `ollama serve` |
| 400 messages required | Missing messages array | Add messages to request |
| 429 Rate limit | Too many requests | Wait 1 minute |
| 500 Ollama error | Ollama down or model missing | `ollama pull <model>` |
| EADDRINUSE port 3000 | Port in use | `PORT=3001 npm run dev` |
| OPENAI_API_KEY not set | No API key for cloud model | Set in `.env` |

---

## File Locations

| Item | Location |
|------|----------|
| Server | `server.js` |
| UI | `public/index.html` + `public/app.js` |
| Config | `.env` (see `.env.example`) |
| API Docs | `API.md` |
| Setup Guide | `README.md` |
| Dev Guide | `DEVELOPMENT.md` |
| Config Guide | `CONFIGURATION.md` |
| This Card | `QUICK_REFERENCE.md` |

---

## Browser URLs

| Purpose | URL |
|---------|-----|
| Chat | http://localhost:3000 |
| Health | http://localhost:3000/health |
| 3D World | http://localhost:3000/game.html |

---

## Performance Tips

1. Use **local models** for instant feedback
2. Use **cloud models** for complex tasks (requires API key)
3. Enable **Stream** for long responses
4. Use **Fetch** transport (default, more compatible)
5. Use **SSE** for improved reliability
6. Check **history cap** if responses are truncated

---

## Support

- 📖 **Full Docs**: See `README.md`, `API.md`, `DEVELOPMENT.md`
- 🔧 **Configuration**: See `CONFIGURATION.md`
- 🐛 **Troubleshooting**: See `README.md` Troubleshooting section
- 💻 **Examples**: See `API_EXAMPLES.sh`

---

**Version**: 1.0.0  
**License**: MIT  
**Last Updated**: December 22, 2025

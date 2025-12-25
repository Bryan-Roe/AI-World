# Copilot Instructions (AI Agents)

This project is a hybrid Node.js + Python app with three pillars:
- Web LLM chat (Express) talking to local Ollama and OpenAI fallback
- Python training suite (language, image, RL) persisting checkpoints
- Optional Three.js 3D world, trainable via Python agents

## Big Picture Flow
- Browser UI ↔ server.js (ESM, logging, rate limits)
- server.js ↔ Ollama (localhost:11434) or OpenAI (requires `OPENAI_API_KEY`)
- Python trainers read from `ai_training/*/data` and write to `ai_training/*/models`

## Run & Validate
```bash
ollama serve && ollama pull gpt-oss-20 llama3.2 qwen2.5
npm install && npm run dev  # http://localhost:3000
```
```powershell
Invoke-WebRequest http://localhost:3000/health
Invoke-WebRequest http://localhost:11434/api/tags
```

## Server API Patterns
- Inputs validated: `/api/chat{,-stream}` expect `messages: [{role,content}], model`
- History trimmed: system + last 12 messages (mirrored in `public/app.js`)
- Model routing: local default `gpt-oss-20`; `gpt-4o(-mini)` -> OpenAI
- Routing rules: any `gpt-oss-*` → Ollama; any `gpt-*` not `gpt-oss-*` or listed in `CLOUD_MODELS` → OpenAI. Configure `DEFAULT_LOCAL_MODEL`, `CLOUD_MODELS` in [server.js](server.js).
- Streaming: `/api/chat-stream` (Fetch) and `/api/chat-sse` (SSE)
- Consistent shape: response `{text, raw}`; 429 limit: 30 req/min/IP

## Frontend Conventions
- Global `messages` with system role; `MAX_HISTORY = 12`
- Streaming renders chunks; `formatContent()` formats code fences/backticks
- Model select includes local and cloud options

## Python Training Patterns
- Each file has top-level `CONFIG` and saves checkpoints under `models/`
- Data discovery under `ai_training/<module>/data` (.txt/.jsonl or folders)
- Scripts: `language_model.py`, `image_classifier.py`, `game_ai.py`, `custom_nn.py`

## Auth Basics
- JWT + bcrypt; endpoints `/auth/register`, `/auth/login`, `/auth/refresh`
- Demo user exists; test users seeded on startup

## Project-Specific Tips
- Use ESM (`type: module`); never expose API keys in client
- Prefer `os.path.join()` and relative roots in Python
- Keep training single-script to avoid CUDA memory conflicts
- New models: add option in `public/index.html`, `ollama pull` locally

## Quick Tasks
- Compare models: POST `/api/multi-chat`
- Docker dev: `docker-compose up -d` then open http://localhost:3000
- Pull models in Docker: `docker exec ollama-local ollama pull <model>`

## Model Routing Matrix
- Local (Ollama): `gpt-oss-20`, `llama3.2`, `qwen2.5`, and any model not matching the cloud rule; uses `OLLAMA_URL`.
- Cloud (OpenAI): `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`, and any `gpt-*` that is not `gpt-oss-*`; requires `OPENAI_API_KEY`.
- Browser (WebLLM): models prefixed with `web-llm:` (e.g., `web-llm:Llama-3.2-1B-Instruct-q4f16_1-MLC`, `web-llm:Phi-3-mini-4k-instruct-q4f16_1-MLC`) handled client-side via `public/webllm-bridge.js` and `public/game.html`.
- Defaults: if `model` is omitted, server uses `DEFAULT_LOCAL_MODEL` (config or `gpt-oss-20`). Cloud list is extended via `CLOUD_MODELS` in [server.js](server.js).

## Try: Multi-Model Compare
```bash
curl -X POST http://localhost:3000/api/multi-chat \
	-H "Content-Type: application/json" \
	-d '{
		"messages": [{"role":"user","content":"Summarize this project"}],
		"models": ["gpt-oss-20","llama3.2","gpt-4o-mini"],
		"aggregator": "length"
	}'
```
- Response includes: `best` (chosen text), `results` (per-model details with `text`, `provider`, `ms`, `status`), `models`, `aggregator`, `totalMs`.
- JS (client): send to `/api/multi-chat` and render `results.map(r => r.model + ': ' + r.text)`.

## Key References
- Server: [server.js](server.js)
- UI: [public/app.js](public/app.js), [public/index.html](public/index.html)
- Trainers: [language_model.py](language_model.py), [image_classifier.py](image_classifier.py), [game_ai.py](game_ai.py)
- Setup: [ai_training_setup.py](ai_training_setup.py), [README.md](README.md)

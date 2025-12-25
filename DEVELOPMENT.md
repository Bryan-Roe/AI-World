# Development & Testing Guide

## Running Tests

### Manual API Testing

Use the scripts in `API_EXAMPLES.sh` or test endpoints directly:

```bash
# Health check
curl http://localhost:3000/health

# Non-streaming chat
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}],
    "model":"gpt-oss-20"
  }'
```

### Testing Streaming Endpoints

**Fetch-Stream (POST):**
```javascript
// In browser console
const res = await fetch('/api/chat-stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: 'Hello' }],
    model: 'gpt-oss-20'
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

**SSE (GET):**
```javascript
// In browser console
const json = JSON.stringify({
  messages: [{ role: 'user', content: 'Tell me a joke' }],
  model: 'llama3.2'
});
const payload = btoa(unescape(encodeURIComponent(json)));
const es = new EventSource(`/api/chat-sse?payload=${encodeURIComponent(payload)}`);

es.onmessage = e => console.log('Chunk:', e.data);
es.addEventListener('done', () => { es.close(); console.log('Done'); });
es.onerror = () => { es.close(); console.error('Error'); };
```

## Debugging

### Check Server Logs
The server uses `morgan` middleware for HTTP logging. All requests/responses are logged to stdout.

### Check Ollama Connection
```bash
# Verify Ollama is responding
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

### Check OpenAI API (if using cloud models)
```bash
# Verify API key works (replace with actual key)
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer sk-your-key-here"
```

## Performance Profiling

### Response Time Measurements
The `/api/multi-chat` endpoint returns timing info for each model:

```bash
curl -X POST http://localhost:3000/api/multi-chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"What is AI?"}],
    "models": ["gpt-oss-20","llama3.2","qwen2.5"]
  }' | jq '.results[] | {model, ms}'
```

This shows latency per model in milliseconds.

### Memory Usage
Monitor Node.js memory:
```bash
# Start server with Node.js memory stats
node --max-old-space-size=4096 server.js
```

## Common Issues & Solutions

### Rate Limiting
If you get `429 Rate limit exceeded`:
- Max 30 requests/min per IP
- Wait 1 minute or use different IP

### Long URLs for SSE
If SSE breaks with "URI too long":
- History is auto-capped to system + 12 messages
- Verify `getTrimmedMessages()` is being called

### Memory Leaks
If memory grows over time:
1. Check browser for unclosed streams
2. Verify rate limit map is cleaned
3. Monitor with `node --inspect server.js`

## Extending the Project

### Adding a New Model
1. Ensure model is available (local via `ollama pull` or cloud via API key)
2. Add option to `public/index.html` model selector
3. Update model recognition logic in `server.js` if needed

### Adding Custom Middleware
Middleware in `server.js` runs in this order:
1. Morgan logging
2. JSON parsing
3. CORS
4. Rate limiting
5. Route handlers

To add authentication:
```javascript
app.use((req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  // Verify token...
  next();
});
```

### Changing History Limit
Edit `MAX_HISTORY` constant:
- In `public/app.js` for SSE: `const MAX_HISTORY = 12;`
- In `server.js` for `/api/chat-stream`: `const MAX_HISTORY = 12;`
- In `server.js` for `/api/chat`: `const MAX_HISTORY = 12;`

## Docker Development

### Build image locally
```bash
docker build -t web-llm-chat:dev .
```

### Run single container (no Ollama)
```bash
docker run -p 3000:3000 \
  -e OPENAI_API_KEY=sk-your-key \
  web-llm-chat:dev
```

### View logs
```bash
docker-compose logs -f web-llm-chat
docker-compose logs -f ollama
```

### Clean up
```bash
docker-compose down -v  # Remove volumes too
```

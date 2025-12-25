# LLM Chat API Documentation

## Base URL
```
http://localhost:3000
```

---

## Endpoints

### 1. Health Check
**GET** `/health`

Check server status.

**Response:**
```json
{ "status": "ok" }
```

---

### 2. Non-Streaming Chat
**POST** `/api/chat`

Send a message and get a full response. Request/response style (no streaming).

**Request Body:**
```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" }
  ],
  "model": "gpt-oss-20"
}
```

**Parameters:**
- `messages` (array, required): Chat history. Each message has `role` ("system", "user", "assistant") and `content`.
- `model` (string, optional): Model name. Default: `"gpt-oss-20"` (local Ollama).
  - Local models: `gpt-oss-20`, `llama3.2`, `qwen2.5`
  - Cloud models: `gpt-4o`, `gpt-4o-mini` (requires `OPENAI_API_KEY`)

**Response:**
```json
{
  "text": "Hello! How can I help you today?",
  "raw": { "model": "gpt-oss-20", "message": { "content": "..." } }
}
```

**Server Behavior:**
- Automatically routes to Ollama for local models, OpenAI for cloud models.
- Trims history to system message + last 12 dialog messages to manage API limits.

---

### 3. Fetch-Stream Chat
**POST** `/api/chat-stream`

Stream response as plain text chunks. Uses HTTP streaming.

**Request Body:**
Same as `/api/chat`

**Response:**
- Content-Type: `text/plain`
- Chunks are written directly as text.

**Example (JavaScript):**
```javascript
const res = await fetch('/api/chat-stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ messages, model: 'gpt-oss-20' })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value, { stream: true });
  console.log(chunk); // Incremental text
}
```

---

### 4. Server-Sent Events (SSE)
**GET** `/api/chat-sse?payload=<base64>`

Stream response using Server-Sent Events. Payload is base64-encoded JSON.

**Query Parameters:**
- `payload` (string, required): Base64-encoded JSON with `{ messages, model }`.

**Response:**
- Content-Type: `text/event-stream`
- Events:
  - `data: <text chunk>` — Incremental text
  - `event: done` — Stream complete
  - `event: error` — Error occurred

**Example (JavaScript with EventSource):**
```javascript
const json = JSON.stringify({ messages, model: 'gpt-oss-20' });
const payload = btoa(unescape(encodeURIComponent(json)));
const url = `/api/chat-sse?payload=${encodeURIComponent(payload)}`;

const es = new EventSource(url);

es.onmessage = (e) => {
  console.log('Chunk:', e.data);
};

es.addEventListener('done', () => {
  console.log('Stream finished');
  es.close();
});

es.onerror = () => {
  console.error('Stream error');
  es.close();
};
```

---

### 5. Multi-Model Orchestration
**POST** `/api/multi-chat`

Query multiple models in parallel and return all responses plus the longest one.

**Request Body:**
```json
{
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Tell me a joke." }
  ],
  "models": ["gpt-oss-20", "llama3.2", "qwen2.5"]
}
```

**Parameters:**
- `messages` (array, required): Chat history.
- `models` (array, optional): List of model names. Default: `["gpt-oss-20", "llama3.2", "qwen2.5"]`

**Response:**
```json
{
  "totalMs": 3500,
  "models": ["gpt-oss-20", "llama3.2", "qwen2.5"],
  "results": [
    { "model": "gpt-oss-20", "ok": true, "ms": 1200, "text": "...", "raw": {...} },
    { "model": "llama3.2", "ok": true, "ms": 950, "text": "...", "raw": {...} },
    { "model": "qwen2.5", "ok": true, "ms": 1350, "text": "...", "raw": {...} }
  ],
  "best": "..." // Longest non-empty response
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "messages array required"
}
```

### 500 Server Error
```json
{
  "error": "Server error",
  "details": "Ollama error / OpenAI error / etc."
}
```

---

## Environment Variables

- `OLLAMA_URL` (default: `http://localhost:11434`): Ollama endpoint.
- `OPENAI_API_KEY`: Your OpenAI API key (required for cloud models).
- `PORT` (default: `3000`): Server port.

---

## History Management

All endpoints automatically cap message history to:
- **1 system message** (if present)
- **Last 12 dialog messages** (user/assistant)

This prevents excessively long requests to providers and improves performance.

---

## Examples

### cURL - Non-Streaming
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"system","content":"Be concise."},{"role":"user","content":"What is AI?"}],
    "model": "gpt-oss-20"
  }'
```

### cURL - Streaming (fetch-stream)
```bash
curl -X POST http://localhost:3000/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Tell me about ML."}],
    "model": "llama3.2"
  }'
```

### JavaScript Fetch - Non-Streaming
```javascript
const response = await fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hello!' }
    ],
    model: 'gpt-oss-20'
  })
});

const data = await response.json();
console.log(data.text);
```

---

## Model Recommendations

- **Local (Free):** `gpt-oss-20`, `llama3.2`, `qwen2.5` (requires Ollama)
- **Cloud (Paid):** `gpt-4o`, `gpt-4o-mini` (requires OPENAI_API_KEY)

For best performance:
1. Run Ollama locally for fastest inference (no API calls).
2. Use cloud models as fallback when local models are unavailable or insufficient.

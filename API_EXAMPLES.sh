# API Test/Demo File
# Use curl or paste directly into app

# ============================================================
# 1. Non-Streaming Chat (standard request/response)
# ============================================================

# Example: Send a message to gpt-oss-20 (local via Ollama)
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "model": "gpt-oss-20"
  }'

# Expected response:
# {
#   "text": "The capital of France is Paris...",
#   "raw": { "model": "gpt-oss-20", ... }
# }

# ============================================================
# 2. Streaming Chat (fetch-stream)
# ============================================================

# Send a streaming request (note: no built-in curl support for text/plain streams, but works in browser fetch)
curl -X POST http://localhost:3000/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Tell me a joke." }
    ],
    "model": "llama3.2"
  }'

# This will stream plain text chunks as they arrive

# ============================================================
# 3. Server-Sent Events (SSE)
# ============================================================

# SSE uses GET with base64-encoded payload
# In the browser, EventSource handles this; curl requires special handling:

# First, encode the payload:
# echo -n '{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"}],"model":"gpt-oss-20"}' | base64

# Then request:
# curl "http://localhost:3000/api/chat-sse?payload=eyJtZXNzYWdlcyI6W3sicm9sZSI6InN5c3RlbSIsImNvbnRlbnQiOiJZb3UgYXJlIGhlbHBmdWwuIn0seyJyb2xlIjoidXNlciIsImNvbnRlbnQiOiJIZWxsbyJ9XSwibW9kZWwiOiJncHQtb3NzLTIwIn0="

# ============================================================
# 4. Multi-Model Orchestration
# ============================================================

# Query multiple models in parallel and get the best response
curl -X POST http://localhost:3000/api/multi-chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "system", "content": "Answer concisely." },
      { "role": "user", "content": "What is 2+2?" }
    ],
    "models": ["gpt-oss-20", "llama3.2", "qwen2.5"]
  }'

# Expected response shows results from all models plus the best one

# ============================================================
# 5. Error Handling
# ============================================================

# Missing messages (will get 400):
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20"}'

# No Ollama running (will get 500 with details):
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Hi"}],
    "model":"gpt-oss-20"
  }'

# ============================================================
# Notes for using cloud models (gpt-4o, gpt-4o-mini)
# ============================================================

# Ensure OPENAI_API_KEY is set in .env:
# OPENAI_API_KEY=sk-your-actual-key-here

# Then use the cloud model in requests:
# "model": "gpt-4o-mini"

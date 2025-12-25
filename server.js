import express from 'express';
import morgan from 'morgan';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { parse as parseToml } from 'toml';
import rateLimit from 'express-rate-limit';
import {
  registerUser,
  loginUser,
  refreshToken as refreshTokenFn,
  verifyToken,
  getUserProfile,
  updateUserPreferences,
  setupTestUsers
} from './auth.js';

dotenv.config();

// Rate limiter for authentication-related routes
const authRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 auth requests per windowMs
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false   // Disable the `X-RateLimit-*` headers
});

const trainingControlLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute window
  max: 10, // limit each IP to 10 training control requests per window
  standardHeaders: true,
  legacyHeaders: false
});

// Load optional TOML configuration and merge with environment
const DEFAULT_CONFIG_PATH = process.env.CONFIG_TOML || path.join(
  process.env.USERPROFILE || process.env.HOME || '',
  '.codex',
  'config.toml'
);

function loadTomlConfig(p) {
  try {
    if (p && fs.existsSync(p)) {
      const raw = fs.readFileSync(p, 'utf-8');
      // Normalize Windows CRLF to LF to avoid parser issues
      const normalized = raw.replace(/\r\n/g, '\n');
      return parseToml(normalized);
    }
  } catch (e) {
    console.warn(`Config load failed from ${p}: ${e.message}`);
  }
  return {};
}

const CFG = loadTomlConfig(DEFAULT_CONFIG_PATH);

const app = express();
const PORT = Number(process.env.PORT || CFG?.server?.port || 3000);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OLLAMA_URL = process.env.OLLAMA_URL || CFG?.ollama?.url || 'http://localhost:11434';
const DEFAULT_LOCAL_MODEL = (CFG?.ollama?.default_model || 'gpt-oss-20');
const CLOUD_DEFAULT_MODEL = (CFG?.openai?.model || 'gpt-4o-mini');
const CLOUD_MODELS = new Set(CFG?.openai?.cloud_models || ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-mini']);
const isCloudModelName = (name) => {
  const m = (name || '').trim();
  if (!m) return false;
  if (CLOUD_MODELS.has(m)) return true;
  return m.startsWith('gpt-') && !m.includes('gpt-oss');
};
const SSE_HEARTBEAT_MS = Number(CFG?.sse?.heartbeat_interval_ms || 25000);
const SSE_RETRY_MS = Number(CFG?.sse?.retry_ms || 3000);
const MORGAN_FORMAT = CFG?.logging?.morgan_format || 'dev';
const JSON_LIMIT = `${CFG?.limits?.max_payload_mb || 10}mb`;
const CHAT_MAX_HISTORY = Number(CFG?.chat?.max_history || 12);
const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS || CFG?.server?.upstream_timeout_ms || 20000);
const STREAM_TIMEOUT_MS = Number(process.env.STREAM_TIMEOUT_MS || CFG?.server?.stream_timeout_ms || 0);
const TRAINING_RUNNER = path.join(process.cwd(), 'training_runner.py');
const TRAINING_LOG_LIMIT = 1500;
const TRAINING_MODULES = new Set([
  'language_model',
  'image_classifier',
  'game_ai',
  'custom_nn',
  'world_generator'
]);
const trainingJobs = new Map();
const resolvePythonPath = () => {
  if (process.platform === 'win32') {
    const candidate = path.join(process.cwd(), '.venv', 'Scripts', 'python.exe');
    if (fs.existsSync(candidate)) return candidate;
  } else {
    const candidate = path.join(process.cwd(), '.venv', 'bin', 'python');
    if (fs.existsSync(candidate)) return candidate;
  }
  return 'python';
};
const getActiveTrainingJob = () => {
  for (const job of trainingJobs.values()) {
    if (job.status === 'running' || job.status === 'stopping') return job;
  }
  return null;
};
const appendTrainingLog = (job, chunk) => {
  const text = chunk.toString('utf8').replace(/\r/g, '\n');
  job.buffer += text;
  const lines = job.buffer.split('\n');
  job.buffer = lines.pop() || '';
  for (const line of lines) {
    if (!line.trim()) continue;
    job.logs.push(line);
    if (job.logs.length > TRAINING_LOG_LIMIT) {
      job.logs.shift();
    }
  }
};
const flushTrainingLog = (job) => {
  if (job.buffer.trim()) {
    job.logs.push(job.buffer.trim());
    if (job.logs.length > TRAINING_LOG_LIMIT) {
      job.logs.shift();
    }
  }
  job.buffer = '';
};
const fetchWithTimeout = async (url, options = {}, timeoutMs = UPSTREAM_TIMEOUT_MS) => {
  if (!timeoutMs || timeoutMs <= 0) {
    return fetch(url, options);
  }
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
};
if (CFG?.server?.trust_proxy === true) {
  app.set('trust proxy', 1);
}
const DATA_DIR = process.env.DATA_DIR || path.join(process.cwd(), 'ai_training', 'language_model', 'data');
const AGENT_PERSONAS = {
  friendly: 'You are a warm, concise guide. Keep answers short, friendly, and actionable.',
  coder: 'You are a focused coding assistant. Return concise answers, bullet steps, and minimal prose. Show code blocks when helpful.',
  coach: 'You are a productivity coach. Give clear next actions and keep replies under 4 sentences.',
  roleplay: 'You are the in-world narrator for the 3D game, describing scenes briefly and helping the player with direction.'
};

// Info logs for environment
if (!OPENAI_API_KEY) {
  console.log('OPENAI_API_KEY not set; will prefer local Ollama if available.');
}
console.log(`Using OLLAMA_URL: ${OLLAMA_URL}`);

// Check Ollama availability at startup
if (process.env.NODE_ENV !== 'test') {
  (async () => {
  try {
    const resp = await fetchWithTimeout(`${OLLAMA_URL}/api/tags`, {}, 3000);
    if (resp.ok) {
      const data = await resp.json();
      const models = data.models?.map(m => m.name).join(', ') || 'none';
      console.log(`✓ Ollama is running with models: ${models}`);
    } else {
      console.warn(`⚠ Ollama responded with status ${resp.status}`);
    }
  } catch (err) {
    console.warn(`⚠ Ollama not available at ${OLLAMA_URL}`);
    console.warn(`   To start Ollama: ollama serve`);
    console.warn(`   To pull a model: ollama pull gpt-oss-20`);
  }
  })();
}

app.use(morgan(MORGAN_FORMAT));
app.use(express.json({ limit: JSON_LIMIT }));
app.use(express.static('public', { index: 'game.html' }));

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Simple rate limiting per IP (configurable requests per minute)
const rateLimitMap = new Map();
const RATE_LIMIT = Number(process.env.RATE_LIMIT_PER_MINUTE || CFG?.server?.rate_limit_per_minute || 30);
const RATE_WINDOW = 60000; // 1 minute

app.use((req, res, next) => {
  const ip = req.ip || 'unknown';
  const now = Date.now();
  
  if (!rateLimitMap.has(ip)) {
    rateLimitMap.set(ip, []);
  }
  
  const timestamps = rateLimitMap.get(ip);
  const recent = timestamps.filter(t => now - t < RATE_WINDOW);
  
  if (recent.length >= RATE_LIMIT) {
    return res.status(429).json({ error: 'Rate limit exceeded. Max 30 requests/minute.' });
  }
  
  recent.push(now);
  rateLimitMap.set(ip, recent);
  next();
});

// Initialize demo user for testing (configurable)
if (CFG?.auth?.demo_user_enabled !== false) {
  setupTestUsers();
}

// Simple health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// ============================================================================
// AUTHENTICATION ENDPOINTS
// ============================================================================

// Register new user
app.post('/auth/register', async (req, res) => {
  try {
    const { username, password, email } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }

    const result = await registerUser(username, password, email || '');
    
    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      user: result.user
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Login user
app.post('/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }

    const result = await loginUser(username, password);
    
    if (!result.success) {
      return res.status(401).json({ error: result.error });
    }

    res.json({
      success: true,
      token: result.token,
      refreshToken: result.refreshToken,
      user: result.user
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Refresh access token
app.post('/auth/refresh', (req, res) => {
  try {
    const { refreshToken } = req.body;

    if (!refreshToken) {
      return res.status(400).json({ error: 'Refresh token required' });
    }

    const result = refreshTokenFn(refreshToken);
    
    if (!result.success) {
      return res.status(401).json({ error: result.error });
    }

    res.json({
      success: true,
      token: result.token
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get user profile (protected)
app.get('/auth/profile', authRateLimiter, verifyToken, (req, res) => {
  try {
    const profile = getUserProfile(req.user.username);
    
    if (!profile) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json(profile);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update user preferences (protected)
app.put('/auth/preferences', authLimiter, verifyToken, (req, res) => {
  try {
    const { preferences } = req.body;

    if (!preferences) {
      return res.status(400).json({ error: 'Preferences required' });
    }

    const result = updateUserPreferences(req.user.username, preferences);
    
    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    res.json({
      success: true,
      preferences: result.preferences
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Agent wrapper: injects persona system prompt and trims history
app.post('/api/agent-chat', async (req, res) => {
  try {
    const { input, history, persona = 'friendly', model } = req.body || {};
    if (!input || typeof input !== 'string') {
      return res.status(400).json({ error: 'input text required' });
    }
    const system = AGENT_PERSONAS[persona] || AGENT_PERSONAS.friendly;
    const dialog = Array.isArray(history) ? history : [];
    const sanitizedHistory = dialog
      .filter(m => m && typeof m.content === 'string' && (m.role === 'user' || m.role === 'assistant'))
      .slice(Math.max(0, dialog.length - 12));
    const messages = [
      { role: 'system', content: system },
      ...sanitizedHistory,
      { role: 'user', content: input }
    ];

    // Reuse the main chat handler by calling it directly
    const response = await fetchWithTimeout(`http://localhost:${PORT}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, model })
    }, UPSTREAM_TIMEOUT_MS);

    const data = await response.json();
    if (!response.ok) {
      return res.status(response.status).json(data);
    }
    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});


// Chat endpoint - routes to Ollama (local) or OpenAI (cloud) based on model
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, model } = req.body;
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array required' });
    }

    // Cap history server-side: keep system + last 12 dialog messages
    const MAX_HISTORY = CHAT_MAX_HISTORY;
    const systemMsg = messages.find(m => m.role === 'system');
    const dialog = messages.filter(m => m.role !== 'system');
    const trimmedMessages = systemMsg
      ? [systemMsg, ...dialog.slice(Math.max(0, dialog.length - MAX_HISTORY))]
      : dialog.slice(Math.max(0, dialog.length - MAX_HISTORY));

    const chosenModel = (model || DEFAULT_LOCAL_MODEL).trim();
    const isCloudModel = isCloudModelName(chosenModel);

    const trimDetails = (value, maxLen = 2000) => {
      const s = typeof value === 'string' ? value : JSON.stringify(value);
      if (!s) return '';
      return s.length > maxLen ? (s.slice(0, maxLen) + '…') : s;
    };

    // Local: Ollama
    if (!isCloudModel) {
      const url = `${OLLAMA_URL}/api/chat`;
      let response;
      try {
        response = await fetchWithTimeout(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: chosenModel,
            messages: trimmedMessages,
            stream: false
          })
        }, UPSTREAM_TIMEOUT_MS);
      } catch (err) {
        console.error('Ollama fetch failed', {
          url,
          model: chosenModel,
          name: err?.name,
          code: err?.code,
          message: err?.message
        });
        return res.status(502).json({
          error: 'Ollama unreachable',
          details: trimDetails(err?.message || 'Failed to connect to Ollama'),
          hint: 'Start Ollama (ollama serve) and ensure the model is pulled (ollama pull gpt-oss-20).',
          ollamaUrl: OLLAMA_URL,
          model: chosenModel
        });
      }

      if (!response.ok) {
        const contentType = response.headers.get('content-type') || '';
        let details = '';
        let parsed = null;
        try {
          if (contentType.includes('application/json')) {
            parsed = await response.json();
            details = trimDetails(parsed);
          } else {
            details = trimDetails(await response.text());
          }
        } catch (err) {
          details = trimDetails(err?.message || 'Failed to read Ollama error body');
        }

        console.error('Ollama error response', {
          url,
          model: chosenModel,
          status: response.status,
          details
        });

        return res.status(response.status).json({
          error: 'Ollama error',
          status: response.status,
          details,
          ollamaUrl: OLLAMA_URL,
          model: chosenModel
        });
      }

      const data = await response.json();
      const reply = data?.message?.content || '';
      return res.json({ text: reply, raw: data });
    }

    // Cloud: OpenAI
    if (!OPENAI_API_KEY) {
      return res.status(400).json({ 
        error: 'OpenAI API key not configured',
        suggestion: 'Add OPENAI_API_KEY to .env file or use a local model like gpt-oss-20',
        guide: 'See OPENAI_API_KEY_SETUP.md for instructions'
      });
    }
    // Fallback: OpenAI Responses API
    const response = await fetchWithTimeout('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: chosenModel || CLOUD_DEFAULT_MODEL,
        // Send a consolidated input string preserving roles for basic chat memory
        input: trimmedMessages.map(m => `${m.role}: ${m.content}`).join('\n\n')
      })
    });

    if (!response.ok) {
      const text = await response.text();
      return res.status(response.status).json({ error: 'OpenAI error', details: text });
    }

    const data = await response.json();
    let outputText = '';
    try {
      // Prefer unified output_text if available from Responses API
      if (typeof data.output_text === 'string') {
        outputText = data.output_text;
      } else {
        const item = data.output?.[0];
        if (item?.type === 'message') {
          const textPart = item.content?.find(c => c.type === 'text');
          outputText = textPart?.text || '';
        }
      }
    } catch (e) {
      outputText = data.output_text || '';
    }
    res.json({ text: outputText, raw: data });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Streaming chat endpoint: writes incremental text chunks to the response
app.post('/api/chat-stream', async (req, res) => {
  try {
    const { messages, model } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array required' });
    }

    // Cap history server-side: keep system + last 12 dialog messages
    const MAX_HISTORY = CHAT_MAX_HISTORY;
    const systemMsg = messages.find(m => m.role === 'system');
    const dialog = messages.filter(m => m.role !== 'system');
    const trimmedMessages = systemMsg
      ? [systemMsg, ...dialog.slice(Math.max(0, dialog.length - MAX_HISTORY))]
      : dialog.slice(Math.max(0, dialog.length - MAX_HISTORY));

    const chosenModel = (model || DEFAULT_LOCAL_MODEL).trim();
    const isCloudModel = isCloudModelName(chosenModel);

    // Prepare streaming response headers
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const decoder = new TextDecoder();
    let lineBuffer = '';

    if (!isCloudModel) {
      // Ollama streaming
      const response = await fetchWithTimeout(`${OLLAMA_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: chosenModel, messages: trimmedMessages, stream: true })
      }, STREAM_TIMEOUT_MS);
      if (!response.ok || !response.body) {
        const text = await response.text().catch(() => '');
        res.status(response.status || 500).end(text || 'Ollama error');
        return;
      }
      const reader = response.body.getReader();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        lineBuffer += chunk;
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const obj = JSON.parse(line);
            const piece = obj?.message?.content || '';
            if (piece) {
              res.write(piece);
            }
            if (obj?.done) {
              break;
            }
          } catch (_) {
            // If parsing fails, write raw line as best-effort
            res.write(line);
          }
        }
      }
      res.end();
      return;
    }

    // OpenAI streaming
    if (!OPENAI_API_KEY) {
      return res.status(400).json({ 
        error: 'OpenAI API key not configured',
        suggestion: 'Add OPENAI_API_KEY to .env file or use a local model like gpt-oss-20',
        guide: 'See OPENAI_API_KEY_SETUP.md for instructions'
      });
    }
    const response = await fetchWithTimeout('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: chosenModel || CLOUD_DEFAULT_MODEL,
        stream: true,
        input: trimmedMessages.map(m => `${m.role}: ${m.content}`).join('\n\n')
      })
    });
    if (!response.ok || !response.body) {
      const text = await response.text().catch(() => '');
      res.status(response.status || 500).end(text || 'OpenAI error');
      return;
    }
    const reader = response.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      // The Responses API uses SSE; extract data lines and parse JSON where possible
      const lines = chunk.split('\n');
      for (const l of lines) {
        const line = l.trim();
        if (!line || line.startsWith(':')) continue; // comment/keepalive
        if (line.startsWith('data:')) {
          const payload = line.slice(5).trim();
          try {
            const obj = JSON.parse(payload);
            const piece = obj?.delta || obj?.output_text?.delta || '';
            if (piece) res.write(piece);
          } catch (_) {
            // Not JSON or not delta; ignore
          }
        }
      }
    }
    res.end();
  } catch (err) {
    console.error(err);
    // In case headers already sent, end the stream
    if (!res.headersSent) {
      res.status(500).json({ error: 'Server error' });
    } else {
      try { res.end(); } catch (_) {}
    }
  }
});

// SSE streaming endpoint (GET) using EventSource
app.get('/api/chat-sse', async (req, res) => {
  try {
    const { payload } = req.query || {};
    if (typeof payload !== 'string' || payload.length === 0) {
      return res.status(400).json({ error: 'payload query param required (base64 JSON {messages, model})' });
    }
    let messages, model;
    try {
      const jsonStr = Buffer.from(payload, 'base64').toString('utf8');
      const parsed = JSON.parse(jsonStr);
      messages = parsed.messages;
      model = parsed.model;
    } catch (_) {
      return res.status(400).json({ error: 'invalid payload encoding' });
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array required' });
    }

    const chosenModel = (model || DEFAULT_LOCAL_MODEL).trim();
    const isCloudModel = isCloudModelName(chosenModel);

    // SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    try { res.flushHeaders(); } catch (_) {}

    const writeData = (text) => {
      if (!text) return;
      res.write(`data: ${text.replace(/\n/g, '\\n')}\n\n`);
    };
    const writeDone = () => {
      res.write('event: done\n');
      res.write('data: done\n\n');
    };

    const decoder = new TextDecoder();

    if (!isCloudModel) {
      // Ollama SSE (adapt JSON lines to SSE)
      const response = await fetchWithTimeout(`${OLLAMA_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: chosenModel, messages, stream: true })
      }, STREAM_TIMEOUT_MS);
      if (!response.ok || !response.body) {
        const text = await response.text().catch(() => '');
        res.write(`event: error\n`);
        res.write(`data: ${text || 'Ollama error'}\n\n`);
        writeDone();
        res.end();
        return;
      }
      const reader = response.body.getReader();
      let lineBuffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        lineBuffer += chunk;
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const obj = JSON.parse(line);
            const piece = obj?.message?.content || '';
            if (piece) writeData(piece);
            if (obj?.done) break;
          } catch (_) {
            writeData(line);
          }
        }
      }
      writeDone();
      res.end();
      return;
    }

    // OpenAI SSE passthrough
    if (!OPENAI_API_KEY) {
      res.write('event: error\n');
      res.write('data: {"error":"OpenAI API key not configured","suggestion":"Add OPENAI_API_KEY to .env or use local model like gpt-oss-20","guide":"See OPENAI_API_KEY_SETUP.md"}\n\n');
      writeDone();
      res.end();
      return;
    }

    const response = await fetchWithTimeout('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: chosenModel || CLOUD_DEFAULT_MODEL,
        stream: true,
        input: messages.map(m => `${m.role}: ${m.content}`).join('\n\n')
      })
    });
    if (!response.ok || !response.body) {
      const text = await response.text().catch(() => '');
      res.write('event: error\n');
      res.write(`data: ${text || 'OpenAI error'}\n\n`);
      writeDone();
      res.end();
      return;
    }
    const reader = response.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      for (const l of lines) {
        const line = l.trim();
        if (!line || line.startsWith(':')) continue;
        if (line.startsWith('data:')) {
          const payload = line.slice(5).trim();
          try {
            const obj = JSON.parse(payload);
            const piece = obj?.delta || obj?.output_text?.delta || '';
            if (piece) writeData(piece);
          } catch (_) {
            // ignore non-JSON
          }
        }
      }
    }
    writeDone();
    res.end();
  } catch (err) {
    console.error(err);
    try {
      res.write('event: error\n');
      res.write('data: Server error\n\n');
      res.end();
    } catch (_) {
      if (!res.headersSent) res.status(500).end('Server error');
    }
  }
});

// Multi-LLM orchestration endpoint: queries multiple models in parallel
app.post('/api/multi-chat', async (req, res) => {
  try {
    const { messages, models } = req.body;
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array required' });
    }
    const modelList = Array.isArray(models) && models.length > 0
      ? models.map(m => String(m).trim())
      : ['gpt-oss-20', 'llama3.2', 'qwen2.5'];

    const startAll = Date.now();

    const queryOllama = async (model) => {
      const t0 = Date.now();
      const resp = await fetchWithTimeout(`${OLLAMA_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages, stream: false })
      }, UPSTREAM_TIMEOUT_MS);
      const t1 = Date.now();
      const rawText = await resp.text();
      let data;
      try { data = JSON.parse(rawText); } catch (_) { data = { rawText }; }
      const text = data?.message?.content || '';
      return { model, provider: 'ollama', ok: resp.ok, status: resp.status, ms: t1 - t0, text, raw: data };
    };

    const queryOpenAI = async (model) => {
      const t0 = Date.now();
      const resp = await fetchWithTimeout('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: model || 'gpt-4o-mini',
          input: messages.map(m => ({ role: m.role, content: m.content }))
        })
      });
      const t1 = Date.now();
      const raw = await resp.json().catch(async () => ({ rawText: await resp.text() }));
      let text = '';
      try {
        const item = raw.output?.[0];
        if (item?.type === 'message') {
          const textPart = item.content?.find(c => c.type === 'text');
          text = textPart?.text || '';
        } else if (typeof raw.output_text === 'string') {
          text = raw.output_text;
        }
      } catch (_) {
        text = raw.output_text || '';
      }
      return { model, provider: 'openai', ok: resp.ok, status: resp.status, ms: t1 - t0, text, raw };
    };

    const tasks = modelList.map(model => {
      const cloudPreferred = isCloudModelName(model);
      if (cloudPreferred) {
        if (!OPENAI_API_KEY) {
          return Promise.resolve({
            model,
            provider: 'openai',
            ok: false,
            status: 400,
            ms: 0,
            text: 'OpenAI API key not configured',
            raw: { error: 'OpenAI API key not configured' }
          });
        }
        return queryOpenAI(model);
      }
      return queryOllama(model);
    }, UPSTREAM_TIMEOUT_MS);

    const results = await Promise.all(tasks);

    // Aggregation strategies
    const promptText = messages.find(m => m.role === 'user')?.content || '';
    const stop = new Set(['the','a','an','and','or','of','to','in','on','for','with','by','is','are','was','were','be','as','at','from']);
    const toks = (s) => String(s).toLowerCase().match(/[a-z0-9]+/g) || [];
    const keywords = toks(promptText).filter(w => !stop.has(w));
    const kwSet = new Set(keywords);

    const lengthBest = () => {
      return results
        .filter(r => r.ok && r.text && r.text.trim().length > 0)
        .sort((a, b) => b.text.length - a.text.length)[0]?.text || '';
    };

    const keywordBest = () => {
      let bestR = null;
      let bestScore = -1;
      for (const r of results) {
        if (!(r.ok && r.text)) continue;
        const rtoks = toks(r.text);
        let overlap = 0;
        for (const t of rtoks) if (kwSet.has(t)) overlap++;
        // Combine keyword overlap and normalized length
        const lenScore = Math.log(1 + r.text.length);
        const score = overlap + 0.3 * lenScore;
        if (score > bestScore) { bestScore = score; bestR = r; }
      }
      return bestR?.text || '';
    };

    const rankByModelBest = () => {
      // Prefer a fixed order if available
      const order = ['gpt-oss-20', 'llama3.2', 'qwen2.5'];
      for (const name of order) {
        const r = results.find(x => x.model === name && x.ok && x.text && x.text.trim());
        if (r) return r.text;
      }
      return lengthBest();
    };

    const semanticBest = async () => {
      return new Promise((resolve) => {
        const payload = JSON.stringify({ prompt: promptText, responses: results });
        const py = spawn('python', ['semantic_rank.py', payload]);
        let stdout = '';
        let stderr = '';
        py.stdout.on('data', d => stdout += d);
        py.stderr.on('data', d => stderr += d);
        py.on('close', () => {
          try {
            const resp = JSON.parse(stdout);
            resolve(resp.best || lengthBest());
          } catch (_) {
            resolve(lengthBest());
          }
        });
        py.on('error', () => resolve(lengthBest()));
      });
    };

    const metaLLMBest = async () => {
      const valid = results.filter(r => r.ok && r.text && r.text.trim());
      if (valid.length === 0) return '';
      if (valid.length === 1) return valid[0].text;

      const judgePrompt = `You are a response quality judge. Given a user query and multiple AI responses, choose the single best response by number (1-${valid.length}).

User Query: "${promptText}"

${valid.map((r, i) => `Response ${i + 1} (${r.model}):\n${r.text}\n`).join('\n')}

Respond with ONLY the number of the best response (1-${valid.length}), nothing else.`;

      try {
        const judgeResp = await fetchWithTimeout(`${OLLAMA_URL}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'gpt-oss-20',
            messages: [{ role: 'user', content: judgePrompt }],
            stream: false
          })
        }, UPSTREAM_TIMEOUT_MS);
        const judgeData = await judgeResp.json();
        const judgeText = judgeData?.message?.content || '';
        const match = judgeText.match(/\d+/);
        if (match) {
          const idx = parseInt(match[0]) - 1;
          if (idx >= 0 && idx < valid.length) {
            return valid[idx].text;
          }
        }
      } catch (_) {
        // Fallback to length on error
      }
      return lengthBest();
    };

    const votingBest = () => {
      const valid = results.filter(r => r.ok && r.text && r.text.trim());
      if (valid.length === 0) return '';
      
      // Simple voting: find most similar responses by first 100 chars
      const votes = new Map();
      for (const r of valid) {
        const sig = r.text.slice(0, 100).toLowerCase().trim();
        votes.set(sig, (votes.get(sig) || 0) + 1);
      }
      
      let maxVotes = 0;
      let bestSig = '';
      for (const [sig, count] of votes) {
        if (count > maxVotes) {
          maxVotes = count;
          bestSig = sig;
        }
      }
      
      const winner = valid.find(r => r.text.slice(0, 100).toLowerCase().trim() === bestSig);
      return winner?.text || lengthBest();
    };

    const agg = (req.body?.aggregator || 'length').toLowerCase();
    let best;
    switch (agg) {
      case 'keyword':
        best = keywordBest();
        break;
      case 'rank-by-model':
        best = rankByModelBest();
        break;
      case 'semantic':
        best = await semanticBest();
        break;
      case 'meta-llm':
        best = await metaLLMBest();
        break;
      case 'voting':
        best = votingBest();
        break;
      case 'length':
      default:
        best = lengthBest();
        break;
    }

    const totalMs = Date.now() - startAll;
    res.json({ totalMs, models: modelList, results, best, aggregator: agg });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Append a collaboration record to collab.jsonl
app.post('/api/collab', async (req, res) => {
  try {
    const { prompt, results, best, aggregator } = req.body || {};
    if (typeof prompt !== 'string' || !Array.isArray(results)) {
      return res.status(400).json({ error: 'prompt string and results array required' });
    }
    const record = {
      prompt,
      results,
      best: typeof best === 'string' ? best : '',
      aggregator: typeof aggregator === 'string' ? aggregator : 'unknown'
    };
    fs.mkdirSync(DATA_DIR, { recursive: true });
    const outPath = path.join(DATA_DIR, 'collab.jsonl');
    fs.appendFileSync(outPath, JSON.stringify(record) + '\n', 'utf-8');
    res.json({ ok: true, path: outPath });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Get dataset statistics
app.get('/api/stats', (req, res) => {
  try {
    const collabPath = path.join(DATA_DIR, 'collab.jsonl');
    if (!fs.existsSync(collabPath)) {
      return res.json({ count: 0, models: {}, aggregators: {} });
    }

    const lines = fs.readFileSync(collabPath, 'utf-8').split('\n').filter(l => l.trim());
    const modelStats = {};
    const aggStats = {};
    let totalTokens = 0;

    for (const line of lines) {
      try {
        const rec = JSON.parse(line);
        const agg = rec.aggregator || 'unknown';
        aggStats[agg] = (aggStats[agg] || 0) + 1;
        
        for (const r of (rec.results || [])) {
          const m = r.model || 'unknown';
          if (!modelStats[m]) modelStats[m] = { count: 0, totalMs: 0, avgMs: 0 };
          modelStats[m].count++;
          modelStats[m].totalMs += (r.ms || 0);
        }
        
        totalTokens += (rec.prompt?.length || 0) + (rec.best?.length || 0);
      } catch (_) {
        // Skip malformed lines
      }
    }

    for (const m in modelStats) {
      modelStats[m].avgMs = Math.round(modelStats[m].totalMs / modelStats[m].count);
    }

    res.json({
      count: lines.length,
      models: modelStats,
      aggregators: aggStats,
      estimatedTokens: totalTokens
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// ============================================================================
// TRAINING ENDPOINTS
// ============================================================================

app.get('/api/training/jobs', (req, res) => {
  const jobs = Array.from(trainingJobs.values()).map(job => ({
    id: job.id,
    module: job.module,
    status: job.status,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    exitCode: job.exitCode,
    pid: job.pid
  }));
  res.json({ jobs });
});

app.get('/api/training/logs/:id', (req, res) => {
  const job = trainingJobs.get(req.params.id);
  if (!job) {
    return res.status(404).json({ error: 'Training job not found' });
  }
  const cursor = Number(req.query.cursor || 0);
  const safeCursor = Number.isFinite(cursor) && cursor >= 0 ? cursor : 0;
  const lines = job.logs.slice(safeCursor);
  res.json({
    id: job.id,
    module: job.module,
    status: job.status,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    exitCode: job.exitCode,
    cursor: job.logs.length,
    lines
  });
});

app.post('/api/training/start', trainingControlLimiter, (req, res) => {
  try {
    const { module, config } = req.body || {};
    const moduleKey = typeof module === 'string' ? module.trim() : '';
    if (!TRAINING_MODULES.has(moduleKey)) {
      return res.status(400).json({ error: 'Unknown training module' });
    }

    const activeJob = getActiveTrainingJob();
    if (activeJob) {
      return res.status(409).json({
        error: 'Training already running',
        activeJob: {
          id: activeJob.id,
          module: activeJob.module,
          status: activeJob.status
        }
      }, UPSTREAM_TIMEOUT_MS);
    }

    if (!fs.existsSync(TRAINING_RUNNER)) {
      return res.status(500).json({ error: 'Training runner not found', path: TRAINING_RUNNER });
    }

    const pythonPath = resolvePythonPath();
    const payload = Buffer.from(JSON.stringify(config || {}), 'utf8').toString('base64');
    const args = ['-u', TRAINING_RUNNER, '--module', moduleKey];
    if (payload) {
      args.push('--config-b64', payload);
    }

    const child = spawn(pythonPath, args, {
      cwd: process.cwd(),
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    }, STREAM_TIMEOUT_MS);

    const jobId = crypto.randomUUID();
    const job = {
      id: jobId,
      module: moduleKey,
      status: 'running',
      startedAt: new Date().toISOString(),
      finishedAt: null,
      exitCode: null,
      logs: [],
      buffer: '',
      pid: child.pid,
      process: child
    };

    trainingJobs.set(jobId, job);

    child.stdout.on('data', data => appendTrainingLog(job, data));
    child.stderr.on('data', data => appendTrainingLog(job, data));
    child.on('error', err => {
      appendTrainingLog(job, `Failed to start training: ${err.message}`);
      flushTrainingLog(job);
      job.status = 'failed';
      job.exitCode = -1;
      job.finishedAt = new Date().toISOString();
    }, STREAM_TIMEOUT_MS);
    child.on('close', (code) => {
      flushTrainingLog(job);
      job.exitCode = code;
      job.finishedAt = new Date().toISOString();
      if (job.status === 'stopping') {
        job.status = 'stopped';
      } else {
        job.status = code === 0 ? 'completed' : 'failed';
      }
    });

    res.json({ id: jobId, status: job.status, module: job.module });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to start training' });
  }
});

app.post('/api/training/stop/:id', trainingControlLimiter, (req, res) => {
  const job = trainingJobs.get(req.params.id);
  if (!job) {
    return res.status(404).json({ error: 'Training job not found' });
  }
  if (job.status !== 'running') {
    return res.status(409).json({ error: 'Training job is not running' });
  }

  job.status = 'stopping';
  if (job.process && !job.process.killed) {
    job.process.kill();
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(job.process.pid), '/t', '/f']);
    }
  }

  res.json({ ok: true });
});

// ============================================================================
// WORLD GENERATION ENDPOINT
// ============================================================================

app.post('/api/generate-world', async (req, res) => {
  try {
    const { prompt } = req.body;
    
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Prompt string required' });
    }

    console.log('Generating world for prompt:', prompt);

    // Spawn Python process to generate world
    const pythonPath = resolvePythonPath();

    const modelPath = path.join(process.cwd(), 'ai_training', 'world_generator', 'models', 'world_generator_final');

    // Check if model exists
    if (!fs.existsSync(modelPath)) {
      return res.status(404).json({
        error: 'World generator model not found',
        message: 'Please run: python world_generator_train.py',
        path: modelPath
      });
    }

    // Create a temporary Python script to generate world
    const promptB64 = Buffer.from(prompt, 'utf8').toString('base64');
    const generateScript = `
import base64
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = ${JSON.stringify(modelPath.replace(/\\/g, '/'))}
PROMPT_B64 = os.environ.get("PROMPT_B64", "")
try:
    PROMPT = base64.b64decode(PROMPT_B64).decode("utf-8", errors="replace")
except Exception:
    PROMPT = ""

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=800,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON
    json_start = generated.find('{')
    json_end = generated.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        world_json = generated[json_start:json_end]
        world = json.loads(world_json)
        print(json.dumps({"success": True, "world": world}))
    else:
        print(json.dumps({"success": False, "error": "No JSON found", "text": generated}))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

    // Run Python script
    const py = spawn(pythonPath, ['-c', generateScript], {
      env: { ...process.env, PROMPT_B64: promptB64, PYTHONUNBUFFERED: '1' }
    });
    
    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    py.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    py.on('close', (code) => {
      if (code !== 0) {
        console.error('Python error:', stderr);
        return res.status(500).json({
          error: 'Generation failed',
          details: stderr,
          hint: 'Make sure transformers and torch are installed'
        });
      }

      try {
        const result = JSON.parse(stdout);
        
        if (result.success) {
          res.json({ world: result.world });
        } else {
          res.status(500).json({
            error: result.error,
            text: result.text,
            message: 'Model needs more training or prompt adjustment'
          });
        }
      } catch (err) {
        console.error('Parse error:', err);
        res.status(500).json({
          error: 'Failed to parse generation output',
          stdout,
          stderr
        });
      }
    });

    py.on('error', (err) => {
      console.error('Spawn error:', err);
      res.status(500).json({
        error: 'Failed to spawn Python process',
        details: err.message
      });
    });

  } catch (err) {
    console.error('Server error:', err);
    res.status(500).json({ error: 'Server error', details: err.message });
  }
});

if (process.env.NODE_ENV !== 'test') {
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`API docs: http://localhost:${PORT} (open in browser)`);
    console.log('API reference: See API.md in project root');
  });
}

export { app, rateLimitMap };

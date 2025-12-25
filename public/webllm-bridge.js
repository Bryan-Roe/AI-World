// Minimal bridge to run LLM fully in browser via WebGPU using WebLLM
// Exposes window.WebLLMBridge with init/ensureInit/chat helpers

// We import ESM build from a CDN and attach a thin wrapper
import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

const state = {
  engine: null,
  currentModelId: null,
  initializing: false,
  currentStream: null,
  cancelRequested: false,
};

function dispatchProgress(progress, text) {
  try {
    window.dispatchEvent(new CustomEvent('webllm-progress', { detail: { progress, text } }));
  } catch {}
}

function dispatchReady(modelId) {
  try { window.dispatchEvent(new CustomEvent('webllm-ready', { detail: { modelId } })); } catch {}
}

async function init(modelId) {
  if (state.initializing) {
    // wait for ongoing init
    while (state.initializing) {
      await new Promise(r => setTimeout(r, 100));
    }
    return { ok: !!state.engine, modelId: state.currentModelId };
  }
  state.initializing = true;
  try {
    // Use prebuilt app config from WebLLM registry
    const cfg = webllm.prebuiltAppConfig?.[modelId];
    if (!cfg) {
      throw new Error(`WebLLM model not found: ${modelId}`);
    }
    // Dispose previous engine if switching models
    if (state.engine && state.currentModelId !== modelId && state.engine.unload) {
      try { await state.engine.unload(); } catch {}
    }
    state.engine = await webllm.CreateMLCEngine(cfg, {
      stream: false,
      initProgressCallback: (report) => {
        const p = Math.max(0, Math.min(100, Math.round((report?.progress ?? 0) * 100)));
        const t = report?.text || 'Loading model…';
        dispatchProgress(p, t);
      },
    });
    state.currentModelId = modelId;
    dispatchProgress(100, 'Ready');
    dispatchReady(modelId);
    return { ok: true, modelId };
  } finally {
    state.initializing = false;
  }
}

async function ensureInit(modelId) {
  if (!state.engine || state.currentModelId !== modelId) {
    return init(modelId);
  }
  return { ok: true, modelId };
}

async function chat(messages) {
  if (!state.engine) throw new Error('WebLLM not initialized');
  // WebLLM provides OpenAI-compatible chat.completions API
  const result = await state.engine.chat.completions.create({
    messages,
    stream: false,
    // temperature etc. can be added here
  });
  const text = result?.choices?.[0]?.message?.content ?? '';
  return text;
}

// Streaming chat: yields deltas via callback and returns full text on completion
async function streamChat(messages, onDelta) {
  if (!state.engine) throw new Error('WebLLM not initialized');
  state.cancelRequested = false;
  const stream = await state.engine.chat.completions.create({
    messages,
    stream: true,
  });
  state.currentStream = stream;
  let full = '';
  // WebLLM stream is an async iterator of chunks
  try {
    for await (const chunk of stream) {
      if (state.cancelRequested) break;
      const delta = chunk?.choices?.[0]?.delta?.content ?? '';
      if (delta) {
        full += delta;
        try { onDelta && onDelta(delta, full, chunk); } catch {}
      }
    }
  } finally {
    state.currentStream = null;
  }
  return full;
}

async function cancel() {
  state.cancelRequested = true;
  try {
    if (state.engine && typeof state.engine.interruptGenerate === 'function') {
      await state.engine.interruptGenerate();
    }
  } catch {}
  try {
    const s = state.currentStream;
    if (s && typeof s.return === 'function') {
      await s.return();
    }
  } catch {}
}

window.WebLLMBridge = {
  init,
  ensureInit,
  chat,
  streamChat,
  cancel,
};

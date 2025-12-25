const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const modelEl = document.getElementById('model');
const personaEl = document.getElementById('persona');
const streamEl = document.getElementById('streamToggle');
const transportEl = document.getElementById('streamTransport');
const clearBtn = document.getElementById('clearChat');
const emptyState = document.getElementById('emptyState');
const sysEl = document.querySelector('.sys');
const multiToggleEl = document.getElementById('multiToggle');
const multiModelsEl = document.getElementById('multiModels');
const batchToggleBtn = document.getElementById('batchToggle');
const batchPanel = document.getElementById('batchPanel');
const closeBatchBtn = document.getElementById('closeBatch');
const batchPromptsEl = document.getElementById('batchPrompts');
const runBatchBtn = document.getElementById('runBatch');
const saveAllBatchBtn = document.getElementById('saveAllBatch');
const batchProgressEl = document.getElementById('batchProgress');
const exportBatchBtn = document.getElementById('exportBatch');
const statsToggleBtn = document.getElementById('statsToggle');
const statsPanel = document.getElementById('statsPanel');
const closeStatsBtn = document.getElementById('closeStats');
const statsContentEl = document.getElementById('statsContent');
const speakToggleEl = document.getElementById('speakToggle');
const micBtn = document.getElementById('micBtn');
const stopBtn = document.getElementById('stopBtn');
const webllmLoader = document.getElementById('webllmLoader');
const webllmLoaderBar = document.getElementById('webllmLoaderBar');
const webllmLoaderText = document.getElementById('webllmLoaderText');

let batchResults = [];

const personaPrompts = {
  friendly: 'You are a warm, concise guide. Keep answers short, friendly, and actionable.',
  coder: 'You are a focused coding assistant. Return concise answers, bullet steps, and minimal prose. Show code blocks when helpful.',
  coach: 'You are a productivity coach. Give clear next actions and keep replies under 4 sentences.',
  roleplay: 'You are the in-world narrator for the 3D game, describing scenes briefly and helping the player with direction.'
};

function getSystemPrompt() {
  const key = personaEl?.value || 'friendly';
  return personaPrompts[key] || personaPrompts.friendly;
}

let messages = [
  { role: 'system', content: getSystemPrompt() }
];

let isLoading = false;
let speakEnabled = false;
let recognition = null;
let micActive = false;
let isStreaming = false;

// Limit SSE payload size to avoid long URLs
const MAX_HISTORY = 12; // last 12 non-system messages
function getTrimmedMessages() {
  const sysIdx = messages.findIndex(m => m.role === 'system');
  const systemMsg = sysIdx >= 0 ? messages[sysIdx] : null;
  const dialog = messages.filter(m => m.role !== 'system');
  const trimmedDialog = dialog.slice(Math.max(0, dialog.length - MAX_HISTORY));
  return systemMsg ? [systemMsg, ...trimmedDialog] : trimmedDialog;
}

function hideEmptyState() {
  if (emptyState) emptyState.style.display = 'none';
}

function showEmptyState() {
  if (emptyState) emptyState.style.display = 'flex';
}

function formatContent(content) {
  // Simple markdown-like formatting for code blocks
  let formatted = content
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
  return formatted;
}

function addMessage(role, content) {
  hideEmptyState();
  const div = document.createElement('div');
  div.className = `msg ${role === 'user' ? 'user' : 'assistant'}`;
  if (role === 'assistant') {
    div.innerHTML = formatContent(content);
    if (speakEnabled && window.speechSynthesis) {
      const utter = new SpeechSynthesisUtterance(content);
      utter.rate = 1.0;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    }
  } else {
    div.textContent = content;
  }
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function showTypingIndicator() {
  hideEmptyState();
  const div = document.createElement('div');
  div.className = 'msg assistant typing';
  div.id = 'typingIndicator';
  div.innerHTML = '<span></span><span></span><span></span>';
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeTypingIndicator() {
  const indicator = document.getElementById('typingIndicator');
  if (indicator) indicator.remove();
}

function setLoading(loading) {
  isLoading = loading;
  sendBtn.disabled = loading;
  inputEl.disabled = loading;
}

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;
  
  inputEl.value = '';
  autoResize();

  messages.push({ role: 'user', content: text });
  addMessage('user', text);

  const model = modelEl.value;
  setLoading(true);
  showTypingIndicator();

  try {
    // Browser-only WebLLM path
    if (typeof window !== 'undefined' && window.WebLLMBridge && model.startsWith('web-llm:')) {
      const modelId = model.split(':', 2)[1];
      if (webllmLoader) {
        webllmLoader.style.display = 'block';
        if (webllmLoaderBar) webllmLoaderBar.style.width = '5%';
        if (webllmLoaderText) webllmLoaderText.textContent = 'Preparing WebGPU…';
      }
      await window.WebLLMBridge.ensureInit(modelId);
      const doStream = !!(streamEl && streamEl.checked);
      if (doStream && window.WebLLMBridge.streamChat) {
        isStreaming = true;
        if (stopBtn) stopBtn.style.display = 'inline-flex';
        removeTypingIndicator();
        const placeholder = addMessage('assistant', '');
        let full = '';
        await window.WebLLMBridge.streamChat(getTrimmedMessages(), (delta) => {
          full += delta || '';
          placeholder.textContent += delta || '';
          chatEl.scrollTop = chatEl.scrollHeight;
        });
        placeholder.innerHTML = formatContent(full);
        messages.push({ role: 'assistant', content: full });
        if (webllmLoader) webllmLoader.style.display = 'none';
        isStreaming = false;
        if (stopBtn) stopBtn.style.display = 'none';
        return;
      } else {
        const reply = await window.WebLLMBridge.chat(getTrimmedMessages());
        removeTypingIndicator();
        messages.push({ role: 'assistant', content: reply });
        addMessage('assistant', reply);
        if (webllmLoader) webllmLoader.style.display = 'none';
        return;
      }
    }

    if (!multiToggleEl?.checked && streamEl && streamEl.checked) {
      const transport = transportEl ? transportEl.value : 'fetch';
      if (transport === 'sse') {
        // SSE streaming via EventSource (GET with base64 payload)
        const json = JSON.stringify({ messages: getTrimmedMessages(), model });
        const payload = btoa(unescape(encodeURIComponent(json)));
        const url = `/api/chat-sse?payload=${encodeURIComponent(payload)}`;
        removeTypingIndicator();
        const placeholder = addMessage('assistant', '');
        let full = '';
        const es = new EventSource(url);
        es.onmessage = (e) => {
          const chunk = e.data || '';
          full += chunk;
          placeholder.textContent += chunk;
          chatEl.scrollTop = chatEl.scrollHeight;
        };
        es.addEventListener('done', () => {
          es.close();
          placeholder.innerHTML = formatContent(full);
          messages.push({ role: 'assistant', content: full });
          setLoading(false);
          inputEl.focus();
        });
        es.onerror = (e) => {
          console.error('SSE error:', e);
          es.close();
          addMessage('assistant', '⚠️ Streaming error');
          setLoading(false);
          inputEl.focus();
        };
        return; // Exit; loading cleared in SSE handlers
      }
      const res = await fetch('/api/chat-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages, model })
      });
      if (!res.ok || !res.body) {
        removeTypingIndicator();
        const text = await res.text();
        throw new Error(text || 'Streaming request failed');
      }
      removeTypingIndicator();
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = '';
      const placeholder = addMessage('assistant', '');
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        full += chunk;
        // Stream as plain text, then format at the end for markdown
        placeholder.textContent += chunk;
        chatEl.scrollTop = chatEl.scrollHeight;
      }
      // Replace placeholder with formatted content and persist in history
      placeholder.innerHTML = formatContent(full);
      messages.push({ role: 'assistant', content: full });
    } else {
      if (multiToggleEl?.checked) {
        // Multi-LLM call
        const models = (multiModelsEl?.value || '').split(',').map(s => s.trim()).filter(Boolean);
        const aggregatorEl = document.getElementById('aggregator');
        const aggregator = aggregatorEl ? aggregatorEl.value : 'length';
        const res = await fetch('/api/multi-chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages, models, aggregator })
        });
        removeTypingIndicator();
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || 'Multi-chat failed');
        }
        renderMultiResults(text, data);
      } else {
        const res = await fetch('/api/agent-chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            persona: personaEl?.value || 'friendly',
            history: messages.filter(m => m.role !== 'system'),
            input: text
          })
        });

        removeTypingIndicator();

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || 'Request failed');
        }
        const reply = data.text || '[No response]';
        messages.push({ role: 'assistant', content: reply });
        addMessage('assistant', reply);
      }
    }
  } catch (e) {
    removeTypingIndicator();
    addMessage('assistant', `⚠️ Error: ${e.message || 'Could not contact server'}`);
    console.error(e);
  } finally {
    setLoading(false);
    inputEl.focus();
    if (webllmLoader) webllmLoader.style.display = 'none';
  }
}

function clearChat() {
  messages = [{ role: 'system', content: getSystemPrompt() }];
  chatEl.innerHTML = '';
  showEmptyState();
  chatEl.appendChild(emptyState);
  inputEl.focus();
}

function applyPersona() {
  messages[0] = { role: 'system', content: getSystemPrompt() };
  updateSysHint();
}

sendBtn.addEventListener('click', sendMessage);

clearBtn.addEventListener('click', clearChat);

inputEl.addEventListener('input', autoResize);

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendMessage();
  }
});

personaEl?.addEventListener('change', applyPersona);

// Update system hint based on selected model (local vs cloud)
function updateSysHint() {
  const val = modelEl.value.trim();
  if (!sysEl) return;
  if (val.startsWith('web-llm:')) {
    sysEl.textContent = 'Using browser WebGPU (WebLLM). No server or API key needed.';
  } else if (val.startsWith('gpt-')) {
    sysEl.textContent = 'Using OpenAI cloud. Set OPENAI_API_KEY in .env';
  } else {
    sysEl.innerHTML = 'Using local Ollama. Pull models with <code>ollama pull modelname</code>';
  }
}

modelEl.addEventListener('change', updateSysHint);
updateSysHint();

// Disable browser-only options if WebGPU is unsupported, and revert on selection
function gateWebLLMOptions() {
  const supported = !!navigator.gpu;
  const opts = Array.from(modelEl?.options || []);
  for (const o of opts) {
    if (typeof o.value === 'string' && o.value.startsWith('web-llm:')) {
      o.disabled = !supported;
      if (!supported) {
        // Append unsupported hint once
        if (!/unsupported\)$/i.test(o.textContent)) {
          o.textContent = o.textContent.replace(/\s*\(browser.*\)$/i, '') + ' (browser - unsupported)';
        }
      }
    }
  }
}

gateWebLLMOptions();

// Restore saved preferences (model, stream) and update hint
try {
  const savedModel = localStorage.getItem('chat:model');
  if (savedModel) {
    const has = Array.from(modelEl.options || []).some(o => o.value === savedModel && !o.disabled);
    if (has) modelEl.value = savedModel;
  }
  const savedStream = localStorage.getItem('chat:stream');
  if (savedStream !== null && streamEl) streamEl.checked = savedStream === '1';
  updateSysHint();
} catch {}

modelEl.addEventListener('change', () => {
  const val = modelEl.value || '';
  if (val.startsWith('web-llm:') && !navigator.gpu) {
    alert('WebGPU is not supported in this browser/device. Please select a local/cloud model or use a WebGPU-capable browser.');
    const fallback = Array.from(modelEl.options).find(o => !o.value.startsWith('web-llm:'));
    if (fallback) modelEl.value = fallback.value;
    updateSysHint();
  }
  try { localStorage.setItem('chat:model', modelEl.value); } catch {}
});

streamEl?.addEventListener('change', () => {
  try { localStorage.setItem('chat:stream', streamEl.checked ? '1' : '0'); } catch {}
});

if (speakToggleEl) {
  speakToggleEl.addEventListener('change', () => {
    speakEnabled = !!speakToggleEl.checked;
    if (!speakEnabled && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  });
}

function initSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const rec = new SR();
  rec.continuous = false;
  rec.interimResults = false;
  rec.lang = 'en-US';
  rec.onresult = (event) => {
    const transcript = event.results?.[0]?.[0]?.transcript;
    if (transcript) {
      inputEl.value = transcript;
      autoResize();
      sendMessage();
    }
  };
  rec.onend = () => {
    micActive = false;
    if (micBtn) micBtn.textContent = '🎙️';
  };
  rec.onerror = () => {
    micActive = false;
    if (micBtn) micBtn.textContent = '🎙️';
  };
  return rec;
}

micBtn?.addEventListener('click', () => {
  if (!recognition) recognition = initSpeechRecognition();
  if (!recognition) {
    alert('Speech recognition not supported in this browser.');
    return;
  }
  if (micActive) {
    recognition.stop();
    return;
  }
  micActive = true;
  micBtn.textContent = '🛑';
  recognition.start();
});

// Stop streaming handler (browser WebLLM)
stopBtn?.addEventListener('click', async () => {
  if (!isStreaming) return;
  try { await window.WebLLMBridge?.cancel?.(); } catch {}
  isStreaming = false;
  if (stopBtn) stopBtn.style.display = 'none';
});

// Focus input on page load
inputEl.focus();

// Batch mode handlers
batchToggleBtn?.addEventListener('click', () => {
  const isVisible = batchPanel.style.display !== 'none';
  batchPanel.style.display = isVisible ? 'none' : 'block';
  if (!isVisible) {
    batchPromptsEl.focus();
  }
});

closeBatchBtn?.addEventListener('click', () => {
  batchPanel.style.display = 'none';
});

runBatchBtn?.addEventListener('click', async () => {
  const prompts = (batchPromptsEl.value || '').split('\n').map(s => s.trim()).filter(Boolean);
  if (prompts.length === 0) {
    batchProgressEl.textContent = 'No prompts to process';
    return;
  }

  const models = (multiModelsEl?.value || '').split(',').map(s => s.trim()).filter(Boolean);
  const aggregatorEl = document.getElementById('aggregator');
  const aggregator = aggregatorEl ? aggregatorEl.value : 'length';

  batchResults = [];
  runBatchBtn.disabled = true;
  saveAllBatchBtn.disabled = true;

  for (let i = 0; i < prompts.length; i++) {
    const prompt = prompts[i];
    batchProgressEl.textContent = `Processing ${i + 1} / ${prompts.length}...`;

    try {
      const messages = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: prompt }
      ];
      const res = await fetch('/api/multi-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages, models, aggregator })
      });
      const data = await res.json();
      if (res.ok) {
        batchResults.push({ prompt, data });
        // Add to chat display
        addMessage('user', prompt);
        renderMultiResults(prompt, data);
      } else {
        batchProgressEl.textContent += ` Error on prompt ${i + 1}: ${data.error || 'unknown'}`;
      }
    } catch (e) {
      batchProgressEl.textContent += ` Error on prompt ${i + 1}: ${e.message}`;
    }
  }

  batchProgressEl.textContent = `Completed ${prompts.length} prompts`;
  runBatchBtn.disabled = false;
  saveAllBatchBtn.disabled = false;
});

saveAllBatchBtn?.addEventListener('click', async () => {
  if (batchResults.length === 0) {
    batchProgressEl.textContent = 'No results to save';
    return;
  }

  saveAllBatchBtn.disabled = true;
  let saved = 0;
  const aggregatorEl = document.getElementById('aggregator');
  const aggregator = aggregatorEl ? aggregatorEl.value : 'unknown';

  for (const item of batchResults) {
    try {
      const { prompt, data } = item;
      const res = await fetch('/api/collab', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          results: data.results || [],
          best: data.best || '',
          aggregator
        })
      });
      if (res.ok) saved++;
    } catch (e) {
      // continue
    }
  }

  batchProgressEl.textContent = `Saved ${saved} / ${batchResults.length} to dataset`;
  saveAllBatchBtn.disabled = false;
});

exportBatchBtn?.addEventListener('click', () => {
  if (batchResults.length === 0) {
    batchProgressEl.textContent = 'No results to export';
    return;
  }

  const aggregatorEl = document.getElementById('aggregator');
  const aggregator = aggregatorEl ? aggregatorEl.value : 'unknown';

  // Generate CSV
  const headers = ['Prompt', 'Best', 'Aggregator', 'Models', 'TotalMs'];
  const rows = batchResults.map(item => {
    const { prompt, data } = item;
    const best = (data.best || '').replace(/"/g, '""');
    const models = (data.models || []).join('|');
    return `"${prompt}","${best}","${aggregator}","${models}","${data.totalMs || 0}"`;
  });

  const csv = [headers.join(','), ...rows].join('\\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `batch_results_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);

  batchProgressEl.textContent = `Exported ${batchResults.length} results to CSV`;
});

statsToggleBtn?.addEventListener('click', async () => {
  const isVisible = statsPanel.style.display !== 'none';
  statsPanel.style.display = isVisible ? 'none' : 'block';
  
  if (!isVisible) {
    try {
      const res = await fetch('/api/stats');
      const stats = await res.json();
      
      let html = `<div style="margin-bottom:16px;"><strong>Dataset Records:</strong> ${stats.count}</div>`;
      
      html += '<div style="margin-bottom:16px;"><strong>Aggregators:</strong><ul style="margin:4px 0; padding-left:20px;">';
      for (const [agg, count] of Object.entries(stats.aggregators || {})) {
        html += `<li>${agg}: ${count}</li>`;
      }
      html += '</ul></div>';
      
      html += '<div style="margin-bottom:16px;"><strong>Model Performance:</strong><table style="width:100%; border-collapse:collapse; margin-top:8px;">';
      html += '<tr style="text-align:left; border-bottom:1px solid var(--border);"><th style="padding:4px;">Model</th><th style="padding:4px;">Queries</th><th style="padding:4px;">Avg Time</th></tr>';
      for (const [model, data] of Object.entries(stats.models || {})) {
        html += `<tr style="border-bottom:1px solid #374151;"><td style="padding:4px;">${model}</td><td style="padding:4px;">${data.count}</td><td style="padding:4px;">${data.avgMs}ms</td></tr>`;
      }
      html += '</table></div>';
      
      html += `<div style="color:#9ca3af; font-size:12px;">Estimated tokens: ~${stats.estimatedTokens?.toLocaleString() || 0}</div>`;
      
      statsContentEl.innerHTML = html;
    } catch (e) {
      statsContentEl.innerHTML = `<div style="color:#ef4444;">Error loading stats: ${e.message}</div>`;
    }
  }
});

closeStatsBtn?.addEventListener('click', () => {
  statsPanel.style.display = 'none';
});

function renderMultiResults(promptText, payload) {
  const { results = [], best = '' } = payload || {};
  // Render aggregated best first
  const bestDiv = document.createElement('div');
  bestDiv.className = 'msg assistant';
  const bestHeader = document.createElement('div');
  bestHeader.style.fontSize = '12px';
  bestHeader.style.color = '#93c5fd';
  bestHeader.textContent = 'Aggregated (best)';
  bestDiv.appendChild(bestHeader);
  const bestBody = document.createElement('div');
  bestBody.innerHTML = formatContent(best || '[No best response]');
  bestDiv.appendChild(bestBody);
  chatEl.appendChild(bestDiv);
  chatEl.scrollTop = chatEl.scrollHeight;

  // Render each model card
  for (const r of results) {
    const card = document.createElement('div');
    card.className = 'msg assistant';
    const meta = document.createElement('div');
    meta.style.fontSize = '12px';
    meta.style.color = '#9ca3af';
    meta.textContent = `${r.model} • ${r.provider} • ${r.ms}ms${r.ok ? '' : ' • error'}`;
    card.appendChild(meta);
    const body = document.createElement('div');
    body.innerHTML = formatContent(r.text || '[No response]');
    card.appendChild(body);
    chatEl.appendChild(card);
  }
  chatEl.scrollTop = chatEl.scrollHeight;

  // Add actions: use best as assistant and save to collab
  const actions = document.createElement('div');
  actions.style.display = 'flex';
  actions.style.gap = '8px';
  actions.style.margin = '8px 0 0 0';
  const useBestBtn = document.createElement('button');
  useBestBtn.textContent = 'Use Best';
  useBestBtn.style.background = '#3b82f6';
  useBestBtn.style.color = 'white';
  useBestBtn.style.border = 'none';
  useBestBtn.style.padding = '6px 10px';
  useBestBtn.style.borderRadius = '6px';
  useBestBtn.addEventListener('click', () => {
    const reply = best || '[No best response]';
    messages.push({ role: 'assistant', content: reply });
    addMessage('assistant', reply);
  });

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save to Dataset';
  saveBtn.style.background = 'transparent';
  saveBtn.style.color = '#93c5fd';
  saveBtn.style.border = '1px solid #93c5fd';
  saveBtn.style.padding = '6px 10px';
  saveBtn.style.borderRadius = '6px';
  saveBtn.addEventListener('click', async () => {
    try {
      const aggregatorEl = document.getElementById('aggregator');
      const aggregator = aggregatorEl ? aggregatorEl.value : 'unknown';
      const res = await fetch('/api/collab', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText, results, best, aggregator })
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || 'Save failed');
      const note = document.createElement('div');
      note.className = 'msg assistant';
      note.textContent = 'Saved to collab.jsonl';
      chatEl.appendChild(note);
      chatEl.scrollTop = chatEl.scrollHeight;
    } catch (e) {
      const note = document.createElement('div');
      note.className = 'msg assistant';
      note.textContent = `⚠️ Save error: ${e.message}`;
      chatEl.appendChild(note);
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  });

  actions.appendChild(useBestBtn);
  actions.appendChild(saveBtn);
  chatEl.appendChild(actions);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// WebLLM progress -> update loader UI
window.addEventListener('webllm-progress', (e) => {
  const p = Math.max(0, Math.min(100, Number(e?.detail?.progress ?? 0)));
  const t = e?.detail?.text || 'Loading…';
  if (webllmLoader) webllmLoader.style.display = 'block';
  if (webllmLoaderBar) webllmLoaderBar.style.width = `${p}%`;
  if (webllmLoaderText) webllmLoaderText.textContent = t;
});

window.addEventListener('webllm-ready', () => {
  if (webllmLoaderBar) webllmLoaderBar.style.width = '100%';
  setTimeout(() => { if (webllmLoader) webllmLoader.style.display = 'none'; }, 400);
});

#!/usr/bin/env node

import readline from 'readline';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';

const execPromise = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
};

const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  warn: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
  header: (msg) => console.log(`\n${colors.bright}${colors.cyan}${msg}${colors.reset}`),
};

// Readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const question = (prompt) =>
  new Promise((resolve) => rl.question(`${colors.bright}${prompt}${colors.reset} `, resolve));

const pause = (msg = 'Press Enter to continue...') =>
  new Promise((resolve) => rl.question(`${colors.yellow}${msg}${colors.reset} `, resolve));

// ============================================================================
// MENU FUNCTIONS
// ============================================================================

async function mainMenu() {
  clear();
  log.header('🚀 LLM Chat Application - Interactive CLI');
  console.log(`
${colors.cyan}Main Menu:${colors.reset}
  1. Setup & Configuration
  2. Server Management
  3. API Testing
  4. Model Management
  5. Deployment
  6. Documentation
  0. Exit
  `);
  
  const choice = await question('Select option:');
  
  switch (choice.trim()) {
    case '1': return setupMenu();
    case '2': return serverMenu();
    case '3': return testingMenu();
    case '4': return modelMenu();
    case '5': return deploymentMenu();
    case '6': return docsMenu();
    case '0': return exit();
    default: 
      log.error('Invalid choice');
      await pause();
      return mainMenu();
  }
}

async function setupMenu() {
  clear();
  log.header('Setup & Configuration');
  console.log(`
${colors.cyan}Setup Options:${colors.reset}
  1. Run environment validator
  2. Configure .env file
  3. Install dependencies
  4. Check Node.js version
  5. Back to main menu
  `);
  
  const choice = await question('Select option:');
  
  switch (choice.trim()) {
    case '1': return await validateEnvironment();
    case '2': return await configureEnv();
    case '3': return await installDeps();
    case '4': return await checkNodeVersion();
    case '5': return mainMenu();
    default:
      log.error('Invalid choice');
      await pause();
      return setupMenu();
  }
}

async function serverMenu() {
  clear();
  log.header('Server Management');
  console.log(`
${colors.cyan}Server Options:${colors.reset}
  1. Start server
  2. Stop server
  3. Restart server
  4. Check server status
  5. View server logs
  6. Back to main menu
  `);
  
  const choice = await question('Select option:');
  
  switch (choice.trim()) {
    case '1': return await startServer();
    case '2': return await stopServer();
    case '3': return await restartServer();
    case '4': return await checkStatus();
    case '5': return await viewLogs();
    case '6': return mainMenu();
    default:
      log.error('Invalid choice');
      await pause();
      return serverMenu();
  }
}

async function testingMenu() {
  clear();
  log.header('API Testing');
  console.log(`
${colors.cyan}Testing Options:${colors.reset}
  1. Test /health endpoint
  2. Test /api/chat (basic)
  3. Test /api/chat-stream
  4. Test /api/chat-sse
  5. Interactive chat
  6. Back to main menu
  `);
  
  const choice = await question('Select option:');
  
  switch (choice.trim()) {
    case '1': return await testHealth();
    case '2': return await testBasicChat();
    case '3': return await testFetchStream();
    case '4': return await testSSE();
    case '5': return await interactiveChat();
    case '6': return mainMenu();
    default:
      log.error('Invalid choice');
      await pause();
      return testingMenu();
  }
}

async function modelMenu() {
  clear();
  log.header('Model Management');
  console.log(`
${colors.cyan}Model Options:${colors.reset}
  1. List available Ollama models
  2. Pull new Ollama model
  3. Remove Ollama model
  4. Test model availability
  5. Back to main menu
  `);
  
  const choice = await question('Select option:');
  
  switch (choice.trim()) {
    case '1': return await listModels();
    case '2': return await pullModel();
    case '3': return await removeModel();
    case '4': return await testModelAvailability();
    case '5': return mainMenu();
    default:
      log.error('Invalid choice');
      await pause();
      return modelMenu();
  }
}

async function deploymentMenu() {
  clear();
  log.header('Deployment');
  console.log(`
${colors.cyan}Deployment Options:${colors.reset}
  1. Docker setup & deployment
  2. Production checklist
  3. Security review
  4. Performance optimization
  5. Back to main menu
  `);
  
  const choice = await question('Select option:');
  
  switch (choice.trim()) {
    case '1': return await dockerSetup();
    case '2': return await productionChecklist();
    case '3': return await securityReview();
    case '4': return await perfOptimization();
    case '5': return mainMenu();
    default:
      log.error('Invalid choice');
      await pause();
      return deploymentMenu();
  }
}

async function docsMenu() {
  clear();
  log.header('Documentation');
  console.log(`
${colors.cyan}Documentation Options:${colors.reset}
  1. Quick start guide
  2. API reference
  3. Configuration guide
  4. Troubleshooting
  5. View project structure
  6. Back to main menu
  `);
  
  const choice = await question('Select option:');
  
  switch (choice.trim()) {
    case '1': return await quickStart();
    case '2': return await apiRef();
    case '3': return await configGuide();
    case '4': return await troubleshoot();
    case '5': return await projectStructure();
    case '6': return mainMenu();
    default:
      log.error('Invalid choice');
      await pause();
      return docsMenu();
  }
}

// ============================================================================
// SETUP FUNCTIONS
// ============================================================================

async function validateEnvironment() {
  clear();
  log.header('Environment Validation');
  
  let passed = 0;
  let warnings = 0;
  let failed = 0;
  
  // Check Node.js
  try {
    const { stdout } = await execPromise('node --version');
    const version = stdout.trim();
    log.success(`Node.js ${version}`);
    passed++;
  } catch {
    log.error('Node.js not installed');
    failed++;
  }
  
  // Check npm
  try {
    const { stdout } = await execPromise('npm --version');
    const version = stdout.trim();
    log.success(`npm ${version}`);
    passed++;
  } catch {
    log.error('npm not installed');
    failed++;
  }
  
  // Check project files
  const required = ['server.js', 'package.json', 'public/index.html'];
  for (const file of required) {
    const filePath = path.join(__dirname, file);
    if (fs.existsSync(filePath)) {
      log.success(`Found: ${file}`);
      passed++;
    } else {
      log.error(`Missing: ${file}`);
      failed++;
    }
  }
  
  // Check dependencies
  const nodeModules = path.join(__dirname, 'node_modules');
  if (fs.existsSync(nodeModules)) {
    log.success('node_modules installed');
    passed++;
  } else {
    log.warn('node_modules not found (run "npm install")');
    warnings++;
  }
  
  // Check .env
  const envPath = path.join(__dirname, '.env');
  if (fs.existsSync(envPath)) {
    log.success('.env file found');
    passed++;
  } else {
    log.warn('.env not configured (using defaults)');
    warnings++;
  }
  
  // Check Ollama
  try {
    const response = await fetch('http://localhost:11434/api/tags', { timeout: 5000 });
    if (response.ok) {
      log.success('Ollama is running');
      passed++;
    } else {
      log.warn('Ollama not responding (expected if not installed)');
      warnings++;
    }
  } catch {
    log.warn('Ollama not detected (install from https://ollama.ai)');
    warnings++;
  }
  
  console.log(`\n${colors.bright}Summary:${colors.reset}`);
  console.log(`${colors.green}Passed: ${passed}${colors.reset}`);
  console.log(`${colors.yellow}Warnings: ${warnings}${colors.reset}`);
  console.log(`${colors.red}Failed: ${failed}${colors.reset}`);
  
  await pause();
  return setupMenu();
}

async function configureEnv() {
  clear();
  log.header('Configure .env File');
  
  const envPath = path.join(__dirname, '.env');
  const defaultContent = `# OpenAI API Configuration (optional)
OPENAI_API_KEY=your_api_key_here

# Server Configuration
PORT=3000

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
`;
  
  if (!fs.existsSync(envPath)) {
    fs.writeFileSync(envPath, defaultContent);
    log.success(`.env created at ${envPath}`);
  } else {
    log.info('.env already exists');
  }
  
  console.log(`\nEditable with any text editor. Path: ${colors.cyan}${envPath}${colors.reset}`);
  console.log(`
Key variables:
${colors.cyan}OPENAI_API_KEY${colors.reset} - For cloud models (gpt-4o, gpt-4o-mini)
${colors.cyan}PORT${colors.reset} - Server port (default 3000)
${colors.cyan}OLLAMA_URL${colors.reset} - Ollama endpoint (default http://localhost:11434)
  `);
  
  await pause();
  return setupMenu();
}

async function installDeps() {
  clear();
  log.header('Installing Dependencies');
  
  try {
    log.info('Running npm install...');
    const { stdout } = await execPromise('npm install', { cwd: __dirname });
    log.success('Dependencies installed successfully');
    console.log(stdout);
  } catch (err) {
    log.error(`Installation failed: ${err.message}`);
  }
  
  await pause();
  return setupMenu();
}

async function checkNodeVersion() {
  clear();
  log.header('Node.js Version Check');
  
  try {
    const { stdout } = await execPromise('node --version');
    console.log(`${colors.cyan}Node.js version:${colors.reset} ${stdout.trim()}`);
    
    const { stdout: npmVersion } = await execPromise('npm --version');
    console.log(`${colors.cyan}npm version:${colors.reset} ${npmVersion.trim()}`);
  } catch (err) {
    log.error(`Failed to check versions: ${err.message}`);
  }
  
  await pause();
  return setupMenu();
}

// ============================================================================
// SERVER FUNCTIONS
// ============================================================================

async function startServer() {
  clear();
  log.header('Start Server');
  
  try {
    const response = await fetch('http://localhost:3000/health');
    if (response.ok) {
      log.warn('Server already running on port 3000');
      await pause();
      return serverMenu();
    }
  } catch {
    // Server not running, proceed with start
  }
  
  log.info('Starting server...');
  log.info('Server will run in background. Use "npm run dev" or "node server.js" in terminal.');
  console.log(`
${colors.yellow}To start manually:${colors.reset}
  npm run dev
  
${colors.yellow}Or in a separate terminal:${colors.reset}
  node server.js
  
Server will be available at: ${colors.cyan}http://localhost:3000${colors.reset}
  `);
  
  await pause();
  return serverMenu();
}

async function stopServer() {
  clear();
  log.header('Stop Server');
  
  console.log(`
${colors.yellow}To stop the server:${colors.reset}
  1. Go to the terminal running the server
  2. Press ${colors.bright}Ctrl+C${colors.reset}
  
Or run: ${colors.cyan}pkill -f "node server.js"${colors.reset} (Unix/macOS)
Or run: ${colors.cyan}taskkill /IM node.exe /F${colors.reset} (Windows PowerShell as Admin)
  `);
  
  await pause();
  return serverMenu();
}

async function restartServer() {
  clear();
  log.header('Restart Server');
  
  console.log(`
${colors.yellow}To restart the server:${colors.reset}
  1. Stop it: Press ${colors.bright}Ctrl+C${colors.reset} in server terminal
  2. Start it again: ${colors.cyan}npm run dev${colors.reset}
  `);
  
  await pause();
  return serverMenu();
}

async function checkStatus() {
  clear();
  log.header('Server Status Check');
  
  try {
    const response = await fetch('http://localhost:3000/health');
    const data = await response.json();
    log.success('Server is running');
    console.log(`Status: ${colors.green}${JSON.stringify(data, null, 2)}${colors.reset}`);
  } catch (err) {
    log.error('Server is not running');
    console.log(`Error: ${colors.red}${err.message}${colors.reset}`);
    console.log(`\nStart the server with: ${colors.cyan}npm run dev${colors.reset}`);
  }
  
  await pause();
  return serverMenu();
}

async function viewLogs() {
  clear();
  log.header('Server Logs');
  
  console.log(`
${colors.yellow}To view server logs:${colors.reset}
  1. Server logs appear in the terminal where you ran ${colors.cyan}npm run dev${colors.reset}
  2. Logs show all incoming requests and responses
  3. Press ${colors.bright}Ctrl+C${colors.reset} in server terminal to stop
  `);
  
  await pause();
  return serverMenu();
}

// ============================================================================
// TESTING FUNCTIONS
// ============================================================================

async function testHealth() {
  clear();
  log.header('Health Endpoint Test');
  
  try {
    const response = await fetch('http://localhost:3000/health');
    const data = await response.json();
    log.success('Health check passed');
    console.log(`${colors.cyan}Response:${colors.reset}`);
    console.log(JSON.stringify(data, null, 2));
  } catch (err) {
    log.error(`Health check failed: ${err.message}`);
    console.log(`\n${colors.yellow}Make sure server is running:${colors.reset} ${colors.cyan}npm run dev${colors.reset}`);
  }
  
  await pause();
  return testingMenu();
}

async function testBasicChat() {
  clear();
  log.header('Basic Chat Test');
  
  const model = await question('Model (default: gpt-oss-20):');
  const msg = await question('Message:');
  
  try {
    log.info('Sending request...');
    const response = await fetch('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model || 'gpt-oss-20',
        messages: [{ role: 'user', content: msg }],
      }),
    });
    
    const data = await response.json();
    if (data.error) {
      log.error(`Error: ${data.error}`);
    } else {
      log.success('Response received');
      console.log(`${colors.cyan}${data.text}${colors.reset}`);
    }
  } catch (err) {
    log.error(`Test failed: ${err.message}`);
  }
  
  await pause();
  return testingMenu();
}

async function testFetchStream() {
  clear();
  log.header('Fetch Stream Test');
  
  console.log(`${colors.yellow}Testing streaming endpoint...${colors.reset}`);
  
  try {
    const response = await fetch('http://localhost:3000/api/chat-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gpt-oss-20',
        messages: [{ role: 'user', content: 'Say hello briefly' }],
      }),
    });
    
    let fullText = '';
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;
      process.stdout.write(chunk);
    }
    
    log.success('\nStream test completed');
  } catch (err) {
    log.error(`Stream test failed: ${err.message}`);
  }
  
  await pause();
  return testingMenu();
}

async function testSSE() {
  clear();
  log.header('SSE Stream Test');
  
  console.log(`
${colors.cyan}SSE (EventSource) Example:${colors.reset}

In your frontend:
${colors.yellow}javascript
const payload = Buffer.from(JSON.stringify({
  model: 'gpt-oss-20',
  messages: [{role: 'user', content: 'Hello'}]
})).toString('base64');

const es = new EventSource(\`/api/chat-sse?payload=\${payload}\`);
es.addEventListener('delta', (e) => {
  const data = JSON.parse(e.data);
  console.log(data.delta);
});
es.addEventListener('done', () => es.close());
  ${colors.reset}`);
  
  await pause();
  return testingMenu();
}

async function interactiveChat() {
  clear();
  log.header('Interactive Chat');
  
  const model = await question('Model (default: gpt-oss-20):');
  const messages = [];
  
  console.log(`\n${colors.cyan}Chat started. Type "exit" to quit.${colors.reset}\n`);
  
  while (true) {
    const userMsg = await question('You: ');
    if (userMsg.toLowerCase() === 'exit') break;
    
    messages.push({ role: 'user', content: userMsg });
    
    try {
      const response = await fetch('http://localhost:3000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model || 'gpt-oss-20',
          messages,
        }),
      });
      
      const data = await response.json();
      if (data.error) {
        console.log(`${colors.red}Error: ${data.error}${colors.reset}`);
      } else {
        console.log(`\n${colors.cyan}Assistant: ${data.text}${colors.reset}\n`);
        messages.push({ role: 'assistant', content: data.text });
      }
    } catch (err) {
      console.log(`${colors.red}Connection error: ${err.message}${colors.reset}`);
    }
  }
  
  await pause();
  return testingMenu();
}

// ============================================================================
// MODEL FUNCTIONS
// ============================================================================

async function listModels() {
  clear();
  log.header('Available Ollama Models');
  
  try {
    const response = await fetch('http://localhost:11434/api/tags');
    const data = await response.json();
    
    if (data.models && data.models.length > 0) {
      log.success(`Found ${data.models.length} model(s)`);
      console.log(`${colors.cyan}`);
      data.models.forEach((m) => {
        console.log(`  • ${m.name} (${m.size} bytes)`);
      });
      console.log(colors.reset);
    } else {
      log.warn('No Ollama models installed');
      console.log(`\nPull models with: ${colors.cyan}ollama pull <model>${colors.reset}`);
    }
  } catch (err) {
    log.error(`Ollama not running: ${err.message}`);
    console.log(`\nStart Ollama: ${colors.cyan}ollama serve${colors.reset}`);
  }
  
  await pause();
  return modelMenu();
}

async function pullModel() {
  clear();
  log.header('Pull Ollama Model');
  
  const model = await question('Model name (e.g., llama3.2, qwen2.5):');
  
  console.log(`\n${colors.yellow}Run this in another terminal:${colors.reset}`);
  console.log(`${colors.cyan}ollama pull ${model}${colors.reset}`);
  
  await pause();
  return modelMenu();
}

async function removeModel() {
  clear();
  log.header('Remove Ollama Model');
  
  const model = await question('Model name to remove:');
  
  console.log(`\n${colors.yellow}Run this in another terminal:${colors.reset}`);
  console.log(`${colors.cyan}ollama rm ${model}${colors.reset}`);
  
  await pause();
  return modelMenu();
}

async function testModelAvailability() {
  clear();
  log.header('Test Model Availability');
  
  const models = ['gpt-oss-20', 'llama3.2', 'qwen2.5'];
  
  try {
    const response = await fetch('http://localhost:11434/api/tags');
    const data = await response.json();
    const installed = data.models ? data.models.map((m) => m.name) : [];
    
    console.log(`${colors.cyan}Model Status:${colors.reset}\n`);
    models.forEach((model) => {
      if (installed.includes(model)) {
        log.success(`${model} is installed`);
      } else {
        log.warn(`${model} not installed`);
      }
    });
  } catch {
    log.error('Cannot reach Ollama. Start it with: ollama serve');
  }
  
  await pause();
  return modelMenu();
}

// ============================================================================
// DEPLOYMENT FUNCTIONS
// ============================================================================

async function dockerSetup() {
  clear();
  log.header('Docker Deployment');
  
  console.log(`
${colors.cyan}Docker Files:${colors.reset}
  • Dockerfile - Container image definition
  • docker-compose.yml - Multi-container orchestration

${colors.cyan}Quick Start:${colors.reset}
  ${colors.bright}1. Build and start:${colors.reset}
     docker-compose up -d

  ${colors.bright}2. View logs:${colors.reset}
     docker-compose logs -f

  ${colors.bright}3. Stop:${colors.reset}
     docker-compose down

${colors.yellow}Prerequisites:${colors.reset}
  • Docker installed (https://docker.com/products/docker-desktop)
  • docker-compose available
  `);
  
  await pause();
  return deploymentMenu();
}

async function productionChecklist() {
  clear();
  log.header('Production Deployment Checklist');
  
  const checks = [
    '✓ Node.js v18+ installed',
    '✓ All dependencies installed (npm install)',
    '✓ .env configured with OPENAI_API_KEY (if using cloud models)',
    '✓ Server port not blocked by firewall',
    '✓ Ollama running (ollama serve)',
    '✓ Models pulled (ollama pull gpt-oss-20)',
    '✓ CORS origin configured for production domain',
    '✓ Rate limiting enabled (30 req/min)',
    '✓ Error handling tested',
    '✓ Load testing performed',
  ];
  
  console.log(`${colors.cyan}Pre-Deployment Checklist:${colors.reset}\n`);
  checks.forEach((check) => console.log(`  ${check}`));
  
  console.log(`\n${colors.cyan}Deployment Steps:${colors.reset}
  1. Run: ${colors.yellow}npm install${colors.reset}
  2. Set: ${colors.yellow}.env variables${colors.reset}
  3. Test: ${colors.yellow}npm run dev${colors.reset}
  4. Deploy: ${colors.yellow}docker-compose up -d${colors.reset}
  5. Monitor: ${colors.yellow}docker-compose logs -f${colors.reset}
  `);
  
  await pause();
  return deploymentMenu();
}

async function securityReview() {
  clear();
  log.header('Security Review');
  
  console.log(`
${colors.cyan}Security Checklist:${colors.reset}

${colors.bright}1. API Keys${colors.reset}
   ✓ OPENAI_API_KEY stored in .env (not in code)
   ✓ .env added to .gitignore
   ✓ No secrets in server.js or client code

${colors.bright}2. Authentication${colors.reset}
   → Implement JWT or OAuth for production
   → Add rate limiting per user (current: per IP)

${colors.bright}3. CORS${colors.reset}
   ✓ Configured but wildcard origin (*) for testing
   → Restrict to specific domain in production: 'https://yourdomain.com'

${colors.bright}4. Input Validation${colors.reset}
   ✓ Message array validation
   ✓ Model name sanitization
   ✓ 10MB JSON limit

${colors.bright}5. Rate Limiting${colors.reset}
   ✓ 30 requests/minute per IP
   → Consider stricter limits for production

${colors.bright}6. Monitoring${colors.reset}
   ✓ Morgan logging enabled
   → Add error tracking (Sentry, DataDog, etc.)
   → Setup health check monitoring
  `);
  
  await pause();
  return deploymentMenu();
}

async function perfOptimization() {
  clear();
  log.header('Performance Optimization');
  
  console.log(`
${colors.cyan}Current Optimizations:${colors.reset}
  ✓ Message history capping (12 messages max)
  ✓ Streaming for large responses
  ✓ Middleware pipeline ordering
  ✓ Efficient JSON parsing

${colors.cyan}Recommended Improvements:${colors.reset}

${colors.bright}1. Response Caching${colors.reset}
   Add Redis caching for common queries
   Example: Cache responses for 1-5 minutes

${colors.bright}2. Connection Pooling${colors.reset}
   Use connection pools for Ollama/OpenAI
   Current: Single connection per request

${colors.bright}3. Load Balancing${colors.reset}
   Deploy multiple instances behind nginx
   Use docker-compose scaling

${colors.bright}4. Database${colors.reset}
   Persist chat history (PostgreSQL, MongoDB)
   Improve retrieval performance

${colors.bright}5. CDN${colors.reset}
   Serve frontend assets via CDN
   Reduce server load

${colors.cyan}Quick Wins:${colors.reset}
   • Enable gzip compression: app.use(compression())
   • Set cache headers on static assets
   • Optimize database queries
   • Profile with: node --prof server.js
  `);
  
  await pause();
  return deploymentMenu();
}

// ============================================================================
// DOCUMENTATION FUNCTIONS
// ============================================================================

async function quickStart() {
  clear();
  log.header('Quick Start Guide');
  
  console.log(`
${colors.cyan}1. Install Dependencies${colors.reset}
   npm install

${colors.cyan}2. Start Ollama (in separate terminal)${colors.reset}
   ollama pull gpt-oss-20  # First time only
   ollama serve

${colors.cyan}3. Start Server${colors.reset}
   npm run dev
   Open: http://localhost:3000

${colors.cyan}4. Chat!${colors.reset}
   • Select model from dropdown
   • Type your message
   • Choose streaming transport (Fetch/SSE)
   • Click Send

${colors.cyan}For Cloud Models (GPT-4o):${colors.reset}
   1. Add to .env: OPENAI_API_KEY=sk-...
   2. Select model from dropdown
   3. Server auto-routes to OpenAI

${colors.cyan}Learn More:${colors.reset}
   • API endpoints: See "API Testing" menu
   • Full docs: Open README.md
   • Configuration: See CONFIGURATION.md
  `);
  
  await pause();
  return docsMenu();
}

async function apiRef() {
  clear();
  log.header('API Reference');
  
  console.log(`
${colors.cyan}1. Health Check${colors.reset}
   GET /health
   Response: { status: "ok" }

${colors.cyan}2. Basic Chat${colors.reset}
   POST /api/chat
   Body: { model, messages }
   Response: { text, raw }

${colors.cyan}3. Fetch Streaming${colors.reset}
   POST /api/chat-stream
   Returns: text/plain stream

${colors.cyan}4. SSE Streaming${colors.reset}
   GET /api/chat-sse?payload=<base64>
   Events: data, delta, done

${colors.cyan}5. Multi-Model${colors.reset}
   POST /api/multi-chat
   Body: { models: [...], messages }
   Response: { results: [...], timings }

${colors.cyan}See full API.md for detailed examples${colors.reset}
  `);
  
  await pause();
  return docsMenu();
}

async function configGuide() {
  clear();
  log.header('Configuration Guide');
  
  console.log(`
${colors.cyan}Environment Variables (.env):${colors.reset}

   OPENAI_API_KEY
   ├─ Required for: gpt-4o, gpt-4o-mini
   ├─ Get from: https://platform.openai.com/api-keys
   └─ Example: sk-proj-abc123...

   PORT
   ├─ Default: 3000
   ├─ Usage: npm run dev -- --port 8080
   └─ Docker: Expose in docker-compose.yml

   OLLAMA_URL
   ├─ Default: http://localhost:11434
   ├─ Docker: http://ollama:11434
   └─ Remote: http://192.168.1.100:11434

${colors.cyan}Model Configuration:${colors.reset}

   Local Models (Ollama):
   • gpt-oss-20 (recommended)
   • llama3.2
   • qwen2.5

   Cloud Models (OpenAI):
   • gpt-4o
   • gpt-4o-mini

${colors.cyan}Routing Logic:${colors.reset}
   Models starting with "gpt-" → OpenAI API
   All others → Ollama (local)
  `);
  
  await pause();
  return docsMenu();
}

async function troubleshoot() {
  clear();
  log.header('Troubleshooting Guide');
  
  console.log(`
${colors.red}Server won't start${colors.reset}
  → Port 3000 in use: lsof -i :3000 (Mac/Linux)
  → Kill process: kill <pid> or taskkill /PID <pid> (Windows)
  → Use different port: PORT=8080 npm run dev

${colors.red}Ollama connection failed${colors.reset}
  → Start Ollama: ollama serve
  → Check URL: http://localhost:11434/api/tags
  → Update OLLAMA_URL if running on different host

${colors.red}OpenAI API errors${colors.reset}
  → Check API key: Set OPENAI_API_KEY in .env
  → Verify key format: sk-proj-...
  → Check quota: https://platform.openai.com/account/billing

${colors.red}Message history limit${colors.reset}
  → System message + 12 dialog messages
  → Old messages automatically trimmed
  → Adjust MAX_HISTORY in server.js

${colors.red}Streaming not working${colors.reset}
  → Try different transport (Fetch/SSE)
  → Check browser console for errors
  → Verify streaming endpoint: /api/chat-stream

${colors.yellow}Still stuck?${colors.reset}
  • Check README.md for more help
  • Review server logs: npm run dev
  • Test endpoints manually: See "API Testing"
  `);
  
  await pause();
  return docsMenu();
}

async function projectStructure() {
  clear();
  log.header('Project Structure');
  
  console.log(`
${colors.cyan}.${colors.reset}
├── ${colors.bright}server.js${colors.reset}
│   ├── /health - Health check
│   ├── /api/chat - Basic chat
│   ├── /api/chat-stream - HTTP streaming
│   ├── /api/chat-sse - SSE streaming
│   └── /api/multi-chat - Multi-model
│
├── ${colors.bright}public/${colors.reset}
│   ├── index.html - Chat UI
│   ├── app.js - Frontend logic
│   └── game.js - 3D world (optional)
│
├── ${colors.bright}ai_training/${colors.reset}
│   ├── language_model/ - LLM fine-tuning
│   ├── image_classifier/ - Image AI
│   ├── game_ai/ - Game training
│   └── custom_nn/ - Custom networks
│
├── ${colors.bright}Documentation${colors.reset}
│   ├── README.md - Overview
│   ├── API.md - API reference
│   ├── CONFIGURATION.md - Config guide
│   ├── DEVELOPMENT.md - Dev guide
│   ├── DEPLOYMENT_CHECKLIST.md - Deploy steps
│   ├── PROJECT_SUMMARY.md - Architecture
│   └── CHANGELOG.md - Version history
│
├── ${colors.bright}Scripts${colors.reset}
│   ├── start.bat / start.sh
│   ├── pull-models.bat / pull-models.sh
│   ├── health-check.bat / health-check.sh
│   ├── setup-validator.bat / setup-validator.sh
│   └── API_EXAMPLES.sh
│
├── ${colors.bright}Docker${colors.reset}
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── ${colors.bright}Config${colors.reset}
    ├── package.json
    ├── .env
    └── .env.example
  `);
  
  await pause();
  return docsMenu();
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function clear() {
  console.clear();
}

async function exit() {
  clear();
  console.log(`${colors.green}👋 Goodbye!${colors.reset}\n`);
  rl.close();
  process.exit(0);
}

// ============================================================================
// START
// ============================================================================

mainMenu().catch((err) => {
  log.error(`Fatal error: ${err.message}`);
  rl.close();
  process.exit(1);
});

# Interactive CLI Menu - User Guide

Complete interactive automation tool for your LLM Chat application.

## Quick Start

### Windows
```bash
# Run the interactive menu
cli-menu.bat

# Or directly with Node.js
node cli-menu.js
```

### macOS / Linux
```bash
# Make executable (first time only)
chmod +x cli-menu.sh

# Run the interactive menu
./cli-menu.sh

# Or directly with Node.js
node cli-menu.js
```

## Menu Structure

### 1. Setup & Configuration
Initialize and configure your project environment.

#### 1.1 Run Environment Validator
Comprehensive environment check that verifies:
- ✓ Node.js installation & version (v18+)
- ✓ npm installation & version
- ✓ Project files (server.js, package.json, public/)
- ✓ Dependencies (node_modules/)
- ✓ Ollama installation & connectivity
- ✓ OpenAI API key configuration
- ✓ Port 3000 availability

**Use this first** when setting up on a new machine.

```
Passed: 7
Warnings: 1  (e.g., Ollama not running)
Failed: 0
```

#### 1.2 Configure .env File
Creates or displays .env configuration file location.

**Key variables:**
```env
OPENAI_API_KEY    # For cloud models (gpt-4o, gpt-4o-mini)
PORT              # Server port (default 3000)
OLLAMA_URL        # Ollama endpoint (default http://localhost:11434)
```

Edit with your preferred text editor.

#### 1.3 Install Dependencies
Runs `npm install` to download all required packages.

**When to use:**
- First-time setup
- After cloning repo
- After updating package.json

#### 1.4 Check Node.js Version
Display your current Node.js and npm versions.

---

### 2. Server Management
Control and monitor the Express server.

#### 2.1 Start Server
Displays instructions for starting the server:
```bash
npm run dev              # Recommended (with watching)
node server.js          # Direct execution
```

The menu cannot directly start the server in background—it will guide you to do so in a separate terminal.

#### 2.2 Stop Server
Instructions for stopping the running server:
- Press `Ctrl+C` in the server terminal
- Or use `taskkill /IM node.exe /F` (Windows Admin)
- Or use `pkill -f "node server.js"` (Unix/macOS)

#### 2.3 Restart Server
Complete restart procedure:
1. Stop the server (Ctrl+C)
2. Start it again (npm run dev)

#### 2.4 Check Server Status
**Automatic health check** via `/health` endpoint.

```
✓ Server is running
Status: { "status": "ok" }
```

Or if not running:
```
✗ Server is not running
Error: connect ECONNREFUSED 127.0.0.1:3000
Start with: npm run dev
```

#### 2.5 View Server Logs
Information about accessing server logs from the terminal running the server.

Logs show:
- HTTP requests & responses
- Errors & warnings
- Performance metrics (via Morgan logging)

---

### 3. API Testing
Interactive endpoint testing without manual curl commands.

#### 3.1 Test /health Endpoint
Quick connectivity check.
```
✓ Health check passed
Response: { "status": "ok" }
```

#### 3.2 Test /api/chat (Basic)
Send a simple chat message and get response.

**Prompts:**
- Model (default: gpt-oss-20)
- Message (your user input)

**Output:**
```
✓ Response received
[AI response text]
```

#### 3.3 Test /api/chat-stream
Test HTTP streaming endpoint.

Streams response word-by-word as it's generated.

#### 3.4 Test /api/chat-sse
Information & example code for Server-Sent Events streaming.

Shows sample frontend code:
```javascript
const es = new EventSource(`/api/chat-sse?payload=${base64}`);
es.addEventListener('delta', (e) => console.log(e.data));
es.addEventListener('done', () => es.close());
```

#### 3.5 Interactive Chat
**Multi-turn conversation** directly in the CLI.

```
Model (default: gpt-oss-20): gpt-oss-20
Chat started. Type "exit" to quit.

You: Hello!
Assistant: Hi there! How can I help you today?

You: What's 2+2?
Assistant: 2+2 equals 4.

You: exit
[Chat ended]
```

Message history automatically maintained across turns.

---

### 4. Model Management
Ollama model operations.

#### 4.1 List Available Ollama Models
Shows all downloaded models with file sizes.

```
✓ Found 2 model(s)
  • gpt-oss-20 (5234567890 bytes)
  • llama3.2 (4123456789 bytes)
```

If none installed:
```
⚠ No Ollama models installed
Pull models with: ollama pull <model>
```

#### 4.2 Pull New Ollama Model
Download model from Ollama registry.

**Steps:**
1. Enter model name (e.g., `qwen2.5`)
2. Menu displays: `ollama pull qwen2.5`
3. Run command in **separate terminal**

**Common models:**
- `gpt-oss-20` - Recommended starting model
- `llama3.2` - Fast, efficient
- `qwen2.5` - Good instruction following
- `mistral` - Code generation

#### 4.3 Remove Ollama Model
Delete a downloaded model.

Menu displays command to run in separate terminal:
```bash
ollama rm gpt-oss-20
```

#### 4.4 Test Model Availability
Check which recommended models are installed.

```
✓ gpt-oss-20 is installed
⚠ llama3.2 not installed
⚠ qwen2.5 not installed
```

---

### 5. Deployment
Production deployment procedures.

#### 5.1 Docker Setup & Deployment
Instructions for containerized deployment.

**Quick start:**
```bash
docker-compose up -d       # Start containers
docker-compose logs -f     # View logs
docker-compose down        # Stop containers
```

**Prerequisites:**
- Docker Desktop installed
- docker-compose available

#### 5.2 Production Checklist
Pre-deployment verification checklist.

**10-point checklist:**
- ✓ Node.js v18+
- ✓ Dependencies installed
- ✓ .env configured
- ✓ Port not blocked
- ✓ Ollama running
- ✓ Models pulled
- ✓ CORS configured
- ✓ Rate limiting enabled
- ✓ Error handling tested
- ✓ Load testing done

**Deployment steps:**
1. `npm install`
2. Configure `.env`
3. Test with `npm run dev`
4. Deploy: `docker-compose up -d`
5. Monitor: `docker-compose logs -f`

#### 5.3 Security Review
Production security checklist.

**Categories:**
- **API Keys** - Secure storage, .gitignore
- **Authentication** - JWT/OAuth recommendations
- **CORS** - Domain restriction (not wildcard)
- **Input Validation** - Current protections
- **Rate Limiting** - Per-IP limits
- **Monitoring** - Logging, error tracking

#### 5.4 Performance Optimization
Performance tuning guide.

**Current optimizations:**
- Message history capping (12 messages)
- Streaming support
- Efficient middleware pipeline
- JSON parsing optimization

**Recommended improvements:**
- Response caching (Redis)
- Connection pooling
- Load balancing (nginx)
- Database integration (PostgreSQL)
- CDN for static assets

---

### 6. Documentation
Access comprehensive project documentation.

#### 6.1 Quick Start Guide
5-minute setup overview.

```
1. npm install
2. ollama serve (separate terminal)
3. npm run dev
4. Open http://localhost:3000
5. Chat!
```

#### 6.2 API Reference
Quick reference for all 5 endpoints.

```
GET /health
POST /api/chat
POST /api/chat-stream
GET /api/chat-sse
POST /api/multi-chat
```

With response format examples.

#### 6.3 Configuration Guide
Environment variable reference.

**Variables:**
- `OPENAI_API_KEY` - Cloud model access
- `PORT` - Server port
- `OLLAMA_URL` - Ollama endpoint

**Model routing logic** explained.

#### 6.4 Troubleshooting Guide
Common issues & solutions.

**Covers:**
- Port 3000 already in use
- Ollama connection failed
- OpenAI API errors
- Message history limits
- Streaming issues

#### 6.5 Project Structure
Visual tree of directory layout.

Shows files, folders, and what each does.

```
. (root)
├── server.js (5 API endpoints)
├── public/ (Frontend UI)
├── ai_training/ (Optional: PyTorch modules)
├── Documentation/ (8 doc files)
├── Scripts/ (Automation tools)
├── Docker/ (Containerization)
└── Config/ (Environment & metadata)
```

---

## Features

### Color-Coded Output
```
✓ Green   - Success / Passed
✗ Red     - Error / Failed
⚠ Yellow  - Warning / Caution
ℹ Blue    - Information
```

### Multi-Terminal Awareness
Menu recognizes when:
- Server already running
- Port in use
- Ollama unavailable
- Dependencies missing

Provides helpful guidance rather than failing.

### Interactive Input Handling
```
Select option: 1
Model (default: gpt-oss-20): llama3.2
Message: Hello there!
```

All inputs have sensible defaults and validation.

### Error Recovery
- Missing dependencies? Offers to install
- Port conflict? Shows how to resolve
- Ollama down? Explains how to start it

---

## Common Workflows

### New User Setup
```
1. Open cli-menu.bat (or ./cli-menu.sh)
2. Choose "1. Setup & Configuration"
3. Choose "1. Run environment validator"
4. Choose "3. Install dependencies"
5. Choose "2. Server Management" → "1. Start server"
6. Open http://localhost:3000 in browser
```

### Before First Chat
```
1. Ensure server running: Menu → Server Management → Check status
2. Pull models: Menu → Model Management → Pull New Model → ollama pull gpt-oss-20
3. Test connectivity: Menu → API Testing → Test /health
4. Interactive chat: Menu → API Testing → Interactive Chat
```

### Debugging Connection Issues
```
1. Menu → Setup → Run environment validator
2. Menu → Server Management → Check server status
3. Menu → Model Management → Test Model Availability
4. Menu → API Testing → Test /health
```

### Deployment Prep
```
1. Menu → Deployment → Production Checklist
2. Menu → Deployment → Security Review
3. Menu → Deployment → Docker Setup
4. Deploy with docker-compose commands shown
```

---

## Tips & Tricks

### Use in Scripts
```bash
# Non-interactive: Chain commands
(echo "1"; echo "1"; sleep 3; echo "0") | node cli-menu.js
```

### Pipe Output to File
```bash
cli-menu.bat > output.txt 2>&1
```

### Combine with Other Tools
```bash
# Check status then view logs
cli-menu.bat  # Choose "Check status"
npm run dev   # In another terminal to see live logs
```

### Extend the Menu
Edit `cli-menu.js` to add custom options:
```javascript
async function myCustomFunction() {
  clear();
  log.header('My Custom Tool');
  // ... your code here
  await pause();
  return mainMenu();
}
```

---

## Troubleshooting

### "Node.js is not installed"
- Install from https://nodejs.org
- Ensure v18 or newer

### Menu exits immediately
- Check terminal encoding (UTF-8)
- Try: `node cli-menu.js` directly
- Check for unsaved interactive input

### Color output not showing
- On Windows: Use PowerShell (not old cmd.exe)
- On Unix: TERM environment variable set

### "Cannot find module" errors
```bash
npm install  # Run dependency installation
```

---

## Size & Performance

- **Script Size**: ~20KB (cli-menu.js)
- **Memory**: ~30-50MB while running
- **Startup**: < 2 seconds
- **Network calls**: Only when testing endpoints

---

## Support

Need help? See in-menu options:
1. **Documentation** → View relevant guides
2. **Troubleshooting** → Common issues
3. **API Testing** → Interactive endpoint tests

Or check project documentation:
- README.md
- API.md
- CONFIGURATION.md

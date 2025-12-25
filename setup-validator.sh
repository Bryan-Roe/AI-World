#!/bin/bash
# setup-validator.sh - Verify environment and dependencies before running

echo ""
echo "=========================================="
echo "  Web LLM Chat - Setup Validator"
echo "=========================================="
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_passed() {
  echo -e "${GREEN}✓${NC} $1"
  ((CHECKS_PASSED++))
}

check_failed() {
  echo -e "${RED}✗${NC} $1"
  ((CHECKS_FAILED++))
}

check_warning() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

echo "1️⃣  Checking System Requirements..."
echo ""

# Check Node.js
if command -v node &> /dev/null; then
  NODE_VERSION=$(node --version)
  check_passed "Node.js installed: $NODE_VERSION"
else
  check_failed "Node.js not found. Install from https://nodejs.org (18+)"
fi

# Check npm
if command -v npm &> /dev/null; then
  NPM_VERSION=$(npm --version)
  check_passed "npm installed: $NPM_VERSION"
else
  check_failed "npm not found. Install Node.js (includes npm)"
fi

echo ""
echo "2️⃣  Checking Project Files..."
echo ""

# Check main files
if [ -f "server.js" ]; then
  check_passed "server.js found"
else
  check_failed "server.js not found"
fi

if [ -f "package.json" ]; then
  check_passed "package.json found"
else
  check_failed "package.json not found"
fi

if [ -d "public" ]; then
  check_passed "public/ directory found"
else
  check_failed "public/ directory not found"
fi

if [ -f ".env.example" ]; then
  check_passed ".env.example found"
  if [ ! -f ".env" ]; then
    check_warning ".env not found - copy from .env.example and configure"
  else
    check_passed ".env file exists"
  fi
else
  check_warning ".env.example not found"
fi

echo ""
echo "3️⃣  Checking Dependencies..."
echo ""

if [ -d "node_modules" ]; then
  check_passed "node_modules/ exists (dependencies installed)"
else
  check_warning "node_modules/ not found - run: npm install"
fi

echo ""
echo "4️⃣  Checking Ollama (Local Models)..."
echo ""

if command -v ollama &> /dev/null; then
  OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "version unknown")
  check_passed "Ollama installed: $OLLAMA_VERSION"
  
  # Check if Ollama is running
  if timeout 2 curl -s http://localhost:11434/api/tags &> /dev/null; then
    check_passed "Ollama is running on http://localhost:11434"
    
    # Check for models
    MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*' | cut -d'"' -f4 | wc -l)
    if [ "$MODELS" -gt 0 ]; then
      check_passed "Found $MODELS model(s) in Ollama"
    else
      check_warning "No models in Ollama - run: ollama pull gpt-oss-20"
    fi
  else
    check_warning "Ollama is not running - start with: ollama serve"
  fi
else
  check_warning "Ollama not installed - for local models, install from https://ollama.com"
fi

echo ""
echo "5️⃣  Checking OpenAI (Cloud Models)..."
echo ""

if [ -f ".env" ]; then
  if grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    check_passed "OPENAI_API_KEY is set in .env"
    
    # Test API key (if present)
    API_KEY=$(grep "OPENAI_API_KEY=" .env | cut -d'=' -f2 | tr -d "'" | tr -d '"')
    if [ ! -z "$API_KEY" ] && [ "$API_KEY" != "sk-your-actual-key" ]; then
      if timeout 5 curl -s https://api.openai.com/v1/models \
        -H "Authorization: Bearer $API_KEY" \
        -H "User-Agent: Web-LLM-Chat" &> /dev/null; then
        check_passed "OpenAI API key is valid"
      else
        check_failed "OpenAI API key test failed - check key is correct"
      fi
    fi
  else
    check_warning "OPENAI_API_KEY not set - for cloud models, set in .env"
  fi
else
  check_warning ".env file not found - create from .env.example"
fi

echo ""
echo "6️⃣  Checking Port Availability..."
echo ""

PORT=${PORT:-3000}
if ! timeout 1 bash -c "echo >/dev/tcp/localhost/$PORT" 2>/dev/null; then
  check_passed "Port $PORT is available"
else
  check_failed "Port $PORT is already in use"
fi

echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
echo -e "${GREEN}Passed:${NC}  $CHECKS_PASSED"
if [ $WARNINGS -gt 0 ]; then
  echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
fi
if [ $CHECKS_FAILED -gt 0 ]; then
  echo -e "${RED}Failed:${NC}  $CHECKS_FAILED"
fi
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
  echo "✅ Setup validation passed!"
  echo ""
  echo "You can now start the server:"
  echo "  npm run dev"
  echo ""
  echo "Then open http://localhost:3000 in your browser"
  exit 0
else
  echo "❌ Setup validation failed. Fix errors above and try again."
  exit 1
fi

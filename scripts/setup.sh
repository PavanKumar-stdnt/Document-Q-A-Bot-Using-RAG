#!/usr/bin/env bash
# scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────
# One-shot setup for the RAG Chatbot (Linux / macOS / WSL)
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────

set -e   # exit immediately on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERR ]${NC} $1"; exit 1; }

echo ""
echo "════════════════════════════════════════════════"
echo "  RAG Chatbot — Free Stack Setup"
echo "════════════════════════════════════════════════"
echo ""

# ── 1. Python version check ─────────────────────────────────────────
info "Checking Python version…"
python3 --version | grep -E "3\.(10|11|12)" > /dev/null 2>&1 \
  || warning "Python 3.10+ recommended. Continuing anyway."

# ── 2. Virtual environment ──────────────────────────────────────────
if [ ! -d "rag-env" ]; then
  info "Creating virtual environment 'rag-env'…"
  python3 -m venv rag-env
else
  info "Virtual environment 'rag-env' already exists."
fi

info "Activating virtual environment…"
source rag-env/bin/activate

# ── 3. Install Python packages ──────────────────────────────────────
info "Installing Python packages (this may take 2-3 minutes)…"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Python packages installed."

# ── 4. Check Ollama ────────────────────────────────────────────────
info "Checking Ollama installation…"
if ! command -v ollama &> /dev/null; then
  warning "Ollama not found. Visit https://ollama.com to install it."
  warning "After installing, run:"
  warning "  ollama pull llama3"
  warning "  ollama pull nomic-embed-text"
else
  info "Ollama found: $(ollama --version)"

  info "Pulling Gemma3:4b (~3.3 GB)… this takes a while on first run."
  ollama pull gemma3:4b

  info "Pulling nomic-embed-text (~274 MB)…"
  ollama pull nomic-embed-text
fi

# ── 5. Copy .env.example → .env ────────────────────────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  info "Created .env from .env.example. Edit it if needed."
else
  info ".env already exists — skipping."
fi

# ── 6. Create docs folder ───────────────────────────────────────────
mkdir -p docs
info "Created ./docs folder. Add your PDF/TXT/MD files there."

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo -e "  ${GREEN}Setup complete!${NC}"
echo ""
echo "  Next steps:"
echo "  1. Add documents:   cp your_file.pdf docs/"
echo "  2. Index them:      python ingest.py"
echo "  3. Test in CLI:     python cli_chat.py"
echo "  4. Launch web UI:   streamlit run app.py"
echo "════════════════════════════════════════════════"
echo ""

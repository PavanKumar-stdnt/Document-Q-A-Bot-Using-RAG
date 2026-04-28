@echo off
REM scripts/setup_windows.bat
REM ─────────────────────────────────────────────────────────────────────
REM One-shot setup for the RAG Chatbot on Windows
REM
REM Usage: Double-click this file OR run from Command Prompt:
REM   scripts\setup_windows.bat
REM ─────────────────────────────────────────────────────────────────────

echo.
echo ================================================
echo   RAG Chatbot — Free Stack Setup (Windows)
echo ================================================
echo.

REM ── 1. Python check ───────────────────────────────────────────────
echo [INFO] Checking Python installation...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Download from https://python.org
    pause
    exit /b 1
)
python --version

REM ── 2. Virtual environment ────────────────────────────────────────
IF NOT EXIST "rag-env" (
    echo [INFO] Creating virtual environment...
    python -m venv rag-env
) ELSE (
    echo [INFO] Virtual environment already exists.
)

echo [INFO] Activating virtual environment...
call rag-env\Scripts\activate.bat

REM ── 3. Install packages ───────────────────────────────────────────
echo [INFO] Installing Python packages (may take 2-3 minutes)...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo [INFO] Python packages installed.

REM ── 4. Copy .env ──────────────────────────────────────────────────
IF NOT EXIST ".env" (
    copy .env.example .env
    echo [INFO] Created .env from .env.example
) ELSE (
    echo [INFO] .env already exists.
)

REM ── 5. Create docs folder ─────────────────────────────────────────
IF NOT EXIST "docs" mkdir docs
echo [INFO] Created .\docs folder. Add your PDF/TXT/MD files there.

REM ── 6. Ollama reminder ────────────────────────────────────────────
echo.
echo [WARN] Make sure Ollama is installed from https://ollama.com
echo [WARN] Then run these commands in a NEW terminal:
echo          ollama pull llama3
echo          ollama pull nomic-embed-text
echo.

REM ── Done ──────────────────────────────────────────────────────────
echo ================================================
echo   Setup complete!
echo.
echo   Next steps:
echo   1. Add documents:  copy your_file.pdf docs\
echo   2. Index them:     python ingest.py
echo   3. Test CLI:       python cli_chat.py
echo   4. Launch web UI:  streamlit run app.py
echo ================================================
echo.
pause

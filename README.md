# 📄 RAG Document Q&A Bot

A fully local **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF, DOCX, TXT, or Markdown documents and ask natural language questions against them. The bot retrieves the most relevant passages from your documents and generates grounded answers with clear citations — it will never make up information that isn't in your files. Everything runs on your own machine using free, open-source models via Ollama — no API keys, no cloud, no cost.

---

## Table of Contents

1. [Tech Stack](#1-tech-stack)
2. [Architecture Overview](#2-architecture-overview)
3. [Chunking Strategy](#3-chunking-strategy)
4. [Embedding Model and Vector Database](#4-embedding-model-and-vector-database)
5. [Setup Instructions](#5-setup-instructions)
6. [Environment Variables](#6-environment-variables)
7. [Example Queries](#7-example-queries)
8. [Known Limitations](#8-known-limitations)

---

## 1. Tech Stack

Every library and tool used in this project, with exact version numbers:

| Tool / Library | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Runtime language |
| **Ollama** | latest | Local model runner — serves LLM and embeddings |
| **Gemma3:4b** (via Ollama) | latest | LLM for answer generation |
| **nomic-embed-text** (via Ollama) | latest | Embedding model for chunks and queries |
| **langchain** | 0.2.16 | RAG pipeline orchestration, prompt templates, memory |
| **langchain-community** | 0.2.16 | ChromaDB integration, document loaders |
| **langchain-ollama** | 0.1.3 | Ollama LLM and embedding wrappers |
| **langchain-core** | 0.2.38 | Base abstractions for chains and retrievers |
| **chromadb** | 0.5.5 | Local persistent vector database |
| **pypdf** | 4.3.1 | PDF text extraction |
| **docx2txt** | 0.8 | DOCX (Word document) text extraction |
| **unstructured** | 0.15.0 | Fallback document parsing |
| **tiktoken** | 0.7.0 | Accurate token counting for chunking |
| **streamlit** | 1.38.0 | Browser-based web UI |
| **ragas** | 0.1.21 | RAG quality evaluation (faithfulness, relevancy) |
| **datasets** | 2.21.0 | RAGAS dataset format support |
| **python-dotenv** | 1.0.1 | `.env` configuration loading |
| **loguru** | 0.7.2 | Structured, colourised logging |

---

## 2. Architecture Overview

The system is split into two clearly separated pipelines:

### Indexing Pipeline *(run once per document set)*

```
┌──────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│   Documents  │     │   Text Chunking      │     │      Embedding        │
│  PDF / DOCX  │────▶│ RecursiveCharacter   │────▶│  nomic-embed-text     │
│  TXT / MD    │     │ 500 tokens / 100 OL  │     │  (Ollama, local)      │
└──────────────┘     └─────────────────────┘     └──────────┬────────────┘
                                                             │
                                                             ▼
                                                   ┌───────────────────┐
                                                   │     ChromaDB      │
                                                   │  (persisted disk) │
                                                   └───────────────────┘
```

**Steps:**
1. `document_loader.py` — reads files from `./docs` or from browser upload using `PyPDFLoader`, `Docx2txtLoader`, or `TextLoader`
2. `text_splitter.py` — applies `RecursiveCharacterTextSplitter` (chunk size 500, overlap 100) and attaches metadata: `source`, `file_type`, `chunk_index`
3. `vector_store.py` — embeds every chunk via `OllamaEmbeddings(nomic-embed-text)` and persists them to a ChromaDB collection at `./chroma_db/`

### Querying Pipeline *(every user question)*

```
┌─────────────┐     ┌──────────────────────────┐     ┌──────────────────┐
│  User Query │────▶│  Embed query              │────▶│  MMR Similarity  │
│             │     │  (nomic-embed-text)        │     │  Search          │
└─────────────┘     └──────────────────────────┘     │  fetch_k=8 → k=4 │
                                                       └────────┬─────────┘
                                                                │
                                                   Top-4 diverse chunks
                                                                │
                                                                ▼
┌────────────────┐     ┌──────────────────────────────────────────────────┐
│  Final Answer  │◀────│  Gemma3:4b (Ollama)                               │
│  + Citations   │     │  System prompt: answer ONLY from retrieved context │
└────────────────┘     └──────────────────────────────────────────────────┘
```

**Steps:**
1. `retriever.py` — embeds the user query and runs MMR similarity search against ChromaDB, returning the top-4 most relevant and diverse chunks
2. `chain.py` — builds a `ConversationalRetrievalChain` with `ConversationBufferMemory` so follow-up questions are resolved in context, then passes chunks to Gemma3:4b
3. The LLM is instructed via system prompt: answer only from retrieved context; if the answer is absent, respond with "I don't have enough information in the provided documents to answer that."

**Key design decisions:**
- **Indexing and querying are fully separated** — `ingest.py` writes vectors once; the chatbot loads them read-only. New documents can be appended without wiping existing data.
- **Conversation memory** — the chain rewrites ambiguous follow-up questions ("what else did it say?") into self-contained queries before retrieval.
- **Hallucination prevention** — the system prompt explicitly forbids the LLM from using its own training knowledge when the answer is not in the context.

---

## 3. Chunking Strategy

**Strategy chosen: Recursive Character Text Splitting**

| Parameter | Value |
|---|---|
| `chunk_size` | 500 tokens |
| `chunk_overlap` | 100 tokens |
| Separator hierarchy | `\n\n` → `\n` → `. ` → ` ` → `""` |

**Why Recursive Character Splitting:**

The splitter tries to break text on natural boundaries first — paragraphs (`\n\n`), then sentences (`\n`, `. `), then words, and finally characters as a last resort. This keeps retrieved chunks coherent: the LLM receives complete thoughts and sentences rather than fragments cut off mid-sentence.

**Why not the alternatives:**
- *Fixed-size splitting* cuts at arbitrary character positions, often mid-sentence, making retrieved chunks hard for the LLM to interpret.
- *Sentence-based splitting* produces chunks of wildly varying sizes — a paragraph with ten short sentences becomes ten tiny chunks, none individually informative enough for good retrieval.

**Why 500 tokens with 100-token overlap:**

A 500-token window is large enough to contain a complete idea (typically 2–4 paragraphs) while staying well within the embedding model's context limit. The 100-token (20%) overlap ensures that sentences crossing a chunk boundary appear in both adjacent chunks — whichever chunk gets retrieved will contain the complete sentence, not a truncated fragment.

Each chunk stores metadata: `source` (filename), `file_type`, and `chunk_index` — used to render source citations in the UI next to every answer.

---

## 4. Embedding Model and Vector Database

### Embedding Model — `nomic-embed-text` via Ollama

**Why nomic-embed-text:**
- Runs entirely locally via Ollama — zero cost, zero internet dependency after the initial download (~274 MB)
- Produces 768-dimensional vectors, rich enough for semantic search over multi-document collections
- Trained specifically for retrieval tasks using contrastive learning on retrieval pairs, outperforming `all-MiniLM-L6-v2` on MTEB benchmarks
- Designed to handle both short queries and longer document passages in the same embedding space — critical for query-to-chunk similarity to work well

### Vector Database — ChromaDB

**Why ChromaDB:**
- Runs fully in-process with Python — no separate server to install, start, or manage
- Persists all vectors to SQLite on disk at `./chroma_db/` — the bot loads instantly on restart without re-embedding anything
- Native LangChain integration through `langchain-community` makes switching to FAISS or Qdrant a one-line change if needed
- Supports MMR (Max Marginal Relevance) natively

**Why MMR retrieval over plain cosine similarity:**

Standard top-k cosine retrieval returns the k most similar chunks — but if your document repeats a key phrase in eight places, all eight end up in the top-4, giving the LLM redundant context. MMR fetches `fetch_k=8` candidates, then iteratively picks the next chunk that maximises *relevance to the query* **and** *difference from already-selected chunks*. The result is 4 diverse chunks that collectively cover the topic from multiple angles, producing more complete answers.

---

## 5. Setup Instructions

Follow these steps exactly from `git clone` to a running bot.

### Prerequisites

- Python 3.11 or higher — download from https://python.org
- 8 GB RAM minimum (16 GB recommended)
- 10 GB free disk space
- Windows 10/11, Ubuntu 20.04+, or macOS 12+

### Step 1 — Install Ollama

Download and install from **https://ollama.com** for your operating system. After installation, verify it works:

```bash
ollama --version
```

### Step 2 — Pull the AI models (one-time download, ~3.6 GB)

```bash
ollama pull gemma3:4b           # LLM for answer generation (~3.3 GB)
ollama pull nomic-embed-text    # Embedding model (~274 MB)
```

Confirm both are downloaded:
```bash
ollama list
# Should show gemma3:4b and nomic-embed-text
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
```

### Step 4 — Create a virtual environment

**Linux / macOS:**
```bash
python3 -m venv rag-env
source rag-env/bin/activate
```

**Windows:**
```cmd
python -m venv rag-env
rag-env\Scripts\activate
```

### Step 5 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6 — Configure environment

```bash
cp .env.example .env
# The defaults work out of the box — no edits needed for local use
```

### Step 7 — Add your documents

Copy PDF, TXT, DOCX, or MD files into the `docs/` folder:

```bash
cp your_document.pdf docs/
cp your_notes.txt docs/
# A sample company policy document is already included at docs/sample_document.txt
```

### Step 8 — Index the documents

```bash
python ingest.py
```

You will see output like:
```
[INFO] STEP 1/3 — Loading documents…
[INFO] STEP 2/3 — Splitting into chunks…
[INFO] STEP 3/3 — Embedding and storing in ChromaDB…
[INFO] Ingestion complete!  Chunks stored: 147
```

### Step 9 — Launch the web UI

Open two terminals:

**Terminal 1** — keep Ollama running:
```bash
ollama serve
```

**Terminal 2** — start the app:
```bash
source rag-env/bin/activate      # Windows: rag-env\Scripts\activate
streamlit run app.py
```

Open **http://localhost:8501** in your browser. You can also upload additional documents directly through the web UI without re-running `ingest.py`.

**Alternative — run the terminal chatbot instead:**
```bash
python cli_chat.py
```

---

## 6. Environment Variables

 All variables have working defaults — the bot runs without changing anything. **Never commit your `.env` file** 

```env
# ── Ollama server ───────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434    # Address where Ollama is listening

# ── Models ─────────────────────────────────────────────────────────────
LLM_MODEL=gemma3:4b                       # Model used for answer generation
EMBEDDING_MODEL=nomic-embed-text          # Model used for embedding chunks

# ── Storage ────────────────────────────────────────────────────────────
CHROMA_PATH=./chroma_db                  # Folder where ChromaDB saves vectors
COLLECTION_NAME=rag_documents            # Name of the ChromaDB collection
DOCS_PATH=./docs                         # Default folder for source documents

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE=500                           # Max tokens per chunk
CHUNK_OVERLAP=100                        # Overlapping tokens between chunks

# ── Retrieval ──────────────────────────────────────────────────────────
RETRIEVER_K=4                            # Number of chunks passed to the LLM
RETRIEVER_FETCH_K=8                      # Candidates fetched before MMR re-ranks

# ── Answer generation ──────────────────────────────────────────────────
LLM_TEMPERATURE=0.1                      # Lower = more factual, less creative
LLM_MAX_TOKENS=1024                      # Maximum length of generated answer
```

**If you switch to a paid LLM later** (e.g. OpenAI GPT-4), add this line — never put the actual key in source code:
```env
OPENAI_API_KEY=sk-...your-key-here...
```

---

## 7. Example Queries

All queries below work against the included `docs/sample_document.txt` (a sample company policy document). Replace these with questions relevant to your own uploaded files.

| # | Sample Question | Expected Answer Theme |
|---|---|---|
| 1 | "What is the annual leave entitlement?" | Employees get 20 days of paid annual leave per year, applied 2 weeks in advance |
| 2 | "How many sick leave days am I allowed?" | 10 days per year with a valid medical certificate |
| 3 | "Can I work from home every day of the week?" | No — maximum 3 days per week, requires manager approval, available during 10 AM–4 PM core hours |
| 4 | "What are the receipt rules for expense claims?" | Receipts required for any expense above $25; submitted within 30 days of the expense |
| 5 | "What password rules does the IT policy require?" | Minimum 12 characters; multi-factor authentication (MFA) mandatory for all company accounts |
| 6 | "What is the customer refund policy?" | *(Not in the documents)* — Bot replies: "I don't have enough information in the provided documents to answer that." |

> **Query 6 is intentional.** It demonstrates the out-of-scope behaviour — the LLM is explicitly instructed not to answer from its own training knowledge when the retrieved context does not contain the answer.

---

## 8. Known Limitations

| Limitation | Why it happens | Suggested workaround |
|---|---|---|
| **Scanned PDFs return no text** | `pypdf` extracts digital text layers only; scanned pages are stored as images with no embedded text | Use PDFs that were created digitally, or pre-process scanned files with an OCR tool such as Adobe Acrobat or `pytesseract` before uploading |
| **Very large files are slow to index** | Each page is loaded, chunked, and embedded in sequence; a 500-page book can take several minutes | Split large documents into chapters or sections before adding them to `docs/` |
| **Answers are limited to retrieved chunks** | The LLM receives at most 4 chunks (~2,000 tokens of context) per question — it cannot read the entire document in one go | Ask targeted, specific questions; rephrase a broad question into smaller sub-questions to get more precise retrieval |
| **No cross-document comparison in one answer** | The retriever fetches the top-4 chunks globally — it does not guarantee chunks from multiple documents unless both are relevant to the query | Use phrasing like "compare X in [doc A] and [doc B]" to guide retrieval toward both files |
| **Chat history is not saved between sessions** | `ConversationBufferMemory` lives in RAM only — restarting the app clears the conversation | Your indexed documents are always preserved in `chroma_db/`; only the chat history is lost on restart |
| **Quality limited by model size** | Gemma3:4b is a 3.3 GB, 4-billion-parameter model — smaller than GPT-4 class models | For factual Q&A over structured documents it performs well; for complex multi-step reasoning, consider switching `LLM_MODEL=llama3` in `.env` |
| **First response is slow (~10–30 sec)** | Ollama loads the model weights into RAM on the first inference call | Wait for the first answer to appear; all subsequent answers in the same session are much faster |
| **Non-English documents have lower accuracy** | `nomic-embed-text` and `gemma3:4b` are trained predominantly on English text | Documents in other languages will index and retrieve, but answer quality and retrieval precision may be reduced |

---

## Project Structure

```
rag-chatbot/
├── app.py                   ← Streamlit web UI (main entry point)
├── ingest.py                ← Index documents from ./docs via CLI
├── cli_chat.py              ← Interactive terminal chatbot
├── evaluate.py              ← RAGAS quality evaluation runner
├── requirements.txt         ← All Python dependencies with versions
├── .env.example             ← Environment variable template
├── .gitignore               ← Excludes chroma_db/, docs/, .env
│
├── src/                     ← Core pipeline modules
│   ├── config.py            ← Central settings (reads .env)
│   ├── logger.py            ← Loguru logging setup
│   ├── document_loader.py   ← Load PDF / DOCX / TXT / MD
│   ├── text_splitter.py     ← Recursive character chunking
│   ├── vector_store.py      ← Embed + persist to ChromaDB
│   ├── retriever.py         ← MMR retriever
│   ├── chain.py             ← ConversationalRetrievalChain
│   └── evaluator.py         ← Evaluation helpers + RAGAS
│
├── tests/
│   └── test_pipeline.py     ← Unit tests (pytest)
│
├── scripts/
│   ├── setup.sh             ← One-shot setup for Linux/macOS
│   ├── setup_windows.bat    ← One-shot setup for Windows
│   └── reset.sh             ← Wipe DB and re-index
│
└── docs/
    └── sample_document.txt  ← Sample company policy for testing
```

## Quick Command Reference

```bash
python ingest.py                       # Index all files in ./docs
python ingest.py --reset               # Wipe DB and re-index from scratch
python ingest.py --docs path/to/dir   # Index a custom folder
streamlit run app.py                   # Web UI at http://localhost:8501
python cli_chat.py                     # Terminal chatbot
python evaluate.py                     # Basic evaluation
python evaluate.py --ragas             # Full RAGAS metrics
pytest tests/ -v                       # Run unit tests
```

---


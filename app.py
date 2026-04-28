# app.py
"""
RAG Chatbot — Enhanced Streamlit Web Application
=================================================
New upload features:
  - Drag-and-drop multi-file upload (PDF / TXT / MD / DOCX)
  - Per-file step-by-step progress during indexing
  - Duplicate detection (skip already-indexed files)
  - Document manager — view all indexed files with metadata
  - Per-document delete from registry
  - Document preview — peek at extracted text chunks
  - Suggested starter questions
  - Two-column layout: Chat | Document Browser
  - Full reset (wipe DB + registry)

Run:
    streamlit run app.py
"""

import json
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st

from src.config import APP_TITLE, APP_ICON, LLM_MODEL, EMBEDDING_MODEL
from src.document_loader import _load_single_file
from src.text_splitter import split_documents
from src.vector_store import (
    embed_and_store,
    load_vector_store,
    vector_store_exists,
    delete_vector_store,
)
from src.retriever import build_retriever
from src.chain import build_chain, ask
from src.logger import logger


# ── Constants ─────────────────────────────────────────────────────────
SUPPORTED_TYPES   = ["pdf", "txt", "md", "docx"]
REGISTRY_FILE     = Path("chroma_db/doc_registry.json")
MAX_PREVIEW_CHARS = 2000
FILE_ICONS        = {"pdf": "📕", "txt": "📄", "md": "📝", "docx": "📘"}


# ════════════════════════════════════════════════════════════════════
# Page config & CSS
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

import streamlit as st

st.markdown(
   """
    <style>
    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        background-color: #0e1117; /* Darker, modern background */
    }

    /* Modern Card Design */
    .doc-card {
        background: rgba(255, 255, 255, 0.05); /* Glassmorphism effect */
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #7d33ff; /* Accent color purple */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    
    .doc-card:hover {
        transform: translateY(-5px); /* Subtle lift on hover */
        border-left: 5px solid #a16eff;
    }

    /* Modern Preview Box */
    .preview-box {
        background: #1a1c24;
        padding: 15px;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        max-height: 400px;
        border: 1px solid #333;
        color: #e0e0e0;
        line-height: 1.6;
    }

    /* Styling the File Uploader to look cleaner */
    [data-testid="stFileUploader"] {
        background: #1a1c24;
        border: 2px dashed #444;
        border-radius: 15px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)



# ════════════════════════════════════════════════════════════════════
# Document Registry helpers
# ════════════════════════════════════════════════════════════════════

def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except Exception:
            pass
    return {}


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def register_file(name: str, pages: int, chunks: int, size_kb: float) -> None:
    reg = load_registry()
    reg[name] = {
        "name":       name,
        "pages":      pages,
        "chunks":     chunks,
        "size_kb":    round(size_kb, 1),
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_registry(reg)


def unregister_file(name: str) -> None:
    reg = load_registry()
    reg.pop(name, None)
    save_registry(reg)


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:10]


# ════════════════════════════════════════════════════════════════════
# Session state
# ════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "messages":       [],
        "chain":          None,
        "db_count":       0,
        "seen_hashes":    set(),
        "preview_name":   None,
        "preview_text":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def rebuild_chain():
    try:
        vs = load_vector_store()
        retriever = build_retriever(vs)
        st.session_state.chain    = build_chain(retriever)
        st.session_state.db_count = vs._collection.count()
    except FileNotFoundError:
        st.session_state.chain    = None
        st.session_state.db_count = 0


init_state()
if st.session_state.chain is None and vector_store_exists():
    rebuild_chain()

registry = load_registry()


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")

    # ── DB health ─────────────────────────────────────────────────
    if st.session_state.db_count > 0:
        st.success(
            f"✅ **{len(registry)} file(s)** · "
            f"**{st.session_state.db_count} chunks** indexed"
        )
    else:
        st.warning("⚠️ No documents indexed yet.")

    st.divider()

    # ── Tabs: Upload | Manage ─────────────────────────────────────
    tab_up, tab_mgr = st.tabs(["⬆️  Upload", "📂  Manage"])

    # ══════════════════════════════════════
    # UPLOAD TAB
    # ══════════════════════════════════════
    with tab_up:
        st.markdown("**Drop files or click to browse**")
        st.caption("PDF · TXT · MD · DOCX  —  multiple files OK")

        uploaded = st.file_uploader(
            "Documents",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # Preview selected files
        if uploaded:
            for f in uploaded:
                kb = len(f.getvalue()) / 1024
                icon = FILE_ICONS.get(Path(f.name).suffix.lstrip("."), "📄")
                st.markdown(
                    f'<span class="metric-pill">{icon} {f.name} &nbsp;{kb:.0f} KB</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

        index_btn = st.button(
            "⚡ Index & Analyse",
            type="primary",
            use_container_width=True,
            disabled=not uploaded,
        )

        # ── Indexing logic ────────────────────────────────────────
        if index_btn and uploaded:
            new_files, skipped = [], []

            for f in uploaded:
                data = f.getvalue()
                h    = file_hash(data)
                if h in st.session_state.seen_hashes or f.name in registry:
                    skipped.append(f.name)
                else:
                    new_files.append((f, data, h))

            if skipped:
                st.info(f"Already indexed — skipped: {', '.join(skipped)}")

            if new_files:
                total_chunks = 0
                progress_bar = st.progress(0, text="Preparing…")
                status       = st.empty()

                for idx, (f, data, h) in enumerate(new_files):
                    fname  = f.name
                    suffix = Path(fname).suffix.lower()

                    # Step 1 — Load
                    progress_bar.progress(
                        int(idx / len(new_files) * 100),
                        text=f"[{idx+1}/{len(new_files)}] Loading {fname}…"
                    )
                    status.markdown(f"📂 **Reading** `{fname}`…")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(data)
                        tmp_path = Path(tmp.name)

                    try:
                        docs = _load_single_file(tmp_path, source_name=fname)
                    finally:
                        tmp_path.unlink(missing_ok=True)

                    if not docs:
                        st.warning(f"⚠️ Could not read `{fname}` — skipped.")
                        continue

                    # Step 2 — Chunk
                    status.markdown(f"✂️ **Chunking** `{fname}` ({len(docs)} sections)…")
                    chunks = split_documents(docs)

                    # Step 3 — Embed & store
                    status.markdown(
                        f"🧠 **Embedding** `{fname}` ({len(chunks)} chunks)…"
                    )
                    embed_and_store(chunks)

                    # Register
                    register_file(fname, len(docs), len(chunks), len(data) / 1024)
                    st.session_state.seen_hashes.add(h)
                    total_chunks += len(chunks)

                progress_bar.progress(100, text="Done!")
                status.empty()

                rebuild_chain()
                st.success(
                    f"✅ Indexed **{len(new_files)}** file(s) · "
                    f"**{total_chunks}** chunks added."
                )
                st.balloons()
                st.rerun()

    # ══════════════════════════════════════
    # MANAGE TAB
    # ══════════════════════════════════════
    with tab_mgr:
        registry = load_registry()
        if not registry:
            st.info("No documents indexed yet.")
        else:
            st.markdown(f"**{len(registry)} document(s) in the database:**")
            for fname, meta in list(registry.items()):
                ext  = Path(fname).suffix.lstrip(".").lower()
                icon = FILE_ICONS.get(ext, "📄")
                with st.expander(f"{icon} {fname}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pages", meta.get("pages", "?"))
                    c2.metric("Chunks", meta.get("chunks", "?"))
                    c3.metric("KB", meta.get("size_kb", "?"))
                    st.caption(f"Indexed: {meta.get('indexed_at','?')}")

                    if st.button(
                        "🗑️ Remove from registry",
                        key=f"del_{fname}",
                        use_container_width=True,
                    ):
                        unregister_file(fname)
                        st.warning(
                            f"Removed `{fname}` from registry.\n\n"
                            "_To remove its vectors too, use Reset DB below._"
                        )
                        st.rerun()

            st.divider()
            if st.button(
                "🔴 Reset Entire Database",
                use_container_width=True,
                help="Deletes ALL vectors and the registry. Re-upload to start again.",
            ):
                delete_vector_store()
                save_registry({})
                st.session_state.update({
                    "chain": None, "db_count": 0,
                    "messages": [], "seen_hashes": set(),
                    "preview_text": None, "preview_name": None,
                })
                st.success("Database wiped. Upload new documents to begin.")
                st.rerun()

    # ── Bottom controls ───────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if vector_store_exists():
            rebuild_chain()
        st.rerun()

    st.caption("🔒 100% local · no API keys · no cost")
    st.caption(f"LLM: `{LLM_MODEL}` · Embed: `{EMBEDDING_MODEL}`")


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════
registry  = load_registry()
has_docs  = bool(registry)

if has_docs:
    chat_col, doc_col = st.columns([3, 2], gap="large")
else:
    chat_col = st.container()
    doc_col  = None


# ══════════════════════════════════════
# CHAT COLUMN
# ══════════════════════════════════════
with chat_col:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption(
        "Ask questions about your uploaded documents — "
        "**Gemma3:4b** + **ChromaDB** · 100% free · fully local"
    )

    # Welcome screen when no docs yet
    if not has_docs:
        st.markdown("---")
        st.markdown("""
### 👈 Get started in 3 easy steps

| Step | Action |
|------|--------|
| 1️⃣  Upload | Open the **Upload tab** in the sidebar — drag your PDF, TXT, MD, or DOCX files |
| 2️⃣  Index  | Click **Index & Analyse** — files are processed locally on your PC |
| 3️⃣  Ask    | Type any question below and get cited answers from your documents |

> 🔒 Everything runs on your machine. No internet required after initial model download.
        """)
        st.stop()

    # ── Render message history ────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources & excerpts", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"- 📄 `{src}`")
                    chunks = msg.get("chunks", [])
                    if chunks:
                        st.caption("Relevant excerpts from retrieved chunks:")
                        for chunk in chunks[:2]:
                            text = chunk.page_content
                            st.markdown(
                                f"> _{text[:300]}{'…' if len(text) > 300 else ''}_"
                            )

    # ── Suggested questions (first visit) ────────────────────────
    if not st.session_state.messages:
        st.markdown("**💡 Suggested questions to get started:**")
        suggestions = [
            "Summarise all the uploaded documents",
            "What are the key points in the documents?",
            "What rules or policies are mentioned?",
            "List the main topics covered",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state["_pending"] = s
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────
    pending = st.session_state.pop("_pending", None)
    user_input = st.chat_input(
        "Ask anything about your documents…",
        disabled=st.session_state.chain is None,
    )
    prompt = pending or user_input

    if prompt:
        if st.session_state.chain is None:
            st.warning("Index your documents first (sidebar → Upload tab).")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                result = ask(st.session_state.chain, prompt)

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("📚 Sources & excerpts", expanded=False):
                    for src in result["sources"]:
                        st.markdown(f"- 📄 `{src}`")
                    for chunk in result["chunks"][:2]:
                        text = chunk.page_content
                        st.markdown(
                            f"> _{text[:300]}{'…' if len(text) > 300 else ''}_"
                        )

        st.session_state.messages.append({
            "role":    "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "chunks":  result["chunks"],
        })


# ══════════════════════════════════════
# DOCUMENT BROWSER COLUMN
# ══════════════════════════════════════
if doc_col:
    with doc_col:
        st.markdown("### 📂 Indexed Documents")

        # Summary metrics
        m1, m2 = st.columns(2)
        m1.metric("Total Files", len(registry))
        m2.metric("Total Chunks", st.session_state.db_count)

        st.markdown("---")

        # Document cards
        for fname, meta in registry.items():
            ext  = Path(fname).suffix.lstrip(".").lower()
            icon = FILE_ICONS.get(ext, "📄")

            st.markdown(
                f"""<div class="doc-card">
                <b>{icon} {fname}</b><br>
                <small>
                  {meta.get('pages','?')} pages &nbsp;·&nbsp;
                  {meta.get('chunks','?')} chunks &nbsp;·&nbsp;
                  {meta.get('size_kb','?')} KB<br>
                  Indexed: {meta.get('indexed_at','?')}
                </small>
                </div>""",
                unsafe_allow_html=True,
            )

            btn1, btn2 = st.columns(2)

            # Ask about this file
            if btn1.button("💬 Ask about", key=f"ask_{fname}", use_container_width=True):
                st.session_state["_pending"] = f"Summarise the document: {fname}"
                st.rerun()

            # Preview chunks
            if btn2.button("👁️ Preview", key=f"prev_{fname}", use_container_width=True):
                try:
                    vs     = load_vector_store()
                    hits   = vs.similarity_search(fname, k=4)
                    chunks = [
                        c.page_content for c in hits
                        if c.metadata.get("source") == fname
                    ]
                    text = "\n\n---\n\n".join(chunks) if chunks else \
                           "\n\n---\n\n".join(c.page_content for c in hits[:3])
                    st.session_state.preview_name = fname
                    st.session_state.preview_text = text[:MAX_PREVIEW_CHARS]
                except Exception as e:
                    st.session_state.preview_text = f"Preview error: {e}"
                    st.session_state.preview_name = fname
                st.rerun()

        # ── Preview panel ─────────────────────────────────────────
        if st.session_state.preview_text:
            st.markdown("---")
            st.markdown(f"#### 👁️ Preview — `{st.session_state.preview_name}`")
            safe_text = (
                st.session_state.preview_text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            st.markdown(
                f'<div class="preview-box">{safe_text}</div>',
                unsafe_allow_html=True,
            )
            if st.button("✖ Close preview", use_container_width=True):
                st.session_state.preview_text = None
                st.session_state.preview_name = None
                st.rerun()

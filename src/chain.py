# src/chain.py
"""
DAY 2 — Steps 2–6: Prompt template + LLM + memory → RAG chain.

Chain type: ConversationalRetrievalChain
  Unlike a simple RetrievalQA chain, this one:
    1. Rewrites follow-up questions to be self-contained
       ("What else did it say?" → "What else did the policy say about X?")
    2. Keeps a rolling conversation history so the LLM has context.

LLM: ChatOllama (Llama 3 running locally — FREE)
"""

from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
)
from src.logger import logger


def build_chain(retriever: VectorStoreRetriever) -> ConversationalRetrievalChain:
    """
    Assemble and return the full RAG chain.

    Args:
        retriever: Built by retriever.py.

    Returns:
        ConversationalRetrievalChain that accepts {"question": str}
        and returns {"answer": str, "source_documents": [...]}
    """

    # ── 1. Local LLM via Ollama ──────────────────────────────────────
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_MAX_TOKENS,
    )
    logger.info(f"LLM ready — model='{LLM_MODEL}', temp={LLM_TEMPERATURE}.")

    # ── 2. Prompt template ───────────────────────────────────────────
    # The {context} placeholder is filled by LangChain with the
    # retrieved chunk text.  The {question} placeholder is the user query.
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    # ── 3. Conversation memory ───────────────────────────────────────
    # output_key="answer" is required when return_source_documents=True
    # because the chain now returns a dict with multiple keys.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # ── 4. Wire into a single chain ──────────────────────────────────
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,   # enables citation display
        verbose=False,
    )
    logger.info("ConversationalRetrievalChain is ready.")
    return chain


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Send a question through the chain and return a structured result.

    Args:
        chain:    Built by build_chain().
        question: User's natural-language question.

    Returns:
        {
          "answer":   str,        — LLM answer text
          "sources":  List[str],  — unique source file names
          "chunks":   List[Document] — raw retrieved chunks
        }
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": [], "chunks": []}

    result = chain.invoke({"question": question})

    answer = result.get("answer", "")
    raw_docs = result.get("source_documents", [])

    # Deduplicate source names for the citation panel
    sources = sorted({
        doc.metadata.get("source", "unknown") for doc in raw_docs
    })

    logger.debug(f"Q: {question!r} → {len(raw_docs)} chunks, sources={sources}")
    return {"answer": answer, "sources": sources, "chunks": raw_docs}

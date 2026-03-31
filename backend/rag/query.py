# query.py
# Load the ChromaDB vector store and answer questions.
# run_query()          → returns a complete dict (used by non-streaming callers)
# run_query_stream()   → generator that yields Server-Sent Event strings
# run_query_for_eval() → evaluation helper that accepts temperature / top_p / chroma_path

import sys
import os
import json
import logging

from chromadb import PersistentClient
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.prompts import PromptTemplate

# Import the singleton embed model from ingest so the model is
# loaded exactly once across the entire process.
from ingest import get_embed_model

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CHROMA_PATH       = os.environ.get("CHROMA_PATH", "chroma")
CHROMA_COLLECTION = "studentcompass"

# ─────────────────────────────────────────────
# LLM setup
# ─────────────────────────────────────────────
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# ─────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    "You are a helpful student advisor for a university. "
    "Answer the student's question using ONLY the context provided below. "
    "If the answer is not in the context, say you don't have enough information "
    "to answer that question.\n\n"
    "Context:\n{context_str}\n\n"
    "Student Question: {query_str}\n\n"
    "Answer:"
)

# ─────────────────────────────────────────────
# Shared index builder
# ─────────────────────────────────────────────
def _build_index(chroma_path: str = CHROMA_PATH, collection_name: str = CHROMA_COLLECTION):
    """Connect to Chroma and return a loaded VectorStoreIndex, or None if empty."""
    chroma_client     = PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    if chroma_collection.count() == 0:
        return None, chroma_collection

    vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=get_embed_model(),   # singleton — no disk load
    )
    return index, chroma_collection


def _build_sources(response) -> list:
    """Deduplicate and format source nodes from a query response."""
    seen: set    = set()
    sources: list = []

    for node in response.source_nodes:
        meta = node.node.metadata
        source = (
            meta.get("original_filename")
            or meta.get("file_path")
            or meta.get("url")
            or meta.get("source")
            or meta.get("blob_name")
            or "Unknown source"
        )
        if source in seen:
            continue
        seen.add(source)

        summary = meta.get("summary")
        sources.append({
            "source":    source,
            "doc_type":  meta.get("doc_type", "unknown"),
            "summary":   summary[:200] + "…" if summary and len(summary) > 200 else summary,
            "blob_name": meta.get("blob_name"),
        })

    return sources


# ─────────────────────────────────────────────
# Non-streaming query (used by CLI / sync callers)
# ─────────────────────────────────────────────
def run_query(question: str) -> dict:
    """
    Query ChromaDB and return a complete structured response.

    Returns
    -------
    { "answer": str, "sources": [ { source, doc_type, summary, blob_name }, … ] }
    """
    if not question or not question.strip():
        return {"answer": "Please provide a question.", "sources": []}

    index, collection = _build_index()

    if index is None:
        return {
            "answer": (
                "The knowledge base is empty. "
                "Please ask an administrator to upload documents."
            ),
            "sources": [],
        }

    query_engine = index.as_query_engine(
        response_mode="compact",
        text_qa_template=QA_PROMPT,
        similarity_top_k=5,
        use_async=False,
        streaming=False,
    )

    response = query_engine.query(question)

    return {
        "answer":  str(response).strip(),
        "sources": _build_sources(response),
    }


# ─────────────────────────────────────────────
# Streaming query — yields Server-Sent Events
# ─────────────────────────────────────────────
def run_query_stream(question: str):
    """
    Generator that streams the LLM answer token-by-token as Server-Sent Events,
    then emits a final 'sources' event.

    SSE format
    ----------
    data: {"type": "token",   "value": "…chunk of text…"}
    data: {"type": "sources", "value": [ … source list … ]}
    data: {"type": "done"}

    Usage in Flask
    --------------
    return Response(run_query_stream(question), mimetype="text/event-stream")
    """
    if not question or not question.strip():
        yield f"data: {json.dumps({'type': 'error', 'value': 'Please provide a question.'})}\n\n"
        return

    index, _ = _build_index()

    if index is None:
        yield (
            f"data: {json.dumps({'type': 'error', 'value': 'The knowledge base is empty.'})}\n\n"
        )
        return

    # Build a streaming query engine
    query_engine = index.as_query_engine(
        response_mode="compact",
        text_qa_template=QA_PROMPT,
        similarity_top_k=5,
        use_async=False,
        streaming=True,        # ← key difference
    )

    try:
        streaming_response = query_engine.query(question)

        # Stream answer tokens as they arrive from Gemini
        for token in streaming_response.response_gen:
            payload = json.dumps({"type": "token", "value": token})
            yield f"data: {payload}\n\n"

        # After the full answer, emit sources
        sources  = _build_sources(streaming_response)
        payload  = json.dumps({"type": "sources", "value": sources})
        yield f"data: {payload}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        logger.error("Streaming query failed: %s", exc)
        payload = json.dumps({"type": "error", "value": str(exc)})
        yield f"data: {payload}\n\n"


# ─────────────────────────────────────────────
# Evaluation query
# Used exclusively by the test pipeline.
# Accepts temperature, top_p, top_k, and a
# custom chroma_path so production is untouched.
# ─────────────────────────────────────────────
def run_query_for_eval(
    question:    str,
    top_k:       int   = 3,
    temperature: float = 0.7,
    top_p:       float = 0.9,
    chroma_path: str   = "rag/chroma_test",
) -> str:
    """
    Run a single query against the *test* ChromaDB using the given
    generation parameters.  Returns the answer string.

    Parameters
    ----------
    question    : The question to ask.
    top_k       : Number of context chunks to retrieve.
    temperature : LLM sampling temperature (0.0 = deterministic, 1.0 = creative).
    top_p       : Nucleus sampling probability threshold.
    chroma_path : Path to the test-only ChromaDB instance.
    """
    if not question or not question.strip():
        return ""

    # Build a fresh LLM with the requested generation parameters.
    # top_p is passed via additional_kwargs for Google GenAI compatibility.
    try:
        llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            additional_kwargs={"top_p": top_p},
        )
    except TypeError:
        # Fallback: older llama-index versions may not accept additional_kwargs
        llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    index, chroma_collection = _build_index(
        chroma_path=chroma_path,
        collection_name="studentcompass_test",
    )

    if index is None:
        logger.warning(
            "run_query_for_eval: test Chroma at %s is empty.", chroma_path
        )
        return ""

    query_engine = index.as_query_engine(
        response_mode="compact",
        text_qa_template=QA_PROMPT,
        similarity_top_k=top_k,
        llm=llm,
        use_async=False,
        streaming=False,
    )

    try:
        response = query_engine.query(question)
        return str(response).strip()
    except Exception as exc:
        logger.error("run_query_for_eval error: %s", exc)
        return ""


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter your question: ")
    else:
        question = " ".join(sys.argv[1:])

    result = run_query(question)

    print("\nQuestion:", question)
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  • {s['source']}  [{s['doc_type']}]")
        if s.get("summary"):
            print(f"    {s['summary']}")

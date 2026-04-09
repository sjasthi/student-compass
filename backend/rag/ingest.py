# ingest.py
# Ingest documents into ChromaDB from local files, URLs, or GCS.
# Supports per-blob ingestion, full GCS ↔ Chroma sync,
# and a GCS-based test ingestion path used by the evaluation pipeline.

import os
import shutil
import tempfile
import logging
import threading
from typing import Optional

from chromadb import PersistentClient
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GCS_BUCKET_NAME   = os.environ.get("GCS_BUCKET_NAME", "your-bucket-name")
CHROMA_PATH       = os.environ.get("CHROMA_PATH", "chroma")
CHROMA_COLLECTION = "studentcompass"
SUPPORTED_EXTS    = {".pdf", ".txt", ".docx", ".doc", ".md"}

CONCEPTUAL_TYPES = {
    "admissions", "financial_aid", "graduation",
    "policies", "registration", "student_support", "tuition_fees",
}

# Lock to serialise concurrent Chroma writes (e.g. background threads)
_chroma_lock = threading.Lock()

# ─────────────────────────────────────────────
# LLM setup
# ─────────────────────────────────────────────
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# ─────────────────────────────────────────────
# Singleton embedding model
# Loaded once when the module is first imported —
# shared by every ingestion call so the model is
# never loaded from disk more than once per process.
# ─────────────────────────────────────────────
logger.info("Loading embedding model (one-time)…")
_EMBED_MODEL = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logger.info("Embedding model ready.")


def get_embed_model() -> HuggingFaceEmbedding:
    """Return the process-wide singleton embedding model."""
    return _EMBED_MODEL


# ─────────────────────────────────────────────
# Chroma helpers
# ─────────────────────────────────────────────
def _get_chroma_collection(chroma_path: str = CHROMA_PATH, collection: str = CHROMA_COLLECTION):
    client = PersistentClient(path=chroma_path)
    return client.get_or_create_collection(collection)


def get_ingested_blob_names() -> set:
    """Return the set of GCS blob_names whose chunks are stored in Chroma."""
    collection = _get_chroma_collection()
    results    = collection.get(include=["metadatas"])
    names: set = set()
    for meta in results.get("metadatas") or []:
        if meta and "blob_name" in meta:
            names.add(meta["blob_name"])
    return names


def remove_blob_from_chroma(blob_name: str) -> int:
    """
    Delete every Chroma node that belongs to *blob_name*.
    Returns the number of deleted nodes.
    """
    with _chroma_lock:
        collection = _get_chroma_collection()
        results    = collection.get(where={"blob_name": {"$eq": blob_name}})
        ids        = results.get("ids") or []
        if ids:
            collection.delete(ids=ids)
            logger.info("Removed %d nodes for blob: %s", len(ids), blob_name)
        else:
            logger.info("No nodes found for blob: %s", blob_name)
        return len(ids)


# ─────────────────────────────────────────────
# Per-blob ingestion (production — GCS-sourced)
# ─────────────────────────────────────────────
def ingest_blob(
    blob_name:         str,
    file_bytes:        bytes,
    original_filename: str,
    doc_type:          Optional[str] = None,
) -> int:
    """
    Ingest a single file (given as raw bytes) into ChromaDB.

    Parameters
    ----------
    blob_name         : GCS object name stored as metadata so the blob can
                        later be removed from Chroma when deleted from GCS.
    file_bytes        : Raw file content.
    original_filename : e.g. "tuition_2024.pdf" – used to detect the file type.
    doc_type          : Optional category label (e.g. "tuition_fees").

    Returns
    -------
    Number of Chroma nodes added.
    """
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        logger.warning("Unsupported extension %s for %s – skipped.", ext, original_filename)
        return 0

    if not doc_type:
        doc_type = "general"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, original_filename)
        with open(tmp_path, "wb") as fh:
            fh.write(file_bytes)

        try:
            documents = SimpleDirectoryReader(
                input_dir=tmpdir,
                recursive=False,
                required_exts=[ext],
            ).load_data()
        except Exception as exc:
            logger.error("Failed to load %s: %s", original_filename, exc)
            return 0

    if not documents:
        logger.warning("No documents parsed from %s", original_filename)
        return 0

    for doc in documents:
        doc.metadata["blob_name"]         = blob_name
        doc.metadata["original_filename"] = original_filename
        doc.metadata["source_type"]       = "gcs"
        doc.metadata["doc_type"]          = doc_type

    if doc_type in CONCEPTUAL_TYPES:
        for doc in documents:
            try:
                prompt  = (
                    "Summarize this document in 4–6 sentences. "
                    "Focus on the main themes, categories, and purpose.\n\n"
                    f"Document:\n{doc.text[:3000]}"
                )
                summary = Settings.llm.complete(prompt).text.strip()
                doc.metadata["summary"] = summary
                logger.info("✓ Summary generated for %s (%s)", original_filename, doc_type)
            except Exception as exc:
                logger.warning("Summary generation failed for %s: %s", original_filename, exc)

    with _chroma_lock:
        chroma_client     = PersistentClient(path=CHROMA_PATH)
        chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
        before            = chroma_collection.count()

        vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        parser          = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=get_embed_model(),
            transformations=[parser],
            metadata_mode=MetadataMode.ALL,
        )

        added = chroma_collection.count() - before

    logger.info("Ingested %d nodes for %s (blob: %s)", added, original_filename, blob_name)
    return added


# ─────────────────────────────────────────────
# GCS test ingestion (evaluation pipeline)
# Downloads every active file from the GCS bucket
# and ingests them into a separate test-only Chroma
# path so the production index is never touched.
# ─────────────────────────────────────────────
def run_gcs_test_ingestion(
    chunk_size:  int = 1024,
    chroma_path: str = "chroma_test",
) -> int:
    """
    Download all active blobs from GCS and ingest them into a
    separate test ChromaDB using the given chunk size.

    Parameters
    ----------
    chunk_size  : Sentence-splitter chunk size to evaluate.
    chroma_path : Path for the test-only ChromaDB instance
                  (kept separate from the production 'chroma' folder).

    Returns
    -------
    Number of embeddings in the test collection after ingestion.
    """
    from google.cloud import storage as gcs_storage

    storage_client = gcs_storage.Client()
    bucket         = storage_client.bucket(GCS_BUCKET_NAME)

    logger.info(
        "GCS test ingestion — chunk_size=%d, chroma_path=%s, bucket=%s",
        chunk_size, chroma_path, GCS_BUCKET_NAME,
    )

    # ── Collect all active blobs ─────────────────────────────────
    active_blobs = []
    for blob in bucket.list_blobs(prefix="uploads/"):
        blob.reload()
        meta = blob.metadata or {}
        if meta.get("status", "active") == "inactive":
            continue
        active_blobs.append(blob)

    if not active_blobs:
        raise ValueError(
            "No active files found in the GCS bucket. "
            "Please upload documents via the Admin page before running a test."
        )

    logger.info("Found %d active blobs in GCS.", len(active_blobs))

    # ── Download each blob into a temp dir and parse ──────────────
    all_documents = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for blob in active_blobs:
            meta              = blob.metadata or {}
            original_filename = meta.get("original_filename", blob.name.split("/")[-1])
            doc_type          = meta.get("doc_type", "general")
            ext               = os.path.splitext(original_filename)[1].lower()

            if ext not in SUPPORTED_EXTS:
                logger.warning("Skipping unsupported file type: %s", original_filename)
                continue

            tmp_path = os.path.join(tmpdir, original_filename)
            try:
                blob.download_to_filename(tmp_path)
            except Exception as exc:
                logger.error("Failed to download %s: %s", blob.name, exc)
                continue

            try:
                docs = SimpleDirectoryReader(
                    input_files=[tmp_path],
                ).load_data()
            except Exception as exc:
                logger.error("Failed to parse %s: %s", original_filename, exc)
                continue

            for doc in docs:
                doc.metadata["blob_name"]         = blob.name
                doc.metadata["original_filename"] = original_filename
                doc.metadata["source_type"]       = "gcs"
                doc.metadata["doc_type"]          = doc_type
                # Flatten — Chroma requires scalar values only
                doc.metadata = {
                    k: v for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float)) or v is None
                }

            all_documents.extend(docs)
            logger.info("Parsed %d doc(s) from %s", len(docs), original_filename)

    if not all_documents:
        raise ValueError(
            "No documents could be parsed from the GCS bucket files. "
            "Check that supported file types (PDF, DOCX, TXT, MD) are uploaded."
        )

    logger.info("Total document objects ready for test ingestion: %d", len(all_documents))

    # ── Build test ChromaDB ───────────────────────────────────────
    chroma_client     = PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection("studentcompass_test")

    vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    parser          = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=100)

    VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        embed_model=get_embed_model(),
        transformations=[parser],
        metadata_mode=MetadataMode.ALL,
    )

    count = chroma_collection.count()
    logger.info(
        "GCS test ingestion complete — chunk_size=%d → %d embeddings in %s",
        chunk_size, count, chroma_path,
    )
    return count


# ─────────────────────────────────────────────
# Full GCS ↔ Chroma sync
# ─────────────────────────────────────────────
def sync_with_gcs() -> dict:
    """
    Reconcile ChromaDB with the current state of the GCS bucket.

    • Files in GCS but not in Chroma  →  ingested.
    • Files in Chroma but not in GCS  →  removed from Chroma.
    """
    from google.cloud import storage as gcs_storage

    storage_client = gcs_storage.Client()
    bucket         = storage_client.bucket(GCS_BUCKET_NAME)

    gcs_files: dict = {}
    for blob in bucket.list_blobs(prefix="uploads/"):
        blob.reload()
        meta = blob.metadata or {}
        if meta.get("status", "active") == "inactive":
            continue
        gcs_files[blob.name] = {
            "original_filename": meta.get("original_filename", blob.name.split("/")[-1]),
            "doc_type":          meta.get("doc_type", "general"),
            "blob":              blob,
        }

    chroma_blobs = get_ingested_blob_names()
    gcs_blob_set = set(gcs_files.keys())
    to_add       = gcs_blob_set - chroma_blobs
    to_remove    = chroma_blobs - gcs_blob_set

    added_nodes   = 0
    removed_nodes = 0
    errors: list  = []

    for blob_name in to_remove:
        try:
            removed_nodes += remove_blob_from_chroma(blob_name)
        except Exception as exc:
            errors.append(f"remove:{blob_name} – {exc}")
            logger.error("Failed to remove %s: %s", blob_name, exc)

    for blob_name in to_add:
        try:
            info        = gcs_files[blob_name]
            file_bytes  = info["blob"].download_as_bytes()
            added_nodes += ingest_blob(
                blob_name,
                file_bytes,
                info["original_filename"],
                info.get("doc_type"),
            )
        except Exception as exc:
            errors.append(f"ingest:{blob_name} – {exc}")
            logger.error("Failed to ingest %s: %s", blob_name, exc)

    summary = {
        "files_added":   len(to_add),
        "files_removed": len(to_remove),
        "nodes_added":   added_nodes,
        "nodes_removed": removed_nodes,
        "errors":        errors,
    }
    logger.info("GCS sync complete: %s", summary)
    return summary


# ─────────────────────────────────────────────
# Legacy full local ingestion (kept for CLI use)
# ─────────────────────────────────────────────
def run_ingestion():
    """Ingest local data/ folder + hardcoded URLs, then run a full GCS sync."""
    documents = []

    data_dir = "data"
    if os.path.exists(data_dir):
        local_docs = SimpleDirectoryReader(
            input_dir=data_dir,
            recursive=True,
            required_exts=[".txt", ".pdf", ".docx"],
        ).load_data()
        documents.extend(local_docs)
        logger.info("Loaded %d local documents.", len(local_docs))

    urls = [
        "https://www.metrostate.edu/about/mission",
        # add more URLs here
    ]
    if urls:
        try:
            url_docs = TrafilaturaWebReader().load_data(urls)
            documents.extend(url_docs)
            logger.info("Loaded %d URL documents.", len(url_docs))
        except Exception as exc:
            logger.warning("URL loading failed: %s", exc)

    for doc in documents:
        path = doc.metadata.get("file_path")
        if path is None:
            src = doc.metadata.get("url") or doc.metadata.get("source") or "unknown_url"
            doc.metadata["source_type"] = "url"
            doc.metadata["doc_type"]    = "web_page"
            doc.metadata["blob_name"]   = f"url:{src}"
        else:
            folder = os.path.basename(os.path.dirname(path))
            doc.metadata["source_type"] = "local_file"
            doc.metadata["doc_type"]    = folder
            doc.metadata["blob_name"]   = f"local:{path}"

    for doc in documents:
        if doc.metadata.get("doc_type") in CONCEPTUAL_TYPES:
            try:
                prompt  = (
                    "Summarize this document in 4–6 sentences. "
                    "Focus on main themes, categories, and purpose.\n\n"
                    f"Document:\n{doc.text[:3000]}"
                )
                summary = Settings.llm.complete(prompt).text.strip()
                doc.metadata["summary"] = summary
            except Exception as exc:
                logger.warning("Summary failed: %s", exc)

    if documents:
        with _chroma_lock:
            chroma_client     = PersistentClient(path=CHROMA_PATH)
            chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
            before            = chroma_collection.count()

            vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            parser          = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=get_embed_model(),
                transformations=[parser],
                metadata_mode=MetadataMode.ALL,
            )

            after = chroma_collection.count()
            logger.info(
                "Local ingestion complete. Nodes: %d → %d (+%d).",
                before, after, after - before,
            )

    try:
        result = sync_with_gcs()
        logger.info("GCS sync after local ingestion: %s", result)
    except Exception as exc:
        logger.warning("GCS sync skipped (no GCS credentials?): %s", exc)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_gcs_test_ingestion(chunk_size=int(sys.argv[1]))
    else:
        run_ingestion()
# gcs_upload.py
# Flask backend that:
#   • Uploads files / URLs to Google Cloud Storage
#   • Automatically keeps ChromaDB in sync when files are added or removed
#   • Exposes /query         for a complete JSON response
#   • Exposes /query/stream  for a token-by-token Server-Sent Events stream
#   • Exposes /sync          for manual full re-synchronisation
#   • Exposes /test/run      for streaming evaluation across RAG, keyword, and prompt-only modes (SSE)
#   • Exposes /test/download for downloading results as CSV

import os
import io
import csv
import json
import shutil
import time
import uuid
import logging
import threading
import requests
import trafilatura

from datetime import timedelta
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from google.cloud import storage
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ingest import ingest_blob, remove_blob_from_chroma, sync_with_gcs, run_gcs_test_ingestion
from query import (
    run_query,
    run_query_stream,
    run_query_for_eval,
    run_keyword_search_for_eval,
    run_prompt_only_for_eval,
)


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GCS_BUCKET_NAME    = os.environ.get("GCS_BUCKET_NAME", "your-bucket-name")
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "md"}
MAX_FILE_SIZE_MB   = 50

# Evaluation / test configuration
TEST_CHROMA_PATH    = os.environ.get("TEST_CHROMA_PATH",    "rag/chroma_test")
GOLD_QUESTIONS_PATH = os.environ.get("GOLD_QUESTIONS_PATH", "gold_questions.json")

# Valid evaluation modes
VALID_MODES = {"rag", "keyword", "prompt_only"}

# ─────────────────────────────────────────────
# GCS Client
# ─────────────────────────────────────────────
storage_client = storage.Client()
bucket         = storage_client.bucket(GCS_BUCKET_NAME)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_blob_name(filename: str) -> str:
    return f"uploads/{uuid.uuid4().hex}_{secure_filename(filename)}"


def delete_old_versions(original_filename: str):
    """Permanently delete older blobs with the same original filename."""
    for blob in list(bucket.list_blobs(prefix="uploads/")):
        blob.reload()
        if (blob.metadata or {}).get("original_filename") == original_filename:
            blob.delete()


# ─────────────────────────────────────────────
# Background ingestion helpers
# ─────────────────────────────────────────────
def _background_ingest(blob_name: str, original_filename: str, doc_type: str = "general"):
    try:
        logger.info("Background ingestion starting for %s", blob_name)
        file_bytes = bucket.blob(blob_name).download_as_bytes()
        count      = ingest_blob(blob_name, file_bytes, original_filename, doc_type)
        logger.info("Background ingestion complete: %d nodes for %s", count, blob_name)
    except Exception as exc:
        logger.error("Background ingestion failed for %s: %s", blob_name, exc)


def _background_remove(blob_name: str):
    try:
        removed = remove_blob_from_chroma(blob_name)
        logger.info("Removed %d nodes for %s", removed, blob_name)
    except Exception as exc:
        logger.error("Background removal failed for %s: %s", blob_name, exc)


def _trigger_background(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()


# ─────────────────────────────────────────────
# Route 1: Upload a File
# ─────────────────────────────────────────────
@app.route("/upload/file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file        = request.files["file"]
    replace_old = request.form.get("replaceOld", "true").lower() == "true"
    doc_type    = request.form.get("docType", "general")

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Permitted: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    file.seek(0, os.SEEK_END)
    if file.tell() / (1024 * 1024) > MAX_FILE_SIZE_MB:
        return jsonify({"error": f"File exceeds {MAX_FILE_SIZE_MB} MB limit"}), 400
    file.seek(0)

    try:
        if replace_old:
            delete_old_versions(file.filename)

        blob_name = generate_blob_name(file.filename)
        blob      = bucket.blob(blob_name)
        blob.upload_from_file(file.stream, content_type=file.content_type or "application/octet-stream", rewind=True)
        blob.metadata = {
            "original_filename": file.filename,
            "replace_old": str(replace_old),
            "doc_type":    doc_type,
            "status":      "active",
            "source":      "file_upload",
        }
        blob.patch()

        _trigger_background(_background_ingest, blob_name, file.filename, doc_type)

        return jsonify({
            "message":           "File uploaded. Indexing started in the background.",
            "blob_name":         blob_name,
            "original_filename": file.filename,
            "doc_type":          doc_type,
            "replace_old":       replace_old,
        }), 200

    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 500


# ─────────────────────────────────────────────
# Route 2: Upload from URL
# ─────────────────────────────────────────────
@app.route("/upload/url", methods=["POST"])
def upload_from_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url         = data["url"].strip()
    replace_old = data.get("replaceOld", True)
    doc_type    = data.get("docType", "general")

    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL — must start with http:// or https://"}), 400

    MIME_TO_EXT = {
        "application/pdf":    ".pdf",
        "text/plain":         ".txt",
        "text/markdown":      ".md",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }

    try:
        with requests.get(url, stream=True, timeout=15) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "application/octet-stream").split(";")[0].strip()
            raw_name = url.split("/")[-1].split("?")[0].strip().strip("/")
            is_html  = content_type in ("text/html", "application/xhtml+xml")

            if raw_name and "." in raw_name and allowed_file(raw_name) and not is_html:
                filename = raw_name
            elif is_html:
                html_bytes = resp.content
                extracted  = trafilatura.extract(
                    html_bytes,
                    include_links=False,
                    include_images=False,
                    include_tables=True,
                    favor_recall=True,
                )
                if not extracted or not extracted.strip():
                    return jsonify({
                        "error": (
                            "Could not extract readable text from this URL. "
                            "The page may require JavaScript or authentication."
                        )
                    }), 400

                base     = raw_name or url.split("/")[2].replace(".", "_") or "webpage"
                base     = base.rsplit(".", 1)[0] if "." in base else base
                filename = f"{base}.txt"

                blob_name = generate_blob_name(filename)
                if replace_old:
                    delete_old_versions(filename)
                blob = bucket.blob(blob_name)
                blob.upload_from_string(extracted.encode("utf-8"), content_type="text/plain")
                blob.metadata = {
                    "original_filename": filename,
                    "replace_old": str(replace_old),
                    "doc_type":    doc_type,
                    "status":      "active",
                    "source_url":  url,
                    "source":      "url_upload",
                }
                blob.patch()

                _trigger_background(_background_ingest, blob_name, filename, doc_type)

                return jsonify({
                    "message":           "URL uploaded. Indexing started in the background.",
                    "blob_name":         blob_name,
                    "original_filename": filename,
                    "source_url":        url,
                    "doc_type":          doc_type,
                    "replace_old":       replace_old,
                }), 200
            else:
                ext = MIME_TO_EXT.get(content_type)
                if ext is None:
                    return jsonify({
                        "error": (
                            f"Could not determine a supported file type from this URL. "
                            f"Content-Type was '{content_type}'. "
                            f"Permitted types: {', '.join(ALLOWED_EXTENSIONS)}"
                        )
                    }), 400
                base     = raw_name or url.split("/")[2].replace(".", "_")
                filename = f"{base}{ext}"

            if not allowed_file(filename):
                return jsonify({"error": f"File type not allowed. Permitted: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

            if replace_old:
                delete_old_versions(filename)

            blob_name = generate_blob_name(filename)
            blob      = bucket.blob(blob_name)
            blob.upload_from_file(resp.raw, content_type=content_type, rewind=False)
            blob.metadata = {
                "original_filename": filename,
                "replace_old": str(replace_old),
                "doc_type":    doc_type,
                "status":      "active",
                "source_url":  url,
                "source":      "url_upload",
            }
            blob.patch()

        _trigger_background(_background_ingest, blob_name, filename, doc_type)

        return jsonify({
            "message":           "URL uploaded. Indexing started in the background.",
            "blob_name":         blob_name,
            "original_filename": filename,
            "source_url":        url,
            "doc_type":          doc_type,
            "replace_old":       replace_old,
        }), 200

    except requests.exceptions.RequestException as exc:
        return jsonify({"error": f"Failed to fetch URL: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 500


# ─────────────────────────────────────────────
# Route 3: List all active files
# ─────────────────────────────────────────────
@app.route("/files", methods=["GET"])
def list_files():
    try:
        file_list = []
        for blob in bucket.list_blobs(prefix="uploads/"):
            blob.reload()
            meta   = blob.metadata or {}
            status = meta.get("status", "active")
            if status == "inactive":
                continue
            file_list.append({
                "blob_name":         blob.name,
                "original_filename": meta.get("original_filename", blob.name.split("/")[-1]),
                "doc_type":          meta.get("doc_type", "general"),
                "size_kb":           round(blob.size / 1024, 2),
                "content_type":      blob.content_type,
                "updated":           blob.updated.isoformat() if blob.updated else None,
                "source":            meta.get("source", "unknown"),
                "source_url":        meta.get("source_url"),
                "status":            status,
            })
        file_list.sort(key=lambda x: x["updated"] or "", reverse=True)
        return jsonify({"files": file_list, "count": len(file_list)}), 200
    except Exception as exc:
        return jsonify({"error": f"Could not list files: {exc}"}), 500


# ─────────────────────────────────────────────
# Route 4: Signed download URL
# ─────────────────────────────────────────────
@app.route("/download-url/<path:blob_name>", methods=["GET"])
def get_signed_url(blob_name):
    try:
        signed_url = bucket.blob(blob_name).generate_signed_url(
            version="v4", expiration=timedelta(minutes=30), method="GET"
        )
        return jsonify({"signed_url": signed_url}), 200
    except Exception as exc:
        return jsonify({"error": f"Could not generate signed URL: {exc}"}), 500


# ─────────────────────────────────────────────
# Route 5: Delete file (GCS + Chroma)
# ─────────────────────────────────────────────
@app.route("/files/<path:blob_name>", methods=["DELETE"])
def delete_file(blob_name):
    try:
        bucket.blob(blob_name).delete()
        _trigger_background(_background_remove, blob_name)
        return jsonify({"message": "File deleted. Removing from index in background.", "blob_name": blob_name}), 200
    except Exception as exc:
        return jsonify({"error": f"Could not delete file: {exc}"}), 500


# ─────────────────────────────────────────────
# Route 6: Manual full sync
# ─────────────────────────────────────────────
@app.route("/sync", methods=["POST"])
def sync_chroma():
    try:
        result = sync_with_gcs()
        return jsonify({"message": "Sync complete.", "details": result}), 200
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        return jsonify({"error": f"Sync failed: {exc}"}), 500


# ─────────────────────────────────────────────
# Route 7: Query — complete JSON response
# ─────────────────────────────────────────────
@app.route("/query", methods=["POST"])
def query_knowledge_base():
    """
    Request:  { "question": "..." }
    Response: { "answer": "...", "sources": [...] }
    """
    data = request.get_json()
    if not data or not data.get("question", "").strip():
        return jsonify({"error": "Request body must contain a non-empty 'question' field."}), 400

    try:
        history = data.get("history") or []
        result = run_query(data["question"].strip(), history=history)
        return jsonify(result), 200
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        return jsonify({"error": f"Query failed: {exc}"}), 500


# ─────────────────────────────────────────────
# Route 8: Query — streaming Server-Sent Events
# ─────────────────────────────────────────────
@app.route("/query/stream", methods=["POST"])
def query_stream():
    """
    Streams the answer token-by-token so the UI can render text as it arrives.

    Request:  { "question": "..." }
    Response: text/event-stream
      data: {"type": "token",   "value": "partial text"}
      data: {"type": "sources", "value": [...]}
      data: {"type": "done"}
      data: {"type": "error",   "value": "message"}   ← on failure
    """
    data = request.get_json()
    if not data or not data.get("question", "").strip():
        return jsonify({"error": "Request body must contain a non-empty 'question' field."}), 400

    question = data["question"].strip()
    history  = data.get("history") or []

    return Response(
        stream_with_context(run_query_stream(question, history=history)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────
# Route 9: Health check
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ═════════════════════════════════════════════
# Evaluation / Test endpoints
# ═════════════════════════════════════════════

# Lazy-loaded scoring model (avoids import cost at startup)
_scoring_model      = None
_scoring_model_lock = threading.Lock()


def _get_scoring_model():
    global _scoring_model
    with _scoring_model_lock:
        if _scoring_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading scoring model for evaluation…")
            _scoring_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Scoring model ready.")
    return _scoring_model


def _score_answer(gold: str, predicted: str) -> int:
    """
    Compare gold vs predicted answer using cosine similarity.
    Returns 0–3 score:  3 = perfect, 2 = good, 1 = partial, 0 = incorrect.
    """
    from sentence_transformers import util as st_util
    model    = _get_scoring_model()
    emb_gold = model.encode(gold,      convert_to_tensor=True)
    emb_pred = model.encode(predicted, convert_to_tensor=True)
    sim      = st_util.cos_sim(emb_gold, emb_pred).item()
    if   sim > 0.85: return 3
    elif sim > 0.70: return 2
    elif sim > 0.50: return 1
    else:            return 0


def _sse(event_type: str, value) -> str:
    """Format a single Server-Sent Event line."""
    return f"data: {json.dumps({'type': event_type, 'value': value})}\n\n"


# ─────────────────────────────────────────────
# Route 10: Run Evaluation — SSE stream
# ─────────────────────────────────────────────
@app.route("/test/run", methods=["POST"])
def run_evaluation():
    """
    Runs an accuracy evaluation over the gold questions file.
    Supports three modes: RAG, keyword search, and prompt-only LLM.
    """

    # ── Parse config ───────────────────────────────────────────────────
    config        = request.get_json(force=True) or {}
    chunk_sizes   = config.get("chunk_sizes",   [500])
    top_k_values  = config.get("top_k_values",  [3])
    temperatures  = config.get("temperatures",  [0.7])
    top_p_values  = config.get("top_p_values",  [0.9])
    num_questions = min(int(config.get("num_questions", 50)), 50)
    requested_modes = config.get("modes", ["rag"])
    modes = [m for m in requested_modes if m in VALID_MODES]
    if not modes:
        modes = ["rag"]

    # ── Parallel question evaluator ─────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def evaluate_single_question(item, top_k, temperature, top_p, chroma_path):
        t_start = time.perf_counter()
        try:
            answer = run_query_for_eval(
                item["question"],
                top_k=top_k,
                temperature=temperature,
                top_p=top_p,
                chroma_path=chroma_path,
            )
            score = _score_answer(item["gold_answer"], answer)
        except Exception as exc:
            logger.warning("RAG question failed: %s", exc)
            score = 0

        latency = round((time.perf_counter() - t_start) * 1000)
        return score, latency

    def evaluate_question_all_topk(item, top_k_values, temperature, top_p, chroma_path):
        """Runs all top_k values for a single question in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=len(top_k_values)) as executor:
            futures = {
                executor.submit(
                    evaluate_single_question,
                    item,
                    top_k,
                    temperature,
                    top_p,
                    chroma_path,
                ): top_k
                for top_k in top_k_values
            }

            for future in as_completed(futures):
                results.append(future.result())

        return results

    # ── SSE generator ───────────────────────────────────────────────────
    def generate():

        # ── Load gold questions ──────────────────────────────────────────
        try:
            with open(GOLD_QUESTIONS_PATH, encoding="utf-8") as f:
                gold_questions = json.load(f)["questions"][:num_questions]
        except FileNotFoundError:
            yield _sse("error", f"gold_questions.json not found at '{GOLD_QUESTIONS_PATH}'.")
            return
        except Exception as exc:
            yield _sse("error", f"Could not load gold_questions.json: {exc}")
            return

        yield _sse("progress", f"Loaded {len(gold_questions)} gold questions.")
        yield _sse("progress", f"Modes selected: {', '.join(modes)}")

        # ── Pre-warm scoring model ───────────────────────────────────────
        yield _sse("progress", "Loading scoring model…")
        try:
            _get_scoring_model()
        except Exception as exc:
            yield _sse("error", f"Could not load scoring model: {exc}")
            return
        yield _sse("progress", "Scoring model ready.")

        all_results            = []
        chroma_folders_created = []

        # ════════════════════════════════════════════════════════════════════
        # RAG MODE
        # ════════════════════════════════════════════════════════════════════
        if "rag" in modes:
            yield _sse("progress", "▶ Starting RAG evaluation…")
            total_rag = len(chunk_sizes) * len(temperatures) * len(top_p_values)
            run_num   = 0

            for chunk_size in chunk_sizes:
                chroma_path = f"{TEST_CHROMA_PATH}_{chunk_size}"
                chroma_folders_created.append(chroma_path)

                yield _sse("progress", f"  Building vector store — chunk_size={chunk_size}…")
                try:
                    count = run_gcs_test_ingestion(chunk_size, chroma_path)
                    yield _sse("progress", f"  Vector store ready: {count} embeddings.")
                except Exception as exc:
                    yield _sse("error", f"Ingestion failed for chunk_size={chunk_size}: {exc}")
                    continue

                for temperature in temperatures:
                    for top_p in top_p_values:
                        run_num += 1
                        label = f"[rag] chunk={chunk_size} temp={temperature} top_p={top_p} [{run_num}/{total_rag}]"
                        yield _sse("progress", f"  Evaluating — {label}")

                        scores = []
                        latencies = []

                        # ── Outer executor: parallelize questions ───────────────
                        with ThreadPoolExecutor(max_workers=10) as question_executor:
                            question_futures = []

                            for item in gold_questions:
                                # Each question runs all top_k values in parallel
                                question_futures.append(
                                    question_executor.submit(
                                        evaluate_question_all_topk,
                                        item,
                                        top_k_values,
                                        temperature,
                                        top_p,
                                        chroma_path,
                                    )
                                )

                            # Collect results as they finish
                            for idx, future in enumerate(as_completed(question_futures), start=1):
                                topk_results = future.result()

                                # Aggregate across all top_k values
                                for score, latency in topk_results:
                                    scores.append(score)
                                    latencies.append(latency)

                                if idx % 10 == 0 or idx == len(gold_questions):
                                    yield _sse("progress", f"    {idx}/{len(gold_questions)} questions evaluated…")

                        # ── Aggregate results ──────────────────────────────────
                        accuracy       = round(sum(scores) / len(scores), 2)
                        avg_latency_ms = round(sum(latencies) / len(latencies))
                        min_latency_ms = min(latencies)
                        max_latency_ms = max(latencies)

                        result = {
                            "mode":            "rag",
                            "chunk_size":      chunk_size,
                            "top_k_values":    top_k_values,
                            "temperature":     temperature,
                            "top_p":           top_p,
                            "accuracy":        accuracy,
                            "avg_latency_ms":  avg_latency_ms,
                            "min_latency_ms":  min_latency_ms,
                            "max_latency_ms":  max_latency_ms,
                            "total_questions": len(gold_questions),
                        }
                        all_results.append(result)
                        yield _sse("result", result)

        # ════════════════════════════════════════════════════════════════════
        # KEYWORD MODE
        # ════════════════════════════════════════════════════════════════════
        if "keyword" in modes:
            yield _sse("progress", "▶ Starting keyword search evaluation…")
            total_kw = len(chunk_sizes) * len(top_k_values)
            run_num  = 0

            for chunk_size in chunk_sizes:
                chroma_path = f"{TEST_CHROMA_PATH}_{chunk_size}"

                if chroma_path not in chroma_folders_created:
                    chroma_folders_created.append(chroma_path)
                    yield _sse("progress", f"  Building vector store — chunk_size={chunk_size}…")
                    try:
                        count = run_gcs_test_ingestion(chunk_size, chroma_path)
                        yield _sse("progress", f"  Vector store ready: {count} embeddings.")
                    except Exception as exc:
                        yield _sse("error", f"Ingestion failed for chunk_size={chunk_size}: {exc}")
                        continue

                for top_k in top_k_values:
                    run_num += 1
                    label = f"[keyword] chunk={chunk_size} top_k={top_k} [{run_num}/{total_kw}]"
                    yield _sse("progress", f"  Evaluating — {label}")

                    scores = []
                    latencies = []

                    # Sequential is fine here (keyword search is fast)
                    for i, item in enumerate(gold_questions):
                        t_start = time.perf_counter()
                        try:
                            answer = run_keyword_search_for_eval(
                                item["question"],
                                top_k=top_k,
                                chroma_path=chroma_path,
                            )
                            score = _score_answer(item["gold_answer"], answer)
                        except Exception as exc:
                            logger.warning("Keyword Q%d failed: %s", i, exc)
                            score = 0

                        latencies.append(round((time.perf_counter() - t_start) * 1000))
                        scores.append(score)

                        if (i + 1) % 10 == 0 or (i + 1) == len(gold_questions):
                            yield _sse("progress", f"    {i + 1}/{len(gold_questions)} questions evaluated…")

                    accuracy       = round(sum(scores) / len(scores), 2)
                    avg_latency_ms = round(sum(latencies) / len(latencies))
                    min_latency_ms = min(latencies)
                    max_latency_ms = max(latencies)

                    result = {
                        "mode":            "keyword",
                        "chunk_size":      chunk_size,
                        "top_k":           top_k,
                        "temperature":     None,
                        "top_p":           None,
                        "accuracy":        accuracy,
                        "avg_latency_ms":  avg_latency_ms,
                        "min_latency_ms":  min_latency_ms,
                        "max_latency_ms":  max_latency_ms,
                        "total_questions": len(scores),
                    }
                    all_results.append(result)
                    yield _sse("result", result)

        # ════════════════════════════════════════════════════════════════════
        # PROMPT-ONLY MODE
        # ════════════════════════════════════════════════════════════════════
        if "prompt_only" in modes:
            yield _sse("progress", "▶ Starting prompt-only evaluation…")
            total_po = len(temperatures) * len(top_p_values)
            run_num  = 0

            for temperature in temperatures:
                for top_p in top_p_values:
                    run_num += 1
                    label = f"[prompt_only] temp={temperature} top_p={top_p} [{run_num}/{total_po}]"
                    yield _sse("progress", f"  Evaluating — {label}")

                    scores = []
                    latencies = []

                    for i, item in enumerate(gold_questions):
                        t_start = time.perf_counter()
                        try:
                            answer = run_prompt_only_for_eval(
                                item["question"],
                                temperature=temperature,
                                top_p=top_p,
                            )
                            score = _score_answer(item["gold_answer"], answer)
                        except Exception as exc:
                            logger.warning("Prompt-only Q%d failed: %s", i, exc)
                            score = 0

                        latencies.append(round((time.perf_counter() - t_start) * 1000))
                        scores.append(score)

                        if (i + 1) % 10 == 0 or (i + 1) == len(gold_questions):
                            yield _sse("progress", f"    {i + 1}/{len(gold_questions)} questions evaluated…")

                    accuracy       = round(sum(scores) / len(scores), 2)
                    avg_latency_ms = round(sum(latencies) / len(latencies))
                    min_latency_ms = min(latencies)
                    max_latency_ms = max(latencies)

                    result = {
                        "mode":            "prompt_only",
                        "chunk_size":      None,
                        "top_k":           None,
                        "temperature":     temperature,
                        "top_p":           top_p,
                        "accuracy":        accuracy,
                        "avg_latency_ms":  avg_latency_ms,
                        "min_latency_ms":  min_latency_ms,
                        "max_latency_ms":  max_latency_ms,
                        "total_questions": len(scores),
                    }
                    all_results.append(result)
                    yield _sse("result", result)

        # ════════════════════════════════════════════════════════════════════
        # CLEANUP
        # ════════════════════════════════════════════════════════════════════
        yield _sse("progress", "✅ Evaluation complete.")
        for folder in chroma_folders_created:
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
            except Exception:
                pass

        yield _sse("done", all_results)

    # ── Return SSE stream ────────────────────────────────────────────────
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────
# Route 11: Download test results as CSV
# ─────────────────────────────────────────────
@app.route("/test/download", methods=["POST"])
def download_test_results():
    """
    Accepts an array of result objects and returns a downloadable CSV.
    Includes a 'mode' column so RAG, keyword, and prompt-only rows are
    clearly labelled in the exported file.

    Request JSON: [ { mode, chunk_size, top_k, temperature, top_p, accuracy, total_questions }, ... ]
    Response:     text/csv attachment
    """
    results = request.get_json(force=True) or []
    if not results:
        return jsonify({"error": "No results provided"}), 400

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["mode", "chunk_size", "top_k", "temperature", "top_p", "accuracy", "avg_latency_ms", "min_latency_ms", "max_latency_ms", "total_questions"],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(results)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=rag_comparison_results.csv"},
    )


# ─────────────────────────────────────────────
# Run server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=True is required for SSE — each streaming response
    # occupies its own thread so other requests are not blocked.
    app.run(debug=True, port=5000, threaded=True)

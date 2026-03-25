# gcs_upload.py
# Flask backend that:
#   • Uploads files / URLs to Google Cloud Storage
#   • Automatically keeps ChromaDB in sync when files are added or removed
#   • Exposes /query         for a complete JSON response
#   • Exposes /query/stream  for a token-by-token Server-Sent Events stream
#   • Exposes /sync          for manual full re-synchronisation

import os
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

from ingest import ingest_blob, remove_blob_from_chroma, sync_with_gcs
from query import run_query, run_query_stream

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

    # Map MIME types to file extensions for URLs with no extension in the path
    MIME_TO_EXT = {
        "application/pdf":    ".pdf",
        "text/plain":         ".txt",
        "text/markdown":      ".md",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        # text/html is handled separately via trafilatura — not listed here
    }

    try:
        with requests.get(url, stream=True, timeout=15) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "application/octet-stream").split(";")[0].strip()

            # Try to get a filename and extension from the URL path first
            raw_name = url.split("/")[-1].split("?")[0].strip().strip("/")

            is_html = content_type in ("text/html", "application/xhtml+xml")

            if raw_name and "." in raw_name and allowed_file(raw_name) and not is_html:
                # URL path already ends with a valid supported extension (non-HTML)
                filename = raw_name
            elif is_html:
                # For HTML pages, extract clean text via trafilatura and store as .txt
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
                # Strip any existing extension so we always save as .txt
                base     = base.rsplit(".", 1)[0] if "." in base else base
                filename = f"{base}.txt"

                if replace_old:
                    delete_old_versions(filename)

                blob_name    = generate_blob_name(filename)
                blob         = bucket.blob(blob_name)
                clean_bytes  = extracted.encode("utf-8")
                blob.upload_from_string(clean_bytes, content_type="text/plain; charset=utf-8")
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
                # Fall back to deriving the extension from the Content-Type header
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
        result = run_query(data["question"].strip())
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
    Streams the answer token-by-token so the UI can render text as it arrives
    rather than waiting for the full Gemini response.

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

    return Response(
        stream_with_context(run_query_stream(question)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


# ─────────────────────────────────────────────
# Route 9: Health check
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ─────────────────────────────────────────────
# Run server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=True is required for SSE — each streaming response
    # occupies its own thread so other requests are not blocked.
    app.run(debug=True, port=5000, threaded=True)

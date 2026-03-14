import os
import uuid
import requests
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend (localhost:3000)

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
    unique_id = uuid.uuid4().hex
    safe_name = secure_filename(filename)
    return f"uploads/{unique_id}_{safe_name}"


def mark_old_versions_inactive(original_filename: str):
    """Set older blobs with same original filename to inactive in metadata."""
    blobs = bucket.list_blobs(prefix="uploads/")
    for blob in blobs:
        blob.reload()
        meta = blob.metadata or {}
        if meta.get("original_filename") == original_filename and meta.get("status") != "inactive":
            meta["status"] = "inactive"
            blob.metadata = meta
            blob.patch()


# ─────────────────────────────────────────────
# Route 1: Upload a File
# ─────────────────────────────────────────────
@app.route("/upload/file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file        = request.files["file"]
    replace_old = request.form.get("replaceOld", "true").lower() == "true"

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Permitted: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if file_size_mb > MAX_FILE_SIZE_MB:
        return jsonify({"error": f"File exceeds {MAX_FILE_SIZE_MB}MB limit"}), 400

    try:
        # Optionally mark old versions inactive
        if replace_old:
            mark_old_versions_inactive(file.filename)

        blob_name    = generate_blob_name(file.filename)
        content_type = file.content_type or "application/octet-stream"
        blob         = bucket.blob(blob_name)

        blob.upload_from_file(file.stream, content_type=content_type, rewind=True)

        blob.metadata = {
            "original_filename": file.filename,
            "replace_old":       str(replace_old),
            "status":            "active",
            "source":            "file_upload"
        }
        blob.patch()

        return jsonify({
            "message":           "File uploaded successfully",
            "blob_name":         blob_name,
            "original_filename": file.filename,
            "replace_old":       replace_old
        }), 200

    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


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

    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL — must start with http:// or https://"}), 400

    try:
        with requests.get(url, stream=True, timeout=15) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "application/octet-stream")
            filename     = url.split("/")[-1].split("?")[0] or "downloaded_file"

            if not allowed_file(filename):
                return jsonify({"error": f"File type not allowed. Permitted: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

            # Optionally mark old versions inactive
            if replace_old:
                mark_old_versions_inactive(filename)

            blob_name = generate_blob_name(filename)
            blob      = bucket.blob(blob_name)

            blob.upload_from_file(response.raw, content_type=content_type, rewind=False)

            blob.metadata = {
                "original_filename": filename,
                "replace_old":       str(replace_old),
                "status":            "active",
                "source_url":        url,
                "source":            "url_upload"
            }
            blob.patch()

        return jsonify({
            "message":           "URL uploaded successfully",
            "blob_name":         blob_name,
            "original_filename": filename,
            "source_url":        url,
            "replace_old":       replace_old
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch URL: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# Route 3: List All Active Files in Bucket
# ─────────────────────────────────────────────
@app.route("/files", methods=["GET"])
def list_files():
    """Return all active files currently stored in the GCS bucket."""
    try:
        blobs     = bucket.list_blobs(prefix="uploads/")
        file_list = []

        for blob in blobs:
            blob.reload()
            meta   = blob.metadata or {}
            status = meta.get("status", "active")

            # Only return active files
            if status == "inactive":
                continue

            file_list.append({
                "blob_name":         blob.name,
                "original_filename": meta.get("original_filename", blob.name.split("/")[-1]),
                "size_kb":           round(blob.size / 1024, 2),
                "content_type":      blob.content_type,
                "updated":           blob.updated.isoformat() if blob.updated else None,
                "source":            meta.get("source", "unknown"),
                "source_url":        meta.get("source_url", None),
                "status":            status
            })

        # Sort newest first
        file_list.sort(key=lambda x: x["updated"] or "", reverse=True)

        return jsonify({"files": file_list, "count": len(file_list)}), 200

    except Exception as e:
        return jsonify({"error": f"Could not list files: {str(e)}"}), 500


# ─────────────────────────────────────────────
# Route 4: Generate Signed Download URL
# ─────────────────────────────────────────────
@app.route("/download-url/<path:blob_name>", methods=["GET"])
def get_signed_url(blob_name):
    try:
        blob       = bucket.blob(blob_name)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=30),
            method="GET"
        )
        return jsonify({"signed_url": signed_url}), 200

    except Exception as e:
        return jsonify({"error": f"Could not generate signed URL: {str(e)}"}), 500


# ─────────────────────────────────────────────
# Route 5: Delete (mark inactive) a file
# ─────────────────────────────────────────────
@app.route("/files/<path:blob_name>", methods=["DELETE"])
def delete_file(blob_name):
    try:
        blob = bucket.blob(blob_name)
        blob.reload()
        meta           = blob.metadata or {}
        meta["status"] = "inactive"
        blob.metadata  = meta
        blob.patch()

        return jsonify({"message": "File marked inactive", "blob_name": blob_name}), 200

    except Exception as e:
        return jsonify({"error": f"Could not delete file: {str(e)}"}), 500


# ─────────────────────────────────────────────
# Run server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)

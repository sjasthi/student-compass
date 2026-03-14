// src/api/uploadService.js
// Centralized API calls to the Python Flask backend

const API_URL = import.meta.env.VITE_APP_API_URL || "http://localhost:5000";

// ─────────────────────────────────────────────
// Upload a file (PDF, DOCX, TXT, MD)
// ─────────────────────────────────────────────
export async function uploadFile(file, replaceOld, onProgress) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("replaceOld", replaceOld.toString());

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    // Track upload progress
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && onProgress) {
        const percent = Math.round((event.loaded / event.total) * 100);
        onProgress(percent);
      }
    });

    xhr.addEventListener("load", () => {
      try {
        const data = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(data);
        } else {
          reject(new Error(data.error || "Upload failed"));
        }
      } catch {
        reject(new Error("Invalid response from server"));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
    xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));

    xhr.open("POST", `${API_URL}/upload/file`);
    xhr.send(formData);
  });
}

// ─────────────────────────────────────────────
// Upload from a URL
// ─────────────────────────────────────────────
export async function uploadFromUrl(url, replaceOld) {
  const response = await fetch(`${API_URL}/upload/url`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ url, replaceOld }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "URL upload failed");
  }

  return data;
}

// ─────────────────────────────────────────────
// Fetch list of active files in GCS
// ─────────────────────────────────────────────
export async function fetchFiles() {
  const response = await fetch(`${API_URL}/files`);
  const data     = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Failed to fetch file list");
  }

  return data.files;
}

// ─────────────────────────────────────────────
// Get a signed download URL for a file
// ─────────────────────────────────────────────
export async function getDownloadUrl(blobName) {
  const encoded  = encodeURIComponent(blobName);
  const response = await fetch(`${API_URL}/download-url/${encoded}`);
  const data     = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Failed to get download URL");
  }

  return data.signed_url;
}

// ─────────────────────────────────────────────
// Delete (mark inactive) a file
// ─────────────────────────────────────────────
export async function deleteFile(blobName) {
  const encoded  = encodeURIComponent(blobName);
  const response = await fetch(`${API_URL}/files/${encoded}`, { method: "DELETE" });
  const data     = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Failed to delete file");
  }

  return data;
}

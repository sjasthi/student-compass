// src/components/Admin.jsx
import { useState, useEffect, useCallback } from "react";
import { uploadFile, uploadFromUrl, fetchFiles } from "../api/uploadService";
import ProgressBar  from "../components/ProgressBar";
import Notification from "../components/Notification";
import FileList     from "../components/FileList";

export default function Admin() {
  // ── Upload form state ──────────────────────
  const [uploadType, setUploadType] = useState("file");
  const [file, setFile]             = useState(null);
  const [url, setUrl]               = useState("");
  const [replaceOld, setReplaceOld] = useState(true);

  // ── UI state ───────────────────────────────
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress]       = useState(null);
  const [notification, setNotification] = useState(null); // { message, type }

  // ── File list state ────────────────────────
  const [files, setFiles]         = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(false);

  // ── Load file list on mount ────────────────
  const loadFiles = useCallback(async () => {
    setLoadingFiles(true);
    try {
      const list = await fetchFiles();
      setFiles(list);
    } catch (err) {
      setNotification({ message: `Could not load files: ${err.message}`, type: "error" });
    } finally {
      setLoadingFiles(false);
    }
  }, []);

  useEffect(() => { loadFiles(); }, [loadFiles]);

  // ── Handle type toggle ─────────────────────
  const handleTypeChange = (type) => {
    setUploadType(type);
    setFile(null);
    setUrl("");
    setNotification(null);
    setProgress(null);
  };

  // ── Handle file input ──────────────────────
  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setNotification(null);
  };

  // ── Disable submit guard ───────────────────
  const isSubmitDisabled = () => {
    if (isUploading) return true;
    if (uploadType === "file") return !file;
    return url.trim() === "";
  };

  // ── Submit handler ─────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsUploading(true);
    setProgress(uploadType === "file" ? 0 : null);
    setNotification(null);

    try {
      if (uploadType === "file") {
        await uploadFile(file, replaceOld, (pct) => setProgress(pct));
      } else {
        await uploadFromUrl(url, replaceOld);
      }

      setNotification({ message: "Upload successful! File is now stored in GCS.", type: "success" });
      setFile(null);
      setUrl("");
      setProgress(null);
      await loadFiles(); // Refresh file list

    } catch (err) {
      setNotification({ message: err.message || "Upload failed. Please try again.", type: "error" });
      setProgress(null);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-semibold mb-1 text-gray-900">Admin Upload</h2>
      <p className="text-sm text-gray-500 mb-6">
        Upload documents to Google Cloud Storage. Supported: PDF, DOCX, TXT, MD, or a document URL.
      </p>

      {/* ── Notification ── */}
      {notification && (
        <div className="mb-4">
          <Notification
            message={notification.message}
            type={notification.type}
            onClose={() => setNotification(null)}
          />
        </div>
      )}

      {/* ── Upload Form ── */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <form onSubmit={handleSubmit} className="space-y-5">

          {/* Upload type toggle */}
          <div className="flex items-center space-x-6">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="uploadType"
                value="file"
                checked={uploadType === "file"}
                onChange={() => handleTypeChange("file")}
                className="text-blue-600"
              />
              <span className="text-sm font-medium text-gray-700">File (PDF / DOCX / TXT)</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="uploadType"
                value="url"
                checked={uploadType === "url"}
                onChange={() => handleTypeChange("url")}
                className="text-blue-600"
              />
              <span className="text-sm font-medium text-gray-700">URL</span>
            </label>
          </div>

          {/* File or URL input */}
          {uploadType === "file" ? (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Select File <span className="text-red-500">*</span>
              </label>
              <input
                type="file"
                accept=".pdf,.doc,.docx,.txt,.md"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {file && (
                <p className="mt-1 text-xs text-gray-400">
                  Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </p>
              )}
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Document URL <span className="text-red-500">*</span>
              </label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/document.pdf"
                className="mt-1 block w-full rounded-md border border-gray-300 shadow-sm px-3 py-2 text-sm focus:border-blue-500 focus:ring-blue-500 focus:outline-none"
              />
            </div>
          )}

          {/* Replace older version checkbox */}
          <div className="flex items-center">
            <input
              id="replaceOld"
              type="checkbox"
              checked={replaceOld}
              onChange={(e) => setReplaceOld(e.target.checked)}
              className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
            <label htmlFor="replaceOld" className="ml-2 block text-sm text-gray-700">
              Replace older version <span className="text-gray-400">(marks previous version inactive)</span>
            </label>
          </div>

          {/* Progress bar */}
          {isUploading && uploadType === "file" && progress !== null && (
            <ProgressBar
              progress={progress}
              label={progress < 100 ? "Uploading..." : "Processing..."}
            />
          )}
          {isUploading && uploadType === "url" && (
            <p className="text-sm text-blue-600 animate-pulse">Fetching and uploading from URL...</p>
          )}

          {/* Submit button */}
          <button
            type="submit"
            disabled={isSubmitDisabled()}
            className={`inline-flex items-center px-5 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white transition-colors ${
              isSubmitDisabled()
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
          >
            {isUploading ? "Uploading..." : "Submit"}
          </button>

        </form>
      </div>

      {/* ── File List ── */}
      {loadingFiles ? (
        <p className="mt-6 text-sm text-gray-400">Loading files...</p>
      ) : (
        <FileList
          files={files}
          onRefresh={loadFiles}
          onError={(msg) => setNotification({ message: msg, type: "error" })}
          onSuccess={(msg) => setNotification({ message: msg, type: "success" })}
        />
      )}
    </div>
  );
}

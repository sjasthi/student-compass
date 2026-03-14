// src/components/FileList.jsx
import { useState } from "react";
import { getDownloadUrl, deleteFile } from "../api/uploadService";

/**
 * FileList component
 * Props:
 *   files       — array of file objects from GCS
 *   onRefresh   — callback to reload file list
 *   onError     — callback(message) for error notifications
 *   onSuccess   — callback(message) for success notifications
 */
export default function FileList({ files, onRefresh, onError, onSuccess }) {
  const [loadingBlob, setLoadingBlob]   = useState(null); // blob being downloaded
  const [deletingBlob, setDeletingBlob] = useState(null); // blob being deleted

  // ── Download ───────────────────────────────
  const handleDownload = async (blobName, filename) => {
    setLoadingBlob(blobName);
    try {
      const signedUrl = await getDownloadUrl(blobName);
      const link      = document.createElement("a");
      link.href       = signedUrl;
      link.download   = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      onError?.(`Download failed: ${err.message}`);
    } finally {
      setLoadingBlob(null);
    }
  };

  // ── Delete ─────────────────────────────────
  const handleDelete = async (blobName) => {
    if (!window.confirm("Mark this file as inactive? It will no longer appear in the list.")) return;
    setDeletingBlob(blobName);
    try {
      await deleteFile(blobName);
      onSuccess?.("File removed successfully.");
      onRefresh?.();
    } catch (err) {
      onError?.(`Delete failed: ${err.message}`);
    } finally {
      setDeletingBlob(null);
    }
  };

  // ── File type icon ─────────────────────────
  const fileIcon = (filename = "") => {
    const ext = filename.split(".").pop().toLowerCase();
    const icons = { pdf: "📄", doc: "📝", docx: "📝", txt: "🗒️", md: "🗒️" };
    return icons[ext] || "📁";
  };

  // ── Format date ────────────────────────────
  const formatDate = (isoString) => {
    if (!isoString) return "—";
    return new Date(isoString).toLocaleDateString("en-US", {
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit"
    });
  };

  if (!files || files.length === 0) {
    return (
      <div className="mt-6 p-6 border border-dashed border-gray-300 rounded-lg text-center text-gray-400 text-sm">
        No files uploaded yet.
      </div>
    );
  }

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800">
          Uploaded Files
          <span className="ml-2 text-sm font-normal text-gray-400">({files.length})</span>
        </h3>
        <button
          onClick={onRefresh}
          className="text-xs text-blue-600 hover:underline"
        >
          ↻ Refresh
        </button>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-sm divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">File</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Source</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Size</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Uploaded</th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-100">
            {files.map((file) => (
              <tr key={file.blob_name} className="hover:bg-gray-50 transition-colors">
                {/* File name */}
                <td className="px-4 py-3 font-medium text-gray-800 max-w-xs truncate">
                  <span className="mr-2">{fileIcon(file.original_filename)}</span>
                  {file.original_filename}
                </td>

                {/* Source */}
                <td className="px-4 py-3 text-gray-500">
                  {file.source === "url_upload" ? (
                    <span title={file.source_url} className="text-blue-500 cursor-help">
                      🔗 URL
                    </span>
                  ) : (
                    <span>📤 File</span>
                  )}
                </td>

                {/* Size */}
                <td className="px-4 py-3 text-gray-500">
                  {file.size_kb >= 1024
                    ? `${(file.size_kb / 1024).toFixed(1)} MB`
                    : `${file.size_kb} KB`}
                </td>

                {/* Date */}
                <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                  {formatDate(file.updated)}
                </td>

                {/* Actions */}
                <td className="px-4 py-3 text-right space-x-2 whitespace-nowrap">
                  <button
                    onClick={() => handleDownload(file.blob_name, file.original_filename)}
                    disabled={loadingBlob === file.blob_name}
                    className="text-blue-600 hover:underline disabled:opacity-40 text-xs font-medium"
                  >
                    {loadingBlob === file.blob_name ? "Getting link..." : "Download"}
                  </button>
                  <span className="text-gray-300">|</span>
                  <button
                    onClick={() => handleDelete(file.blob_name)}
                    disabled={deletingBlob === file.blob_name}
                    className="text-red-500 hover:underline disabled:opacity-40 text-xs font-medium"
                  >
                    {deletingBlob === file.blob_name ? "Removing..." : "Remove"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

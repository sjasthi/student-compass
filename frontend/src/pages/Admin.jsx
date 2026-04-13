// src/pages/Admin.jsx
import { useState, useEffect, useCallback } from "react";
import { uploadFile, uploadFromUrl, fetchFiles, syncChroma } from "../api/uploadService";
import ProgressBar  from "../components/ProgressBar";
import Notification from "../components/Notification";
import FileList     from "../components/FileList";

const DOC_TYPE_OPTIONS = [
  { value: "general",         label: "General" },
  { value: "admissions",      label: "Admissions" },
  { value: "financial_aid",   label: "Financial Aid" },
  { value: "graduation",      label: "Graduation" },
  { value: "policies",        label: "Policies" },
  { value: "registration",    label: "Registration" },
  { value: "student_support", label: "Student Support" },
  { value: "tuition_fees",    label: "Tuition & Fees" },
  { value: "web_page",        label: "Web Page" },
];

export default function Admin() {
  const [uploadType,   setUploadType]   = useState("file");
  const [file,         setFile]         = useState(null);
  const [url,          setUrl]          = useState("");
  const [replaceOld,   setReplaceOld]   = useState(true);
  const [docType,      setDocType]      = useState("general");
  const [isUploading,  setIsUploading]  = useState(false);
  const [isSyncing,    setIsSyncing]    = useState(false);
  const [progress,     setProgress]     = useState(null);
  const [notification, setNotification] = useState(null);
  const [files,        setFiles]        = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(false);

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

  const handleTypeChange = (type) => {
    setUploadType(type);
    setFile(null);
    setUrl("");
    setNotification(null);
    setProgress(null);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setNotification(null);
  };

  const isSubmitDisabled = () => {
    if (isUploading) return true;
    if (uploadType === "file") return !file;
    return url.trim() === "";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsUploading(true);
    setProgress(uploadType === "file" ? 0 : null);
    setNotification(null);
    try {
      if (uploadType === "file") {
        await uploadFile(file, replaceOld, docType, (pct) => setProgress(pct));
      } else {
        await uploadFromUrl(url, replaceOld, docType);
      }
      setNotification({
        message: "Upload successful! The document is being indexed in the background.",
        type: "success",
      });
      setFile(null);
      setUrl("");
      setProgress(null);
      await loadFiles();
    } catch (err) {
      setNotification({ message: err.message || "Upload failed. Please try again.", type: "error" });
      setProgress(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSync = async () => {
    setIsSyncing(true);
    setNotification(null);
    try {
      const result = await syncChroma();
      const d      = result?.details || {};
      setNotification({
        message:
          `Sync complete — ${d.files_added ?? 0} file(s) indexed, ` +
          `${d.files_removed ?? 0} file(s) removed from index.` +
          (d.errors?.length ? ` ${d.errors.length} error(s) occurred.` : ""),
        type: d.errors?.length ? "error" : "success",
      });
    } catch (err) {
      setNotification({ message: `Sync failed: ${err.message}`, type: "error" });
    } finally {
      setIsSyncing(false);
    }
  };

  // ─────────────────────────────────────────────
  // Update an existing file in-place.
  // newFile      — the File object chosen by the admin
  // existingFile — the file record from GCS being replaced
  //
  // We rename newFile to existingFile.original_filename so the backend's
  // delete_old_versions() correctly matches and removes the old blob,
  // then uploads the replacement under the same logical name and category.
  // ─────────────────────────────────────────────
  const handleUpdate = async (newFile, existingFile) => {
    setIsUploading(true);
    setProgress(0);
    setNotification(null);
    try {
      // Create a renamed copy so the backend identifies it as the same document
      const renamedFile = new File([newFile], existingFile.original_filename, {
        type: newFile.type || "application/octet-stream",
      });

      await uploadFile(
        renamedFile,
        true,                    // replaceOld — deletes the previous version
        existingFile.doc_type,   // preserve the existing category
        (pct) => setProgress(pct),
      );

      setNotification({
        message: `"${existingFile.original_filename}" updated. The new version is being indexed in the background.`,
        type: "success",
      });
      setProgress(null);
      await loadFiles();
    } catch (err) {
      setNotification({ message: err.message || "Update failed. Please try again.", type: "error" });
      setProgress(null);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-2xl font-semibold text-gray-900">Admin Upload</h2>
        <button
          onClick={handleSync}
          disabled={isSyncing}
          title="Re-sync ChromaDB with all files currently in GCS"
          className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md border transition-colors
            ${isSyncing
              ? "border-gray-300 text-gray-400 cursor-not-allowed bg-gray-50"
              : "border-blue-300 text-blue-700 bg-blue-50 hover:bg-blue-100"
            }`}
        >
          <svg
            className={`h-4 w-4 ${isSyncing ? "animate-spin" : ""}`}
            xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
          >
            <path stroke="currentColor" strokeWidth="2" strokeLinecap="round"
              strokeLinejoin="round" d="M4 4v5h.582M20 20v-5h-.581M5.636 15A9 9 0 1 0 6.5 6.5" />
          </svg>
          {isSyncing ? "Syncing…" : "Sync Index"}
        </button>
      </div>

      <p className="text-sm text-gray-500 mb-6">
        Upload documents to Google Cloud Storage — they are automatically indexed
        for the Student Compass chatbot. Supported: PDF, DOCX, TXT, MD, or a URL.
      </p>

      {notification && (
        <div className="mb-4">
          <Notification
            message={notification.message}
            type={notification.type}
            onClose={() => setNotification(null)}
          />
        </div>
      )}

      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <form onSubmit={handleSubmit} className="space-y-5">

          <div className="flex items-center space-x-6">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input type="radio" name="uploadType" value="file"
                checked={uploadType === "file"}
                onChange={() => handleTypeChange("file")}
                className="text-blue-600" />
              <span className="text-sm font-medium text-gray-700">File (PDF / DOCX / TXT / MD)</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input type="radio" name="uploadType" value="url"
                checked={uploadType === "url"}
                onChange={() => handleTypeChange("url")}
                className="text-blue-600" />
              <span className="text-sm font-medium text-gray-700">URL</span>
            </label>
          </div>

          {uploadType === "file" ? (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Select File <span className="text-red-500">*</span>
              </label>
              <input type="file" accept=".pdf,.doc,.docx,.txt,.md"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer p-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
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
              <input type="url" value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/document.pdf"
                className="mt-1 block w-full rounded-md border border-gray-300 shadow-sm px-3 py-2 text-sm focus:border-blue-500 focus:ring-blue-500 focus:outline-none" />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Document Category
            </label>
            <select value={docType} onChange={(e) => setDocType(e.target.value)}
              className="block w-full rounded-md border border-gray-300 shadow-sm px-3 py-2 text-sm focus:border-blue-500 focus:ring-blue-500 focus:outline-none">
              {DOC_TYPE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <p className="mt-1 text-xs text-gray-400">
              Conceptual categories (Admissions, Financial Aid, etc.) receive an
              AI-generated summary to improve answer quality.
            </p>
          </div>

          <div className="flex items-center">
            <input id="replaceOld" type="checkbox"
              checked={replaceOld}
              onChange={(e) => setReplaceOld(e.target.checked)}
              className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500" />
            <label htmlFor="replaceOld" className="ml-2 block text-sm text-gray-700">
              Replace older version{" "}
              <span className="text-gray-400">(deletes previous copy in GCS &amp; index)</span>
            </label>
          </div>

          {isUploading && progress !== null && (
            <ProgressBar progress={progress} label={progress < 100 ? "Uploading…" : "Processing…"} />
          )}
          {isUploading && uploadType === "url" && (
            <p className="text-sm text-blue-600 animate-pulse">Fetching and uploading from URL…</p>
          )}

          <button type="submit" disabled={isSubmitDisabled()}
            className={`inline-flex items-center px-5 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white transition-colors ${
              isSubmitDisabled() ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
            } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}>
            {isUploading ? "Uploading…" : "Upload"}
          </button>
        </form>
      </div>

      {loadingFiles ? (
        <p className="mt-6 text-sm text-gray-400">Loading files…</p>
      ) : (
        <FileList
          files={files}
          onRefresh={loadFiles}
          onError={(msg)  => setNotification({ message: msg, type: "error" })}
          onSuccess={(msg) => setNotification({ message: msg, type: "success" })}
          onUpdate={handleUpdate}
        />
      )}
    </div>
  );
}

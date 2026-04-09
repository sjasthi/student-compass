// testService.js
// API calls for the evaluation test pipeline.
// streamEvaluation() uses SSE to stream progress and results in real time.
// Supports a `modes` parameter to run RAG, keyword search, and prompt-only
// baselines in a single evaluation pass.

const API_URL = import.meta.env.VITE_APP_API_URL || "http://localhost:5000";

/**
 * Stream a baseline comparison evaluation run via Server-Sent Events.
 *
 * @param {object}   config
 * @param {number[]} config.chunk_sizes   - Chunk sizes to test
 * @param {number[]} config.top_k_values  - Top-K retrieval values to test
 * @param {number[]} config.temperatures  - Temperature values to test
 * @param {number[]} config.top_p_values  - Top-P values to test
 * @param {number}   config.num_questions - Number of gold questions to use
 * @param {string[]} config.modes         - Modes to run: "rag" | "keyword" | "prompt_only"
 * @param {function} onProgress           - Called with (string) status messages
 * @param {function} onResult             - Called with each result object as it arrives
 * @param {AbortSignal} signal            - AbortController signal to cancel the stream
 * @returns {Promise<Array>}              - Resolves to the full results array when done
 */
export async function streamEvaluation(config, onProgress, onResult, signal) {
  const response = await fetch(`${API_URL}/test/run`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(config),
    signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Server error (${response.status})`);
  }

  const reader     = response.body.getReader();
  const decoder    = new TextDecoder();
  let   buffer     = "";
  let   allResults = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by double newlines
    const parts = buffer.split("\n\n");
    buffer      = parts.pop(); // keep incomplete trailing chunk

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;

      let event;
      try { event = JSON.parse(line.slice(5).trim()); }
      catch { continue; }

      if (event.type === "progress") {
        onProgress?.(event.value);
      } else if (event.type === "result") {
        allResults.push(event.value);
        onResult?.(event.value);
      } else if (event.type === "done") {
        allResults = event.value || allResults;
      } else if (event.type === "error") {
        throw new Error(event.value);
      }
    }
  }

  return allResults;
}

/**
 * Download test results as a CSV file.
 * Triggers a browser download dialog.
 * Includes mode column so RAG, keyword, and prompt-only results are
 * clearly labelled in the exported file.
 *
 * @param {Array} results - Array of result objects from the evaluation
 */
export async function downloadResultsCSV(results) {
  const response = await fetch(`${API_URL}/test/download`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(results),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || "Download failed");
  }

  const blob = await response.blob();
  const url  = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href     = url;
  link.download = "rag_comparison_results.csv";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

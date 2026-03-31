// src/pages/Test.jsx
// Admin evaluation page.
// Lets the admin select parameter combinations (chunk_size, top_k,
// temperature, top_p), runs the RAG accuracy test, streams live
// progress, and displays a scored results table.

import { useState, useRef, useCallback } from "react";
import Notification from "../components/Notification.jsx";
import { streamEvaluation, downloadResultsCSV } from "../api/testService.js";

// ─────────────────────────────────────────────
// Available parameter values
// ─────────────────────────────────────────────
const PARAM_OPTIONS = {
  chunkSizes:   [200, 300, 500, 800, 1000, 1200],
  topKValues:   [1, 2, 3, 5],
  temperatures: [0.0, 0.2, 0.4, 0.7, 1.0],
  topPValues:   [0.7, 0.8, 0.9, 0.95, 1.0],
};

const NUM_QUESTION_OPTIONS = [10, 20, 50];

// Score thresholds for colour coding
function accuracyColor(accuracy) {
  if (accuracy >= 2.5) return "text-green-700 bg-green-50 border-green-200";
  if (accuracy >= 2.0) return "text-blue-700  bg-blue-50  border-blue-200";
  if (accuracy >= 1.5) return "text-yellow-700 bg-yellow-50 border-yellow-200";
  return "text-red-700 bg-red-50 border-red-200";
}

function accuracyLabel(accuracy) {
  if (accuracy >= 2.5) return "Excellent ✅";
  if (accuracy >= 2.0) return "Good 👍";
  if (accuracy >= 1.5) return "Fair ⚠️";
  return "Poor ❌";
}

// ─────────────────────────────────────────────
// Multi-select checkbox group
// ─────────────────────────────────────────────
function CheckGroup({ label, options, selected, onChange }) {
  const toggle = (val) =>
    onChange(
      selected.includes(val)
        ? selected.filter((v) => v !== val)
        : [...selected, val].sort((a, b) => a - b)
    );

  return (
    <div>
      <p className="text-sm font-medium text-gray-700 mb-1">{label}</p>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => {
          const active = selected.includes(opt);
          return (
            <button
              key={opt}
              type="button"
              onClick={() => toggle(opt)}
              className={`px-3 py-1 text-sm rounded-full border font-medium transition-colors ${
                active
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-gray-600 border-gray-300 hover:border-blue-400 hover:text-blue-600"
              }`}
            >
              {opt}
            </button>
          );
        })}
      </div>
      {selected.length === 0 && (
        <p className="text-xs text-red-500 mt-1">Select at least one value.</p>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Main page component
// ─────────────────────────────────────────────
export default function Test() {
  // Parameter selections
  const [chunkSizes,   setChunkSizes]   = useState([500]);
  const [topKValues,   setTopKValues]   = useState([3]);
  const [temperatures, setTemperatures] = useState([0.7]);
  const [topPValues,   setTopPValues]   = useState([0.9]);
  const [numQuestions, setNumQuestions] = useState(50);

  // Run state
  const [isRunning,     setIsRunning]     = useState(false);
  const [log,           setLog]           = useState([]);
  const [results,       setResults]       = useState([]);
  const [notification,  setNotification]  = useState(null);
  const abortRef = useRef(null);
  const logEndRef = useRef(null);

  // Computed: how many test runs will be executed
  const totalRuns = chunkSizes.length * topKValues.length * temperatures.length * topPValues.length;

  const isValid =
    chunkSizes.length > 0 &&
    topKValues.length > 0 &&
    temperatures.length > 0 &&
    topPValues.length > 0;

  const appendLog = useCallback((msg) => {
    setLog((prev) => [...prev, msg]);
    // Auto-scroll the log panel
    setTimeout(() => logEndRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }, []);

  const handleRun = async () => {
    if (!isValid || isRunning) return;

    // Cancel any prior run
    if (abortRef.current) abortRef.current.abort();
    const controller   = new AbortController();
    abortRef.current   = controller;

    setIsRunning(true);
    setLog([]);
    setResults([]);
    setNotification(null);

    try {
      await streamEvaluation(
        {
          chunk_sizes:   chunkSizes,
          top_k_values:  topKValues,
          temperatures:  temperatures,
          top_p_values:  topPValues,
          num_questions: numQuestions,
        },
        (msg)    => appendLog(msg),
        (result) => setResults((prev) => [...prev, result]),
        controller.signal,
      );
      setNotification({ message: "Evaluation complete! See results below.", type: "success" });
    } catch (err) {
      if (err.name === "AbortError") {
        appendLog("⛔ Test stopped by user.");
      } else {
        setNotification({ message: err.message || "Evaluation failed.", type: "error" });
        appendLog(`❌ Error: ${err.message}`);
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleDownload = async () => {
    try {
      await downloadResultsCSV(results);
    } catch (err) {
      setNotification({ message: `Download failed: ${err.message}`, type: "error" });
    }
  };

  // Sort results: best accuracy first for easy reading
  const sortedResults = [...results].sort((a, b) => b.accuracy - a.accuracy);

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-900">Evaluation Test</h2>
        <p className="text-sm text-gray-500 mt-1">
          Run RAG accuracy tests against the gold question set. Choose which
          parameter values to test — each combination counts as one experiment run.
        </p>
      </div>

      {notification && (
        <Notification
          message={notification.message}
          type={notification.type}
          onClose={() => setNotification(null)}
        />
      )}

      {/* ── Parameter Selection ────────────────────────────────────── */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-5">
        <h3 className="text-base font-semibold text-gray-800">Test Parameters</h3>

        <CheckGroup
          label="Chunk Sizes"
          options={PARAM_OPTIONS.chunkSizes}
          selected={chunkSizes}
          onChange={setChunkSizes}
        />
        <CheckGroup
          label="Top-K (retrieval)"
          options={PARAM_OPTIONS.topKValues}
          selected={topKValues}
          onChange={setTopKValues}
        />
        <CheckGroup
          label="Temperature"
          options={PARAM_OPTIONS.temperatures}
          selected={temperatures}
          onChange={setTemperatures}
        />
        <CheckGroup
          label="Top-P (nucleus sampling)"
          options={PARAM_OPTIONS.topPValues}
          selected={topPValues}
          onChange={setTopPValues}
        />

        {/* Questions selector */}
        <div>
          <label className="text-sm font-medium text-gray-700">
            Questions per run
          </label>
          <div className="flex gap-2 mt-1">
            {NUM_QUESTION_OPTIONS.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setNumQuestions(n)}
                className={`px-3 py-1 text-sm rounded-full border font-medium transition-colors ${
                  numQuestions === n
                    ? "bg-blue-600 text-white border-blue-600"
                    : "bg-white text-gray-600 border-gray-300 hover:border-blue-400 hover:text-blue-600"
                }`}
              >
                {n}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Max 50 gold questions available.
          </p>
        </div>

        {/* Run summary + action buttons */}
        <div className="flex items-center gap-4 pt-1">
          <div className="text-sm text-gray-500">
            <span className="font-semibold text-gray-800">{totalRuns}</span> experiment
            {totalRuns !== 1 ? "s" : ""} ×{" "}
            <span className="font-semibold text-gray-800">{numQuestions}</span> questions
            {" "}= up to{" "}
            <span className="font-semibold text-gray-800">
              {(totalRuns * numQuestions).toLocaleString()}
            </span>{" "}
            LLM calls
          </div>

          <div className="ml-auto flex gap-2">
            {isRunning ? (
              <button
                onClick={handleStop}
                className="px-4 py-2 text-sm font-medium rounded-md border border-red-300 text-red-600 bg-red-50 hover:bg-red-100 transition-colors"
              >
                ⛔ Stop
              </button>
            ) : (
              <button
                onClick={handleRun}
                disabled={!isValid}
                className={`px-5 py-2 text-sm font-medium rounded-md shadow-sm text-white transition-colors ${
                  isValid
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-400 cursor-not-allowed"
                }`}
              >
                ▶ Run Test
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── Score Rubric ───────────────────────────────────────────── */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 text-sm text-gray-600">
        <span className="font-semibold text-gray-700">Score rubric: </span>
        <span className="text-green-700 font-medium">3 = Perfect</span>
        {" · "}
        <span className="text-blue-700 font-medium">2 = Good</span>
        {" · "}
        <span className="text-yellow-700 font-medium">1 = Partial</span>
        {" · "}
        <span className="text-red-700 font-medium">0 = Incorrect</span>
        <span className="ml-2 text-gray-400">
          (based on cosine similarity to gold answers)
        </span>
      </div>

      {/* ── Live Progress Log ─────────────────────────────────────── */}
      {(isRunning || log.length > 0) && (
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-700">
              Progress Log
              {isRunning && (
                <span className="ml-2 inline-flex items-center gap-1 text-blue-600">
                  <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10"
                      stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Running…
                </span>
              )}
            </h3>
            <span className="text-xs text-gray-400">{log.length} messages</span>
          </div>
          <div className="h-52 overflow-y-auto p-4 font-mono text-xs text-gray-600 space-y-0.5 bg-gray-950">
            {log.map((line, i) => (
              <div
                key={i}
                className={
                  line.startsWith("❌") || line.startsWith("⛔")
                    ? "text-red-400"
                    : line.startsWith("✅")
                    ? "text-green-400"
                    : line.startsWith("▶")
                    ? "text-yellow-300 font-semibold"
                    : "text-gray-300"
                }
              >
                {line}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}

      {/* ── Results Table ─────────────────────────────────────────── */}
      {results.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-700">
              Results
              <span className="ml-2 text-xs font-normal text-gray-400">
                ({results.length} / {totalRuns} runs complete — sorted by accuracy)
              </span>
            </h3>
            <button
              onClick={handleDownload}
              disabled={isRunning}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md border border-blue-300 text-blue-700 bg-blue-50 hover:bg-blue-100 transition-colors disabled:opacity-40"
            >
              ⬇ Download CSV
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-sm divide-y divide-gray-100">
              <thead className="bg-gray-50">
                <tr>
                  {[
                    "Chunk Size",
                    "Top-K",
                    "Temperature",
                    "Top-P",
                    "Questions",
                    "Avg Score",
                    "Rating",
                  ].map((h) => (
                    <th
                      key={h}
                      className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {sortedResults.map((r, i) => (
                  <tr key={i} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-2.5 font-medium text-gray-800">{r.chunk_size}</td>
                    <td className="px-4 py-2.5 text-gray-600">{r.top_k}</td>
                    <td className="px-4 py-2.5 text-gray-600">{r.temperature}</td>
                    <td className="px-4 py-2.5 text-gray-600">{r.top_p}</td>
                    <td className="px-4 py-2.5 text-gray-600">{r.total_questions}</td>
                    <td className="px-4 py-2.5">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border ${accuracyColor(
                          r.accuracy
                        )}`}
                      >
                        {r.accuracy.toFixed(2)} / 3.00
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-xs font-medium text-gray-600">
                      {accuracyLabel(r.accuracy)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Best result callout */}
          {!isRunning && sortedResults.length > 0 && (
            <div className="px-4 py-3 bg-green-50 border-t border-green-100 text-sm text-green-800">
              <span className="font-semibold">Best configuration: </span>
              chunk={sortedResults[0].chunk_size}, top_k={sortedResults[0].top_k},{" "}
              temp={sortedResults[0].temperature}, top_p={sortedResults[0].top_p}{" "}
              → <span className="font-semibold">{sortedResults[0].accuracy.toFixed(2)}</span> avg score
            </div>
          )}
        </div>
      )}
    </div>
  );
}

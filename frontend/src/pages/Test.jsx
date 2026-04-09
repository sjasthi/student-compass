// Test.jsx
// Admin evaluation page.
// Lets the admin select parameter combinations (chunk_size, top_k,
// temperature, top_p) AND which evaluation modes to run (RAG,
// keyword search, prompt-only). Streams live progress and displays
// a scored results table with per-mode columns and a summary card.

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

const MODE_OPTIONS = [
  {
    id:    "rag",
    label: "RAG",
    desc:  "Retrieve + generate",
    color: "bg-blue-600 text-white border-blue-600",
    badge: "bg-blue-100 text-blue-700",
  },
  {
    id:    "keyword",
    label: "Keyword search",
    desc:  "Retrieval only, no LLM",
    color: "bg-amber-500 text-white border-amber-500",
    badge: "bg-amber-100 text-amber-800",
  },
  {
    id:    "prompt_only",
    label: "Prompt-only LLM",
    desc:  "LLM only, no retrieval",
    color: "bg-gray-600 text-white border-gray-600",
    badge: "bg-gray-100 text-gray-700",
  },
];

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
// Comparison summary card
// ─────────────────────────────────────────────
function ComparisonSummary({ results, activeModes }) {
  // Compute average score and average latency per mode
  const modeScores    = {};
  const modeLatencies = {};
  for (const mode of activeModes) {
    const rows = results.filter((r) => r.mode === mode);
    if (rows.length === 0) {
      modeScores[mode]    = null;
      modeLatencies[mode] = null;
    } else {
      modeScores[mode]    = rows.reduce((sum, r) => sum + r.accuracy, 0) / rows.length;
      const lats          = rows.filter((r) => r.avg_latency_ms != null).map((r) => r.avg_latency_ms);
      modeLatencies[mode] = lats.length ? Math.round(lats.reduce((a, b) => a + b, 0) / lats.length) : null;
    }
  }

  const ranked = activeModes
    .filter((m) => modeScores[m] !== null)
    .sort((a, b) => modeScores[b] - modeScores[a]);

  const winner = ranked[0];

  const modeInfo = Object.fromEntries(MODE_OPTIONS.map((m) => [m.id, m]));

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700">Comparison summary</h3>
        <p className="text-xs text-gray-400 mt-0.5">
          Average score across all parameter configurations
        </p>
      </div>

      <div className="p-4 grid grid-cols-3 gap-3">
        {activeModes.map((mode) => {
          const score = modeScores[mode];
          const info  = modeInfo[mode];
          const isWinner = mode === winner;
          return (
            <div
              key={mode}
              className={`rounded-lg border p-3 text-center ${
                isWinner
                  ? "border-green-300 bg-green-50"
                  : "border-gray-200 bg-gray-50"
              }`}
            >
              {isWinner && (
                <div className="text-xs font-semibold text-green-700 mb-1">
                  Winner 🏆
                </div>
              )}
              <span
                className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full mb-2 ${info.badge}`}
              >
                {info.label}
              </span>
              <div className="text-2xl font-bold {isWinner ? 'text-green-700' : 'text-gray-600'}">
                {score !== null ? score.toFixed(2) : "—"}
              </div>
              <div className="text-xs text-gray-400 mt-0.5">avg / 3.00</div>
              {modeLatencies[mode] != null && (
                <div className="text-xs text-gray-500 mt-1">
                  {modeLatencies[mode].toLocaleString()} ms avg
                </div>
              )}
            </div>
          );
        })}
      </div>

      {ranked.length >= 2 && (
        <div className="px-4 pb-4 text-xs text-gray-500">
          RAG outperforms the keyword baseline by{" "}
          <span className="font-semibold text-gray-700">
            {modeScores["rag"] !== null && modeScores["keyword"] !== null
              ? (modeScores["rag"] - modeScores["keyword"]).toFixed(2)
              : "—"}
          </span>{" "}
          points and the prompt-only baseline by{" "}
          <span className="font-semibold text-gray-700">
            {modeScores["rag"] !== null && modeScores["prompt_only"] !== null
              ? (modeScores["rag"] - modeScores["prompt_only"]).toFixed(2)
              : "—"}
          </span>{" "}
          points on average.
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Main page component
// ─────────────────────────────────────────────
export default function TestWithComparison() {
  // Parameter selections
  const [chunkSizes,   setChunkSizes]   = useState([500]);
  const [topKValues,   setTopKValues]   = useState([3]);
  const [temperatures, setTemperatures] = useState([0.7]);
  const [topPValues,   setTopPValues]   = useState([0.9]);
  const [numQuestions, setNumQuestions] = useState(50);

  // Mode selection — all three on by default
  const [activeModes, setActiveModes] = useState(["rag", "keyword", "prompt_only"]);

  // Run state
  const [isRunning,    setIsRunning]    = useState(false);
  const [log,          setLog]          = useState([]);
  const [results,      setResults]      = useState([]);
  const [notification, setNotification] = useState(null);
  const abortRef  = useRef(null);
  const logEndRef = useRef(null);

  // Computed: how many test runs will be executed
  const paramCombos = chunkSizes.length * topKValues.length * temperatures.length * topPValues.length;
  const totalRuns   = paramCombos * activeModes.length;

  const isValid =
    chunkSizes.length > 0 &&
    topKValues.length > 0 &&
    temperatures.length > 0 &&
    topPValues.length > 0 &&
    activeModes.length > 0;

  const appendLog = useCallback((msg) => {
    setLog((prev) => [...prev, msg]);
    setTimeout(() => logEndRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }, []);

  const toggleMode = (modeId) => {
    setActiveModes((prev) =>
      prev.includes(modeId)
        ? prev.filter((m) => m !== modeId)
        : [...prev, modeId]
    );
  };

  const handleRun = async () => {
    if (!isValid || isRunning) return;

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

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
          modes:         activeModes,
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

  // Group results by param config key, then by mode within each group
  const grouped = {};
  for (const r of results) {
    const key = `${r.chunk_size}-${r.top_k}-${r.temperature}-${r.top_p}`;
    if (!grouped[key]) grouped[key] = { meta: r, byMode: {} };
    grouped[key].byMode[r.mode] = r;
  }
  const groupedRows = Object.values(grouped).sort((a, b) => {
    const bestA = Math.max(...Object.values(a.byMode).map((r) => r.accuracy));
    const bestB = Math.max(...Object.values(b.byMode).map((r) => r.accuracy));
    return bestB - bestA;
  });

  const modeInfo = Object.fromEntries(MODE_OPTIONS.map((m) => [m.id, m]));

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-900">Evaluation Test</h2>
        <p className="text-sm text-gray-500 mt-1">
          Run accuracy tests across RAG, keyword search, and prompt-only modes.
          Each parameter combination is tested in every selected mode so results
          are directly comparable.
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

        {/* ── Mode selector (new) ───────────────────────────────────── */}
        <div>
          <p className="text-sm font-medium text-gray-700 mb-2">Evaluation modes</p>
          <div className="flex flex-wrap gap-3">
            {MODE_OPTIONS.map((mode) => {
              const active = activeModes.includes(mode.id);
              return (
                <button
                  key={mode.id}
                  type="button"
                  onClick={() => toggleMode(mode.id)}
                  className={`flex flex-col items-start px-4 py-2.5 rounded-lg border text-left transition-colors ${
                    active
                      ? mode.color
                      : "bg-white text-gray-600 border-gray-300 hover:border-gray-400"
                  }`}
                >
                  <span className="text-sm font-semibold">{mode.label}</span>
                  <span className={`text-xs mt-0.5 ${active ? "opacity-80" : "text-gray-400"}`}>
                    {mode.desc}
                  </span>
                </button>
              );
            })}
          </div>
          {activeModes.length === 0 && (
            <p className="text-xs text-red-500 mt-1">Select at least one mode.</p>
          )}
        </div>

        {/* Run summary + action buttons */}
        <div className="flex items-center gap-4 pt-1">
          <div className="text-sm text-gray-500">
            <span className="font-semibold text-gray-800">{paramCombos}</span> config
            {paramCombos !== 1 ? "s" : ""}{" "}×{" "}
            <span className="font-semibold text-gray-800">{activeModes.length}</span> mode
            {activeModes.length !== 1 ? "s" : ""}{" "}×{" "}
            <span className="font-semibold text-gray-800">{numQuestions}</span> questions
            {" "}={" "}
            <span className="font-semibold text-gray-800">
              {(totalRuns * numQuestions).toLocaleString()}
            </span>{" "}
            total evaluations
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
          (cosine similarity vs gold answers — same scorer for all modes)
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
                    : line.includes("[keyword]")
                    ? "text-amber-300"
                    : line.includes("[prompt_only]")
                    ? "text-gray-400"
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

      {/* ── Comparison Summary ────────────────────────────────────── */}
      {results.length > 0 && !isRunning && activeModes.length > 1 && (
        <ComparisonSummary results={results} activeModes={activeModes} />
      )}

      {/* ── Results Table ─────────────────────────────────────────── */}
      {groupedRows.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-700">
              Results
              <span className="ml-2 text-xs font-normal text-gray-400">
                ({groupedRows.length} config{groupedRows.length !== 1 ? "s" : ""} — sorted by best score)
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
            <table className="min-w-full text-sm divide-y divide-gray-100" style={{tableLayout:"fixed"}}>
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide w-24">Chunk</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide w-16">Top-K</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide w-20">Temp</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide w-16">Top-P</th>
                  {activeModes.map((mode) => (
                    <th key={mode} className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide w-32">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${modeInfo[mode]?.badge}`}>
                        {modeInfo[mode]?.label}
                      </span>
                    </th>
                  ))}
                  {activeModes.map((mode) => (
                    <th key={`lat-${mode}`} className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide w-32">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${modeInfo[mode]?.badge}`}>
                        {modeInfo[mode]?.label}
                      </span>
                      <span className="block text-gray-400 font-normal normal-case mt-0.5">avg ms</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {groupedRows.map((group, i) => {
                  const { meta, byMode } = group;
                  // Find best mode for this config row
                  const bestScore = Math.max(
                    ...activeModes
                      .filter((m) => byMode[m])
                      .map((m) => byMode[m].accuracy)
                  );
                  return (
                    <tr key={i} className="hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-2.5 font-medium text-gray-800">{meta.chunk_size}</td>
                      <td className="px-4 py-2.5 text-gray-600">{meta.top_k}</td>
                      <td className="px-4 py-2.5 text-gray-600">{meta.temperature}</td>
                      <td className="px-4 py-2.5 text-gray-600">{meta.top_p}</td>
                      {activeModes.map((mode) => {
                        const r = byMode[mode];
                        if (!r) {
                          return (
                            <td key={mode} className="px-4 py-2.5 text-gray-300 text-xs">—</td>
                          );
                        }
                        const isWinner = r.accuracy === bestScore && activeModes.length > 1;
                        return (
                          <td key={mode} className="px-4 py-2.5">
                            <span
                              className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border ${accuracyColor(r.accuracy)} ${isWinner ? "ring-2 ring-green-400 ring-offset-1" : ""}`}
                            >
                              {r.accuracy.toFixed(2)}
                            </span>
                          </td>
                        );
                      })}
                      {activeModes.map((mode) => {
                        const r = byMode[mode];
                        if (!r || r.avg_latency_ms == null) {
                          return <td key={`lat-${mode}`} className="px-4 py-2.5 text-gray-300 text-xs">—</td>;
                        }
                        return (
                          <td key={`lat-${mode}`} className="px-4 py-2.5">
                            <span className="text-xs text-gray-700 font-medium">
                              {r.avg_latency_ms.toLocaleString()}
                            </span>
                            <span className="text-xs text-gray-400 ml-1">
                              ({r.min_latency_ms}–{r.max_latency_ms})
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Best config callout */}
          {!isRunning && groupedRows.length > 0 && (
            <div className="px-4 py-3 bg-green-50 border-t border-green-100 text-sm text-green-800">
              <span className="font-semibold">Best RAG configuration: </span>
              {(() => {
                const ragRows = results.filter((r) => r.mode === "rag");
                if (!ragRows.length) return "No RAG results yet.";
                const best = [...ragRows].sort((a, b) => b.accuracy - a.accuracy)[0];
                return `chunk=${best.chunk_size}, top_k=${best.top_k}, temp=${best.temperature}, top_p=${best.top_p} → ${best.accuracy.toFixed(2)} avg score`;
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// src/services/api.js
// Non-streaming fallback for any component that prefers a simple Promise.
// Home.jsx uses /query/stream directly for better perceived performance.

const API_BASE = import.meta.env.VITE_APP_API_URL || 'http://localhost:5000';

/**
 * Send a question and wait for the complete answer (non-streaming).
 *
 * @param   {string} question
 * @returns {Promise<{ answer: string, sources: Array }>}
 */
export async function askQuestion(question) {
  const response = await fetch(`${API_BASE}/query`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ question }),
  });

  if (!response.ok) {
    let errorMsg = `Server error (${response.status})`;
    try {
      const err = await response.json();
      errorMsg  = err.error || errorMsg;
    } catch { /* ignore */ }
    throw new Error(errorMsg);
  }

  return response.json();
}

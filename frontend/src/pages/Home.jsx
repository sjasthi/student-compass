// src/pages/Home.jsx
// Uses /query/stream (SSE) so the answer renders token-by-token
// instead of waiting for the full Gemini response.

import React, { useState, useRef } from 'react';
import QuestionBar    from '../components/QuestionBar.jsx';
import AnswerCard     from '../components/AnswerCard.jsx';
import SourcesCard    from '../components/SourcesCard.jsx';
import DisclaimerCard from '../components/DisclaimerCard.jsx';

const API_BASE = import.meta.env.VITE_APP_API_URL || 'http://localhost:5000';

function Home() {
  const [question, setQuestion] = useState('');
  const [answer,   setAnswer]   = useState('');
  const [sources,  setSources]  = useState([]);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState('');

  // Abort controller so a new question cancels any in-flight stream
  const abortRef = useRef(null);

  const handleAsk = async () => {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError('');
    setAnswer('');
    setSources([]);

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: trimmed }),
        signal:  controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Server error (${response.status})`);
      }

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const parts = buffer.split('\n\n');
        buffer      = parts.pop(); // keep any incomplete trailing chunk

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;

          let event;
          try { event = JSON.parse(line.slice(5).trim()); }
          catch { continue; }

          if      (event.type === 'token')   setAnswer(prev => prev + event.value);
          else if (event.type === 'sources') setSources(event.value || []);
          else if (event.type === 'error')   throw new Error(event.value);
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Stream error:', err);
      setError(err.message || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 space-y-6 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold text-center">Student Compass</h1>
      <p className="text-center text-gray-500 text-sm">
        Ask any question about admissions, registration, financial aid, and more.
      </p>

      <QuestionBar
        question={question}
        setQuestion={setQuestion}
        onAsk={handleAsk}
        loading={loading}
      />

      {/* Spinner — shown until first token arrives */}
      {loading && answer === '' && (
        <div className="flex items-center justify-center space-x-2 text-blue-600">
          <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg"
            fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10"
              stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          <span className="text-sm">Searching the knowledge base…</span>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="rounded-md bg-red-50 border border-red-200 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Answer streams in; sources appear once the stream finishes */}
      {answer && (
        <>
          <AnswerCard answer={answer} isStreaming={loading} />
          {sources.length > 0 && <SourcesCard sources={sources} />}
        </>
      )}

      <DisclaimerCard />
    </div>
  );
}

export default Home;
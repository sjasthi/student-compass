// src/pages/Home.jsx
// Conversation thread UI with sliding-window history.
// Sends only the last 3 turns to the backend on each request
// so the Gemini prompt stays small regardless of conversation length.

import React, { useState, useRef, useEffect } from 'react';
import QuestionBar    from '../components/QuestionBar.jsx';
import AnswerCard     from '../components/AnswerCard.jsx';
import SourcesCard    from '../components/SourcesCard.jsx';
import DisclaimerCard from '../components/DisclaimerCard.jsx';

const API_BASE      = import.meta.env.VITE_APP_API_URL || 'http://localhost:5000';
const HISTORY_WINDOW = 3;   // turns sent to backend — matches server-side constant

function Home() {
  const [question,  setQuestion]  = useState('');
  const [history,   setHistory]   = useState([]);   // [ { question, answer, sources } ]
  const [liveAnswer, setLiveAnswer] = useState('');
  const [liveSources, setLiveSources] = useState([]);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState('');

  const abortRef   = useRef(null);
  const bottomRef  = useRef(null);

  // Auto-scroll to bottom as new tokens arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [liveAnswer, history]);

  const handleNewConversation = () => {
    if (abortRef.current) abortRef.current.abort();
    setHistory([]);
    setLiveAnswer('');
    setLiveSources([]);
    setError('');
    setQuestion('');
  };

  const handleAsk = async () => {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError('');
    setLiveAnswer('');
    setLiveSources([]);
    setQuestion('');

    // Sliding window: send only the last HISTORY_WINDOW completed turns
    const windowedHistory = history.slice(-HISTORY_WINDOW).map(({ question, answer }) => ({
      question,
      answer,
    }));

    let fullAnswer = '';

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: trimmed, history: windowedHistory }),
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

        const parts = buffer.split('\n\n');
        buffer      = parts.pop();

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;

          let event;
          try { event = JSON.parse(line.slice(5).trim()); }
          catch { continue; }

          if (event.type === 'token') {
            fullAnswer += event.value;
            setLiveAnswer(prev => prev + event.value);
          } else if (event.type === 'sources') {
            setLiveSources(event.value || []);
          } else if (event.type === 'error') {
            throw new Error(event.value);
          }
        }
      }

      // Commit the completed turn to history
      setHistory(prev => [
        ...prev,
        { question: trimmed, answer: fullAnswer, sources: liveSources },
      ]);
      setLiveAnswer('');
      setLiveSources([]);

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
    <div className="min-h-screen bg-gray-50 flex flex-col">

      {/* ── Header ── */}
      <div className="sticky top-0 z-10 bg-white border-b border-gray-200 px-4 py-3
                      flex items-center justify-between max-w-3xl mx-auto w-full">
        <h1 className="text-xl font-bold">Student Compass</h1>
        {history.length > 0 && (
          <button
            onClick={handleNewConversation}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            New conversation
          </button>
        )}
      </div>

      {/* ── Conversation thread ── */}
      <div className="flex-1 max-w-3xl mx-auto w-full px-4 py-6 space-y-6">

        {history.length === 0 && !loading && !liveAnswer && (
          <p className="text-center text-gray-400 text-sm mt-10">
            Ask any question about admissions, registration, financial aid, and more.
          </p>
        )}

        {/* Completed turns */}
        {history.map((turn, i) => (
          <div key={i} className="space-y-3">
            {/* Question bubble */}
            <div className="flex justify-end">
              <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm
                              px-4 py-2 max-w-[80%] text-sm shadow-sm">
                {turn.question}
              </div>
            </div>
            {/* Answer */}
            <AnswerCard answer={turn.answer} isStreaming={false} />
            {turn.sources?.length > 0 && <SourcesCard sources={turn.sources} />}
          </div>
        ))}

        {/* Live (in-progress) turn */}
        {(loading || liveAnswer) && (
          <div className="space-y-3">
            {/* The question that was just submitted — shown at top of live turn */}
            {loading && liveAnswer === '' && (
              <div className="flex justify-end">
                <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm
                                px-4 py-2 max-w-[80%] text-sm shadow-sm opacity-70">
                  {/* placeholder while waiting for first token */}
                </div>
              </div>
            )}

            {/* Spinner before first token */}
            {loading && liveAnswer === '' && (
              <div className="flex items-center space-x-2 text-blue-600">
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg"
                  fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10"
                    stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span className="text-sm">Searching the knowledge base…</span>
              </div>
            )}

            {/* Streaming answer */}
            {liveAnswer && (
              <AnswerCard answer={liveAnswer} isStreaming={loading} />
            )}
            {liveSources.length > 0 && <SourcesCard sources={liveSources} />}
          </div>
        )}

        {/* Error state */}
        {error && !loading && (
          <div className="rounded-md bg-red-50 border border-red-200 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Sticky input bar ── */}
      <div className="sticky bottom-0 bg-white border-t border-gray-200 px-4 py-3
                      max-w-3xl mx-auto w-full space-y-2">
        <QuestionBar
          question={question}
          setQuestion={setQuestion}
          onAsk={handleAsk}
          loading={loading}
        />
        <DisclaimerCard />
      </div>

    </div>
  );
}

export default Home;

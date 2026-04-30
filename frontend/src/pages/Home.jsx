// src/pages/Home.jsx
// Conversation thread UI with sliding-window history.
// Sends only the last 3 turns to the backend on each request
// so the Gemini prompt stays small regardless of conversation length.

import React, { useState, useRef, useEffect } from 'react';
import QuestionBar from '../components/QuestionBar.jsx';
import AnswerCard from '../components/AnswerCard.jsx';
import SourcesCard from '../components/SourcesCard.jsx';
import DisclaimerCard from '../components/DisclaimerCard.jsx';

const API_BASE = import.meta.env.VITE_APP_API_URL || 'http://localhost:5000';
const HISTORY_WINDOW = 3;

function Home() {
  const [question, setQuestion] = useState('');
  const [history, setHistory] = useState([]);
  const [liveAnswer, setLiveAnswer] = useState('');
  const [liveSources, setLiveSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const abortRef = useRef(null);
  const bottomRef = useRef(null);

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

  const handleSuggestedQuestion = (value) => {
    setQuestion(value);
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

    const windowedHistory = history.slice(-HISTORY_WINDOW).map(({ question, answer }) => ({
      question,
      answer,
    }));

    let fullAnswer = '';
    let finalSources = [];

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, history: windowedHistory }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Server error (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop();

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;

          let event;
          try {
            event = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }

          if (event.type === 'token') {
            fullAnswer += event.value;
            setLiveAnswer((prev) => prev + event.value);
          } else if (event.type === 'sources') {
            finalSources = event.value || [];
            setLiveSources(event.value || []);
          } else if (event.type === 'error') {
            throw new Error(event.value);
          }
        }
      }

      setHistory((prev) => [
        ...prev,
        { question: trimmed, answer: fullAnswer, sources: finalSources },
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

  const suggestedQuestions = [
    'What are the admission requirements?',
    'How do I apply for financial aid?',
    'When are important deadlines?',
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <div className="sticky top-0 z-10 bg-white border-b border-gray-200 px-4 py-4 max-w-3xl mx-auto w-full">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">Student Compass</h1>
          {history.length > 0 && (
            <button
              onClick={handleNewConversation}
              className="text-sm font-medium text-blue-600 hover:text-blue-700"
            >
              New conversation
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 max-w-5xl mx-auto w-full px-4 py-10">
        {history.length === 0 && !loading && !liveAnswer && (
          <div className="space-y-10">
            <div className="text-center max-w-2xl mx-auto">
              <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-blue-100">
                <svg
                  className="h-8 w-8 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 2l2.5 5.5L20 10l-5.5 2.5L12 18l-2.5-5.5L4 10l5.5-2.5L12 2z"
                  />
                </svg>
              </div>

              <h2 className="text-4xl font-bold tracking-tight text-gray-900">
                How can I <span className="text-black-600">help you</span> today?
              </h2>

              <p className="mt-5 text-lg text-gray-500">
                Ask any question about admissions, registration, financial aid, and more.
              </p>
              <p className="mt-2 text-lg text-gray-500">
                I&apos;ll search our documents and provide you with accurate information.
              </p>
            </div>

            <div className="max-w-4xl mx-auto">
              <div className="flex items-center gap-4 mb-6">
                <div className="h-px flex-1 bg-gray-200" />
                <p className="text-black-600 font-semibold text-lg">Try asking about</p>
                <div className="h-px flex-1 bg-gray-200" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {suggestedQuestions.map((item, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestedQuestion(item)}
                    className="bg-white border border-gray-200 rounded-2xl p-5 text-left shadow-sm hover:border-blue-300 hover:shadow-md transition"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="text-xl font-semibold text-gray-900">
                          {index === 0 && 'Admissions'}
                          {index === 1 && 'Financial Aid'}
                          {index === 2 && 'Deadlines'}
                        </h3>
                        <p className="mt-3 text-base text-gray-500">{item}</p>
                      </div>
                      <span className="text-blue-600 text-2xl leading-none">›</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {(history.length > 0 || loading || liveAnswer || error) && (
          <div className="max-w-3xl mx-auto space-y-6">
            {history.map((turn, i) => (
              <div key={i} className="space-y-3">
                <div className="flex justify-end">
                  <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-2 max-w-[80%] text-sm shadow-sm">
                    {turn.question}
                  </div>
                </div>
                <AnswerCard answer={turn.answer} isStreaming={false} />
                {turn.sources?.length > 0 && <SourcesCard sources={turn.sources} />}
              </div>
            ))}

            {(loading || liveAnswer) && (
              <div className="space-y-3">
                {loading && liveAnswer === '' && (
                  <div className="flex items-center space-x-2 text-blue-600">
                    <svg
                      className="animate-spin h-4 w-4"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8v8H4z"
                      />
                    </svg>
                    <span className="text-sm">Searching the knowledge base…</span>
                  </div>
                )}

                {liveAnswer && <AnswerCard answer={liveAnswer} isStreaming={loading} />}
                {liveSources.length > 0 && <SourcesCard sources={liveSources} />}
              </div>
            )}

            {error && !loading && (
              <div className="rounded-xl bg-red-50 border border-red-200 p-4 text-sm text-red-700">
                {error}
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="sticky bottom-0 bg-gray-50 px-4 pb-8">
        <div className="max-w-4xl mx-auto space-y-4">
          <QuestionBar
            question={question}
            setQuestion={setQuestion}
            onAsk={handleAsk}
            loading={loading}
          />
          <DisclaimerCard />
        </div>
      </div>
    </div>
  );
}

export default Home;
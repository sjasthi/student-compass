// src/components/AnswerCard.jsx
// Displays the answer returned from the backend.
// Accepts an optional isStreaming prop to show a blinking cursor
// while tokens are still arriving from the SSE stream.

import React from 'react';

function AnswerCard({ answer, isStreaming = false }) {
  return (
    <section className="p-4 border-2 border-gray-300 rounded-lg bg-white space-y-2">
      <h2 className="text-lg font-semibold">Answer</h2>
      <p className="text-sm text-gray-800 whitespace-pre-wrap">
        {answer || 'Your answer will appear here once you ask a question.'}
        {/* Blinking cursor shown while the stream is still coming in */}
        {isStreaming && answer && (
          <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-0.5 align-middle" />
        )}
      </p>
    </section>
  );
}

export default AnswerCard;

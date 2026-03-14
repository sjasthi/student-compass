import React from 'react';

/**
 * Card component displaying the answer returned from the backend.
 *
 * Props:
 * - answer: string containing the generated answer
 */
function AnswerCard({ answer }) {
  return (
    <section className="p-4 border-2 border-gray-300 rounded-lg bg-white space-y-2">
      <h2 className="text-lg font-semibold">Answer</h2>
      <p className="text-sm text-gray-800 whitespace-pre-wrap">
        {answer || 'Your answer will appear here once you ask a question.'}
      </p>
    </section>
  );
}

export default AnswerCard;
import React from 'react';

/**
 * Input component for submitting a question.
 *
 * Props:
 * - question: the current value of the input
 * - setQuestion: function to update the question state
 * - onAsk: handler invoked when the user clicks the Ask button
 * - loading: boolean indicating whether a request is in progress
 */
function QuestionBar({ question, setQuestion, onAsk, loading }) {
  // Capture Enter key press to submit the question
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      onAsk();
    }
  };

  return (
    <section className="p-4 border-2 border-gray-300 rounded-lg bg-white space-y-2">
      <h2 className="text-lg font-semibold">Ask a Question</h2>
      <div className="flex gap-3">
        <input
          type="text"
          className="flex-1 border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-gray-400"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your question..."
        />
        <button
          className="border border-gray-500 rounded-md px-5 py-3 text-sm font-medium hover:bg-gray-100 disabled:opacity-60"
          onClick={onAsk}
          disabled={loading || !question.trim()}
        >
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </div>
    </section>
  );
}

export default QuestionBar;
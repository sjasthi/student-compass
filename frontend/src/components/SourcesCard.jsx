import React from 'react';

/**
 * Card component to display sources used to generate the answer.
 *
 * Props:
 * - sources: array of objects containing at least a `title` and optional `url`
 */
function SourcesCard({ sources }) {
  const hasSources = Array.isArray(sources) && sources.length > 0;
  return (
    <section className="p-4 border-2 border-gray-300 rounded-lg bg-white space-y-2">
      <h2 className="text-lg font-semibold">Sources</h2>
      {hasSources ? (
        <ul className="list-disc pl-5 space-y-1 text-sm">
          {sources.map((src, idx) => (
            <li key={idx}>
              {src.url ? (
                <a
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 underline"
                >
                  {src.title || src.url}
                </a>
              ) : (
                <span>{src.title}</span>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-gray-600">Sources will appear here after a question is asked.</p>
      )}
    </section>
  );
}

export default SourcesCard;
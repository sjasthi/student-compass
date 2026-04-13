// src/components/SourcesCard.jsx
// Displays sources returned by the backend.
// Each source object: { source, doc_type, summary, blob_name }

import React from 'react';

const TYPE_LABELS = {
  admissions:      "Admissions",
  financial_aid:   "Financial Aid",
  graduation:      "Graduation",
  policies:        "Policies",
  registration:    "Registration",
  student_support: "Student Support",
  tuition_fees:    "Tuition & Fees",
  web_page:        "Web Page",
  general:         "General",
  unknown:         "Unknown",
};

function SourcesCard({ sources }) {
  const hasSources = Array.isArray(sources) && sources.length > 0;

  return (
    <section className="p-4 border-2 border-gray-300 rounded-lg bg-white space-y-2">
      <h2 className="text-lg font-semibold">Sources</h2>
      {hasSources ? (
        <ul className="space-y-2 text-sm">
          {sources.map((src, idx) => (
            <li key={idx} className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-medium text-gray-800">
                  {src.source}
                </span>
                {src.doc_type && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">
                    {TYPE_LABELS[src.doc_type] || src.doc_type}
                  </span>
                )}
              </div>
              {src.summary && (
                <p className="text-xs text-gray-500 ml-0.5">{src.summary}</p>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-gray-600">
          Sources will appear here after a question is asked.
        </p>
      )}
    </section>
  );
}

export default SourcesCard;

// src/components/ProgressBar.jsx

/**
 * ProgressBar component
 * Props:
 *   progress — number 0–100
 *   label    — optional label string
 */
export default function ProgressBar({ progress, label }) {
  if (progress === null || progress === undefined) return null;

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>{label}</span>
          <span>{progress}%</span>
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
        <div
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

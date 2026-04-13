// src/components/Notification.jsx
import { useEffect, useState } from "react";

/**
 * Notification / Toast component
 * Props:
 *   message  — string to display
 *   type     — "success" | "error" | "info"
 *   onClose  — callback when dismissed
 */
export default function Notification({ message, type = "info", onClose }) {
  const [visible, setVisible] = useState(true);

  // Auto-dismiss after 5 seconds
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
      onClose?.();
    }, 5000);
    return () => clearTimeout(timer);
  }, [message, onClose]);

  if (!visible || !message) return null;

  const styles = {
    success: {
      container: "bg-green-50 border border-green-300 text-green-800",
      icon:      "✅",
    },
    error: {
      container: "bg-red-50 border border-red-300 text-red-800",
      icon:      "❌",
    },
    info: {
      container: "bg-blue-50 border border-blue-300 text-blue-800",
      icon:      "ℹ️",
    },
  };

  const { container, icon } = styles[type] || styles.info;

  return (
    <div className={`flex items-start justify-between p-3 rounded-md text-sm ${container}`}>
      <span>
        {icon} {message}
      </span>
      <button
        onClick={() => { setVisible(false); onClose?.(); }}
        className="ml-4 font-bold opacity-60 hover:opacity-100"
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  );
}

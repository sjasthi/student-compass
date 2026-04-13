import React from 'react';

/**
 * Card component to display a disclaimer to users.
 */
function DisclaimerCard() {
  return (
    <section className="p-4 border-2 border-gray-300 rounded-lg bg-white space-y-2">
      <h2 className="text-lg font-semibold">Disclaimer</h2>
      <p className="text-sm text-gray-700">
        This tool provides informational guidance only and does not replace official academic advising or university offices.
      </p>
    </section>
  );
}

export default DisclaimerCard;
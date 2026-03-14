/**
 * Sends a question to the backend API and returns the parsed JSON response.
 *
 * @param {string} question - The question to ask the backend.
 * @returns {Promise<object>} Response data containing `answer` and `sources`.
 */
export async function askQuestion(question) {
  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! Status: ${response.status}`);
  }
  return await response.json();
}
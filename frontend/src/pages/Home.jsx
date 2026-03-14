import React, { useState } from 'react';
import QuestionBar from '../components/QuestionBar.jsx';
import AnswerCard from '../components/AnswerCard.jsx';
import SourcesCard from '../components/SourcesCard.jsx';
import DisclaimerCard from '../components/DisclaimerCard.jsx';
import { askQuestion } from '../services/api.js';

/**
 * Home page component for the Student Compass application.
 * It holds the state for the question, answer, sources, and loading indicator.
 */
function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  /**
   * Handler invoked when the user clicks the Ask button.
   * Sends the question to the backend and updates the answer and sources accordingly.
   */
  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const data = await askQuestion(question);
      setAnswer(data?.answer || '');
      setSources(data?.sources || []);
    } catch (error) {
      console.error('Error fetching answer:', error);
      setAnswer('An error occurred while fetching the answer.');
      setSources([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 space-y-6">
      <h1 className="text-3xl font-bold text-center">Student Compass</h1>
      <QuestionBar
        question={question}
        setQuestion={setQuestion}
        onAsk={handleAsk}
        loading={loading}
      />
      <AnswerCard answer={answer} />
      <SourcesCard sources={sources} />
      <DisclaimerCard />
    </div>
  );
}

export default Home;
import React, { useState } from 'react';
import Header from './components/Header';
import AssessmentForm from './components/AssessmentForm';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (formData) => {
    setIsLoading(true);
    setError(null);
    try {
      // Assuming backend is running on default localhost:8000
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction from server.');
      }

      const result = await response.json();
      setPredictionResult(result);
    } catch (err) {
      console.error(err);
      setError('An error occurred while connecting to the backend. Is it running?');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPredictionResult(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{ flex: 1, padding: '2rem 0' }}>
        {error && (
          <div className="container" style={{ marginBottom: '1rem' }}>
            <div style={{ backgroundColor: '#fff', borderLeft: '4px solid #000', padding: '1rem', boxShadow: 'var(--shadow-sm)' }}>
              <p style={{ margin: 0, color: '#111' }}>{error}</p>
            </div>
          </div>
        )}

        {!predictionResult ? (
          <AssessmentForm onSubmit={handleSubmit} isLoading={isLoading} />
        ) : (
          <ResultsDashboard result={predictionResult} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}

export default App;

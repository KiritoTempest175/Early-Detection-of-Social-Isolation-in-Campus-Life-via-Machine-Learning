import React from 'react';
import { ArrowLeft, AlertCircle, CheckCircle, Info } from 'lucide-react';
import RadarChartComponent from './RadarChartComponent';

export default function ResultsDashboard({ result, onReset }) {
  const getAlertIcon = (status) => {
    switch (status) {
      case 'warning': return <AlertCircle size={20} color="#111111" />;
      case 'success': return <CheckCircle size={20} color="#111111" />;
      default: return <Info size={20} color="#111111" />;
    }
  };

  return (
    <div className="container animate-in">
      <button
        onClick={onReset}
        className="btn-secondary flex items-center gap-2 mb-8"
        style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}
      >
        <ArrowLeft size={16} /> Back to Assessment
      </button>

      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Your Results</h2>
        <p>Based on your recent assessment.</p>
      </div>

      <div className="flex flex-col gap-6">
        <div className="card" style={{ textAlign: 'center', backgroundColor: '#fafafa' }}>
          <p style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Risk Category</p>
          <h3 style={{ fontSize: '2rem', margin: 0 }}>{result.risk_category}</h3>

          <div className="flex justify-center gap-4 mt-4">
            <div style={{ padding: '1rem', borderRight: '1px solid var(--color-border)' }}>
              <p style={{ margin: 0, fontSize: '0.8rem' }}>Social Density Score</p>
              <strong style={{ fontSize: '1.2rem' }}>{result.social_density_score.toFixed(1)}</strong>
            </div>
            <div style={{ padding: '1rem' }}>
              <p style={{ margin: 0, fontSize: '0.8rem' }}>Behavioral Profile</p>
              <strong style={{ fontSize: '1.2rem' }}>{result.cluster_label}</strong>
            </div>
          </div>
        </div>

        <div className="card flex items-center gap-4" style={{ borderLeft: '4px solid var(--color-text)' }}>
          <div>{getAlertIcon(result.recommendation.status)}</div>
          <div>
            <h4 style={{ margin: 0 }}>{result.recommendation.title}</h4>
            <p style={{ margin: 0, marginTop: '0.25rem', color: 'var(--color-text)' }}>{result.recommendation.message}</p>
          </div>
        </div>

        <div className="card mt-4">
          <h4 style={{ textAlign: 'center' }}>Social Profile Analysis</h4>
          <RadarChartComponent
            userScores={result.user_scores}
            campusBaseline={result.campus_baseline}
          />
        </div>
      </div>
    </div>
  );
}

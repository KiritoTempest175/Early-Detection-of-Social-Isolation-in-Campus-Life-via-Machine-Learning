import { Activity, ArrowLeft, ArrowRight, ShieldCheck } from 'lucide-react';
import ComparisonBars from './ComparisonBars';

export default function ResultsDashboard({ result, onReset, onOverview, saved }) {
  const density = Number(result.social_density_score);
  const anomaly = Number(result.anomaly_score);
  const isUnusual = Number.isFinite(anomaly) && anomaly < 0;
  return <section className="result-view">
    <div className="result-actions"><button type="button" className="dark-button" onClick={onReset}><ArrowLeft size={17}/> New assessment</button><button type="button" className="plain-button" onClick={onOverview}>View live dashboard <ArrowRight size={16}/></button></div>
    <div className="result-banner"><span className="eyebrow">YOUR EXPLORATORY PROFILE</span><div className="result-banner-grid"><div><h2>{result.risk_category}</h2><p>A descriptive pattern of the answers you entered. This is not a clinical conclusion.</p></div><div className="density-score"><span>SOCIAL DENSITY INDEX</span><strong>{Number.isFinite(density) ? density.toFixed(2) : '—'}</strong><small>Derived engagement indicator</small></div></div></div>
    <div className="result-panels">
      <article className="panel"><span className="eyebrow">BEHAVIORAL CLUSTER</span><h3>{result.cluster_label}</h3><p>K-Means groups similar response patterns; the cluster is not a measure of mental health.</p><div className="result-metric"><span>Statistical unusualness</span><strong>{isUnusual ? 'Outside typical range' : 'Within typical range'}</strong></div><p className="microcopy">Isolation Forest detects uncommon combinations, including positive ones. Raw decision score: {Number.isFinite(anomaly) ? anomaly.toFixed(3) : '—'} (below zero means unusual). It is not a probability or a risk score.</p></article>
      <article className="panel"><span className="eyebrow">OPTIONAL SUGGESTION</span><h3>{result.recommendation.title}</h3><p>{result.recommendation.message}</p><div className="private-note"><ShieldCheck size={17}/>{saved ? 'A summary was saved in this browser only. Your individual answers were not saved.' : 'Could not save locally; check browser storage settings.'}</div></article>
    </div>
    <section className="panel result-chart"><div className="panel-heading"><div><span className="eyebrow">COMPARATIVE INDICATORS</span><h3>Your result vs. the demo benchmark</h3></div></div><p className="panel-description">Readable side-by-side bars replace the radar pointers. Higher is not automatically better.</p><ComparisonBars userScores={result.user_scores} campusBaseline={result.campus_baseline}/></section>
    <p className="result-disclaimer"><Activity size={16}/> Survey analytics cannot diagnose loneliness or mental health conditions. Consider qualified campus support if you need help.</p>
  </section>;
}

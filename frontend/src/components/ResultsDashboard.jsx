import { ArrowLeft, Activity, ShieldCheck, ChartNoAxesCombined } from 'lucide-react';
import RadarChartComponent from './RadarChartComponent';

export default function ResultsDashboard({ result, onReset }) {
  const density = Number(result.social_density_score);
  const anomaly = Number(result.anomaly_score);
  return <section className="result-view">
    <button className="plain-button" onClick={onReset}><ArrowLeft size={17}/> Start another assessment</button>
    <div className="result-banner"><span className="eyebrow">YOUR EXPLORATORY PROFILE</span><div className="result-banner-grid"><div><h2>{result.risk_category}</h2><p>Descriptive response pattern — not a clinical risk category or a diagnosis.</p></div><div className="density-score"><span>SOCIAL DENSITY INDEX</span><strong>{Number.isFinite(density) ? density.toFixed(2) : '—'}</strong><small>Derived engagement indicator</small></div></div></div>
    <div className="result-panels"><article className="panel"><span className="eyebrow">BEHAVIORAL CLUSTER</span><h3>{result.cluster_label}</h3><p>Groups describe similarities among survey responses. They do not identify a mental health condition.</p><div className="result-metric"><span>Anomaly decision score</span><strong>{Number.isFinite(anomaly) ? anomaly.toFixed(3) : '—'}</strong></div></article><article className="panel"><span className="eyebrow">OPTIONAL SUGGESTION</span><h3>{result.recommendation.title}</h3><p>{result.recommendation.message}</p><div className="private-note"><ShieldCheck size={17}/> Your responses were not saved.</div></article></div>
    <section className="panel result-chart"><div className="panel-heading"><div><span className="eyebrow">COMPARATIVE INDICATORS</span><h3>You vs. the sample average</h3></div><ChartNoAxesCombined size={19}/></div><p className="panel-description">Normalized to 0–100. Greater screen time is not necessarily a positive outcome.</p><RadarChartComponent userScores={result.user_scores} campusBaseline={result.campus_baseline}/></section>
    <p className="result-disclaimer"><Activity size={16}/> Survey analytics cannot diagnose social isolation or mental health concerns. Consider qualified campus support if you need help.</p>
  </section>;
}

import { Activity, ArrowLeft, ArrowRight, ShieldCheck, Lightbulb, MapPinned } from 'lucide-react';
import { ClusterMap, SignalTiles } from './DataVisuals';

export default function ResultsDashboard({ result, analytics, history = [], onReset, onOverview, saved }) {
  const density = Number(result.social_density_score);
  const anomaly = Number(result.anomaly_score);
  const percentile = Number(result.density_percentile);
  const unusual = Number.isFinite(anomaly) && anomaly < 0;
  return <section className="result-view">
    <div className="result-actions"><button type="button" className="dark-button" onClick={onReset}><ArrowLeft size={17}/> New assessment</button><button type="button" className="plain-button" onClick={onOverview}>See dashboard <ArrowRight size={16}/></button></div>
    <div className="result-banner"><span className="eyebrow">YOUR ENGAGEMENT PROFILE</span><div className="result-banner-grid"><div><h2>{result.risk_category}</h2><p>A descriptive interpretation of your answers, not a diagnosis or mental-health score.</p></div><div className="density-score"><span>SOCIAL DENSITY INDEX</span><strong>{Number.isFinite(density) ? density.toFixed(2) : '—'}</strong><small>Derived engagement indicator</small></div></div></div>
    <div className="result-snapshot">
      <article className="panel snapshot-panel"><span className="eyebrow">YOUR CLUSTER</span><h3>{result.cluster_label}</h3><p>K-Means grouped this pattern with similar answers.</p></article>
      <article className="panel snapshot-panel"><span className="eyebrow">DATASET POSITION</span><h3>{Number.isFinite(percentile) ? Math.round(percentile) + 'th percentile' : 'Unavailable'}</h3><p>Share of reference examples at or below your social engagement index; not a risk probability.</p></article>
      <article className="panel snapshot-panel"><span className="eyebrow">UNUSUALNESS</span><h3>{unusual ? 'Less common pattern' : 'Within typical range'}</h3><p>Statistical outliers can be healthy or unhealthy. Not a risk prediction.</p></article>
    </div>
    <div className="analytics-grid result-visuals">
      <section className="panel analytics-wide">
        <div className="panel-heading"><div><span className="eyebrow">YOUR POSITION</span><h3>Where you sit among the clusters</h3></div><MapPinned size={19}/></div>
        <p className="panel-description">Your orange point is plotted against the reference examples. The axes use normalized social engagement and social wellness.</p>
        {analytics ? <ClusterMap analytics={analytics} history={history} current={result} compact/> : <div className="no-data"><p>The cluster map could not load. Your other results remain available.</p></div>}
      </section>
      <section className="panel"><div className="panel-heading"><div><span className="eyebrow">YOUR SIGNALS</span><h3>Quick indicator heatmap</h3></div></div>
        <p className="panel-description">Six comparable dimensions, each 0–100. Darker tiles indicate a larger value, not that the value is better.</p>
        <SignalTiles userScores={result.user_scores} baseline={result.campus_baseline}/>
      </section>
    </div>
    <section className="panel action-panel"><div className="panel-heading"><div><span className="eyebrow">PRACTICAL NEXT STEPS</span><h3>Where you could improve</h3></div><Lightbulb size={19}/></div><p className="panel-description">Suggestions reflect your answers and preferences, not a clinical determination.</p>
      <div className="insight-grid">{(result.insights || [{title: result.recommendation.title, detail:'Optional reflection',action:result.recommendation.message}]).map((idea,i)=><article className="insight-card" key={idea.title}><span className="insight-index">0{i+1}</span><h4>{idea.title}</h4><p>{idea.detail}</p><div className="insight-action"><ArrowRight size={15}/><span>{idea.action}</span></div></article>)}</div>
    </section>
    <p className="private-note"><ShieldCheck size={17}/>{saved ? 'Only your result summary was saved in this browser. The full questionnaire was not recorded.' : 'Browser storage is unavailable; this assessment was not retained.'}</p>
    <p className="result-disclaimer"><Activity size={16}/> Exploratory clustering cannot diagnose loneliness or mental health concerns. Seek trusted or qualified support if you need it.</p>
  </section>;
}

import { useCallback, useEffect, useState } from 'react';
import { Activity, ArrowRight, BookOpen, ChevronRight, Clock3, ClipboardList, HeartHandshake, LayoutDashboard, Menu, Plus, RefreshCw, ShieldCheck, Trash2, TrendingUp, Users, X } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, LineChart, Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import AssessmentForm from './components/AssessmentForm';
import ResultsDashboard from './components/ResultsDashboard';
import { readHistory, storeAssessment, eraseHistory, HISTORY_LIMIT } from './history';
import { ClusterMap, DensityHistogram, DensityHeatmap, CorrelationHeatmap } from './components/DataVisuals';

const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');
const PALETTE = ['#292922', '#a89a84', '#ded2bd'];
const NAV = [
  { id: 'overview', title: 'Overview', Icon: LayoutDashboard },
  { id: 'assessment', title: 'Assessment', Icon: ClipboardList },
  { id: 'method', title: 'Methodology', Icon: BookOpen },
];
async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(API + path, { ...options, signal: controller.signal, cache: 'no-store', headers: { 'Content-Type': 'application/json' } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The service could not complete this request.');
    return data;
  } finally { clearTimeout(timeout); }
}
function Kpi({ icon: Icon, name, value, note }) {
  return <article className="kpi"><div className="kpi-head"><span>{name}</span><Icon size={19} strokeWidth={1.7}/></div><div className="kpi-value">{value}</div><p>{note}</p></article>;
}
function Overview({ data, analytics, error, refresh, open, history, clearHistory }) {
  const demo = data?.mode === 'synthetic_demo';
  const groups = data?.distribution || [];
  const sum = groups.reduce((n, g) => n + g.count, 0);
  const ordered = history.slice(0, 15).reverse();
  const trend = ordered.map((entry, i) => ({ step: i + 1, score: Number(entry.density.toFixed(3)), profile: entry.category }));
  const breakdown = Object.entries(history.reduce((map, entry) => {
    map[entry.category] = (map[entry.category] || 0) + 1;
    return map;
  }, {})).sort((a, b) => b[1] - a[1]);
  const localAverage = history.length ? history.reduce((sum, r) => sum + r.density, 0) / history.length : 0;
  return <>
    <div className="intro"><span className="eyebrow">CAMPUS INTELLIGENCE / OVERVIEW</span><h1>Understanding connection,<br/><em>one pattern at a time.</em></h1><p>An exploratory view of student engagement, academic routines, and social well-being.</p></div>
    <section className="hero"><div className="hero-content"><div className="hero-label"><span className="signal-dot"/> RESEARCH DASHBOARD <span className="separator"/> {demo ? 'SYNTHETIC DEMO' : 'CAMPUS SIGNALS'}</div><h2>See the patterns.<br/><span>Support the people.</span></h2><p>Explore broad campus-life trends and reflect on your own routine with a private, non-clinical assessment.</p><button className="light-button" onClick={() => open('assessment', true)}>Start an assessment <ArrowRight size={16}/></button></div><div className="hero-graphic" aria-hidden="true"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="orbit orbit-three"/><div className="orbit-symbol"><Activity size={33}/></div></div></section>
    <div className="section-title"><div><span className="eyebrow">AT A GLANCE</span><h2>Survey overview</h2></div><button className="plain-button" onClick={refresh}><RefreshCw size={15}/> Refresh data</button></div>
    {error && <p className="warning" role="status">{error}</p>}
    <div className="kpis">
      <Kpi icon={Users} name={demo ? "Synthetic records" : "Survey records"} value={data ? data.sample_size.toLocaleString() : '—'} note={demo ? 'Generated demonstration records' : 'Source survey observations'}/>
      <Kpi icon={Activity} name="Social density" value={data ? data.average_social_density.toFixed(2) : '—'} note="Mean derived engagement index"/>
      <Kpi icon={Clock3} name="Weekly time alone" value={data ? data.average_hours_alone + 'h' : '—'} note={demo ? "Generated weekly estimate" : "Average reported hours"}/>
      <Kpi icon={HeartHandshake} name="High loneliness ratings" value={data ? data.high_loneliness_share + '%' : '—'} note="Ratings of 4–5 / 5; not diagnostic"/>
    </div>
    <div className="panels">
      <section className="panel"><div className="panel-heading"><div><span className="eyebrow">DATA BREAKDOWN</span><h3>Engagement patterns</h3></div><span className="muted-label">Cluster distribution</span></div>
        {sum > 0 ? <div className="donut-content"><div className="donut"><ResponsiveContainer width="100%" height={248}><PieChart><Pie data={groups} dataKey="count" nameKey="name" innerRadius={68} outerRadius={98} paddingAngle={2} stroke="#fffdfa" strokeWidth={4}>{groups.map((g, i) => <Cell key={g.name} fill={PALETTE[i]}/>)}</Pie><Tooltip formatter={v => [v + ' responses', 'Count']}/></PieChart></ResponsiveContainer><div className="donut-total"><strong>{sum}</strong><small>responses</small></div></div><div className="cluster-list">{groups.map((g, i) => <div className="cluster" key={g.name}><div><span className="cluster-name"><i style={{ background: PALETTE[i] }}/>{g.name}</span><strong>{Math.round(100 * g.count / sum)}%</strong></div><div className="track"><span style={{ background: PALETTE[i], width: 100 * g.count / sum + '%' }}/></div></div>)}</div></div> : <div className="no-data"><h3>Overview unavailable</h3><p>Configure a private dataset or enable explicit synthetic demo mode on the API.</p></div>}
        <p className="panel-footnote">Clusters are descriptive patterns, not medical or behavioral diagnoses.</p>
      </section>
      <section className="panel"><div className="panel-heading"><div><span className="eyebrow">SAMPLE INDICATORS</span><h3>Survey benchmarks</h3></div></div><p className="panel-description">Normalized means, 0–100. Higher values aren't always better (for example, screen time).</p>
        {data ? <div className="benchmarks">{Object.entries(data.baseline).map(([name, value]) => <div className="benchmark" key={name}><div><span>{name}</span><strong>{Math.round(value)}<small>/100</small></strong></div><div className="track"><span style={{ width: Math.max(0, Math.min(100, value)) + '%' }}/></div></div>)}</div> : <div className="no-data"><p>Benchmarks appear when the model is ready.</p></div>}
      </section>
    </div>


    <section className="analytics-section" aria-labelledby="analytics-heading">
      <div className="section-title"><div><span className="eyebrow">DATASET EXPLORER</span><h2 id="analytics-heading">Patterns across the dataset</h2></div><span className="dataset-caption">{analytics?.mode === 'synthetic_demo' ? '520 generated records' : 'Aggregated research data'}</span></div>
      <p className="history-intro">Explore how social engagement, well-being, and academic routines vary across the sample. Each generated example appears in the cluster plot, alongside your saved assessments. These plots are descriptive, not diagnostic.</p>
      {analytics ? <>
        <div className="analytics-grid">
          <section className="panel analytics-wide"><div className="panel-heading"><div><span className="eyebrow">CLUSTER MAP</span><h3>Where different patterns sit</h3></div><span className="muted-label">Engagement × social wellness</span></div>
            <ClusterMap analytics={analytics} history={history}/>
          </section>
          <section className="panel"><div className="panel-heading"><div><span className="eyebrow">FREQUENCY</span><h3>Social density distribution</h3></div></div><p className="panel-description">How many records fall in each 10-point engagement band.</p><DensityHistogram analytics={analytics}/></section>
          <section className="panel"><div className="panel-heading"><div><span className="eyebrow">FREQUENCY HEATMAP</span><h3>Engagement × wellness</h3></div></div><p className="panel-description">Each square counts records within one of 25 bands.</p><DensityHeatmap analytics={analytics}/></section>
          <section className="panel analytics-wide"><div className="panel-heading"><div><span className="eyebrow">RELATIONSHIPS</span><h3>Indicator correlation heatmap</h3></div><span className="muted-label">Seven selected variables</span></div><p className="panel-description">Explore patterns across the complete reference dataset, not just the average.</p><CorrelationHeatmap analytics={analytics}/></section>
        </div>
      </> : <div className="panel no-data"><p>Detailed analytics are loading or temporarily unavailable. Refresh data to retry.</p></div>}
    </section>
    <section className="history-dashboard" aria-labelledby="history-heading">
      <div className="section-title"><div><span className="eyebrow">YOUR PRIVATE WORKSPACE</span><h2 id="history-heading">Assessment activity</h2></div><div className="history-actions">{history.length > 0 && <button type="button" className="plain-button danger-button" onClick={clearHistory}><Trash2 size={15}/> Clear history</button>}<button type="button" className="dark-button" onClick={() => open('assessment', true)}><Plus size={16}/> New assessment</button></div></div>
      <p className="history-intro">Updates immediately when you finish a check-in. Only summary scores are saved in this browser, not your 22 answers; nothing here is shared with other visitors. Your data stays on this device (up to {HISTORY_LIMIT} recent sessions).</p>
      {history.length > 0 ? <>
        <div className="kpis local-kpis">
          <Kpi icon={ClipboardList} name="Your assessments" value={history.length.toLocaleString()} note="Saved on this browser"/>
          <Kpi icon={Activity} name="Average engagement index" value={localAverage.toFixed(2)} note="From your saved summaries"/>
          <Kpi icon={TrendingUp} name="Latest pattern" value={history[0].category} note="Descriptive, not a diagnosis"/>
          <Kpi icon={Clock3} name="Latest check-in" value={new Date(history[0].created_at).toLocaleDateString()} note="Stored locally, no account required"/>
        </div>
        <div className="panels history-panels">
          <article className="panel"><div className="panel-heading"><div><span className="eyebrow">PERSONAL HISTORY</span><h3>Engagement index over time</h3></div><span className="muted-label">Last {trend.length} assessments</span></div><p className="panel-description">Sequential sessions, oldest to newest, up to 15. An index is not a mental-health score.</p><div className="trend-chart" role="img" aria-label="Your derived social engagement index across recent assessments"><ResponsiveContainer width="100%" height={235}><LineChart data={trend} margin={{top:10,right:14,left:-20,bottom:4}}><CartesianGrid stroke="#e9e3d9" vertical={false}/><XAxis dataKey="step" tick={{fill:'#878073',fontSize:11}} axisLine={false} tickLine={false}/><YAxis domain={[0,1]} tick={{fill:'#878073',fontSize:11}} axisLine={false} tickLine={false}/><Tooltip formatter={v => [Number(v).toFixed(2),'Engagement index']} labelFormatter={v=>'Session '+v} contentStyle={{background:'#fffdfa',border:'1px solid #e8e2d9',borderRadius:9}}/><Line type="monotone" dataKey="score" stroke="#35362d" strokeWidth={3} dot={trend.length === 1 ? {r:4} : false} activeDot={{r:5}} isAnimationActive={false}/></LineChart></ResponsiveContainer></div></article>
          <article className="panel"><div className="panel-heading"><div><span className="eyebrow">YOUR SESSION MIX</span><h3>Observed response patterns</h3></div></div><p className="panel-description">Counts of your own exploratory profiles, not population prevalence.</p><div className="local-distribution">{breakdown.map(([label,count]) => <div className="benchmark" key={label}><div><span>{label}</span><strong>{count} / {history.length}</strong></div><div className="track"><span style={{width:(count / history.length * 100)+'%',background:'#494a40'}}/></div></div>)}</div></article>
        </div>
        <section className="panel recent-history"><div className="panel-heading"><div><span className="eyebrow">SAVED ON THIS DEVICE</span><h3>All saved assessment summaries</h3></div></div><div className="history-scroll"><table><thead><tr><th>When</th><th>Profile</th><th>Social density</th><th>Pattern signal</th></tr></thead><tbody>{history.map((entry) => <tr key={entry.id}><td>{new Date(entry.created_at).toLocaleString()}</td><td>{entry.category}</td><td>{entry.density.toFixed(2)}</td><td>{entry.anomaly < 0 ? 'Unusual' : 'Typical'}</td></tr>)}</tbody></table></div></section>
      </> : <div className="panel empty-history"><div className="empty-symbol"><Activity size={24}/></div><h3>No assessments saved yet</h3><p>Complete a check-in to start a private trend chart and recent-history log. The synthetic survey charts above won't change.</p><button className="dark-button" onClick={() => open('assessment', true)}>Start first assessment <ArrowRight size={16}/></button></div>}
    </section>
    <div className="privacy-banner"><div className="privacy-icon"><ShieldCheck size={24}/></div><div><h3>Designed around privacy.</h3><p>Your original answers are not saved. After a successful check-in, a score-only summary is saved in your browser; synthetic benchmarks stay separate.</p></div><button className="plain-button" onClick={() => open('method')}>Learn more <ChevronRight size={16}/></button></div>
    <footer>CAMPUS SIGNALS <span>Academic research prototype · Not clinical screening</span></footer>
  </>;
}
function Method({ open }) {
  return <section className="method-page"><span className="eyebrow">TRANSPARENCY / RESEARCH</span><h1>How Campus Signals works.</h1><p className="method-lead">This exploratory prototype does not provide a medical diagnosis or clinically validated predictions. Synthetic example data is used for this public demo.</p><div className="methods"><article className="panel"><span>01 / INPUT</span><h3>Self-reported indicators</h3><p>22 questions cover student routines, academics, social engagement, loneliness and perceived well-being. No name or email is requested.</p></article><article className="panel"><span>02 / PATTERNS</span><h3>Unsupervised ML</h3><p>Preprocessing, a derived social-density index, K-Means clustering and Isolation Forest describe similarities and unusual patterns.</p></article><article className="panel"><span>03 / CONTEXT</span><h3>Reflect and compare</h3><p>You receive a descriptive profile and synthetic benchmark comparisons. A summary of each check-in stays in your browser until you clear it.</p></article></div><section className="panel ethics"><ShieldCheck size={27}/><h2>Limitations & ethical use</h2><p>The survey is not representative of all students. Loneliness is itself an assessment input, so this is not independent detection of loneliness. Anomaly scores are not risk probabilities. Predictions have not been validated for diagnosis, student discipline, admissions, or welfare surveillance.</p><p>The other two uploaded datasets relate to stress and mental health and are not linked to respondents in this campus survey. They must not be joined row-for-row. No personally identifiable survey data should be published.</p><p>Your answers are processed transiently by the API, but assessment summaries (not raw answers) are stored in this browser for your private history. Clear them from Overview at any time. Participation must be voluntary. If loneliness or distress affects daily life, speak with a trusted person or qualified professional.</p></section><button className="dark-button" onClick={() => open('assessment', true)}>Start assessment <ArrowRight size={17}/></button></section>;
}
export default function App() {
  const [page, setPage] = useState('overview');
  const [menu, setMenu] = useState(false);
  const [overview, setOverview] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [assessmentError, setAssessmentError] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState(readHistory);
  const [saved, setSaved] = useState(true);
  const [formKey, setFormKey] = useState(0);
  useEffect(() => {
    const onStorage = () => setHistory(readHistory());
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);
  const refresh = useCallback(async () => {
    const [overviewResponse, analyticsResponse] = await Promise.allSettled([
      request('/api/overview'), request('/api/analytics')
    ]);
    if (overviewResponse.status === 'fulfilled') {
      setOverview(overviewResponse.value);
      setError('');
    } else {
      setOverview(null);
      setError(overviewResponse.reason?.message || 'API unavailable');
    }
    setAnalytics(analyticsResponse.status === 'fulfilled' ? analyticsResponse.value : null);
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  function open(tab, fresh = false) {
    if (fresh && tab === 'assessment') {
      setResult(null);
      setAssessmentError('');
      setFormKey(i => i + 1);
    }
    setPage(tab);
    setMenu(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function clearHistory() {
    if (!window.confirm('Erase saved assessment summaries from this browser? This cannot be undone.')) return;
    try {
      eraseHistory();
      setHistory([]);
    } catch {
      setError('Browser storage could not be cleared.');
    }
  }
  async function submit(values) {
    setLoading(true); setAssessmentError('');
    try {
      const response = await request('/predict', { method: 'POST', body: JSON.stringify(values) });
      let stored = true;
      try { setHistory(storeAssessment(response)); }
      catch { stored = false; }
      setSaved(stored);
      setResult(response);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    catch (e) { setAssessmentError(e.name === 'AbortError' ? 'Request timed out. Please retry.' : e.message); }
    finally { setLoading(false); }
  }
  return <div className="layout">
    <aside className={'sidebar ' + (menu ? 'shown' : '')}><div className="identity"><div className="identity-icon"><Activity size={20}/></div><div><strong>campus<span>.</span>signals</strong><small>SOCIAL CONNECTION LAB</small></div></div><div className="nav-caption">WORKSPACE</div><nav aria-label="Main navigation">{NAV.map(({id,title,Icon}) => <button key={id} className={'nav-link ' + (page === id ? 'selected' : '')} onClick={() => open(id)} aria-current={page === id ? 'page' : undefined}><Icon size={19}/>{title}{page === id && <ChevronRight className="nav-arrow" size={16}/>}</button>)}</nav><div className="side-footer"><div className="side-activity"><span className="eyebrow">YOUR ACTIVITY</span><strong>{history.length} <small>saved sessions</small></strong><p>{history[0]?.category || 'No check-ins yet'}</p><button type="button" className="side-action" onClick={() => open('assessment', true)}><Plus size={15}/> New assessment <ArrowRight size={14}/></button></div><small>PRIVATE TO THIS BROWSER</small></div></aside>
    {menu && <button className="mobile-shade" aria-label="Close navigation" onClick={() => setMenu(false)}/>}
    <div className="workspace"><header className="topbar"><div className="breadcrumbs"><button className="hamburger" aria-label="Toggle navigation" onClick={() => setMenu(!menu)}>{menu ? <X size={21}/> : <Menu size={21}/>}</button>Workspace <ChevronRight size={15}/> <strong>{NAV.find(n => n.id === page)?.title}</strong></div><div className="top-right"><span className={'health ' + (overview ? 'online' : '')}><i/>{overview ? 'Model online' : 'Model unavailable'}</span><span className="badge-avatar">CS</span></div></header><main className="main">
      {page === 'overview' && <Overview data={overview} analytics={analytics} error={error} refresh={refresh} open={open} history={history} clearHistory={clearHistory}/>}
      {page === 'method' && <Method open={open}/>}
      {page === 'assessment' && <div className="assessment-page"><div className="intro"><span className="eyebrow">PRIVATE REFLECTION / ASSESSMENT</span><h1>Your campus check-in.</h1><p>Reflect on your campus routine. Raw answers are not saved; a score-only summary of each completed assessment is stored in this browser.</p></div>{result ? <ResultsDashboard result={result} analytics={analytics} history={history.filter(item => item.id !== result.user_id)} onReset={() => open('assessment', true)} onOverview={() => open('overview')} saved={saved}/> : <>{assessmentError && <div className="warning" role="alert">{assessmentError}</div>}<AssessmentForm key={formKey} onSubmit={submit} isLoading={loading}/></>}</div>}
    </main></div>
  </div>;
}

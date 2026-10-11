import { useCallback, useEffect, useState } from 'react';
import { Activity, ArrowRight, BookOpen, ChevronRight, Clock3, ClipboardList, HeartHandshake, LayoutDashboard, Menu, RefreshCw, ShieldCheck, Users, X } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import AssessmentForm from './components/AssessmentForm';
import ResultsDashboard from './components/ResultsDashboard';

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
function Overview({ data, error, refresh, open }) {
  const demo = data?.mode === 'synthetic_demo';
  const groups = data?.distribution || [];
  const sum = groups.reduce((n, g) => n + g.count, 0);
  return <>
    <div className="intro"><span className="eyebrow">CAMPUS INTELLIGENCE / OVERVIEW</span><h1>Understanding connection,<br/><em>one pattern at a time.</em></h1><p>An exploratory view of student engagement, academic routines, and social well-being.</p></div>
    <section className="hero"><div className="hero-content"><div className="hero-label"><span className="signal-dot"/> RESEARCH DASHBOARD <span className="separator"/> {demo ? 'SYNTHETIC DEMO' : 'CAMPUS SIGNALS'}</div><h2>See the patterns.<br/><span>Support the people.</span></h2><p>Explore broad campus-life trends and reflect on your own routine with a private, non-clinical assessment.</p><button className="light-button" onClick={() => open('assessment')}>Start an assessment <ArrowRight size={16}/></button></div><div className="hero-graphic" aria-hidden="true"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="orbit orbit-three"/><div className="orbit-symbol"><Activity size={33}/></div></div></section>
    <div className="section-title"><div><span className="eyebrow">AT A GLANCE</span><h2>Survey overview</h2></div><button className="plain-button" onClick={refresh}><RefreshCw size={15}/> Refresh data</button></div>
    {error && <p className="warning" role="status">{error}</p>}
    {demo && <p className="demo-note"><ShieldCheck size={17}/> Synthetic demonstration only. Figures do not describe actual students or campuses.</p>}
    <div className="kpis">
      <Kpi icon={Users} name="Survey records" value={data ? data.sample_size.toLocaleString() : '—'} note={demo ? 'Generated demonstration records' : 'Source survey observations'}/>
      <Kpi icon={Activity} name="Social density" value={data ? data.average_social_density.toFixed(2) : '—'} note="Mean derived engagement index"/>
      <Kpi icon={Clock3} name="Weekly time alone" value={data ? data.average_hours_alone + 'h' : '—'} note="Average self-reported hours"/>
      <Kpi icon={HeartHandshake} name="High loneliness ratings" value={data ? data.high_loneliness_share + '%' : '—'} note="Self-report 4–5 / 5, not diagnosis"/>
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
    <div className="privacy-banner"><div className="privacy-icon"><ShieldCheck size={24}/></div><div><h3>Designed around privacy.</h3><p>Assessments are processed without saving your answers. The dashboard uses only anonymized aggregate calculations.</p></div><button className="plain-button" onClick={() => open('method')}>Learn more <ChevronRight size={16}/></button></div>
    <footer>CAMPUS SIGNALS <span>Academic research prototype · Not clinical screening</span></footer>
  </>;
}
function Method({ open }) {
  return <section className="method-page"><span className="eyebrow">TRANSPARENCY / RESEARCH</span><h1>How Campus Signals works.</h1><p className="method-lead">This is an exploratory research prototype. It does not provide a medical diagnosis, clinically validated risk prediction, or individual institutional screening.</p><div className="methods"><article className="panel"><span>01 / INPUT</span><h3>Self-reported indicators</h3><p>22 questions cover student routines, academics, social engagement, loneliness and perceived well-being. No name or email is requested.</p></article><article className="panel"><span>02 / PATTERNS</span><h3>Unsupervised ML</h3><p>Preprocessing, a derived social-density index, K-Means clustering and Isolation Forest describe similarities and unusual patterns.</p></article><article className="panel"><span>03 / CONTEXT</span><h3>Reflect and compare</h3><p>Respondents receive a profile and comparisons with sample means, alongside optional non-clinical support suggestions.</p></article></div><section className="panel ethics"><ShieldCheck size={27}/><h2>Limitations & ethical use</h2><p>The survey is not representative of all students. Loneliness is itself an assessment input, so this is not independent detection of loneliness. Anomaly scores are not risk probabilities. Predictions have not been validated for diagnosis, student discipline, admissions, or welfare surveillance.</p><p>The other two uploaded datasets relate to stress and mental health and are not linked to respondents in this campus survey. They must not be joined row-for-row. No personally identifiable survey data should be published.</p><p>Participation must be voluntary. If loneliness or distress is affecting daily life, speak with a trusted person or qualified support professional.</p></section><button className="dark-button" onClick={() => open('assessment')}>Start private assessment <ArrowRight size={17}/></button></section>;
}
export default function App() {
  const [page, setPage] = useState('overview');
  const [menu, setMenu] = useState(false);
  const [overview, setOverview] = useState(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [assessmentError, setAssessmentError] = useState('');
  const [loading, setLoading] = useState(false);
  const refresh = useCallback(async () => {
    try { const data = await request('/api/overview'); setOverview(data); setError(''); }
    catch (e) { setOverview(null); setError(e.message || 'API unavailable'); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  function open(tab) { setPage(tab); setMenu(false); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  async function submit(values) {
    setLoading(true); setAssessmentError('');
    try { setResult(await request('/predict', { method: 'POST', body: JSON.stringify(values) })); }
    catch (e) { setAssessmentError(e.name === 'AbortError' ? 'Request timed out. Please retry.' : e.message); }
    finally { setLoading(false); }
  }
  return <div className="layout">
    <aside className={'sidebar ' + (menu ? 'shown' : '')}><div className="identity"><div className="identity-icon"><Activity size={20}/></div><div><strong>campus<span>.</span>signals</strong><small>SOCIAL CONNECTION LAB</small></div></div><div className="nav-caption">WORKSPACE</div><nav aria-label="Main navigation">{NAV.map(({id,title,Icon}) => <button key={id} className={'nav-link ' + (page === id ? 'selected' : '')} onClick={() => open(id)} aria-current={page === id ? 'page' : undefined}><Icon size={19}/>{title}{page === id && <ChevronRight className="nav-arrow" size={16}/>}</button>)}</nav><div className="side-footer"><div className="project-meta"><ShieldCheck size={21}/><div><strong>Academic research</strong><p>Privacy-first prototype</p></div></div><small>© 2026 CAMPUS SIGNALS</small></div></aside>
    {menu && <button className="mobile-shade" aria-label="Close navigation" onClick={() => setMenu(false)}/>}
    <div className="workspace"><header className="topbar"><div className="breadcrumbs"><button className="hamburger" aria-label="Toggle navigation" onClick={() => setMenu(!menu)}>{menu ? <X size={21}/> : <Menu size={21}/>}</button>Workspace <ChevronRight size={15}/> <strong>{NAV.find(n => n.id === page)?.title}</strong></div><div className="top-right"><span className={'health ' + (overview ? 'online' : '')}><i/>{overview ? 'Model online' : 'Model unavailable'}</span><span className="badge-avatar">CS</span></div></header><main className="main">
      {page === 'overview' && <Overview data={overview} error={error} refresh={refresh} open={open}/>}
      {page === 'method' && <Method open={open}/>}
      {page === 'assessment' && <div className="assessment-page"><div className="intro"><span className="eyebrow">PRIVATE REFLECTION / ASSESSMENT</span><h1>Your campus check-in.</h1><p>Reflect on your campus routine. Your responses are not stored by this application.</p></div>{overview?.mode === 'synthetic_demo' && <p className="demo-note"><ShieldCheck size={17}/> Demo mode: assessment is compared to synthetic data, not real students.</p>}{result ? <ResultsDashboard result={result} onReset={() => { setResult(null); setAssessmentError(''); }}/> : <>{assessmentError && <div className="warning" role="alert">{assessmentError}</div>}<AssessmentForm onSubmit={submit} isLoading={loading}/></>}</div>}
    </main></div>
  </div>;
}

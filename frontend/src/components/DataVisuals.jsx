import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, Cell } from 'recharts';

export const GROUPS = ['Lower Engagement', 'Study Focused', 'Socially Active'];
export const GROUP_COLORS = ['#ad896e', '#bdb09b', '#343a30'];
const bounds = (n) => Math.max(0, Math.min(100, Number(n) || 0));
const tipStyle = { border: '1px solid #e2d9cc', borderRadius: 10, background: '#fffdfa', color: '#36352f', fontSize: 12 };
const groupColor = (name) => GROUP_COLORS[GROUPS.indexOf(name)] || '#998773';

export function ClusterMap({ analytics, history = [], current = null, compact = false }) {
  const points = analytics?.points || [];
  const centerPoints = analytics?.centroids || [];
  const historyPoints = history.map((item, i) => ({
    x: bounds(item.scores?.['Social Density']),
    y: bounds(item.scores?.Wellbeing),
    category: item.category,
    label: 'Your check-in #' + (history.length - i),
    recorded: item.created_at,
  }));
  const currentPoint = current ? [{
    x: bounds(current.user_scores?.['Social Density']),
    y: bounds(current.user_scores?.Wellbeing),
    label: 'Current assessment',
    category: current.risk_category,
  }] : [];
  return <div className="cluster-viz">
    <div className="scatter-key">
      {GROUPS.map((name) => <span key={name}><i style={{background:groupColor(name)}}/>{name}</span>)}
      {(historyPoints.length > 0 || current) && <span><i style={{background:'#c58434'}}/>Your assessments</span>}
    </div>
    <div className={'scatter-graph ' + (compact ? 'scatter-compact' : '')} role="img" aria-label="Social engagement versus perceived social wellness, including synthetic cluster patterns and your recent assessments">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{top:12,right:18,bottom:30,left:0}}>
          <CartesianGrid stroke="#ede6db" strokeDasharray="3 3"/>
          <XAxis type="number" dataKey="x" name="Engagement" domain={[0,100]} tick={{fontSize:11,fill:'#716b63'}} tickLine={false} axisLine={false} label={{value:'Social engagement index (0–100)',position:'insideBottom',offset:-16,fontSize:11,fill:'#716b63'}}/>
          <YAxis type="number" dataKey="y" name="Wellness" domain={[0,100]} tick={{fontSize:11,fill:'#716b63'}} tickLine={false} axisLine={false} label={{value:'Social wellness (0–100)',angle:-90,position:'insideLeft',fontSize:11,fill:'#716b63',dy:65}}/>
          <Tooltip cursor={{strokeDasharray:'4 4'}} contentStyle={tipStyle} formatter={(v,name) => [Math.round(Number(v)) + ' / 100',name]}/>
          {GROUPS.map((name) => <Scatter key={name} name={name} data={(points.length ? points : centerPoints.filter(p=>p.name===name)).filter(p=>points.length ? p.cluster===name : true)} fill={groupColor(name)} fillOpacity={points.length ? .62 : 1} isAnimationActive={false}/>)}
          {historyPoints.length > 0 && <Scatter name="Your saved assessments" data={historyPoints} fill="#c58434" isAnimationActive={false}/>}
          {currentPoint.length > 0 && <Scatter name="Current assessment" data={currentPoint} fill="#df8c15" isAnimationActive={false}/>}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
    <p className="chart-caption">{points.length ? 'Each dot represents one generated example; orange dots represent assessments in your browser.' : 'Dots represent cluster averages only; no individual student survey records are shared.'} Horizontal position is derived social engagement, vertical position is self-reported social wellness. This does not imply clinical risk.</p>
  </div>;
}

export function DensityHistogram({ analytics }) {
  const bins = analytics?.density_histogram || [];
  return <div className="histogram-viz" role="img" aria-label="Distribution of derived social engagement indices in the reference dataset">
    <ResponsiveContainer width="100%" height={235}>
      <BarChart data={bins} margin={{top:14,right:12,bottom:12,left:-19}}>
        <CartesianGrid stroke="#eee8dd" vertical={false}/>
        <XAxis dataKey="range" tick={{fontSize:10,fill:'#746b60'}} tickLine={false} axisLine={false}/>
        <YAxis allowDecimals={false} tick={{fontSize:11,fill:'#746b60'}} tickLine={false} axisLine={false}/>
        <Tooltip contentStyle={tipStyle} formatter={v=>[v+' records','Count']}/>
        <Bar dataKey="count" radius={[4,4,0,0]} isAnimationActive={false}>{bins.map((b,i)=><Cell key={b.range} fill={i<3?'#d4bfa8':i<7?'#9e998b':'#373a32'}/>)}</Bar>
      </BarChart>
    </ResponsiveContainer>
    <p className="chart-caption">Counts by derived engagement decile. A higher index means more measured social activity, not necessarily improved health.</p>
  </div>;
}

function neutralHeat(count,max) {
  const intensity = max ? count/max : 0;
  const alpha = .08 + intensity * .85;
  return `rgba(51, 58, 45, ${alpha.toFixed(3)})`;
}
export function DensityHeatmap({ analytics }) {
  const values = analytics?.density_wellbeing_heatmap || [];
  const max = Math.max(0,...values.map(v=>v.count));
  return <div className="heatmap-wrap">
    <div className="heat-ylabel">Wellness ↑</div>
    <div className="heat-grid" role="img" aria-label="Five by five heatmap of reference counts by engagement and social wellness">
      {[4,3,2,1,0].map(y => [0,1,2,3,4].map(x=>{
        const cell = values.find(v=>v.x===x && v.y===y);
        return <div key={x+'-'+y} className="heat-cell" style={{background:neutralHeat(cell?.count||0,max),color:(max>0 && (cell?.count||0)/max>.55) ? "#fffdfa" : "#2a2f28"}} title={`Engagement ${x*20}–${(x+1)*20}, wellness ${y*20}–${(y+1)*20}: ${cell?.count||0} records`} aria-label={`${cell?.count||0} records at engagement band ${x+1}, wellness band ${y+1}`}>{cell?.count||0}</div>;
      }))}
    </div>
    <div className="heat-axis"><span>Lower engagement</span><span>Higher engagement →</span></div>
    <p className="chart-caption">Cell counts across all reference records. Darker means more examples, not greater risk.</p>
  </div>;
}
function correlationColor(v) {
  const n = Math.max(-1,Math.min(1,Number(v)||0));
  if(n>=0) return `rgba(49, 70, 54, ${(.09+.75*n).toFixed(2)})`;
  return `rgba(169, 110, 76, ${(.09+.75*Math.abs(n)).toFixed(2)})`;
}
export function CorrelationHeatmap({ analytics }) {
  const labels = analytics?.correlation?.labels || [];
  const rows = analytics?.correlation?.values || [];
  return <div className="correlation-wrapper">
    <div className="correlation-scroll"><table className="correlation-table" aria-label="Correlation matrix of seven dataset indicators"><thead><tr><th>Indicator</th>{labels.map((label,i)=><th key={i} title={label}>{label}</th>)}</tr></thead>
      <tbody>{labels.map((label,r)=><tr key={label}><th>{label}</th>{labels.map((column,c)=>{
        const n = rows[r]?.[c] ?? 0;
        return <td key={column} style={{background:correlationColor(n),color:Math.abs(n)>.60 ? "#fffdfa" : "#292b25"}} title={label+' × '+column+': '+Number(n).toFixed(2)}>{Number(n).toFixed(2)}</td>;
      })}</tr>)}</tbody></table></div>
    <div className="correlation-legend"><span className="negative-key"/> Negative relationship <span className="nearzero-key"/> Near zero <span className="positive-key"/> Positive relationship</div>
    <p className="chart-caption">Pearson correlations between selected inputs, from −1 to +1. Correlation does not show causation.</p>
  </div>;
}

const SIGNAL_DESCRIPTIONS = {
  'Social Density':'Derived engagement index',
  'Academic Focus':'Relative study time',
  'Social Activity':'Meals, events, outings',
  'Wellbeing':'Self-rated social wellness',
  'Sleep Duration':'Hours, not sleep quality',
  'Screen Time':'Use level, not a benefit',
};
export function SignalTiles({ userScores = {}, baseline = {} }) {
  const order = ['Social Density','Social Activity','Wellbeing','Academic Focus','Sleep Duration','Screen Time'];
  return <div className="signal-tiles">{order.filter(k=>Number.isFinite(userScores[k])).map(key=>{
    const v = bounds(userScores[key]);
    const avg = bounds(baseline[key]);
    return <div className="signal-tile" key={key}><span>{key}</span><div className="signal-tile-score">{Math.round(v)} <small>/100</small></div><div className="signal-tile-bottom"><span className="signal-swatch" style={{background:neutralHeat(v,100)}}/><span>Reference {Math.round(avg)}</span></div><p>{SIGNAL_DESCRIPTIONS[key]}</p></div>;
  })}</div>;
}

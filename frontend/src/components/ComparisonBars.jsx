const EXPLANATIONS = {
  'Social Density': 'Derived social engagement index',
  'Academic Focus': 'Relative self-study time',
  'Screen Time': 'Phone usage; higher is not necessarily better',
  'Wellbeing': 'Self-reported social wellness',
  'Sleep Duration': 'Relative hours slept, not quality',
  'Social Activity': 'Meals, events and social outings',
};
const bounded = (n) => Math.max(0, Math.min(100, Number(n) || 0));

export default function ComparisonBars({ userScores = {}, campusBaseline = {} }) {
  return <div className="compare-list">
    <div className="compare-legend"><span><i className="legend-you"/> Your result</span><span><i className="legend-demo"/> Synthetic benchmark</span></div>
    {Object.entries(userScores).map(([label, value]) => <div className="compare-item" key={label}>
      <div className="compare-top"><div><strong>{label}</strong><small>{EXPLANATIONS[label] || 'Relative indicator'}</small></div><span>{Math.round(bounded(value))} / {Math.round(bounded(campusBaseline[label]))}</span></div>
      <div className="compare-tracks" aria-label={label + ': you ' + Math.round(bounded(value)) + ', synthetic benchmark ' + Math.round(bounded(campusBaseline[label]))}>
        <div className="track"><span className="track-you" style={{ width: bounded(value) + '%' }}/></div>
        <div className="track"><span className="track-demo" style={{ width: bounded(campusBaseline[label]) + '%' }}/></div>
      </div>
    </div>)}
    <p className="panel-footnote">The two values per row are Your result / Synthetic benchmark, normalized to 0–100. They are not percentages of students or clinical scores.</p>
  </div>;
}

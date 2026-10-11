// Browser-local, result-summary-only history. No survey answers are saved.
const KEY = 'campus-signals-assessments-v1';
export const HISTORY_LIMIT = 250;

export function readHistory() {
  try {
    const records = JSON.parse(window.localStorage.getItem(KEY) || '[]');
    if (!Array.isArray(records)) return [];
    return records.filter(r =>
      r && typeof r.id === 'string' && typeof r.created_at === 'string'
      && Number.isFinite(r.density) && typeof r.category === 'string'
    ).slice(0, HISTORY_LIMIT);
  } catch { return []; }
}

export function storeAssessment(result) {
  const record = {
    id: result.user_id,
    created_at: new Date().toISOString(),
    category: result.risk_category,
    cluster: result.cluster_label,
    density: Number(result.social_density_score),
    anomaly: Number(result.anomaly_score),
    scores: Object.fromEntries(
      Object.entries(result.user_scores || {}).filter(([, v]) => Number.isFinite(v))
    ),
  };
  // The same request/session is never saved twice.
  const records = [record, ...readHistory().filter(r => r.id !== record.id)].slice(0, HISTORY_LIMIT);
  window.localStorage.setItem(KEY, JSON.stringify(records));
  return records;
}

export function eraseHistory() {
  window.localStorage.removeItem(KEY);
}

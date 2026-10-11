import test from 'node:test';
import assert from 'node:assert/strict';
import { readHistory, storeAssessment, eraseHistory, HISTORY_LIMIT } from '../src/history.js';

const items = new Map();
globalThis.window = {
  localStorage: {
    getItem(key) { return items.get(key) ?? null; },
    setItem(key, value) { items.set(key, value); },
    removeItem(key) { items.delete(key); }
  }
};
const sample = (id) => ({
  user_id: id,
  social_density_score: 0.65,
  anomaly_score: -0.075,
  risk_category: 'Socially Active',
  cluster_label: 'Socially Active',
  user_scores: {'Social Density': 80},
  // These must NEVER enter storage.
  friend_group_size: 13,
  loneliness_score: 5,
});

test('each successful assessment creates a local summary only', () => {
  eraseHistory();
  storeAssessment(sample('one'));
  const list = storeAssessment(sample('two'));
  assert.equal(list.length, 2);
  assert.equal(readHistory()[0].id, 'two');
  const text = [...items.values()].join('');
  assert.ok(!text.includes('friend_group_size'));
  assert.ok(!text.includes('loneliness_score'));
  assert.ok(!text.includes('13'));
});

test('idempotent session writes and bounded retained history', () => {
  eraseHistory();
  storeAssessment(sample('same'));
  storeAssessment(sample('same'));
  assert.equal(readHistory().length, 1);
  for (let i = 0; i < HISTORY_LIMIT + 4; i++) storeAssessment(sample('entry-'+i));
  assert.equal(readHistory().length, HISTORY_LIMIT);
  assert.equal(readHistory()[0].id, 'entry-'+(HISTORY_LIMIT+3));
});

test('history can be erased; malformed data is ignored', () => {
  eraseHistory();
  assert.deepEqual(readHistory(), []);
  items.set('campus-signals-assessments-v1', '{not valid JSON');
  assert.deepEqual(readHistory(), []);
  eraseHistory();
  assert.equal(items.size, 0);
});

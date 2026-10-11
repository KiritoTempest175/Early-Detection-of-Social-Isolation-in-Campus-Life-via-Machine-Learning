import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip
} from 'recharts';

export default function RadarChartComponent({ userScores, campusBaseline }) {
  // Combine user scores and campus baseline into an array of objects for recharts
  const data = Object.keys(userScores).map((key) => {
    // Format key for display (e.g. "social_wellness" -> "Social Wellness")
    const formattedKey = key
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    return {
      subject: formattedKey,
      User: userScores[key],
      Baseline: campusBaseline[key] || 0,
    };
  });

  return (
    <div style={{ width: '100%', height: 330, marginTop: '0.5rem' }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="var(--color-border)" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />

          <Radar
            name="You"
            dataKey="User"
            stroke="#292a24"
            fill="#292a24"
            fillOpacity={0.6}
          />
          <Radar
            name="Campus Baseline"
            dataKey="Baseline"
            stroke="#b7a38a"
            fill="#dfd0b9"
            fillOpacity={0.4}
          />

          <Tooltip
            contentStyle={{
              backgroundColor: '#fffdfa',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              boxShadow: 'var(--shadow-sm)'
            }}
          />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

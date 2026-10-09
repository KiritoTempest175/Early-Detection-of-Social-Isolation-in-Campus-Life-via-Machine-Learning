import React from 'react';
import { Activity } from 'lucide-react';

export default function Header() {
  return (
    <header style={{
      borderBottom: '1px solid var(--color-border)',
      padding: '1rem 2rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
      backgroundColor: '#ffffff'
    }}>
      <Activity size={24} color="var(--color-text)" />
      <h1 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>Campus AI</h1>
    </header>
  );
}

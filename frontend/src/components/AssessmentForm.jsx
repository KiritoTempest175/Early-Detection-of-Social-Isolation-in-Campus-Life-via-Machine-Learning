import React, { useState } from 'react';
import { ChevronRight, Loader2 } from 'lucide-react';

export default function AssessmentForm({ onSubmit, isLoading }) {
  const [formData, setFormData] = useState({
    living_status: 'Hostelite',
    friend_group_size: 3,
    friend_influence: 3,
    cgpa: 3.0,
    class_attendance: 4,
    daily_self_study_hours: 2,
    society_memberships: 1,
    society_events_attended: 2,
    meals_with_friends: 5,
    hours_alone_weekly: 10,
    common_room_frequency: 3,
    social_outings: 2,
    assignments_due: 3,
    workload_stress: 5,
    workload_management: 3,
    screen_time: 4,
    sleep_hours: 7,
    social_wellness_rating: 3,
    mood_rating: 7,
    loneliness_score: 2,
    peer_satisfaction: 7,
    peer_helpfulness: 4,
  });

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' || type === 'range' ? Number(value) : value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="container animate-in">
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Your social assessment</h2>
        <p>Reflect on your recent campus routines. Responses are processed in memory, not saved. Results are exploratory and non-clinical.</p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-8">

        {/* Section 1: Academic & lifestyle */}
        <section className="card flex flex-col gap-4">
          <h3 style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Academic & Lifestyle</h3>

          <div className="flex gap-4" style={{ flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 45%' }}>
              <label>Living Status</label>
              <select name="living_status" value={formData.living_status} onChange={handleChange} required>
                <option value="Hostelite">Hostelite</option>
                <option value="Day Scholar">Day Scholar</option>
              </select>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Current CGPA (0.0 - 4.0)</label>
              <input type="number" name="cgpa" step="0.01" min="0" max="4.0" value={formData.cgpa} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Class Attendance (1-5)</label>
              <input type="number" name="class_attendance" min="1" max="5" value={formData.class_attendance} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Daily Self-Study Hours</label>
              <input type="number" name="daily_self_study_hours" min="0" value={formData.daily_self_study_hours} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Assignments Due This Week</label>
              <input type="number" name="assignments_due" min="0" value={formData.assignments_due} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Sleep Hours (Weekdays)</label>
              <input type="number" name="sleep_hours" min="0" value={formData.sleep_hours} onChange={handleChange} required />
            </div>
          </div>
        </section>

        {/* Section 2: Social Activity */}
        <section className="card flex flex-col gap-4">
          <h3 style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Social Activity</h3>

          <div className="flex gap-4" style={{ flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 45%' }}>
              <label>Friend Group Size</label>
              <input type="number" name="friend_group_size" min="0" value={formData.friend_group_size} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Meals with Friends (per week)</label>
              <input type="number" name="meals_with_friends" min="0" value={formData.meals_with_friends} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Social Outings (This week)</label>
              <input type="number" name="social_outings" min="0" value={formData.social_outings} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Society Memberships</label>
              <input type="number" name="society_memberships" min="0" value={formData.society_memberships} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Society Events Attended (Semester)</label>
              <input type="number" name="society_events_attended" min="0" value={formData.society_events_attended} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Common Room Frequency (1-5)</label>
              <input type="number" name="common_room_frequency" min="1" max="5" value={formData.common_room_frequency} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Hours Alone Weekly</label>
              <input type="number" name="hours_alone_weekly" min="0" value={formData.hours_alone_weekly} onChange={handleChange} required />
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Daily Screen Time (Hours)</label>
              <input type="number" name="screen_time" min="0" value={formData.screen_time} onChange={handleChange} required />
            </div>
          </div>
        </section>

        {/* Section 3: Well-being & peers */}
        <section className="card flex flex-col gap-4">
          <h3 style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Well-being & Perceptions</h3>

          <div className="flex gap-4" style={{ flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 45%' }}>
              <label>Friend Influence (1-5)</label>
              <input type="range" name="friend_influence" min="1" max="5" value={formData.friend_influence} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.friend_influence}</div>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Workload Stress (1-10)</label>
              <input type="range" name="workload_stress" min="1" max="10" value={formData.workload_stress} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.workload_stress}</div>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Workload Management (1-5)</label>
              <input type="range" name="workload_management" min="1" max="5" value={formData.workload_management} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.workload_management}</div>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Social Wellness Rating (1-5)</label>
              <input type="range" name="social_wellness_rating" min="1" max="5" value={formData.social_wellness_rating} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.social_wellness_rating}</div>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Mood Rating (1-10)</label>
              <input type="range" name="mood_rating" min="1" max="10" value={formData.mood_rating} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.mood_rating}</div>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Loneliness Score (1-5)</label>
              <input type="range" name="loneliness_score" min="1" max="5" value={formData.loneliness_score} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.loneliness_score}</div>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Peer Satisfaction (1-10)</label>
              <input type="range" name="peer_satisfaction" min="1" max="10" value={formData.peer_satisfaction} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.peer_satisfaction}</div>
            </div>

            <div style={{ flex: '1 1 45%' }}>
              <label>Peer Helpfulness (1-5)</label>
              <input type="range" name="peer_helpfulness" min="1" max="5" value={formData.peer_helpfulness} onChange={handleChange} style={{ width: '100%' }} />
              <div style={{ textAlign: 'center', fontSize: '0.8rem' }}>{formData.peer_helpfulness}</div>
            </div>
          </div>
        </section>

        <div className="flex justify-center mt-4 mb-8">
          <button type="submit" className="btn-primary flex items-center gap-2" disabled={isLoading} style={{ width: '100%', maxWidth: '300px', justifyContent: 'center' }}>
            {isLoading ? <Loader2 className="animate-spin" size={20} /> : 'View engagement patterns'}
            {!isLoading && <ChevronRight size={20} />}
          </button>
        </div>

      </form>
    </div>
  );
}

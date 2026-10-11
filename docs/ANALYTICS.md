# Dataset analytics, visualization, and assessment results

The public frontend uses a deliberately generated 520-record demo. All 520 **synthetic** entries are plotted in the social-engagement vs. social-wellness scatter graph. The dashboard includes:
- A cluster distribution donut chart, descriptive 3-cluster breakdown, and survey KPI overview
- A 10-bin social engagement histogram
- A 5×5 engagement/wellness count heatmap
- A 7-feature Pearson correlation heatmap (correlations are not causal)
- A cluster scatter plot with each synthetic example and the local browser's saved assessments
- A trend line, counts, profile mix, and the full local history table (up to retained history limit).

Each result view shows an orange point on the same cohort scatter chart, the engagement percentile of the **reference distribution** (not medical-risk probability), compact normalized indicators, and contextual reflection suggestions. An Isolation Forest anomaly is *not* a wellness diagnosis or isolation classification.

Privacy model:
- The synthetic dataset does not describe real individuals or campus demographics.
- In a future authorized survey deployment, `/api/analytics` exposes **aggregated cells, correlation coefficients and cluster centroids only**, never survey-row points.
- The backend does **not** store questionnaire answers, prediction history, or user profiles.
- Each user's own result summaries are stored only in that browser's localStorage; other visitors never see them. Session data can be erased with Clear History.
- No authentication or shared cross-device record history has been implemented; adding it would require explicit consent, access controls, encryption, retention/deletion policies and institutional ethics review.

The user-facing recurring synthetic banner has been removed. Data provenance remains labeled in the Overview hero, dataset section, and methodology because synthetic research metrics must not be presented as real campus observations.

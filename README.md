# Campus Signals — Social Connection Research Dashboard

**An exploratory campus-engagement ML research project.** Responsive beige-and-charcoal React dashboard; FastAPI API; unsupervised K-Means and Isolation Forest inference. Built for coursework and portfolio demonstration.

**Authors:** Muhammad Mobeen, Huzaifa Zaman  
**Course:** Machine Learning (NUTECH, BSAI-2024)  
**Instructor:** Lec Faiza Khan

> **Research limitations:** This system is not a medical device, a clinical screening test, a validated predictor of future isolation, or a decision system for students. The questionnaire includes directly self-reported loneliness, so it does not independently detect that condition. Do not use outputs for surveillance, admissions, discipline, grading, or targeted interventions. Use only with consent.

## Product

- Overview dashboard: total survey records, average derived social density, weekly hours alone, high self-reported loneliness frequency, aggregate cluster distribution, and normalized sample comparisons.
- Accessible assessment of 22 survey inputs with instant, transient profile results and comparison radar chart.
- Methodology, limitations, voluntary participation and privacy disclosures.
- Explicitly labeled **synthetic demo mode**, so the public preview works without exposing student records.
- Responsive UI with minimal beige, warm neutral, and charcoal design. No tracking SDK or user account is required.
- JSON API with CORS allowlist, basic IP throttling, restricted input ranges, security response headers, and sanitized server errors.

## Structure

```
backend/    FastAPI endpoints, inference engine and Pydantic schemas
frontend/   Vite / React dashboard, assessments, and charts
src/        Original research pipelines (standalone; require local authorized data)
tests/      API validation and privacy smoke tests
.github/    GitHub Actions build and test workflow
```

The two research pipelines remain in `src/`. The independent supervised comparison evaluates campus loneliness, stress type, and depression self-report as three **different** tasks with separate splits and preprocessing; it does not concatenate incompatible records or targets. The dashboard uses a separate unsupervised exploratory model.

To run the offline research scripts locally, install `research-requirements.txt` and place authorized private CSV files under `dataset/`:\n\n```bash\npip install -r research-requirements.txt\npython src/multi_dataset_pipeline.py\n# Or run the original exploratory clustering experiment:\npython src/social_isolation_pipeline.py\n```\n\nReports are generated under `outputs/` and exclude individual predictions. Treat performance numbers as internal exploratory holdout scores.\n\n## Local setup

### 1. API

Python 3.11+:

```bash
python -m venv .venv
# Activate .venv for your platform
pip install -r backend/requirements.txt
```

For a **synthetic demo** (not actual student data):

```bash
# Linux / macOS
DEMO_MODE=true uvicorn backend.main:app --reload
```

Windows PowerShell:

```powershell
$env:DEMO_MODE = "true"
python -m uvicorn backend.main:app --reload
```

To use the authorized **real survey**, set `DEMO_MODE=false` and `DATASET_PATH` to an accessible, private location of `ML-Dataset.csv`. The backend reads the data at startup, strips name/timestamp/institution metadata immediately after preprocessing, and retains only numerical modeling fields and living-status grouping in memory. It does not expose individual training rows.

The raw survey includes directly identifying fields, and its storage, permission and deletion requirements are your responsibility. Do **not** commit it, upload it to a public host, or put it in the frontend's public folder. See `.gitignore`.

### 2. Frontend

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/predict`, `/health`, and `/api/*` to port 8000 during development.

For a separately hosted frontend, set `VITE_API_URL` to the actual HTTPS FastAPI origin when building. It is a **public frontend setting, not a secret**. Set `CORS_ORIGINS` on the backend to the exact frontend origin(s), comma separated.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Lightweight service details |
| GET | `/health` | Model status and whether input is survey or synthetic demo |
| GET | `/api/overview` | Aggregate-only, non-identifying training sample statistics |
| POST | `/predict` | Validated assessment, transient cluster profile and recommendation |
| GET | `/docs` | OpenAPI interface |

The model returns 503 when neither a usable private dataset nor an explicitly enabled demo is available. It never silently fabricates a survey. The frontend renders a truthful empty state.

**Privacy:** No assessment answers are saved to disk or written to a training CSV. In-memory inference returns an ephemeral session ID. Logs should never include submitted payloads. For a real campus deployment, apply consent and retention policies, secure the private source data, add deployment-edge rate limiting, monitoring without personal data, access controls for non-public aggregate statistics, and independent model evaluation before use.

**ML caveats:** Cluster semantics are heuristic; Isolation Forest scores are anomaly decision values, not probabilities; normalized radar metrics may be misleading outside the original data distribution. Real-world representativeness, subgroup fairness, calibration, repeatability, independent validation and security review have not yet been established.

## CI

The GitHub Actions workflow runs API tests, frontend build, and frontend lint. Run locally:

```bash
pip install pytest httpx
python -m pytest -q tests
cd frontend && npm ci && npm run build && npm run lint
```

## Hosting

- **Render demo API:** `render.yaml` includes an explicitly synthetic mode deployment, with `DEMO_MODE=true`. It does *not* imply clinical or survey validity.
- **Netlify / similar static hosts:** Build `frontend/` with `npm run build` and publish `frontend/dist`. Set `VITE_API_URL` to the Render HTTPS URL; `frontend/public/_redirects` supports SPA fallback.
- Never store student survey data inside build artifacts, Git history, client bundles, or public repository settings.

## Data

The following source files must remain private and are **not** distributed in the Git repository:

- `ML-Dataset.csv` — named campus student survey used by the dashboard research engine
- `Stress Dataset.csv` — unrelated survey for a separate research pipeline
- `Student Mental health.csv` — unrelated health questionnaire for separate research comparison

Do not evaluate supervised models with target-derived predictors, perform preprocessing on evaluation data before splitting, or report research metrics as verified deployment accuracy without a reproducible, leak-free holdout evaluation.

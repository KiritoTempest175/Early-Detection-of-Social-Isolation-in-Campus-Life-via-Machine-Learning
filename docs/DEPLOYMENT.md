# Campus Signals hosting

## Service URLs
- **Frontend project:** https://app.netlify.com/projects/campus-signals
- **Desired public frontend domain:** https://campus-signals.huzaifazaman.com
- **Built-in frontend domain:** https://campus-signals.netlify.app
- **Backend API:** https://campus-signals-api.onrender.com
- **Backend health:** https://campus-signals-api.onrender.com/health
- **Render dashboard:** https://dashboard.render.com/web/srv-db5kfah42hec7399ptig

## Render
The API was created from the repository's real default branch `kiritotempest175-add-ml-pipeline` with:
- Python 3.11, free plan, Singapore region, auto deploy on commit.
- Build: `pip install -r backend/requirements.txt`
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- `DEMO_MODE=true` (only deterministic synthetic records; **no student survey files**).
- `CORS_ORIGINS=https://campus-signals.huzaifazaman.com,https://campus-signals.netlify.app`

Render's free plan may spin down after inactivity. The first request after idle can be slow.

## Netlify
The Netlify project exists but needs Git-backed deployment from the dashboard:

1. Open [campus-signals](https://app.netlify.com/projects/campus-signals).
2. Connect the GitHub repository `KiritoTempest175/Early-Detection-of-Social-Isolation-in-Campus-Life-via-Machine-Learning`.
3. Choose **production branch** `kiritotempest175-add-ml-pipeline` (this is currently the actual default branch, not `main`).
4. The repository-root `netlify.toml` defines base directory `frontend`, command `npm ci && npm run build`, output `frontend/dist`, and the public `VITE_API_URL` value.
5. Confirm the first deploy is `Ready` and that the site calls `/api/overview` on Render, **not** localhost.
6. In Netlify's **Domain management**, add `campus-signals.huzaifazaman.com`. Confirm the exact target Netlify gives you, then add a matching CNAME record at the DNS provider that manages `huzaifazaman.com`. Usually the CNAME points at `campus-signals.netlify.app`, but the Netlify verification screen is the source of truth.
7. Wait for DNS verification and the HTTPS certificate to be issued. Test the HTTPS frontend and assessment in a real browser.

The project should stay public only with synthetic demo data until review of student privacy, informed consent and model evaluation.

## Quick checks

```bash
curl -fsS https://campus-signals-api.onrender.com/health
curl -fsS https://campus-signals-api.onrender.com/api/overview
```

Expected health: `{"status":"ready","mode":"synthetic_demo"}`. Dashboard shows a prominent synthetic-data disclosure.

**Never upload the three survey CSV files to GitHub, Netlify public assets or the Render demo instance.**

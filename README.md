# Early Detection of Social Isolation in Campus Life via Machine Learning

A robust, modular ML pipeline for detecting social isolation anomalies among university students using behavioral, academic, and social indicators.

**Authors:** Muhammad Mobeen, Huzaifa Zaman
**Course:** Machine Learning (NUTECH, BSAI-2024)
**Instructor:** Lec Faiza Khan

---

## Project Structure

```
ML_pipeline/
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
│
├── src/                       # Python ML pipeline scripts
│   ├── social_isolation_pipeline.py   # Unsupervised pipeline (K-Means + Isolation Forest)
│   └── multi_dataset_pipeline.py      # Supervised pipeline (5 classifiers, 2 dataset sets)
│
├── dataset/                   # Local, private datasets (not included in Git)
│
├── outputs/                   # Generated plots and reports (not included in Git)
│
├── backend/                   # FastAPI backend (REST API)
│   ├── main.py                        # Application entry point
│   ├── ml_engine.py                   # ML model engine
│   ├── models.py                      # Pydantic data models
│   ├── router.py                      # API routes
│   ├── privacy.py                     # Privacy utilities
│   └── requirements.txt               # Python dependencies
│
├── frontend/                  # Vite + React web dashboard
│   ├── src/                           # React components
│   ├── public/                        # Static assets
│   └── package.json                   # Node.js dependencies
```

Survey datasets and generated results are intentionally excluded: the source
surveys contain identifying and sensitive student information. Use only data
you are authorized to process, keep it under `dataset/`, and do not commit it.

---

## Quick Start

Place the authorized `ML-Dataset.csv` file in `dataset/` before starting the
backend or running the social-isolation pipeline. To run the multi-dataset
pipeline, also place `Stress Dataset.csv` and `Student Mental health.csv` there.
These files are not distributed with this repository.

Install the Python dependencies once:
```bash
pip install -r backend/requirements.txt
```

### 1. Run the Social Isolation Pipeline (Unsupervised)
```bash
python src/social_isolation_pipeline.py
```

### 2. Run the Multi-Dataset Comparison Pipeline (Supervised)
```bash
python src/multi_dataset_pipeline.py
```

### 3. Start the Backend API
```bash
cd backend
uvicorn main:app --reload
```

### 4. Start the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```

---

## Pipelines

### Social Isolation Pipeline (`src/social_isolation_pipeline.py`)
- **Phase 1:** Data Ingestion & Preprocessing (CGPA sanitization, contextual imputation, outlier clipping)
- **Phase 2:** Feature Engineering (Social Density Score)
- **Phase 3:** K-Means Clustering + Isolation Forest anomaly detection
- **Phase 4:** False Positive Mitigation (High-Academic Exception Filter)
- **Phase 5:** Evaluation (PR Curves, ROC-AUC, MAE, Silhouette, Cross-Validation)

### Multi-Dataset Pipeline (`src/multi_dataset_pipeline.py`)
Trains 5 classifiers on 2 dataset sets:
- **Set 1:** ML-Dataset.csv (target: loneliness >= 4)
- **Set 2:** Stress Dataset + Student Mental Health (target: distress/depression)

**Models:** Random Forest, Logistic Regression, SVM, KNN, Gradient Boosting

**Output:** Side-by-side comparison report with accuracy, precision, recall, F1, confusion matrices, ROC curves.

---

## Tech Stack

- **Python:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Vite, React

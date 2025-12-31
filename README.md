# Primeira Liga — Match Outcome ML Prediction

**Description:**
- Lightweight project to build match-level features from Football-Data CSVs, train a simple model (odds-only or feature-based), and run batch predictions for upcoming fixtures.

**Requirements:**
- Python 3.8+ (tested). Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Quick Usage**

- Build consolidated league history (aggregates seasons and rolling features):

```bash
python3 scripts/build_league_history.py
```

- Build upcoming matches (odds-only CSV) from a season file:

```bash
python3 scripts/build_upcoming_matches.py --season_csv data/raw/P1_2526.csv --round 16 --out data/processed/upcoming_matches.csv
```

- Train a model (example):

```bash
python3 training/train_model.py
```

- Evaluate a trained model:

```bash
python3 training/evaluate_model.py
```

- Batch predict using a trained model and the upcoming matches CSV:

```bash
python3 inference/batch_predict.py --model model/match_logreg.pkl --input data/processed/upcoming_matches.csv --output data/processed/predicted_matches.csv
```

**Files & Outputs**
- `data/raw/`: raw season CSVs (Football-Data format).
- `data/processed/league_history.csv`: consolidated match features (produced by `scripts/build_league_history.py`).
- `data/processed/upcoming_matches.csv`: odds-only fixtures for prediction (produced by `scripts/build_upcoming_matches.py`).
- `data/processed/predicted_matches.csv`: predictions produced by `inference/batch_predict.py`.
- `model/`: trained model artifacts (pickles).

**Notes & Recommendations**
- A `.gitignore` was added to exclude raw data and model pickles; keep large datasets out of source control.
- `requirements.txt` lists minimal dependencies (`pandas`, `numpy`, `scikit-learn`, `joblib`). Pin versions if you need reproducible environments.
- Date parsing uses `dayfirst=True` and falls back to `dateutil`; ensure input CSV dates match Football-Data formatting.

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, log_loss

FEATURES = ["AvgH", "AvgD", "AvgA"]
TARGET = "FTR"  # H / D / A

DATA_PATH = Path("data/processed/league_history.csv")
MODEL_PATH = Path("model/match_logreg.pkl")
METRICS_PATH = Path("model/odds_only_metrics.json")

def get_temporal_split(df: pd.DataFrame):
    # Use last season as test if available
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique().tolist())
        test_season = seasons[-1]
        train_df = df[df["season"] != test_season].copy()
        test_df = df[df["season"] == test_season].copy()
        return train_df, test_df, f"season_last_as_test({test_season})"
    # Fallback: last 20% by date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy(), "last_20_percent_as_test"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only relevant columns
    df = df[[c for c in df.columns if c in FEATURES + [TARGET, "season", "date"]]].copy()

    train_df, test_df, split_name = get_temporal_split(df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(str)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(str)

    # Preprocess: impute missing odds + scale
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), FEATURES)
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        class_weight=None,
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.named_steps["clf"].classes_)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    ll = log_loss(y_test, proba, labels=classes)

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=classes).tolist()

    metrics = {
        "model": "odds_only_logreg",
        "split": split_name,
        "features": FEATURES,
        "classes": classes,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "log_loss": ll,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete")
    print(f"- Model saved to: {MODEL_PATH}")
    print(f"- Metrics saved to: {METRICS_PATH}")
    print(f"- Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | LogLoss: {ll:.4f}")

if __name__ == "__main__":
    main()
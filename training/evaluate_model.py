import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

FEATURES = [
    "home_points_last5",
    "home_gf_last5",
    "home_ga_last5",
    "home_rest_days",
    "away_points_last5",
    "away_gf_last5",
    "away_ga_last5",
    "away_rest_days",
    "AvgH",
    "AvgD",
    "AvgA",
]

TARGET = "FTR"  # H / D / A

DATA_PATH = Path("data/processed/league_history.csv")
MODEL_PATH = Path("model/match_logreg.pkl")
OUT_METRICS_PATH = Path("model/metrics_evaluation.json")
OUT_TEST_PREDS_PATH = Path("data/test_set_predictions.csv")


def get_classes(model):
    """Get class order robustly for Pipeline or plain estimator."""
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps") and "clf" in model.named_steps and hasattr(model.named_steps["clf"], "classes_"):
        return list(model.named_steps["clf"].classes_)
    raise AttributeError("Could not find model classes_ attribute.")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Prefer a temporal split if 'season' exists (more realistic),
    # otherwise use a stratified random split.
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique().tolist())
        test_season = seasons[-1]
        train_df = df[df["season"] != test_season].copy()
        test_df = df[df["season"] == test_season].copy()
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[TARGET].astype(str)
        )

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(str)

    model = joblib.load(MODEL_PATH)

    classes = get_classes(model)

    # Predict labels and probabilities
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # Core metrics
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=classes).tolist()

    # Probabilistic metrics
    ll = log_loss(y_test, proba, labels=classes)

    # Multiclass ROC AUC (one-vs-rest, macro)
    try:
        roc_auc_macro = roc_auc_score(
            y_test, proba, labels=classes, multi_class="ovr", average="macro"
        )
    except Exception:
        roc_auc_macro = None

    # Multiclass PR AUC (macro, one-vs-rest)
    y_bin = label_binarize(y_test, classes=classes)  # shape (n_samples, n_classes)
    pr_auc_macro = average_precision_score(y_bin, proba, average="macro")

    # Class distribution (useful instead of "positive_rate")
    class_dist = y_test.value_counts(normalize=True).to_dict()

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "log_loss": ll,
        "roc_auc_ovr_macro": roc_auc_macro,
        "pr_auc_ovr_macro": pr_auc_macro,
        "classes": classes,
        "class_distribution_test": class_dist,
        "classification_report": report,
        "confusion_matrix": cm,
        "n_test": int(len(y_test)),
        "split": "season_last_as_test" if "season" in df.columns else "stratified_random_80_20",
    }

    OUT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions (include identifiers if available)
    keep_cols = []
    for c in ["date", "home_team", "away_team", "season"]:
        if c in test_df.columns:
            keep_cols.append(c)

    out = test_df[keep_cols + FEATURES].copy()
    out["true_FTR"] = y_test.values
    out["pred_FTR"] = preds

    # Save probabilities with correct class mapping
    for idx, cls in enumerate(classes):
        out[f"pred_proba_{cls}"] = proba[:, idx]

    OUT_TEST_PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TEST_PREDS_PATH, index=False)

    print("✅ Evaluation complete")
    print(f"- Metrics saved to: {OUT_METRICS_PATH}")
    print(f"- Test predictions saved to: {OUT_TEST_PREDS_PATH}")
    print(f"- Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | LogLoss: {ll:.4f}")


if __name__ == "__main__":
    main()
from pathlib import Path
import argparse
import joblib
import pandas as pd

# Odds-only features (must match training)
FEATURES = ["AvgH", "AvgD", "AvgA"]

# Default paths
DEFAULT_MODEL_PATH = Path("model/match_logreg.pkl")
DEFAULT_INPUT_PATH = Path("data/processed/upcoming_matches.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/predicted_matches.csv")

# Optional columns to keep in output if present
ID_COLS = ["date", "season", "home_team", "away_team"]

# Columns that should never be used as model inputs
LABEL_LIKE_COLS = ["FTR", "true_FTR", "result", "label"]

def get_classes(model):
    """
    Get class order robustly for a Pipeline or a plain estimator.
    For LogisticRegression inside a Pipeline, classes are stored on the final estimator.
    """
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return list(clf.classes_)

    raise AttributeError("Could not find model classes_ attribute.")


def main():
    parser = argparse.ArgumentParser(description="Batch prediction for match outcome (odds-only).")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained model .pkl")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Path to upcoming_matches.csv")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to output predicted_matches.csv")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    # --- Safety checks ---
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path} (train odds-only model first)")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # --- Load model and input ---
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)

    # Drop label-like columns if someone accidentally includes them
    cols_to_drop = [c for c in LABEL_LIKE_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Validate feature columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns in input: {missing}\n"
            f"Expected features: {FEATURES}\n"
            f"Input columns: {list(df.columns)}"
        )

    # Prepare feature matrix
    X = df[FEATURES].copy()

    # --- Predict ---
    preds = model.predict(X)
    proba = model.predict_proba(X)
    classes = get_classes(model)  # probability columns follow this order

    # --- Build output ---
    out_cols = [c for c in ID_COLS if c in df.columns]
    out = df[out_cols].copy() if out_cols else pd.DataFrame(index=df.index)

    out["pred_FTR"] = preds

    # Add per-class probabilities, correctly mapped
    for idx, cls in enumerate(classes):
        out[f"pred_proba_{cls}"] = proba[:, idx]

    # Confidence + margin (helps interpretation)
    proba_cols = [f"pred_proba_{c}" for c in classes]
    out["pred_confidence"] = out[proba_cols].max(axis=1)
    out["pred_margin"] = out[proba_cols].apply(lambda r: r.nlargest(2).iloc[0] - r.nlargest(2).iloc[1], axis=1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("✅ Batch prediction complete (odds-only)")
    print(f"- Model:  {model_path}")
    print(f"- Input:  {input_path} (rows: {len(df)})")
    print(f"- Output: {output_path}")

    # Print a helpful preview
    preview_cols = [c for c in ["date", "home_team", "away_team", "pred_FTR"] if c in out.columns]
    for c in ["H", "D", "A"]:
        pc = f"pred_proba_{c}"
        if pc in out.columns:
            preview_cols.append(pc)
    preview_cols += ["pred_confidence", "pred_margin"]

    print("\nPreview (first 10 rows):")
    print(out[preview_cols].head(10).to_string(index=False))

    if "pred_proba_H" in out.columns:
        print("\nTop 10 by predicted Home-win probability:")
        print(out.sort_values("pred_proba_H", ascending=False)[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
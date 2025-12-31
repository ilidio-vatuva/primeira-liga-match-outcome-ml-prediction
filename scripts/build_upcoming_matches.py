from pathlib import Path
import argparse
import pandas as pd

MATCHES_PER_ROUND = 9  # Primeira Liga: 18 teams => 9 matches per round

# Odds-only features
ODDS_COLS = ["AvgH", "AvgD", "AvgA"]

# Output columns (production-like: no labels)
OUTPUT_COLS = ["date", "home_team", "away_team", "AvgH", "AvgD", "AvgA"]

# Columns we expect from Football-Data
REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam"]


def parse_date(series: pd.Series) -> pd.Series:
    # Football-Data typically uses dd/mm/yy or dd/mm/yyyy
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure odds columns exist (some seasons may not have them)
    for c in ODDS_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.copy()
    df["Date"] = parse_date(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()

    # Normalize output naming
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
    })

    return df


def select_by_round(df: pd.DataFrame, round_n: int) -> pd.DataFrame:
    """
    Select matches by round index assuming 9 matches per round.
    This requires that df is in chronological order and includes matches from the start of the season.
    """
    df = df.sort_values("date").reset_index(drop=True)
    start = (round_n - 1) * MATCHES_PER_ROUND
    end = round_n * MATCHES_PER_ROUND
    return df.iloc[start:end].copy()


def select_by_date_window(df: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    """
    Select matches in a date window (inclusive).
    """
    d1 = pd.to_datetime(date_from, errors="coerce")
    d2 = pd.to_datetime(date_to, errors="coerce")
    if pd.isna(d1) or pd.isna(d2):
        raise ValueError("Invalid --date-from or --date-to. Use YYYY-MM-DD.")

    return df[(df["date"] >= d1) & (df["date"] <= d2)].copy()


def select_unplayed_only(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select fixtures that appear in the CSV but do not have a final result yet.
    Football-Data sometimes includes future fixtures with missing FTR/FTHG/FTAG.
    """
    # If FTR exists, filter rows where it's empty/NaN
    if "FTR" in original_df.columns:
        mask = original_df["FTR"].isna() | (original_df["FTR"].astype(str).str.strip() == "")
        idx = original_df[mask].index
        return cleaned_df.loc[idx].copy()

    # If no FTR column exists, fallback: return empty
    return cleaned_df.iloc[0:0].copy()


def main():
    parser = argparse.ArgumentParser(description="Build upcoming_matches.csv using odds only (AvgH/AvgD/AvgA).")
    parser.add_argument("--season_csv", type=str, required=True, help="Path to season CSV (e.g., data/raw/P1_2526.csv)")
    parser.add_argument("--out", type=str, default="data/processed/upcoming_matches.csv", help="Output CSV path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--round", type=int, help="Round number to select (assumes 9 matches per round).")
    group.add_argument("--date-from", type=str, help="Start date YYYY-MM-DD (inclusive). Requires --date-to.")
    group.add_argument("--unplayed-only", action="store_true", help="Select only fixtures without results (if present).")

    parser.add_argument("--date-to", type=str, help="End date YYYY-MM-DD (inclusive). Used with --date-from.")

    args = parser.parse_args()

    season_path = Path(args.season_csv)
    out_path = Path(args.out)

    if not season_path.exists():
        raise FileNotFoundError(f"Season CSV not found: {season_path}")

    original_df = pd.read_csv(season_path)
    df = clean_df(original_df)

    # Selection
    if args.round is not None:
        selected = select_by_round(df, args.round)
    elif args.date_from is not None:
        if not args.date_to:
            raise ValueError("--date-from requires --date-to.")
        selected = select_by_date_window(df, args.date_from, args.date_to)
    else:
        selected = select_unplayed_only(original_df, df)

    if selected.empty:
        raise ValueError(
            "No matches found for the selection.\n"
            "If you used --round: your CSV may not include that full round yet.\n"
            "If you used --unplayed-only: the CSV may not contain future fixtures.\n"
            "Try --date-from/--date-to or update the season CSV."
        )

    # Output: production-like (no labels)
    out = selected[["date", "home_team", "away_team"] + ODDS_COLS].copy()

    # Use ISO date format
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("✅ upcoming_matches.csv created (odds-only)")
    print(f"- Input:  {season_path}")
    print(f"- Output: {out_path}")
    print(f"- Rows:   {len(out)}")
    print("\nPreview:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()



# python scripts/build_upcoming_matches.py --season_csv data/raw/P1_2526.csv --round 16 --out data/processed/upcoming_matches.csv
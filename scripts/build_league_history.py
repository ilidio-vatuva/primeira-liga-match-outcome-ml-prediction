from pathlib import Path
from typing import Union, Optional
import re
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/league_history.csv")

# Football-Data typical columns:
MIN_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLS = ["AvgH", "AvgD", "AvgA"]   # may or may not exist depending on season
def parse_date(series: pd.Series) -> pd.Series:
    # Football-Data often uses dd/mm/yy or dd/mm/yyyy
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def infer_season_from_filename(path: Union[Path, str]) -> Optional[str]:
    """Try to infer season from filename patterns like 2425, 2324, etc.
    Returns a string like '2024/25' or None.
    Accepts a `Path` or string; uses the filename stem for matching.
    """
    stem = Path(path).stem
    m = re.search(r"(\d{2})(\d{2})", stem)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    # Heuristic: 2425 -> 2024/25
    if 0 <= a <= 99 and 0 <= b <= 99:
        return f"20{a:02d}/20{b:02d}"
    return None

def season_from_date(dt: pd.Timestamp) -> str:
    """
    For leagues that span years: season starts roughly in July.
    If month >= 7 => season = YYYY/(YYYY+1)
    else => (YYYY-1)/YYYY
    """
    y = dt.year
    if dt.month >= 7:
        return f"{y}/{y+1}"
    return f"{y-1}/{y}"

def points_for_side(ftr: pd.Series, side: str) -> pd.Series:
    """
    ftr: H/D/A
    side: 'home' or 'away'
    """
    if side == "home":
        return ftr.map({"H": 3, "D": 1, "A": 0}).astype("Int64")
    else:
        return ftr.map({"A": 3, "D": 1, "H": 0}).astype("Int64")

def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR.resolve()}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in MIN_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} is missing required columns: {missing}")

        df = df.copy()
        df["season"] = infer_season_from_filename(f)  # may be None; we'll fill later
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Keep only columns we need (+ odds if present)
    keep = MIN_COLS + ["season"] + [c for c in ODDS_COLS if c in df.columns]
    df = df[keep].copy()

    df["Date"] = parse_date(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"]).copy()

    # Fill season when not provided by filename
    df["season"] = df["season"].fillna(df["Date"].apply(season_from_date))

    # Sort chronologically (important for rolling features)
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    # Build "long" table: one match => two team rows (home and away)
    home = df[["Date", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    home.rename(columns={"HomeTeam": "team", "AwayTeam": "opponent"}, inplace=True)
    home["is_home"] = 1
    home["gf"] = home["FTHG"]
    home["ga"] = home["FTAG"]
    home["points"] = points_for_side(home["FTR"], "home")

    away = df[["Date", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    away.rename(columns={"AwayTeam": "team", "HomeTeam": "opponent"}, inplace=True)
    away["is_home"] = 0
    away["gf"] = away["FTAG"]
    away["ga"] = away["FTHG"]
    away["points"] = points_for_side(away["FTR"], "away")

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "Date"]).reset_index(drop=True)

    # Rest days: days since last match for the team (pre-match)
    long["prev_date"] = long.groupby("team")["Date"].shift(1)
    long["rest_days"] = (long["Date"] - long["prev_date"]).dt.days
    # For first match, rest_days is NaN; set to a neutral value (e.g., 7)
    long["rest_days"] = long["rest_days"].fillna(7).clip(lower=0)

    # Rolling last5 sums (pre-match) -> shift(1) prevents leakage
    def rolling5_sum(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(5, min_periods=1).sum()

    long["points_last5"] = long.groupby("team")["points"].transform(rolling5_sum)
    long["gf_last5"] = long.groupby("team")["gf"].transform(rolling5_sum)
    long["ga_last5"] = long.groupby("team")["ga"].transform(rolling5_sum)

    # Create match_id to merge back to match-level
    df = df.reset_index(drop=True)
    df["match_id"] = np.arange(len(df))

    # Home rows in long correspond to df rows in chronological order? Not guaranteed after groupby sort.
    # So we merge by (Date, team, opponent) for robustness.
    home_long = long[long["is_home"] == 1].copy()
    away_long = long[long["is_home"] == 0].copy()

    # Match keys
    df_key = df[["match_id", "Date", "HomeTeam", "AwayTeam"]].copy()

    home_key = home_long.rename(columns={"team": "HomeTeam", "opponent": "AwayTeam"})[
        ["Date", "HomeTeam", "AwayTeam", "points_last5", "gf_last5", "ga_last5", "rest_days"]
    ]
    away_key = away_long.rename(columns={"team": "AwayTeam", "opponent": "HomeTeam"})[
        ["Date", "HomeTeam", "AwayTeam", "points_last5", "gf_last5", "ga_last5", "rest_days"]
    ]

    m = df_key.merge(home_key, on=["Date", "HomeTeam", "AwayTeam"], how="left").rename(columns={
        "points_last5": "home_points_last5",
        "gf_last5": "home_gf_last5",
        "ga_last5": "home_ga_last5",
        "rest_days": "home_rest_days",
    })

    m = m.merge(away_key, on=["Date", "HomeTeam", "AwayTeam"], how="left").rename(columns={
        "points_last5": "away_points_last5",
        "gf_last5": "away_gf_last5",
        "ga_last5": "away_ga_last5",
        "rest_days": "away_rest_days",
    })

    # Bring label + odds + season
    base = df[["match_id", "Date", "season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    for c in ODDS_COLS:
        if c in df.columns:
            base[c] = df[c]

    out = base.merge(m, on="match_id", how="left")

    # Final tidy names
    out = out.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team"
    }).drop(columns=["match_id"])

    # Ensure odds exist even if missing in some seasons
    for c in ODDS_COLS:
        if c not in out.columns:
            out[c] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"OK -> {OUT_PATH} | rows: {len(out)} | seasons: {out['season'].nunique()}")

if __name__ == "__main__":
    main()
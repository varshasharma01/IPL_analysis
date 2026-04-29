"""
Run this ONCE locally before deploying:
    python preprocess.py

It reads your raw large files and writes aggregated_data.zip (~2-5MB).
Commit that zip to your repo instead of the raw CSVs.
"""

import pandas as pd
import zipfile
import os
import io

# ── Paths to your raw source files ───────────────────────────────────────────
BB_FILE   = "IPL_ball_by_ball.zip"   # or "src/IPL_ball_by_ball.zip"
TEAM_FILE = "ipl_team.csv"           # or "src/ipl_team.csv"
OUT_ZIP   = "aggregated_data.zip"    # what gets committed / uploaded

if not os.path.exists(BB_FILE):
    BB_FILE = os.path.join("src", BB_FILE)
if not os.path.exists(TEAM_FILE):
    TEAM_FILE = os.path.join("src", TEAM_FILE)

print("Loading ball-by-ball data (this may take a minute)...")

bb_cols = [
    'season', 'match_id', 'runs_total', 'runs_batter', 'wicket_kind',
    'batting_team', 'over', 'valid_ball', 'bat_pos', 'extra_type',
    'review_decision', 'bowler', 'fielders', 'non_striker_pos'
]
dtypes = {
    'match_id': 'int32', 'runs_total': 'int8', 'runs_batter': 'int8',
    'over': 'int8', 'bat_pos': 'int8', 'valid_ball': 'int8'
}

df = pd.read_csv(BB_FILE, compression='zip', usecols=bb_cols, dtype=dtypes, low_memory=False)
df2 = pd.read_csv(TEAM_FILE)

df.columns  = df.columns.str.strip()
df2.columns = df2.columns.str.strip()
df['season']  = df['season'].astype(str)
df2['season'] = df2['season'].astype(str)

years = sorted(df['season'].unique(), key=lambda x: int(x.split('/')[0]) if x.split('/')[0].isdigit() else x)
mapping = {y: str(i+1) for i, y in enumerate(years)}
df['season_num'] = df['season'].map(mapping)

def get_phase(over):
    if over < 6:  return 'Powerplay'
    if over < 15: return 'Middle'
    return 'Death'

df['phase'] = df['over'].map(get_phase)

print("Computing aggregations...")

# ── Overview aggregations ─────────────────────────────────────────────────────
season_match_counts = (
    df2.groupby('season', sort=False)['season']
    .count()
    .rename('match_count')
    .reset_index()
)
season_match_counts.columns = ['season', 'match_count']

team_runs = (
    df.groupby('batting_team')['runs_total']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
team_runs.columns = ['batting_team', 'total_runs']

run_distribution = (
    df['runs_total']
    .value_counts()
    .sort_index()
    .reset_index()
)
run_distribution.columns = ['runs_per_ball', 'frequency']

# Per-season summary for metrics
season_summary = df.groupby('season_num').agg(
    matches   = ('match_id',   'nunique'),
    total_runs = ('runs_total', 'sum'),
    wickets    = ('wicket_kind', lambda x: x.notna().sum()),
    sixes      = ('runs_batter', lambda x: (x == 6).sum()),
    fours      = ('runs_batter', lambda x: (x == 4).sum()),
).reset_index()

# All-season totals (single-row summary)
totals = pd.DataFrame([{
    'matches':     df['match_id'].nunique(),
    'total_runs':  int(df['runs_total'].sum()),
    'wickets':     int(df['wicket_kind'].notna().sum()),
    'sixes':       int((df['runs_batter'] == 6).sum()),
    'fours':       int((df['runs_batter'] == 4).sum()),
    'seasons':     df2['season'].nunique(),
}])

# ── Team Analysis aggregations ────────────────────────────────────────────────
winner_counts      = df2['winner'].value_counts().reset_index()
winner_counts.columns = ['winner', 'wins']

toss_decision      = df2['toss_decision'].value_counts().reset_index()
toss_decision.columns = ['decision', 'count']

target_runs        = df2[['target_runs']].dropna()

venue_counts       = df2['venue'].value_counts().head(10).reset_index()
venue_counts.columns = ['venue', 'count']

result_margin      = df2[['result_margin']].dropna()

match_type_counts  = df2['match_type'].value_counts().reset_index()
match_type_counts.columns = ['match_type', 'count']

matches_per_season = df2['season'].value_counts().reset_index()
matches_per_season.columns = ['season', 'count']

super_over_counts  = df2['super_over'].value_counts().reset_index()
super_over_counts.columns = ['super_over', 'count']

city_counts        = df2['city'].value_counts().head(10).reset_index()
city_counts.columns = ['city', 'count']

pom_counts         = df2['player_of_match'].value_counts().head(10).reset_index()
pom_counts.columns = ['player', 'count']

# ── Ball-by-Ball aggregations ─────────────────────────────────────────────────
valid_ball_counts  = df['valid_ball'].value_counts().reset_index()
valid_ball_counts.columns = ['valid_ball', 'count']

runs_per_delivery  = df['runs_total'].value_counts().reset_index()
runs_per_delivery.columns = ['runs', 'count']

bat_pos_counts     = df['bat_pos'].value_counts().reset_index()
bat_pos_counts.columns = ['bat_pos', 'count']

over_counts        = df['over'].value_counts().sort_index().reset_index()
over_counts.columns = ['over', 'count']

extra_type_counts  = df[df['extra_type'].notna()]['extra_type'].value_counts().reset_index()
extra_type_counts.columns = ['extra_type', 'count']

review_counts      = df[df['review_decision'].notna()]['review_decision'].value_counts().reset_index()
review_counts.columns = ['review_decision', 'count']

bowler_workload    = df['bowler'].value_counts().head(10).reset_index()
bowler_workload.columns = ['bowler', 'deliveries']

fielder_counts     = df['fielders'].value_counts().head(10).reset_index()
fielder_counts.columns = ['fielder', 'count']

non_striker_pos    = df['non_striker_pos'].value_counts().reset_index()
non_striker_pos.columns = ['position', 'count']

phase_counts       = df['phase'].value_counts().reset_index()
phase_counts.columns = ['phase', 'count']

# ── Pack everything into a zip ────────────────────────────────────────────────
frames = {
    # Overview
    "season_match_counts.csv": season_match_counts,
    "team_runs.csv":           team_runs,
    "run_distribution.csv":    run_distribution,
    "season_summary.csv":      season_summary,
    "totals.csv":              totals,
    # Team Analysis
    "winner_counts.csv":       winner_counts,
    "toss_decision.csv":       toss_decision,
    "target_runs.csv":         target_runs,
    "venue_counts.csv":        venue_counts,
    "result_margin.csv":       result_margin,
    "match_type_counts.csv":   match_type_counts,
    "matches_per_season.csv":  matches_per_season,
    "super_over_counts.csv":   super_over_counts,
    "city_counts.csv":         city_counts,
    "pom_counts.csv":          pom_counts,
    # Ball-by-Ball
    "valid_ball_counts.csv":   valid_ball_counts,
    "runs_per_delivery.csv":   runs_per_delivery,
    "bat_pos_counts.csv":      bat_pos_counts,
    "over_counts.csv":         over_counts,
    "extra_type_counts.csv":   extra_type_counts,
    "review_counts.csv":       review_counts,
    "bowler_workload.csv":     bowler_workload,
    "fielder_counts.csv":      fielder_counts,
    "non_striker_pos.csv":     non_striker_pos,
    "phase_counts.csv":        phase_counts,
}

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for name, frame in frames.items():
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        zf.writestr(name, buf.getvalue())

size_kb = os.path.getsize(OUT_ZIP) / 1024
print(f"\nDone! Written: {OUT_ZIP}  ({size_kb:.1f} KB)")
print("Now commit this file to your repo and redeploy on Render.")
print("You can delete the raw IPL_ball_by_ball.zip from the repo entirely.")
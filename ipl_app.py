import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import requests
import warnings
import io
import base64
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

st.set_page_config(layout='wide', page_title='IPL Analysis with AI Explainer')


# ── API key ───────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

GEMINI_API_KEY = get_api_key()
BACKEND_URL = "https://ipl-analysis-uq5a.onrender.com"


# ── Data loading (heavily optimized) ─────────────────────────────────────────
@st.cache_data
def load_data():
    bb_cols = [
        'season', 'match_id', 'runs_total', 'runs_batter', 'wicket_kind',
        'batting_team', 'over', 'valid_ball', 'bat_pos', 'extra_type',
        'review_decision', 'bowler', 'fielders', 'non_striker_pos'
    ]
    dtypes_dict = {
        'match_id':    'int32',
        'runs_total':  'int8',
        'runs_batter': 'int8',
        'over':        'int8',
        'bat_pos':     'int8',
        'valid_ball':  'int8',
    }

    file_path = 'IPL_ball_by_ball.zip'
    if not os.path.exists(file_path):
        file_path = os.path.join('src', 'IPL_ball_by_ball.zip')

    team_path = 'ipl_team.csv'
    if not os.path.exists(team_path):
        team_path = os.path.join('src', 'ipl_team.csv')

    try:
        df1 = pd.read_csv(
            file_path,
            compression='zip',
            usecols=bb_cols,
            dtype=dtypes_dict,
            low_memory=False,
        )
        df2 = pd.read_csv(team_path)
    except Exception as e:
        st.error(f"Critical Error: Could not load data file. {e}")
        st.stop()

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    df1['season'] = df1['season'].astype(str)
    df2['season'] = df2['season'].astype(str)

    years = sorted(
        df1['season'].unique(),
        key=lambda x: int(x.split('/')[0]) if x.split('/')[0].isdigit() else x
    )
    mapping = {year: str(i + 1) for i, year in enumerate(years)}
    df1['season_num'] = df1['season'].map(mapping)

    # FIX 1: Convert high-cardinality string columns to category dtype
    # Saves ~60-80% memory on these columns
    for col in ['batting_team', 'wicket_kind', 'extra_type',
                 'review_decision', 'bowler', 'fielders', 'non_striker_pos']:
        if col in df1.columns:
            df1[col] = df1[col].astype('category')

    # FIX 2: Add match_phase as a typed categorical (not object)
    def get_phase(over):
        if over < 6:
            return 'Powerplay'
        elif over < 15:
            return 'Middle'
        return 'Death'

    df1['phase'] = pd.Categorical(
        df1['over'].map(get_phase),
        categories=['Powerplay', 'Middle', 'Death'],
        ordered=True
    )

    for col in ['winner', 'toss_decision', 'venue', 'match_type',
                'super_over', 'city', 'player_of_match']:
        if col in df2.columns:
            df2[col] = df2[col].astype('category')

    return df1, df2


df, df2 = load_data()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_image_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=80)  # lower dpi = less RAM
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return b64


def ai_explainer_ui(fig, key_id):
    if st.button("Explain with AI", key=key_id):
        with st.spinner("Analyzing with AI..."):
            try:
                img_b64 = get_image_base64(fig)
                headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
                response = requests.post(
                    f"{BACKEND_URL}/explain-chart",
                    json={"base64_image": img_b64},
                    headers=headers,
                    timeout=30,
                )
                if response.status_code == 200:
                    st.info(response.json()['explanation'])
                elif response.status_code == 401:
                    st.error("API Error: Unauthorized (API Key mismatch)")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("IPL Analytics Dashboard")
analysis_mode = st.sidebar.radio(
    "Select Analysis Level:",
    ["Overview", "Ball-by-Ball Analysis", "IPL Team Analysis (2008-2025)"]
)

st.title(f"🏏 {analysis_mode}")


# ── Overview ──────────────────────────────────────────────────────────────────
if analysis_mode == "Overview":
    available_seasons = sorted(
        df2['season'].unique().tolist(),
        key=lambda x: int(x) if str(x).isdigit() else x
    )
    options = ["All Seasons"] + available_seasons
    selected_option = st.selectbox("Select Season Focus", options)

    # FIX 3: Use boolean mask instead of df.copy() — avoids duplicating the full dataframe
    if selected_option == "All Seasons":
        mask1 = slice(None)       # selects everything without a copy
        mask2 = slice(None)
        title_prefix = "All Time"
        df_f  = df
        df2_f = df2
    else:
        sel_season = selected_option
        df_f  = df[df['season_num'] == sel_season]
        df2_f = df2[df2['season'] == sel_season]
        title_prefix = f"Season {selected_option}"

    total_seasons = df2_f['season'].nunique()
    total_matches = df_f['match_id'].nunique()
    total_runs    = pd.to_numeric(df_f['runs_total'], errors='coerce').sum()
    total_wickets = df_f['wicket_kind'].notna().sum()
    total_sixes   = (df_f['runs_batter'] == 6).sum()
    total_fours   = (df_f['runs_batter'] == 4).sum()

    st.subheader(f"📊 {title_prefix} Summary")
    cols = st.columns(6)
    for col, label, val in zip(cols, ["Seasons","Matches","Total Runs","Wickets","6s","4s"],
                                [total_seasons, total_matches, f"{int(total_runs):,}",
                                 total_wickets, total_sixes, total_fours]):
        col.metric(label, val)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Matches per Season")
            season_counts = (
                df2_f['season']
                .value_counts()
                .sort_index(key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else x))
            )
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(season_counts.index.astype(str), season_counts.values,
                     marker='o', color='#1f77b4', linewidth=2)
            ax1.set_xlabel("Season", fontsize=11)
            ax1.set_ylabel("Matches Played", fontsize=11)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            fig1.tight_layout()
            st.pyplot(fig1)
            ai_explainer_ui(fig1, "overview_season")
            plt.close(fig1)  # FIX 4: always close immediately after render

    with col2:
        with st.container(border=True):
            st.subheader("Top 10 Run Scoring Teams")
            team_runs = (
                df_f.groupby('batting_team', observed=True)['runs_total']
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            colors = plt.cm.viridis_r([i / max(len(team_runs) - 1, 1) for i in range(len(team_runs))])
            ax2.barh(team_runs.index[::-1], team_runs.values[::-1], color=colors)
            ax2.set_xlabel("Total Runs", fontsize=11)
            ax2.set_ylabel("Team", fontsize=11)
            ax2.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
            )
            ax2.grid(axis='x', alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2)
            ai_explainer_ui(fig2, "overview_teams")
            plt.close(fig2)

    st.divider()
    st.subheader("Run Distribution per Ball")
    run_counts = df_f['runs_total'].value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.bar(run_counts.index.astype(str), run_counts.values, color='pink', alpha=0.8)
    ax3.set_xlabel("Runs per Ball", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    ai_explainer_ui(fig3, "overview_distribution")
    plt.close(fig3)


# ── IPL Team Analysis ─────────────────────────────────────────────────────────
elif analysis_mode == "IPL Team Analysis (2008-2025)":

    # FIX 5: Pre-compute value_counts once — avoids recomputing inside lambdas
    winner_order  = df2['winner'].value_counts().index.tolist()
    venue_vc      = df2['venue'].value_counts().head(10)
    city_vc       = df2['city'].value_counts().head(10)
    pom_vc        = df2['player_of_match'].value_counts().head(10)

    plots = [
        ("Most Wins by Team",
         lambda: sns.countplot(data=df2, y='winner', order=winner_order, palette='viridis')),

        ("Toss Decision Preference",
         lambda: plt.pie(df2['toss_decision'].value_counts(),
                         labels=df2['toss_decision'].value_counts().index.astype(str).tolist(),
                         autopct='%1.1f%%', startangle=140)),

        ("Distribution of Target Runs",
         lambda: sns.histplot(x=df2['target_runs'], bins=10, kde=True)),

        ("Top 10 Venues by Match Count",
         lambda: sns.barplot(x=venue_vc.values, y=venue_vc.index.tolist())),

        ("Spread of Result Margin",
         lambda: sns.boxplot(x=df2['result_margin'])),

        ("Match Type Distribution",
         lambda: sns.countplot(data=df2, x='match_type')),

        ("Matches Played per Season",
         lambda: sns.countplot(data=df2, x='season')),

        ("Super Over Matches Distribution",
         lambda: sns.countplot(data=df2, x='super_over')),

        ("Top 10 Cities by Match Count",
         lambda: sns.barplot(x=city_vc.values, y=city_vc.index.tolist())),

        ("Top 10 Players of the Match",
         lambda: sns.barplot(x=pom_vc.values, y=pom_vc.index.tolist())),
    ]

    for i, (title, func) in enumerate(plots):
        st.subheader(f"{i+1}. {title}")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            func()
            if any(kw in title for kw in ("Venue", "Season", "City")):
                plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            ai_explainer_ui(fig, f"team_btn_{i}")
        plt.close(fig)   # FIX 4 applied to every loop iteration
        st.divider()


# ── Ball-by-Ball Analysis ─────────────────────────────────────────────────────
elif analysis_mode == "Ball-by-Ball Analysis":

    # FIX 5: Pre-compute expensive aggregations once
    bowler_vc   = df['bowler'].value_counts().head(10)
    fielder_vc  = df['fielders'].value_counts().head(10)
    extras_df   = df[df['extra_type'].notna()]
    review_df   = df[df['review_decision'].notna()]
    review_vc   = review_df['review_decision'].value_counts()
    phase_vc    = df['phase'].value_counts()

    bb_plots = [
        ("Valid vs Extra Deliveries",
         lambda: sns.countplot(data=df, x='valid_ball', palette='Set1')),

        ("Runs Scored per Delivery",
         lambda: sns.countplot(data=df, x='runs_total', palette='viridis')),

        ("Deliveries Faced by Batting Position",
         lambda: sns.countplot(data=df, x='bat_pos', palette='plasma')),

        ("Ball Data Volume per Over",
         lambda: sns.histplot(x=df['over'], bins=20, kde=True, color='orange')),

        ("Breakdown of Extras",
         lambda: sns.countplot(data=extras_df, y='extra_type', palette='magma')),

        ("DRS Review Decisions",
         lambda: plt.pie(review_vc, labels=review_vc.index.tolist(), autopct='%1.1f%%')),

        ("Top 10 Bowlers by Workload",
         lambda: sns.barplot(x=bowler_vc.values, y=bowler_vc.index,
                             hue=bowler_vc.index, palette='rocket', legend=False)),

        ("Top 10 Fielder Involvements",
         lambda: sns.barplot(x=fielder_vc.values, y=fielder_vc.index,
                             hue=fielder_vc.index, palette='flare', legend=False)),

        ("Non-Striker Position Distribution",
         lambda: sns.countplot(data=df, x='non_striker_pos',
                               hue='non_striker_pos', palette='Set2', legend=False)),

        ("Ball Distribution by Match Phase",
         lambda: plt.pie(phase_vc, labels=phase_vc.index.tolist(), autopct='%1.1f%%')),
    ]

    for i, (title, func) in enumerate(bb_plots):
        st.subheader(f"{i+1}. {title}")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            func()
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            ai_explainer_ui(fig, f"bb_btn_{i}")
        plt.close(fig)   # FIX 4 applied to every loop iteration
        st.divider()
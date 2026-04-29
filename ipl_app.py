"""
Streamlit IPL Dashboard — lightweight version.
Reads aggregated_data.zip (~3MB) instead of the raw 200MB CSV.
Run preprocess.py locally first to generate that file.
"""

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
import zipfile
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

st.set_page_config(layout='wide', page_title='IPL Analysis with AI Explainer')


# ── Config ────────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

GEMINI_API_KEY = get_api_key()
BACKEND_URL    = "https://ipl-analysis-uq5a.onrender.com"


# ── Load pre-aggregated data (~3MB total) ─────────────────────────────────────
@st.cache_data
def load_data():
    """
    Reads aggregated_data.zip produced by preprocess.py.
    All heavy computation was done offline — this is just CSV reads.
    Total RAM footprint: ~10-20MB.
    """
    zip_path = "aggregated_data.zip"
    if not os.path.exists(zip_path):
        zip_path = os.path.join("src", zip_path)

    if not os.path.exists(zip_path):
        st.error(
            "aggregated_data.zip not found. "
            "Run preprocess.py locally first, then commit the output file."
        )
        st.stop()

    def read(name):
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(name) as f:
                return pd.read_csv(f)

    d = {}
    tables = [
        # Overview
        "season_match_counts", "team_runs", "run_distribution",
        "season_summary", "totals",
        # Team Analysis
        "winner_counts", "toss_decision", "target_runs", "venue_counts",
        "result_margin", "match_type_counts", "matches_per_season",
        "super_over_counts", "city_counts", "pom_counts",
        # Ball-by-Ball
        "valid_ball_counts", "runs_per_delivery", "bat_pos_counts",
        "over_counts", "extra_type_counts", "review_counts",
        "bowler_workload", "fielder_counts", "non_striker_pos", "phase_counts",
    ]
    for t in tables:
        d[t] = read(f"{t}.csv")
    return d

data = load_data()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_image_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=80)
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
                    st.error("API Error: Unauthorized")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")


def make_chart(title, plot_fn, key_id):
    """Render a chart column + AI explainer column, then free the figure."""
    st.subheader(title)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots()
        plot_fn(ax)
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        ai_explainer_ui(fig, key_id)
    plt.close(fig)
    st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("IPL Analytics Dashboard")
analysis_mode = st.sidebar.radio(
    "Select Analysis Level:",
    ["Overview", "Ball-by-Ball Analysis", "IPL Team Analysis (2008-2025)"]
)

st.title(f"🏏 {analysis_mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if analysis_mode == "Overview":
    totals = data["totals"].iloc[0]

    st.subheader("📊 All Time Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Seasons",    int(totals["seasons"]))
    c2.metric("Matches",    int(totals["matches"]))
    c3.metric("Total Runs", f"{int(totals['total_runs']):,}")
    c4.metric("Wickets",    int(totals["wickets"]))
    c5.metric("6s",         int(totals["sixes"]))
    c6.metric("4s",         int(totals["fours"]))

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Matches per Season")
            smc = data["season_match_counts"].copy()
            smc = smc.sort_values(
                'season',
                key=lambda s: pd.to_numeric(s, errors='coerce').fillna(0)
            )
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(smc['season'].astype(str), smc['match_count'],
                     marker='o', color='#1f77b4', linewidth=2)
            ax1.set_xlabel("Season"); ax1.set_ylabel("Matches Played")
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            fig1.tight_layout()
            st.pyplot(fig1)
            ai_explainer_ui(fig1, "overview_season")
            plt.close(fig1)

    with col2:
        with st.container(border=True):
            st.subheader("Top 10 Run Scoring Teams")
            tr = data["team_runs"]
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            colors = plt.cm.viridis_r([i / max(len(tr)-1, 1) for i in range(len(tr))])
            ax2.barh(tr['batting_team'][::-1], tr['total_runs'][::-1], color=colors)
            ax2.set_xlabel("Total Runs"); ax2.set_ylabel("Team")
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
    rd = data["run_distribution"]
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.bar(rd['runs_per_ball'].astype(str), rd['frequency'], color='pink', alpha=0.8)
    ax3.set_xlabel("Runs per Ball"); ax3.set_ylabel("Frequency")
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    ai_explainer_ui(fig3, "overview_distribution")
    plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
# IPL TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "IPL Team Analysis (2008-2025)":

    def plot_winner(ax):
        wc = data["winner_counts"]
        sns.barplot(x='wins', y='winner', data=wc, palette='viridis', ax=ax)

    def plot_toss(ax):
        td = data["toss_decision"]
        ax.pie(td['count'], labels=td['decision'].tolist(), autopct='%1.1f%%', startangle=140)

    def plot_target(ax):
        tr = data["target_runs"].dropna()
        sns.histplot(x=tr['target_runs'], bins=10, kde=True, ax=ax)

    def plot_venue(ax):
        vc = data["venue_counts"]
        sns.barplot(x='count', y='venue', data=vc, ax=ax)

    def plot_margin(ax):
        rm = data["result_margin"].dropna()
        sns.boxplot(x=rm['result_margin'], ax=ax)

    def plot_match_type(ax):
        mt = data["match_type_counts"]
        sns.barplot(x='match_type', y='count', data=mt, ax=ax)

    def plot_season(ax):
        ms = data["matches_per_season"]
        ms = ms.sort_values('season', key=lambda s: pd.to_numeric(s, errors='coerce').fillna(0))
        sns.barplot(x='season', y='count', data=ms, ax=ax)
        ax.tick_params(axis='x', rotation=45)

    def plot_super_over(ax):
        so = data["super_over_counts"]
        sns.barplot(x='super_over', y='count', data=so, ax=ax)

    def plot_city(ax):
        cc = data["city_counts"]
        sns.barplot(x='count', y='city', data=cc, ax=ax)

    def plot_pom(ax):
        pm = data["pom_counts"]
        sns.barplot(x='count', y='player', data=pm, ax=ax)

    plots = [
        ("Most Wins by Team",            plot_winner),
        ("Toss Decision Preference",     plot_toss),
        ("Distribution of Target Runs",  plot_target),
        ("Top 10 Venues by Match Count", plot_venue),
        ("Spread of Result Margin",      plot_margin),
        ("Match Type Distribution",      plot_match_type),
        ("Matches Played per Season",    plot_season),
        ("Super Over Matches",           plot_super_over),
        ("Top 10 Cities by Match Count", plot_city),
        ("Top 10 Players of the Match",  plot_pom),
    ]

    for i, (title, fn) in enumerate(plots):
        make_chart(f"{i+1}. {title}", fn, f"team_btn_{i}")


# ═══════════════════════════════════════════════════════════════════════════════
# BALL-BY-BALL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "Ball-by-Ball Analysis":

    def plot_valid_ball(ax):
        vb = data["valid_ball_counts"]
        sns.barplot(x='valid_ball', y='count', data=vb, palette='Set1', ax=ax)

    def plot_runs_delivery(ax):
        rd = data["runs_per_delivery"]
        sns.barplot(x='runs', y='count', data=rd, palette='viridis', ax=ax)

    def plot_bat_pos(ax):
        bp = data["bat_pos_counts"]
        sns.barplot(x='bat_pos', y='count', data=bp, palette='plasma', ax=ax)

    def plot_over_vol(ax):
        oc = data["over_counts"]
        sns.barplot(x='over', y='count', data=oc, color='orange', ax=ax)

    def plot_extras(ax):
        et = data["extra_type_counts"]
        sns.barplot(x='count', y='extra_type', data=et, palette='magma', ax=ax)

    def plot_drs(ax):
        rc = data["review_counts"]
        ax.pie(rc['count'], labels=rc['review_decision'].tolist(), autopct='%1.1f%%')

    def plot_bowler(ax):
        bw = data["bowler_workload"]
        sns.barplot(x='deliveries', y='bowler', data=bw, palette='rocket', ax=ax)

    def plot_fielder(ax):
        fc = data["fielder_counts"]
        sns.barplot(x='count', y='fielder', data=fc, palette='flare', ax=ax)

    def plot_non_striker(ax):
        ns = data["non_striker_pos"]
        sns.barplot(x='position', y='count', data=ns, palette='Set2', ax=ax)

    def plot_phase(ax):
        pc = data["phase_counts"]
        ax.pie(pc['count'], labels=pc['phase'].tolist(), autopct='%1.1f%%')

    bb_plots = [
        ("Valid vs Extra Deliveries",          plot_valid_ball),
        ("Runs Scored per Delivery",           plot_runs_delivery),
        ("Deliveries Faced by Batting Position", plot_bat_pos),
        ("Ball Data Volume per Over",          plot_over_vol),
        ("Breakdown of Extras",                plot_extras),
        ("DRS Review Decisions",               plot_drs),
        ("Top 10 Bowlers by Workload",         plot_bowler),
        ("Top 10 Fielder Involvements",        plot_fielder),
        ("Non-Striker Position Distribution",  plot_non_striker),
        ("Ball Distribution by Match Phase",   plot_phase),
    ]

    for i, (title, fn) in enumerate(bb_plots):
        make_chart(f"{i+1}. {title}", fn, f"bb_btn_{i}")
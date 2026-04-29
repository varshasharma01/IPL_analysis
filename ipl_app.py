import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
import logging
import os
import requests
import warnings
import io
import base64
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from google import genai

load_dotenv() 


from PIL import Image
from google import genai 

# Client initialize karein
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def get_api_key():
    # 1. Try to get it from st.secrets first
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except:
        pass
    
    # 2. If that fails, look in Environment variables
    return os.environ.get("GEMINI_API_KEY")

GEMINI_API_KEY = get_api_key()

@st.fragment
def render_chart_section(title, func, key_id):
    with st.container(border=True):
        st.subheader(title)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            func() # Ye plotting function execute karega
            if any(x in title for x in ["Venue", "Season", "City"]):
                plt.xticks(rotation=45)
            st.pyplot(fig, use_container_width=True) # Container width se jumping rukegi
        with col2:
            ai_explainer_ui(fig, key_id)
        plt.close(fig)
        

# Now, use this variable throughout your app
# GROQ_API_KEY = get_api_key()


# Set to 'ERROR' to only see major issues, or 'INFO' to see everything
logging.basicConfig(level=logging.ERROR)

# Ignore specific warnings like the FutureWarning from Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore the DtypeWarning from Pandas
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

st.set_page_config(layout='wide', page_title='IPL Analysis with AI Explainer')

@st.cache_data
def load_data():
    # read raw data and normalize the season column so filtering is reliable
    df1 = pd.read_csv('IPL_ball_by_ball.zip',compression='zip', low_memory=False) # Add low_memory=False
    df2 = pd.read_csv('ipl_team.csv')

    # strip stray whitespace
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # unify season column types: use string everywhere for simplicity
    df1['season'] = df1['season'].astype(str)
    df2['season'] = df2['season'].astype(str)

    # create integer season IDs in the ball‑by‑ball dataframe so it lines up with
    # the numeric "season" values used in df2.  The two sources have identical
    # logical seasons, just labelled differently (year strings vs 1..18).
    years = sorted(df1['season'].unique(), key=lambda x: int(x.split('/')[0]) if x.split('/')[0].isdigit() else x)
    mapping = {year: str(i + 1) for i, year in enumerate(years)}
    df1['season_num'] = df1['season'].map(mapping)

    # example mapping is available if you ever want to show the year text
    # alongside the numeric id in the UI
    # print("season mapping", mapping)

    return df1, df2

df, df2 = load_data()


st.sidebar.title("IPL Analytics Dashboard")
analysis_mode = st.sidebar.radio(
    "Select Analysis Level:",
    ["Overview","Ball-by-Ball Analysis", "IPL Team Analysis (2008-2025)"]
)

def get_image_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Use the new Render URL instead of localhost
BACKEND_URL = "https://ipl-analysis-uq5a.onrender.com"

def ai_explainer_ui(fig, key_id):
    if st.button("Explain with AI", key=key_id):
        with st.spinner("Analyzing with AI..."):
            try:
                img_b64 = get_image_base64(fig)
                headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
                
                # Point to the live backend
                response = requests.post(
                    f"{BACKEND_URL}/explain-chart", 
                    json={"base64_image": img_b64},
                    headers=headers
                )
                
                if response.status_code == 200:
                    st.info(response.json()['explanation'])
                elif response.status_code == 401:
                    st.error("API Error: Unauthorized (API Key mismatch)")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

st.title(f"🏏 {analysis_mode}")

if analysis_mode == "Overview":

    available_seasons = sorted(df2['season'].unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
    options = ["All Seasons"] + available_seasons
    selected_option = st.selectbox("Select Season Focus", options)

    if selected_option == "All Seasons":
        df_filtered = df.copy()
        df2_filtered = df2.copy()
        title_prefix = "All Time"
    else:
        sel_season = selected_option
        df_filtered = df[df['season_num'] == sel_season]
        df2_filtered = df2[df2['season'] == sel_season]
        title_prefix = f"Season {selected_option}"

    total_seasons  = df2_filtered['season'].nunique()
    total_matches  = df_filtered['match_id'].nunique()
    total_runs     = pd.to_numeric(df_filtered['runs_total'], errors='coerce').sum()
    total_wickets  = df_filtered['wicket_kind'].notna().sum()
    total_sixes    = (df_filtered['runs_batter'] == 6).sum()
    total_fours    = (df_filtered['runs_batter'] == 4).sum()

    st.subheader(f"📊 {title_prefix} Summary")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Seasons",     total_seasons)
    m_col2.metric("Matches",     total_matches)
    m_col3.metric("Total Runs",  f"{int(total_runs):,}")
    m_col4.metric("Wickets",     total_wickets)
    m_col5.metric("6s",          total_sixes)
    m_col6.metric("4s",          total_fours)

    # ── Chart row ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Matches per Season")

            season_counts = (
                df2_filtered['season']
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
            fig1.tight_layout()          # ← instance method, not plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)              # ← free memory immediately
            ai_explainer_ui(fig1, "overview_season")

    with col2:
        with st.container(border=True):
            st.subheader("Top 10 Run Scoring Teams")

            team_runs = (
                df_filtered.groupby('batting_team')['runs_total']
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            fig2, ax2 = plt.subplots(figsize=(8, 5))   # ← plt.subplots, NOT Figure()
            colors = plt.cm.viridis_r(
                [i / max(len(team_runs) - 1, 1) for i in range(len(team_runs))]
            )
            ax2.barh(team_runs.index[::-1], team_runs.values[::-1], color=colors[::1])
            ax2.set_xlabel("Total Runs", fontsize=11)
            ax2.set_ylabel("Team", fontsize=11)
            ax2.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
            )
            ax2.grid(axis='x', alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            ai_explainer_ui(fig2, "overview_teams")

    st.divider()

    # ── Run distribution ─────────────────────────────────────────────────────
    st.subheader("Run Distribution per Ball")

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    run_counts = df_filtered['runs_total'].value_counts().sort_index()
    ax3.bar(run_counts.index.astype(str), run_counts.values, color='pink', alpha=0.8)
    ax3.set_xlabel("Runs per Ball", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    ai_explainer_ui(fig3, "overview_distribution")
elif analysis_mode == "IPL Team Analysis (2008-2025)":
    
    plots = [
        
        ("Most Wins by Team", 
         lambda: sns.countplot(
             data=df2, 
             y='winner', 
             order=df2['winner'].value_counts().index.tolist(), 
             palette='viridis')),
        
        
        ("Toss Decision Preference", 
         lambda: plt.pie(
             df2['toss_decision'].value_counts(), 
             labels=df2['toss_decision'].value_counts().index.astype(str).tolist(), 
             autopct='%1.1f%%', 
             startangle=140)),
        
        
        ("Distribution of Target Runs", 
         lambda: sns.histplot(
             x=df2['target_runs'], 
             bins=10, 
             kde=True)),
        
        
        ("Top 10 Venues by Match Count", 
         lambda: sns.barplot(
             x=df2['venue'].value_counts().head(10).values, 
             y=df2['venue'].value_counts().head(10).index.tolist())),
        
        
        ("Spread of Result Margin", 
         lambda: sns.boxplot(
             x=df2['result_margin'])),
        
        
        ("Match Type Distribution", 
         lambda: sns.countplot(
             data=df2, 
             x='match_type')),
        
        
        ("Matches Played per Season", 
         lambda: sns.countplot(
             data=df2, 
             x='season')),
        
        
        ("Super Over Matches Distribution", 
         lambda: sns.countplot(
             data=df2, 
             x='super_over')),
        
        
        ("Top 10 Cities by Match Count", 
         lambda: sns.barplot(
             x=df2['city'].value_counts().head(10).values, 
             y=df2['city'].value_counts().head(10).index.tolist())),
        
        
        ("Top 10 Players of the Match", 
         lambda: sns.barplot(
             x=df2['player_of_match'].value_counts().head(10).values, 
             y=df2['player_of_match'].value_counts().head(10).index.tolist()))
    ]


    for i, (title, func) in enumerate(plots):
        st.subheader(f"{i+1}. {title}")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            func()
            
            if "Venue" in title or "Season" in title or "City" in title:
                plt.xticks(rotation=45)
                
            st.pyplot(fig)
        with col2:
            ai_explainer_ui(fig, f"team_btn_{i}")
    st.divider()
        
elif analysis_mode == "Ball-by-Ball Analysis":

    def get_phase(over):
        if over < 6: return 'Powerplay'
        elif over < 15: return 'Middle'
        else: return 'Death'
    df['phase'] = df['over'].apply(get_phase)

    bb_plots = [
        ("Valid vs Extra Deliveries", lambda: sns.countplot(data=df, x='valid_ball', palette='Set1')),
        ("Runs Scored per Delivery", lambda: sns.countplot(data=df, x='runs_total', palette='viridis')),
        ("Deliveries Faced by Batting Position", lambda: sns.countplot(data=df, x='bat_pos', palette='plasma')),
        ("Ball Data Volume per Over", lambda: sns.histplot(x=df['over'], bins=20, kde=True, color='orange')),
        ("Breakdown of Extras", lambda: sns.countplot(data=df[df['extra_type'].notna()], y='extra_type', palette='magma')),
        ("DRS Review Decisions", lambda: plt.pie(df[df['review_decision'].notna()]['review_decision'].value_counts(), labels=df[df['review_decision'].notna()]['review_decision'].value_counts().index.tolist(), autopct='%1.1f%%')),
        ("Top 10 Bowlers by Workload", lambda: sns.barplot(
            x=df['bowler'].value_counts().head(10).values, 
            y=df['bowler'].value_counts().head(10).index, 
            hue=df['bowler'].value_counts().head(10).index, 
            palette='rocket', 
            legend=False
        )),
        ("Top 10 Fielder involvements", lambda: sns.barplot(
            x=df['fielders'].value_counts().head(10).values, 
            y=df['fielders'].value_counts().head(10).index, 
            hue=df['fielders'].value_counts().head(10).index, 
            palette='flare', 
            legend=False
        )),
        ("Non-Striker Position Distribution", lambda: sns.countplot(
            data=df, 
            x='non_striker_pos', 
            hue='non_striker_pos', 
            palette='Set2', 
            legend=False
        )),
        ("Ball Distribution by Match Phase", lambda: plt.pie(x=df['phase'].value_counts(), labels=df['phase'].value_counts().index.tolist(), autopct='%1.1f%%'))
    ]

    for i, (title, func) in enumerate(bb_plots):
        st.subheader(f"{i+1}. {title}")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            func()
            st.pyplot(fig)
        with col2:
            ai_explainer_ui(fig, f"bb_btn_{i}")
        st.divider()
        
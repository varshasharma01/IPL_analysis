import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import base64
import requests

st.set_page_config(layout='wide', page_title='IPL Analysis with AI Explainer')

@st.cache_data
def load_data():
    # read raw data and normalize the season column so filtering is reliable
    df1 = pd.read_csv('IPL_ball_by_ball.csv', dtype={
        'season': str  # keep as string to accommodate mixed formats like "2007/08"
    })
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

def ai_explainer_ui(fig, key_id):
    if st.button("Explain with AI", key=key_id):
        with st.spinner("Analyzing with AI..."):
            try:
                img_b64 = get_image_base64(fig)
                response = requests.post("http://127.0.0.1:8000/explain-chart", json={"base64_image": img_b64})
                if response.status_code == 200:
                    st.info(response.json()['explanation'])
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Connection Error: {e}")

st.title(f"🏏 {analysis_mode}")

if analysis_mode == "Overview":
    
    st.title("IPL Cricket Analytics Dashboard")

    # Create the "All" option for the season selector
    # derive options from the cleaned season column (strings)
    available_seasons = sorted(df2['season'].unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
    options = ["All Seasons"] + available_seasons

    selected_option = st.selectbox("Select Season Focus", options)

    # after normalization both dataframes use text seasons, so filtering is straightforward
    if selected_option == "All Seasons":
        df_filtered = df.copy()
        df2_filtered = df2.copy()
        title_prefix = "All Time"
    else:
        # selected_option holds the season number as a string (e.g. '1', '2', ...)
        sel_season = selected_option
        df_filtered = df[df['season_num'] == sel_season]
        df2_filtered = df2[df2['season'] == sel_season]
        title_prefix = f"Season {selected_option}"

    # compute aggregates once the filtered frames are ready
    total_seasons = df2_filtered['season'].nunique()
    total_matches = df_filtered['match_id'].nunique()
    total_runs = pd.to_numeric(df_filtered['runs_total'], errors='coerce').sum()
    total_wickets = df_filtered['wicket_kind'].notna().sum()
    total_sixes = df_filtered[df_filtered['runs_batter'] == 6].shape[0]
    total_fours = df_filtered[df_filtered['runs_batter'] == 4].shape[0]

    # 3. Calculate Metrics based on filtered data
    

    # 4. Display Metrics
    st.subheader(f"📊 {title_prefix} Summary")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

    m_col1.metric("Seasons", total_seasons)
    m_col2.metric("Matches", total_matches)
    m_col3.metric("Total Runs", f"{int(total_runs):,}")
    m_col4.metric("Wickets", total_wickets)
    m_col5.metric("6s", total_sixes)
    m_col6.metric("4s", total_fours)
    col1, col2 = st.columns(2)
    

    with col1:
        st.subheader("Matches per Season")
        # Using df2 for season trends
        season_counts = df2_filtered['season'].value_counts().sort_index()
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=season_counts.index, y=season_counts.values, marker='o', color='#1f77b4', ax=ax1)
        ax1.set_xlabel("Season")
        ax1.set_ylabel("Matches Played")
        st.pyplot(fig1)
        ai_explainer_ui(fig1, "overview_season")
    
    with col2:
        
        st.subheader("Top 10 Run Scoring Teams")
        team_runs = (
            df_filtered.groupby('batting_team')['runs_total']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=team_runs.values, y=team_runs.index, palette="viridis", ax=ax2)
        ax2.set_xlabel("Total Runs")
        ax2.set_ylabel("Team")
        st.pyplot(fig2)
        ai_explainer_ui(fig2, "overview_teams")
        
    st.divider()

    # 6. Run Distribution
    st.subheader("Run Distribution per Ball")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.histplot(data=df_filtered, x='runs_total', bins=8, kde=True, color='orange', ax=ax3)
    ax3.set_xlabel("Runs per Ball")
    st.pyplot(fig3)
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
        ("Top 10 Bowlers by Workload", lambda: sns.barplot(x=df['bowler'].value_counts().head(10).values, y=df['bowler'].value_counts().head(10).index, palette='rocket')),
        ("Top 10 Fielder involvements", lambda: sns.barplot(x=df['fielders'].value_counts().head(10).values, y=df['fielders'].value_counts().head(10).index, palette='flare')),
        ("Non-Striker Position Distribution", lambda: sns.countplot(data=df, x='non_striker_pos', palette='Set2')),
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

# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import io
# import base64
# import requests

# st.set_page_config(layout = 'wide', page_title='IPL Analysis with AI Explainer')

# df = pd.read_csv('IPL_ball_by_ball.csv')
# df2 = pd.read_csv('ipl_team.csv')

# st.set_page_config(layout="wide", page_title="IPL Analytics AI")

# st.sidebar.title("IPL Analytics Dashboard")
# analysis_mode = st.sidebar.radio(
#     "Select Analysis Level:",
#     ["Ball-by-Ball Analysis", "IPL Team Analysis (2008-2025)"]
# )

# def get_image_base64(fig):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches='tight')
#     return base64.b64encode(buf.getvalue()).decode('utf-8')

# st.title(f"🏏 {analysis_mode}")

# if analysis_mode == "IPL Team Analysis (2008-2025)":
    
#     # 1. Wicket Kind Analysis
#     st.subheader("1. Distribution of Wicket Types")
    
#     # Create Two Columns: Left for Plot, Right for AI
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         fig, ax = plt.subplots()
#         # Use your univariate logic here
#         sns.countplot(data=df, y='wicket_kind', order=df['wicket_kind'].value_counts().index, palette='viridis', ax=ax)
#         plt.title("Wicket Kind Frequency")
#         st.pyplot(fig)
        
#     with col2:
#         st.write("### AI Insight")
#         if st.button("Explain this Chart", key="btn_wicket"):
#             # Convert the current plot to base64
#             img_b64 = get_image_base64(fig)
            
#             # Call your FastAPI (Ensure your FastAPI is running!)
#             try:
#                 response = requests.post(
#                     "http://127.0.0.1:8000/explain-chart", 
#                     json={"base64_image": img_b64}
#                 )
#                 if response.status_code == 200:
#                     st.info(response.json()['explanation'])
#                 else:
#                     st.error("Failed to connect to AI server.")
#             except Exception as e:
#                 st.error(f"Error: {e}")

#     st.divider() # Separation between your 10 plots
    
# if you want to ignore a file to upload on github, you can create a .gitignore file in the root of your repository and add the filename or pattern you want to ignore.
# For example, if you want to ignore all CSV files, you can add the following line to your .gitignore file:
# *.csv


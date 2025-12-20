import io
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="COVID-19 Analytics Dashboard", layout="wide", page_icon="📊")

# Custom color scheme - feel free to change these colors!
PRIMARY_COLOR = "#6C63FF"
SECONDARY_COLOR = "#FF6B9D"
ACCENT_COLOR = "#00D9FF"

st.markdown(f"""
<style>
body {{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}}

.block-container {{
  padding-top: 1.5rem;
}}

.main-header {{
  background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
  padding: 2rem;
  border-radius: 20px;
  margin-bottom: 2rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  color: white;
}}

.stTabs [data-baseweb="tab-list"] {{
  background: linear-gradient(90deg, rgba(108, 99, 255, 0.1) 0%, rgba(255, 107, 157, 0.1) 100%);
  border-radius: 12px;
  padding: 0.5rem;
  gap: 0.5rem;
}}

.stTabs [data-baseweb="tab"] {{
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  padding: 0.8rem 1.5rem;
  transition: all 0.3s ease;
  font-weight: 600;
  border: 2px solid transparent;
}}

.stTabs [data-baseweb="tab"]:hover {{
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-3px);
  border-color: {ACCENT_COLOR};
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
  color: white;
}}

/* Custom metric cards */
[data-testid="stMetric"] {{
  background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
  padding: 1rem;
  border-radius: 15px;
  border: 1px solid rgba(255,255,255,0.2);
  backdrop-filter: blur(10px);
}}

/* Sidebar styling */
section[data-testid="stSidebar"] > div {{
  background: linear-gradient(180deg, rgba(108, 99, 255, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
}}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {{
    h1, h2, h3, h4, h5, p, label, span, div {{
        color: #f5f5f5 !important;
    }}
}}

/* Custom button styling */
.stButton > button {{
  background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
  color: white;
  border: none;
  border-radius: 10px;
  padding: 0.5rem 1.5rem;
  font-weight: 600;
  transition: all 0.3s ease;
}}

.stButton > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(108, 99, 255, 0.4);
}}

/* Info boxes */
.info-box {{
  background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(108, 99, 255, 0.1));
  padding: 1.5rem;
  border-radius: 15px;
  border-left: 4px solid {ACCENT_COLOR};
  margin: 1rem 0;
}}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1 style='margin:0; font-size: 2.5rem;'>📊 COVID-19 Analytics Dashboard</h1>
  <p style='margin:0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;'>
    Interactive exploration and analysis of US COVID-19 pandemic data
  </p>
  <p style='margin:0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.7;'>
    Developed by Arham Yunus Awan | Programming for AI Course Project
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading COVID-19 data...")
def load_covid_data():
    try:
        url = "https://raw.githubusercontent.com/COVID19Tracking/covid-tracking-data/master/data/states_daily_4pm_et.csv"
        df = pd.read_csv(url, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['state', 'date'])
        
        numeric_cols = ['positive', 'negative', 'hospitalizedCurrently', 'inIcuCurrently', 
                       'onVentilatorCurrently', 'death', 'recovered', 'positiveIncrease', 
                       'negativeIncrease', 'deathIncrease', 'hospitalizedIncrease']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

with st.spinner('🔄 Loading data...'):
    df = load_covid_data()

if df.empty:
    st.error("⚠️ Unable to load data. Please check your connection and try again.")
    st.stop()

# Sidebar with custom styling
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.markdown("---")

states = sorted(df['state'].unique())

# Initialize session state
if 'selected_states' not in st.session_state:
    st.session_state['selected_states'] = ['CA', 'NY', 'TX', 'FL']

selected_states = st.sidebar.multiselect(
    '🗺️ Select States to Analyze',
    options=states,
    default=st.session_state['selected_states']
)

date_min = df['date'].min()
date_max = df['date'].max()
date_range = st.sidebar.date_input(
    '📅 Date Range',
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

if len(date_range) == 2:
    filtered_df = df[
        (df['state'].isin(selected_states)) &
        (df['date'] >= pd.to_datetime(date_range[0])) &
        (df['date'] <= pd.to_datetime(date_range[1]))
    ]
else:
    filtered_df = df[df['state'].isin(selected_states)]

st.sidebar.markdown('---')
st.sidebar.markdown('### ⚡ Quick Actions')

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button('🔥 Top 5', use_container_width=True):
        top_states = df.groupby('state')['positive'].max().nlargest(5).index.tolist()
        st.session_state['selected_states'] = top_states
        st.rerun()

with col_b:
    if st.button('🔄 Reset', use_container_width=True):
        st.session_state['selected_states'] = ['CA', 'NY', 'TX', 'FL']
        st.rerun()

# Display data info
st.sidebar.markdown('---')
st.sidebar.markdown('### 📊 Data Info')
st.sidebar.info(f"""
**Records**: {len(df):,}  
**States**: {len(df['state'].unique())}  
**Time Span**: {(df['date'].max() - df['date'].min()).days} days
""")

tabs = st.tabs(["🏠 Dashboard", "📈 Trend Analysis", "🌍 Geographic View", "🔬 Deep Dive", "💾 Export Data"])

# TAB 1: Dashboard Overview
with tabs[0]:
    st.markdown("### 📊 Key Performance Indicators")
    
    latest_data = df[df['date'] == df['date'].max()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_cases = latest_data['positive'].sum()
        st.metric('💉 Total Cases', f'{total_cases:,.0f}')
    with col2:
        total_deaths = latest_data['death'].sum()
        st.metric('⚰️ Total Deaths', f'{total_deaths:,.0f}')
    with col3:
        hospitalized = latest_data['hospitalizedCurrently'].sum()
        st.metric('🏥 Hospitalized', f'{hospitalized:,.0f}')
    with col4:
        mortality_rate = (total_deaths / total_cases * 100) if total_cases > 0 else 0
        st.metric('📉 Mortality Rate', f'{mortality_rate:.2f}%')
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Daily Cases Timeline")
        if not filtered_df.empty:
            daily_cases = filtered_df.groupby('date')['positiveIncrease'].sum().reset_index()
            fig = px.area(daily_cases, x='date', y='positiveIncrease', 
                         title='Daily New COVID-19 Cases',
                         labels={'positiveIncrease': 'New Cases', 'date': 'Date'})
            fig.update_traces(line_color=PRIMARY_COLOR, fillcolor=f'rgba(108, 99, 255, 0.3)')
            fig.update_layout(hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Quick Stats")
        if not filtered_df.empty:
            avg_daily = filtered_df['positiveIncrease'].mean()
            max_daily = filtered_df['positiveIncrease'].max()
            total_new = filtered_df['positiveIncrease'].sum()
            
            st.markdown(f"""
            <div class="info-box">
            <strong>Average Daily Cases:</strong><br>{avg_daily:,.0f}
            </div>
            <div class="info-box">
            <strong>Peak Daily Cases:</strong><br>{max_daily:,.0f}
            </div>
            <div class="info-box">
            <strong>Total New Cases:</strong><br>{total_new:,.0f}
            </div>
            """, unsafe_allow_html=True)

# TAB 2: Trends
with tabs[1]:
    st.markdown("### 📊 Multi-Metric Trend Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        metric_option = st.selectbox(
            '📌 Select Primary Metric',
            ['positiveIncrease', 'deathIncrease', 'hospitalizedCurrently', 
             'positive', 'death', 'recovered'],
            format_func=lambda x: x.replace('Increase', ' (Daily)').replace('Currently', ' (Current)').title()
        )
    
    with col2:
        chart_type = st.radio("Chart Type", ["Line", "Area"], horizontal=True)
    
    if not filtered_df.empty and selected_states:
        fig = go.Figure()
        
        for idx, state in enumerate(selected_states):
            state_data = filtered_df[filtered_df['state'] == state]
            
            if chart_type == "Area":
                fig.add_trace(go.Scatter(
                    x=state_data['date'],
                    y=state_data[metric_option],
                    mode='lines',
                    name=state,
                    fill='tonexty' if idx > 0 else 'tozeroy',
                    line=dict(width=2)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=state_data['date'],
                    y=state_data[metric_option],
                    mode='lines+markers',
                    name=state,
                    line=dict(width=3),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title=f'{metric_option.title()} Over Time',
            xaxis_title='Date',
            yaxis_title=metric_option.title(),
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📉 Smoothed Trends (7-Day Moving Average)")
        
        rolling_data = filtered_df.groupby('date')[metric_option].sum().rolling(window=7, center=True).mean().reset_index()
        fig2 = px.line(rolling_data, x='date', y=metric_option,
                      title='7-Day Rolling Average (Smoothed)',
                      labels={metric_option: f'{metric_option} (7-day avg)', 'date': 'Date'})
        fig2.update_traces(line_color=SECONDARY_COLOR, line_width=3)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# TAB 3: Geographic Comparison
with tabs[2]:
    st.markdown("### 🗺️ State-by-State Analysis")
    
    comparison_metric = st.selectbox(
        '📊 Choose Comparison Metric',
        ['Total Cases', 'Total Deaths', 'Mortality Rate', 'Peak Daily Cases', 'Current Hospitalizations']
    )
    
    latest = df[df['date'] == df['date'].max()].copy()
    
    if comparison_metric == 'Total Cases':
        latest['metric'] = latest['positive']
        title = '🦠 Total COVID-19 Cases'
    elif comparison_metric == 'Total Deaths':
        latest['metric'] = latest['death']
        title = '⚰️ Total COVID-19 Deaths'
    elif comparison_metric == 'Mortality Rate':
        latest['metric'] = (latest['death'] / latest['positive'] * 100).fillna(0)
        title = '📉 Mortality Rate (%)'
    elif comparison_metric == 'Current Hospitalizations':
        latest['metric'] = latest['hospitalizedCurrently']
        title = '🏥 Current Hospitalizations'
    else:
        peak_cases = df.groupby('state')['positiveIncrease'].max().reset_index()
        peak_cases.columns = ['state', 'metric']
        latest = latest.merge(peak_cases, on='state', how='left')
        title = '🔥 Peak Daily Cases'
    
    top_20 = latest.nlargest(20, 'metric')
    
    fig = px.bar(top_20, x='state', y='metric',
                title=f'{title} by State (Top 20)',
                labels={'metric': comparison_metric, 'state': 'State'},
                color='metric',
                color_continuous_scale='Viridis')
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🏥 Healthcare System Impact")
    
    if not filtered_df.empty:
        hosp_data = filtered_df.groupby('date')[['hospitalizedCurrently', 'inIcuCurrently', 'onVentilatorCurrently']].sum().reset_index()
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hosp_data['date'], y=hosp_data['hospitalizedCurrently'],
                                 mode='lines', name='🏥 Hospitalized', 
                                 line=dict(color='#FF6B6B', width=3),
                                 fill='tonexty'))
        fig2.add_trace(go.Scatter(x=hosp_data['date'], y=hosp_data['inIcuCurrently'],
                                 mode='lines', name='🚨 ICU', 
                                 line=dict(color='#FFA07A', width=3)))
        fig2.add_trace(go.Scatter(x=hosp_data['date'], y=hosp_data['onVentilatorCurrently'],
                                 mode='lines', name='🫁 Ventilator', 
                                 line=dict(color='#FF4500', width=3)))
        
        fig2.update_layout(
            title='Healthcare Capacity Utilization Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Patients',
            hovermode='x unified',
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

# TAB 4: Deep Dive / EDA
with tabs[3]:
    st.markdown("### 🔬 Statistical Deep Dive")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('📊 Total Records', f'{len(df):,}')
    with col2:
        st.metric('🗺️ States/Territories', len(df['state'].unique()))
    with col3:
        st.metric('📅 Days of Data', f'{(df["date"].max() - df["date"].min()).days}')
    
    st.markdown("---")
    
    tab_a, tab_b, tab_c = st.tabs(["📊 Correlation", "📈 Distribution", "📋 Summary Stats"])
    
    with tab_a:
        st.markdown("#### Correlation Heatmap")
        corr_cols = ['positive', 'death', 'hospitalizedCurrently', 'inIcuCurrently', 
                     'positiveIncrease', 'deathIncrease']
        corr_data = df[corr_cols].dropna()
        
        if not corr_data.empty:
            corr_matrix = corr_data.corr()
            
            fig = px.imshow(corr_matrix,
                           labels=dict(color="Correlation"),
                           x=corr_matrix.columns,
                           y=corr_matrix.columns,
                           color_continuous_scale='RdBu_r',
                           aspect="auto",
                           title='Feature Correlation Matrix',
                           zmin=-1, zmax=1)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_b:
        st.markdown("#### Distribution Analysis")
        dist_metric = st.selectbox('Select metric for distribution analysis',
                                  ['positiveIncrease', 'deathIncrease', 'hospitalizedCurrently'])
        
        if not filtered_df.empty:
            fig = px.histogram(filtered_df, x=dist_metric, nbins=50,
                              title=f'Distribution of {dist_metric}',
                              labels={dist_metric: dist_metric},
                              color_discrete_sequence=[PRIMARY_COLOR],
                              marginal="box")
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_c:
        st.markdown("#### Statistical Summary")
        summary_stats = filtered_df[['positive', 'death', 'hospitalizedCurrently', 
                                     'positiveIncrease', 'deathIncrease']].describe()
        st.dataframe(summary_stats, use_container_width=True)

# TAB 5: Export
with tabs[4]:
    st.markdown("### 💾 Data Export & Downloads")
    
    st.markdown("#### 📄 Preview Filtered Data")
    st.dataframe(filtered_df.head(25), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### ⬇️ Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Filtered Data (CSV)",
            data=csv,
            file_name=f'covid_filtered_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        full_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Complete Dataset (CSV)",
            data=full_csv,
            file_name=f'covid_complete_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col3:
        if not filtered_df.empty:
            state_summary = filtered_df.groupby('state').agg({
                'positive': 'max',
                'death': 'max',
                'hospitalizedCurrently': 'mean',
                'positiveIncrease': 'sum',
                'deathIncrease': 'sum'
            }).round(2)
            summary_csv = state_summary.to_csv().encode('utf-8')
            st.download_button(
                label="📥 State Summary (CSV)",
                data=summary_csv,
                file_name=f'state_summary_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
                use_container_width=True
            )
    
    st.markdown("---")
    st.markdown("#### 📊 Aggregated State Statistics")
    
    if not filtered_df.empty:
        state_summary = filtered_df.groupby('state').agg({
            'positive': 'max',
            'death': 'max',
            'hospitalizedCurrently': 'mean',
            'positiveIncrease': 'sum',
            'deathIncrease': 'sum'
        }).round(2)
        
        state_summary.columns = ['Total Cases', 'Total Deaths', 'Avg Hospitalized', 
                                'Total New Cases', 'Total New Deaths']
        st.dataframe(state_summary, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(255, 107, 157, 0.1)); border-radius: 15px;'>
    <strong style='font-size: 1.1rem;'>🎓 Programming for AI Course Project</strong><br>
    <span style='font-size: 0.9rem;'>Developed by <strong>Arham Yunus Awan</strong></span><br>
    <span style='font-size: 0.85rem; opacity: 0.8;'>Data Source: The COVID Tracking Project | Dataset Last Updated: March 2021</span>
</div>
""", unsafe_allow_html=True)
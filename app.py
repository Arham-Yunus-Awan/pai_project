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

st.set_page_config(page_title="COVID-19 Data Explorer", layout="wide", page_icon="🦠")

st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #e0f7fa 0%, #fbe9e7 100%) !important;
}

.block-container {
  padding-top: 2rem;
}

.glass-header {
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  background: rgba(255, 255, 255, 0.28);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

.stTabs [data-baseweb="tab-list"] {
  backdrop-filter: blur(10px) saturate(150%);
  background: rgba(255,255,255,0.2);
  border-radius: 16px;
  padding: 0.4rem 0.6rem;
}

.stTabs [data-baseweb="tab"] {
  backdrop-filter: blur(8px) saturate(160%);
  background: rgba(255, 255, 255, 0.35);
  border-radius: 14px;
  margin: 0 6px;
  padding: 0.6rem 1rem;
  transition: 0.25s ease;
  font-weight: 600;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(255, 255, 255, 0.55);
  transform: translateY(-2px);
}

h1, h2, h3, h4, h5, p, label, span {
  color: #2d2d2d !important;
  font-weight: 500 !important;
}

@media (prefers-color-scheme: dark) {
    h1, h2, h3, h4, h5, p, label, span, div {
        color: #f5f5f5 !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #f5f5f5 !important;
        background: rgba(255, 255, 255, 0.15) !important;
    }

    .small-text {
        color: #e0e0e0 !important;
    }

    .glass-header, 
    .stTabs [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab"],
    [data-testid="stSlider"] > div,
    section[data-testid="stSidebar"] > div {
        background: rgba(50, 50, 50, 0.35) !important;
        backdrop-filter: blur(14px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
}

[data-testid="stSlider"] > div {
  backdrop-filter: blur(10px) saturate(180%);
  background: rgba(255, 255, 255, 0.45);
  padding: 1rem;
  border-radius: 16px;
}

.metric-card {
  backdrop-filter: blur(10px) saturate(180%);
  background: rgba(255, 255, 255, 0.45);
  padding: 1.2rem;
  border-radius: 16px;
  margin: 0.5rem 0;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card petals">
  <div style='display:flex; align-items:center; justify-content:space-between'>
    <div>
      <h1 class='display'>COVID-19 Data Explorer 🦠</h1>
      <div class='small-muted'>Comprehensive analysis of US COVID-19 tracking data — explore trends, compare states, and visualize the pandemic.</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:0.9rem; color:#334;'>Made for Sir Zeeshan, with love from Enigjes 💕💦</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_covid_data():
    url = "https://raw.githubusercontent.com/COVID19Tracking/covid-tracking-data/master/data/states_daily_4pm_et.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values(['state', 'date'])
    
    numeric_cols = ['positive', 'negative', 'hospitalizedCurrently', 'inIcuCurrently', 
                   'onVentilatorCurrently', 'death', 'recovered', 'positiveIncrease', 
                   'negativeIncrease', 'deathIncrease', 'hospitalizedIncrease']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

df = load_covid_data()

st.sidebar.header("🔍 Filters & Controls")

states = sorted(df['state'].unique())
selected_states = st.sidebar.multiselect(
    'Select States',
    options=states,
    default=['CA', 'NY', 'TX', 'FL']
)

date_min = df['date'].min()
date_max = df['date'].max()
date_range = st.sidebar.date_input(
    'Date Range',
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
st.sidebar.write('**Quick Presets**')
if st.sidebar.button('Top 5 Most Affected'):
    top_states = df.groupby('state')['positive'].max().nlargest(5).index.tolist()
    selected_states = top_states
    st.rerun()

if st.sidebar.button('Reset Filters'):
    selected_states = ['CA', 'NY', 'TX', 'FL']
    st.rerun()

tabs = st.tabs(["📊 Overview", "📈 Trends", "🗺️ State Comparison", "📉 EDA & Stats", "💾 Data Export"])

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Key Metrics Summary')
    
    latest_data = df[df['date'] == df['date'].max()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_cases = latest_data['positive'].sum()
        st.metric('Total Cases', f'{total_cases:,.0f}')
    with col2:
        total_deaths = latest_data['death'].sum()
        st.metric('Total Deaths', f'{total_deaths:,.0f}')
    with col3:
        hospitalized = latest_data['hospitalizedCurrently'].sum()
        st.metric('Currently Hospitalized', f'{hospitalized:,.0f}')
    with col4:
        mortality_rate = (total_deaths / total_cases * 100) if total_cases > 0 else 0
        st.metric('Mortality Rate', f'{mortality_rate:.2f}%')
    
    st.markdown('---')
    st.subheader('Daily New Cases Over Time')
    
    if not filtered_df.empty:
        daily_cases = filtered_df.groupby('date')['positiveIncrease'].sum().reset_index()
        fig = px.line(daily_cases, x='date', y='positiveIncrease', 
                     title='Daily New Cases',
                     labels={'positiveIncrease': 'New Cases', 'date': 'Date'})
        fig.update_traces(line_color='#FF6B6B', line_width=2)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Temporal Trends Analysis')
    
    metric_option = st.selectbox(
        'Select Metric to Visualize',
        ['positiveIncrease', 'deathIncrease', 'hospitalizedCurrently', 
         'positive', 'death', 'recovered']
    )
    
    if not filtered_df.empty and selected_states:
        fig = go.Figure()
        for state in selected_states:
            state_data = filtered_df[filtered_df['state'] == state]
            fig.add_trace(go.Scatter(
                x=state_data['date'],
                y=state_data[metric_option],
                mode='lines',
                name=state,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f'{metric_option} Over Time by State',
            xaxis_title='Date',
            yaxis_title=metric_option,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('---')
        st.subheader('7-Day Rolling Average')
        
        rolling_data = filtered_df.groupby('date')[metric_option].sum().rolling(window=7).mean().reset_index()
        fig2 = px.area(rolling_data, x='date', y=metric_option,
                      title='7-Day Rolling Average',
                      labels={metric_option: f'{metric_option} (7-day avg)', 'date': 'Date'})
        fig2.update_traces(line_color='#4ECDC4', fillcolor='rgba(78, 205, 196, 0.3)')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('State-by-State Comparison')
    
    comparison_metric = st.selectbox(
        'Comparison Metric',
        ['Total Cases', 'Total Deaths', 'Mortality Rate', 'Peak Daily Cases']
    )
    
    latest = df[df['date'] == df['date'].max()].copy()
    
    if comparison_metric == 'Total Cases':
        latest['metric'] = latest['positive']
        title = 'Total COVID-19 Cases by State'
    elif comparison_metric == 'Total Deaths':
        latest['metric'] = latest['death']
        title = 'Total COVID-19 Deaths by State'
    elif comparison_metric == 'Mortality Rate':
        latest['metric'] = (latest['death'] / latest['positive'] * 100).fillna(0)
        title = 'Mortality Rate by State (%)'
    else:
        peak_cases = df.groupby('state')['positiveIncrease'].max().reset_index()
        peak_cases.columns = ['state', 'metric']
        latest = latest.merge(peak_cases, on='state', how='left')
        title = 'Peak Daily Cases by State'
    
    top_20 = latest.nlargest(20, 'metric')
    
    fig = px.bar(top_20, x='state', y='metric',
                title=f'{title} (Top 20)',
                labels={'metric': comparison_metric, 'state': 'State'},
                color='metric',
                color_continuous_scale='Reds')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.subheader('Hospitalization Trends')
    
    if not filtered_df.empty:
        hosp_data = filtered_df.groupby('date')[['hospitalizedCurrently', 'inIcuCurrently', 'onVentilatorCurrently']].sum().reset_index()
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hosp_data['date'], y=hosp_data['hospitalizedCurrently'],
                                 mode='lines', name='Hospitalized', line=dict(color='#FF6B6B', width=2)))
        fig2.add_trace(go.Scatter(x=hosp_data['date'], y=hosp_data['inIcuCurrently'],
                                 mode='lines', name='ICU', line=dict(color='#FFA07A', width=2)))
        fig2.add_trace(go.Scatter(x=hosp_data['date'], y=hosp_data['onVentilatorCurrently'],
                                 mode='lines', name='On Ventilator', line=dict(color='#FF4500', width=2)))
        
        fig2.update_layout(
            title='Hospitalization Status Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Patients',
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Exploratory Data Analysis & Statistics')
    
    st.markdown('**Dataset Overview**')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Records', f'{len(df):,}')
    with col2:
        st.metric('States Covered', len(df['state'].unique()))
    with col3:
        st.metric('Date Range', f'{(df["date"].max() - df["date"].min()).days} days')
    
    st.markdown('---')
    st.markdown('**Correlation Analysis**')
    
    corr_cols = ['positive', 'death', 'hospitalizedCurrently', 'inIcuCurrently', 
                 'positiveIncrease', 'deathIncrease']
    corr_data = df[corr_cols].dropna()
    
    if not corr_data.empty:
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu',
                       aspect="auto",
                       title='Feature Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('**Statistical Summary**')
    
    summary_stats = filtered_df[['positive', 'death', 'hospitalizedCurrently', 
                                 'positiveIncrease', 'deathIncrease']].describe()
    st.dataframe(summary_stats)
    
    st.markdown('---')
    st.markdown('**Distribution Analysis**')
    
    dist_metric = st.selectbox('Select metric for distribution',
                              ['positiveIncrease', 'deathIncrease', 'hospitalizedCurrently'])
    
    if not filtered_df.empty:
        fig = px.histogram(filtered_df, x=dist_metric, nbins=50,
                          title=f'Distribution of {dist_metric}',
                          labels={dist_metric: dist_metric},
                          color_discrete_sequence=['#4ECDC4'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Data Export & Raw Data')
    
    st.markdown('**Sample Data**')
    st.dataframe(filtered_df.head(20))
    
    st.markdown('---')
    st.markdown('**Download Options**')
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f'covid_data_filtered_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    
    with col2:
        full_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=full_csv,
            file_name=f'covid_data_complete_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    
    st.markdown('---')
    st.markdown('**Data Summary by State**')
    
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
        st.dataframe(state_summary)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:18px; padding:12px; background: rgba(255,255,255,0.6); border-radius:12px'>
<strong>Made by ENIGJES</strong><br>
Data Source: The COVID Tracking Project | Last Updated: March 2021
</div>
""", unsafe_allow_html=True)
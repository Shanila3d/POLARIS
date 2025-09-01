import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from backend_v2 import PolarisBackendV2 as PolarisBackend

# Page config
st.set_page_config(
    page_title="POLARIS - ESG What-If Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    h1 {color: #1e3d59; font-family: 'Arial Black';}
    h2 {color: #2e5266;}
</style>
""", unsafe_allow_html=True)

# Initialize backend
@st.cache_resource
def load_backend():
    return PolarisBackend()

backend = load_backend()

# Title
st.title("🌍 POLARIS: Predictive ESG & Labor Analytics Dashboard")
st.markdown("**Interactive What-If Scenario Modeling for Financial vs Non-Financial Sectors**")

# Sidebar controls
st.sidebar.header("Scenario Configuration")

scenario_type = st.sidebar.selectbox(
    "Scenario Type",
    ["carbon_tax", "esg_mandate", "talent_crisis", "supply_shock", "green_transition"],
    format_func=lambda x: {
        'carbon_tax': '💰 Carbon Tax Implementation',
        'esg_mandate': '📋 ESG Disclosure Mandate',
        'talent_crisis': '👥 ESG Talent Crisis',
        'supply_shock': '🏭 Supply Chain Disruption',
        'green_transition': '🌱 Green Transition Acceleration'
    }[x]
)

magnitude = st.sidebar.slider(
    "Shock Magnitude",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Multiplier for shock intensity"
)

timing = st.sidebar.radio(
    "Implementation Timing",
    ["Immediate", "Phased", "Delayed"],
    help="How the shock is applied over time"
)

sector_filter = st.sidebar.radio(
    "Sector Focus",
    ["Both", "Financial", "Non-Financial"]
)

# Main dashboard
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.header("Current State")
    current = backend.get_current_state(sector_filter if sector_filter != "Both" else "all")
    
    st.metric("Avg ESG Score", f"{current['avg_esg_score']:.1f}")
    st.metric("ESG Jobs/Company", f"{current['avg_esg_jobs']:.1f}")
    st.metric("Companies", f"{current['n_companies']:,}")
    st.caption(f"As of {current['quarter']}")

with col2:
    st.header("Scenario Impact Projection")
    
    # Generate predictions
    predictions = backend.apply_live_scenario(
        scenario_type, magnitude, timing, sector_filter
    )
    
    if not predictions.empty:
        # Aggregate by quarter
        quarterly = predictions.groupby(['year', 'quarter'])['prediction'].mean().reset_index()
        quarterly['date'] = pd.to_datetime(
            quarterly['year'].astype(str) + 'Q' + quarterly['quarter'].astype(str)
        )
        
        # Create projection chart
        fig = go.Figure()
        
        # Baseline
        baseline = backend.data['base_predictions']
        baseline_quarterly = baseline.groupby(['year', 'quarter'])['pred_target_esg_1q'].mean().reset_index()
        baseline_quarterly['date'] = pd.to_datetime(
            baseline_quarterly['year'].astype(str) + 'Q' + baseline_quarterly['quarter'].astype(str)
        )
        
        fig.add_trace(go.Scatter(
            x=baseline_quarterly['date'],
            y=baseline_quarterly['pred_target_esg_1q'],
            mode='lines',
            name='Baseline',
            line=dict(color='gray', dash='dash')
        ))
        
        # Scenario
        fig.add_trace(go.Scatter(
            x=quarterly['date'],
            y=quarterly['prediction'],
            mode='lines+markers',
            name='Scenario Impact',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="ESG Score Trajectory",
            xaxis_title="Quarter",
            yaxis_title="ESG Score",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.header("Impact Summary")
    
    if not predictions.empty:
        # Calculate impacts
        baseline_mean = backend.data['base_predictions']['pred_target_esg_1q'].mean()
        scenario_mean = predictions['prediction'].mean()
        impact = ((scenario_mean - baseline_mean) / baseline_mean) * 100
        
        st.metric(
            "ESG Impact",
            f"{impact:+.1f}%",
            delta=f"{scenario_mean - baseline_mean:+.1f} points"
        )
        
        # Sector breakdown if both
        if sector_filter == "Both":
            st.subheader("Sector Comparison")
            
            # Mock sector impacts (replace with actual calculation)
            sector_impacts = pd.DataFrame({
                'Sector': ['Financial', 'Non-Financial'],
                'Impact': [impact * 1.2, impact * 0.8]  # Example multipliers
            })
            
            fig_bar = px.bar(
                sector_impacts,
                x='Sector',
                y='Impact',
                color='Impact',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="Differential Impact"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# Bottom section - Comparative Analysis
st.header("Multi-Scenario Comparison")

col4, col5 = st.columns(2)

with col4:
    st.subheader("Configure Scenarios")
    
    # Allow multiple scenario configuration
    scenarios = []
    
    with st.expander("Scenario 1", expanded=True):
        s1_name = st.text_input("Name", "Baseline", key="s1_name")
        scenarios.append({
            'name': s1_name,
            'type': 'carbon_tax',
            'magnitude': 0,
            'timing': 'Immediate',
            'sector': 'Both'
        })
    
    with st.expander("Scenario 2"):
        s2_name = st.text_input("Name", "Moderate Carbon Tax", key="s2_name")
        s2_type = st.selectbox("Type", ["carbon_tax", "esg_mandate"], key="s2_type")
        s2_mag = st.slider("Magnitude", 0.5, 3.0, 1.0, key="s2_mag")
        scenarios.append({
            'name': s2_name,
            'type': s2_type,
            'magnitude': s2_mag,
            'timing': 'Phased',
            'sector': 'Both'
        })

with col5:
    st.subheader("Comparative Results")
    
    if st.button("Run Comparison"):
        comparison = backend.get_scenario_comparison(scenarios)
        
        if not comparison.empty:
            # Create comparison chart
            fig_comp = go.Figure()
            
            for scenario_name in comparison['scenario_name'].unique():
                scenario_data = comparison[comparison['scenario_name'] == scenario_name]
                quarterly = scenario_data.groupby(['year', 'quarter'])['prediction'].mean().reset_index()
                
                fig_comp.add_trace(go.Scatter(
                    x=quarterly.index,
                    y=quarterly['prediction'],
                    mode='lines',
                    name=scenario_name
                ))
            
            fig_comp.update_layout(
                title="Scenario Comparison",
                xaxis_title="Time Period",
                yaxis_title="ESG Score",
                height=350
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)

# Footer
st.markdown("---")
st.caption("POLARIS v1.0 | MSc Thesis - Bank of England Collaboration | Data: Refinitiv Eikon + LinkedIn")
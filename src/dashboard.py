import sys
sys.path.append('../scripts') 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from user_engagement_analysis import aggregate_metrics, perform_kmeans_clustering
from experience_analytics import plot_clusters_2d, plot_tcp_per_handset

# Load your data here
data = pd.read_csv('../telecom.csv')

# Dashboard pages
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["User Overview", "User Engagement", "Experience Analysis", "Satisfaction Analysis"]
)

if page == "User Overview":
    st.title("User Overview Analysis")
    st.write("User overview KPIs and key insights.")
    
    # Example plot (replace with your actual KPIs)
    fig, ax = plt.subplots()
    data['Handset Manufacturer'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

elif page == "User Engagement":
    st.title("User Engagement Analysis")
    
    # Engagement metrics (from your function in user_engagement_analysis.py)
    engagement_metrics = aggregate_metrics(data)
    st.write("Engagement metrics for the top customers.")
    
    # Visualize engagement metrics
    fig, ax = plt.subplots()
    engagement_metrics['metric'].plot(kind='bar', ax=ax)
    st.pyplot(fig)

elif page == "Experience Analysis":
    st.title("Experience Analysis")
    
    # Example 2D Clustering Plot (from experience_analytics.py)
    plot_clusters_2d(data)
    st.pyplot()

elif page == "Satisfaction Analysis":
    st.title("Satisfaction Analysis")
    
    # Satisfaction score analysis
    fig, ax = plt.subplots()
    st.write("Satisfaction scores based on engagement and experience analysis.")
    # You can add more plots/metrics as required here
    ax.hist(data['satisfaction_score'])
    st.pyplot(fig)

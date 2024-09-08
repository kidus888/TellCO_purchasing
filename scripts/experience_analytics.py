import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

def preprocess_data(df):
    # Check data types
    print(df.dtypes)

    # Select relevant columns (numerical and categorical separately)
    numerical_cols = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                      'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    categorical_cols = ['Handset Type']
    
    # Ensure the selected columns are in the DataFrame
    for col in numerical_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the DataFrame.")
    
    # Handle missing values in numerical columns
    imputer_mean = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_mean.fit_transform(df[numerical_cols])

    # Handle missing values in categorical columns
    imputer_mode = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

    # Handle outliers (clipping)
    for col in numerical_cols:
        lower, upper = np.percentile(df[col], [1, 99])
        df[col] = np.clip(df[col], lower, upper)

    return df

# Task 3.1: Aggregation per customer
def aggregate_per_customer(df):
    agg_funcs = {
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Handset Type': 'first'
    }
    return df.groupby('Bearer Id').agg(agg_funcs)

# Task 3.2: Compute top, bottom, and most frequent values
def compute_top_bottom_frequent(df, column):
    top_10 = df[column].nlargest(10)
    bottom_10 = df[column].nsmallest(10)
    most_frequent = df[column].value_counts().head(10)
    return top_10, bottom_10, most_frequent

# Task 3.3: Distribution and analysis by handset type
def analysis_by_handset_type(df):
    throughput_dist = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
    tcp_retrans_dist = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()
    return throughput_dist, tcp_retrans_dist



# Task 3.4: K-means clustering
def perform_kmeans_clustering(df, k=3):
    X = df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans


# Plot throughput per handset using Plotly
def plot_throughput_per_handset(throughput_dist):
    fig = px.bar(throughput_dist, x=throughput_dist.index, y=throughput_dist.values, 
                 labels={'x': 'Handset Type', 'y': 'Average Throughput (kbps)'},
                 title='Average Throughput per Handset Type')
    fig.show()

# Plot TCP retransmission per handset using Plotly
def plot_tcp_per_handset(tcp_retrans_dist):
    fig = px.bar(tcp_retrans_dist, x=tcp_retrans_dist.index, y=tcp_retrans_dist.values, 
                 labels={'x': 'Handset Type', 'y': 'TCP DL Retransmission (Bytes)'},
                 title='TCP Retransmission per Handset Type')
    fig.show()

# Function to visualize clusters in 3D

def plot_clusters_3d(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot RTT vs Throughput vs TCP Retransmission colored by cluster
    scatter = ax.scatter(df['Avg RTT DL (ms)'], df['Avg Bearer TP DL (kbps)'], df['TCP DL Retrans. Vol (Bytes)'],
                         c=df['Cluster'], cmap='viridis', marker='o')

    ax.set_title('K-Means Clustering (3D) - RTT, Throughput, TCP Retransmission')
    ax.set_xlabel('Avg RTT DL (ms)')
    ax.set_ylabel('Avg Bearer TP DL (kbps)')
    ax.set_zlabel('TCP DL Retrans. Vol (Bytes)')
    
    # Colorbar for clusters
    fig.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.show()

# Function to visualize clusters in 2D
def plot_clusters_2d(df):
    # Plot RTT vs Throughput colored by cluster
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df['Avg RTT DL (ms)'], df['Avg Bearer TP DL (kbps)'], 
                          c=df['Cluster'], cmap='viridis', marker='o')
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('K-Means Clustering (2D) - RTT vs Throughput')
    plt.xlabel('Avg RTT DL (ms)')
    plt.ylabel('Avg Bearer TP DL (kbps)')
    plt.grid(True)
    plt.show()
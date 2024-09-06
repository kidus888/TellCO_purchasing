import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def aggregate_metrics(df):
    """ Aggregate session metrics per customer (MSISDN) """
    agg_data = df.groupby("MSISDN/Number").agg({
        'Bearer Id': 'count',  # Session frequency
        'Dur. (ms)': 'sum',  # Total session duration
        'HTTP DL (Bytes)': 'sum',  # Total download traffic
        'HTTP UL (Bytes)': 'sum'   # Total upload traffic
    }).reset_index()
    
    # Rename columns for clarity
    agg_data.columns = ['MSISDN', 'Session Frequency', 'Total Duration (ms)', 
                        'Total DL Traffic (Bytes)', 'Total UL Traffic (Bytes)']
    
    # Total traffic (download + upload)
    agg_data['Total Traffic (Bytes)'] = agg_data['Total DL Traffic (Bytes)'] + agg_data['Total UL Traffic (Bytes)']
    
    return agg_data

def top_10_customers_per_metric(agg_data):
    """ Report top 10 customers per engagement metric """
    metrics = ['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']
    top_customers = {}
    
    for metric in metrics:
        top_customers[metric] = agg_data.nlargest(10, metric)[['MSISDN', metric]]
    
    return top_customers

def normalize_metrics(agg_data):
    """ Normalize the session engagement metrics """
    scaler = StandardScaler()
    agg_data[['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']] = scaler.fit_transform(
        agg_data[['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']])
    
    return agg_data

def perform_kmeans_clustering(agg_data, k=3):
    """ Perform K-means clustering to group customers into k clusters """
    kmeans = KMeans(n_clusters=k, random_state=42)
    agg_data['Cluster'] = kmeans.fit_predict(agg_data[['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']])
    
    return agg_data, kmeans

def compute_cluster_stats(agg_data):
    """ Compute minimum, maximum, average, and total engagement metrics for each cluster """
    cluster_stats = agg_data.groupby('Cluster').agg({
        'Session Frequency': ['min', 'max', 'mean', 'sum'],
        'Total Duration (ms)': ['min', 'max', 'mean', 'sum'],
        'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
    })
    return cluster_stats

def compute_cluster_stats(agg_data):
    """ Compute minimum, maximum, average, and total engagement metrics for each cluster """
    cluster_stats = agg_data.groupby('Cluster').agg({
        'Session Frequency': ['min', 'max', 'mean', 'sum'],
        'Total Duration (ms)': ['min', 'max', 'mean', 'sum'],
        'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
    })
    return cluster_stats

def aggregate_traffic_per_application(df):
    """ Aggregate total traffic per application and get the top 10 most engaged users per app """
    app_traffic = df.groupby('Handset Type').agg({
        'HTTP DL (Bytes)': 'sum',  # Total download traffic per app
        'HTTP UL (Bytes)': 'sum'   # Total upload traffic per app
    }).reset_index()
    
    # Total traffic per application
    app_traffic['Total Traffic (Bytes)'] = app_traffic['HTTP DL (Bytes)'] + app_traffic['HTTP UL (Bytes)']
    
    return app_traffic.nlargest(10, 'Total Traffic (Bytes)')


def elbow_method(agg_data):
    """ Use the Elbow Method to determine the optimal number of clusters """
    distortions = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(agg_data[['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']])
        distortions.append(kmeans.inertia_)
    
    # Plot elbow graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, distortions, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scripts.user_engagement_analysis import perform_kmeans_clustering, compute_cluster_stats
from scripts.experience_analytics import preprocess_data, perform_kmeans_clustering as experience_clustering

def assign_scores(user_data):
    # Get user data for engagement and experience
    engagement_data = preprocess_data(user_data)
    experience_data = preprocess_data(user_data)

    # Engagement clustering
    engagement_cluster_centers, engagement_labels = perform_kmeans_clustering(engagement_data)
    less_engaged_cluster_center = engagement_cluster_centers[0]  # Assume cluster 0 is less engaged

    # Experience clustering
    experience_cluster_centers, experience_labels = experience_clustering(experience_data)
    worst_experience_cluster_center = experience_cluster_centers[0]  # Assume cluster 0 is worst experience

    # Calculate Euclidean distance (Engagement Score)
    user_engagement_scores = euclidean_distances(engagement_data, [less_engaged_cluster_center]).flatten()

    # Calculate Euclidean distance (Experience Score)
    user_experience_scores = euclidean_distances(experience_data, [worst_experience_cluster_center]).flatten()

    # Combine into a DataFrame for further steps
    user_scores = user_data.copy()
    user_scores['Engagement Score'] = user_engagement_scores
    user_scores['Experience Score'] = user_experience_scores

    return user_scores

def calculate_satisfaction_score(user_scores):
    # Satisfaction score is the average of both engagement and experience scores
    user_scores['Satisfaction Score'] = (user_scores['Engagement Score'] + user_scores['Experience Score']) / 2
    
    # Sort by Satisfaction Score and get top 10 customers
    top_10_customers = user_scores.sort_values(by='Satisfaction Score', ascending=True).head(10)
    
    return top_10_customers


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def build_regression_model(user_scores):
    # Use engagement and experience scores as features
    X = user_scores[['Engagement Score', 'Experience Score']]
    y = user_scores['Satisfaction Score']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and calculate performance metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    
    return model

from sklearn.cluster import KMeans

def cluster_scores(user_scores):
    kmeans = KMeans(n_clusters=2, random_state=42)
    user_scores['Cluster'] = kmeans.fit_predict(user_scores[['Engagement Score', 'Experience Score']])
    
    return user_scores

def aggregate_per_cluster(user_scores):
    cluster_agg = user_scores.groupby('Cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean'
    }).reset_index()

    return cluster_agg


import mlflow

def track_model(model):
    mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("MSE", mean_squared_error(y_test, y_pred))
        
        # Track artifacts, e.g., model or plots
        mlflow.sklearn.log_model(model, "model")

    print("Model tracked successfully.")

import numpy as n
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from user_engagement_analysis import perform_kmeans_clustering as engagement_clustering
from experience_analytics import perform_kmeans_clustering as experience_clustering

def assign_scores(df):
    # Perform k-means clustering for engagement
    engagement_clusters, engagement_model = engagement_clustering(df[['Engagement_Feature1', 'Engagement_Feature2']], k=3)
    experience_clusters, experience_model = experience_clustering(df[['Experience_Feature1', 'Experience_Feature2']], k=3)

    # Identify the least engaged and worst experience clusters
    least_engaged_cluster = engagement_clusters['cluster'].value_counts().idxmax()
    worst_experience_cluster = experience_clusters['cluster'].value_counts().idxmin()

    # Compute engagement and experience scores based on Euclidean distance
    df['engagement_score'] = euclidean_distances(df[['Engagement_Feature1', 'Engagement_Feature2']], 
                                                  engagement_model.cluster_centers_[least_engaged_cluster])
    df['experience_score'] = euclidean_distances(df[['Experience_Feature1', 'Experience_Feature2']], 
                                                  experience_model.cluster_centers_[worst_experience_cluster])
    
    return df

def compute_satisfaction_score(df):
    # Average engagement and experience scores to get satisfaction score
    df['satisfaction_score'] = (df['engagement_score'] + df['experience_score']) / 2
    return df

def top_satisfied_customers(df, top_n=10):
    # Sort customers by satisfaction score and get top 10
    top_customers = df.nlargest(top_n, 'satisfaction_score')
    return top_customers

def predict_satisfaction(df):
    X = df[['engagement_score', 'experience_score']]
    y = df['satisfaction_score']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    print(f'Mean Squared Error: {mse}')
    return model

def kmeans_on_scores(df, k=2):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k, random_state=42)
    df['score_cluster'] = model.fit_predict(df[['engagement_score', 'experience_score']])
    return df

def aggregate_scores_by_cluster(df):
    cluster_aggregates = df.groupby('score_cluster').agg({
        'satisfaction_score': 'mean',
        'experience_score': 'mean'
    })
    return cluster_aggregates

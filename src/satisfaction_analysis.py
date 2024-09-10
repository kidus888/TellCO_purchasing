import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class EngagementExperienceAnalytics:
    def __init__(self, user_data):
        self.user_data = user_data
    
    def calculate_engagement_experience_scores(self, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(self.user_data)
        
        cluster_centers = kmeans.cluster_centers_
        less_engaged_center = cluster_centers[0]
        worst_experience_center = cluster_centers[1]
        
        engagement_scores = euclidean_distances(self.user_data, [less_engaged_center]).flatten()
        experience_scores = euclidean_distances(self.user_data, [worst_experience_center]).flatten()
        
        self.user_data['engagement_score'] = engagement_scores
        self.user_data['experience_score'] = experience_scores
    
    def calculate_satisfaction(self):
        self.user_data['satisfaction_score'] = self.user_data[['engagement_score', 'experience_score']].mean(axis=1)
        return self.user_data.nlargest(10, 'satisfaction_score')
    
    def build_regression_model(self):
        features = self.user_data[['engagement_score', 'experience_score']]
        target = self.user_data['satisfaction_score']
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return model, mse
    
    def run_kmeans_on_scores(self, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        scores = self.user_data[['engagement_score', 'experience_score']]
        kmeans.fit(scores)
        self.user_data['cluster'] = kmeans.labels_

    def aggregate_scores_per_cluster(self):
        return self.user_data.groupby('cluster').agg({
            'engagement_score': 'mean',
            'experience_score': 'mean',
            'satisfaction_score': 'mean'
        }).reset_index()

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import time

def train_model(df):
    # Split data
    X = df[['engagement_score', 'experience_score']]
    y = df['satisfaction_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log model training session
    with mlflow.start_run():
        start_time = time.time()
        model = LinearRegression()
        
        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Log metrics
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        
        # Log model artifacts
        mlflow.sklearn.log_model(model, "model")
        
        # Track time taken
        end_time = time.time()
        mlflow.log_metric("training_duration", end_time - start_time)

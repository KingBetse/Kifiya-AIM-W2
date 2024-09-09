import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
# import mysql.connector
# import mlflow

# Task 4.1: Calculate Engagement and Experience Scores
def calculate_scores(df, engagement_clusters, experience_clusters):
    """
    Calculate engagement and experience scores based on Euclidean distances
    to the centroids of the respective clusters.
    """
    # Calculate the centroid for engagement and experience clusters
    engagement_centroid = engagement_clusters.mean().values
    experience_centroid = experience_clusters.mean().values

    # Calculate engagement scores using pairwise distances
    df['Engagement Score'] = pairwise_distances(df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']], 
                                                  [engagement_centroid]).flatten()
    
    # Calculate experience scores using pairwise distances
    df['Experience Score'] = pairwise_distances(df[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']], 
                                                  [experience_centroid]).flatten()
    
    return df

# Task 4.2: Calculate Satisfaction Score and Report Top 10 Customers
def calculate_satisfaction(df):
    """
    Calculate satisfaction scores and return the top 10 customers
    based on their satisfaction scores.
    """
    df['Satisfaction Score'] = (df['Engagement Score'] + df['Experience Score']) / 2
    top_customers = df.nlargest(10, 'Satisfaction Score')
    return top_customers[['User ID', 'Satisfaction Score']]

# Task 4.3: Build a Regression Model
def regression_model(df):
    """
    Build a linear regression model to predict satisfaction scores
    based on engagement and experience scores. Return the model and its MSE.
    """
    X = df[['Engagement Score', 'Experience Score']]
    y = df['Satisfaction Score']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)  # Train the model
    
    y_pred = model.predict(X_test)  # Make predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error
    
    return model, mse

# Task 4.4: Run K-Means on Engagement and Experience Scores
def kmeans_clustering(df):
    """
    Perform K-Means clustering on engagement and experience scores
    and assign the cluster labels to the DataFrame.
    """
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Engagement & Experience Cluster'] = kmeans.fit_predict(df[['Engagement Score', 'Experience Score']])
    return df

# Task 4.5: Aggregate Average Scores per Cluster
def aggregate_scores(df):
    """
    Aggregate average satisfaction and experience scores per cluster.
    """
    cluster_summary = df.groupby('Engagement & Experience Cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean'
    }).reset_index()
    return cluster_summary

# Task 4.6: Export to MySQL Database
def export_to_mysql(df):
    """
    Export engagement, experience, and satisfaction scores to a MySQL database.
    """
    connection = mysql.connector.connect(
        host='localhost', 
        user='your_username', 
        password='your_password', 
        database='your_database'
    )
    cursor = connection.cursor()

    # Create table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_scores (
            UserID INT,
            EngagementScore FLOAT,
            ExperienceScore FLOAT,
            SatisfactionScore FLOAT
        )
    ''')

    # Insert data into the database
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO user_scores (UserID, EngagementScore, ExperienceScore, SatisfactionScore) 
            VALUES (%s, %s, %s, %s)
        ''', (row['User ID'], row['Engagement Score'], row['Experience Score'], row['Satisfaction Score']))

    connection.commit()  # Commit the transaction
    cursor.close()  # Close the cursor
    connection.close()  # Close the connection

# Task 4.7: Model Deployment Tracking
def track_model_deployment(model, X_train, X_test, mse):
    """
    Track model parameters and metrics using MLflow.
    """
    mlflow.start_run()  # Start a new MLflow run
    mlflow.log_param("model_type", "Linear Regression")  # Log model type
    mlflow.log_param("train_size", len(X_train))  # Log training size
    mlflow.log_param("test_size", len(X_test))  # Log testing size
    mlflow.log_metric("mse", mse)  # Log mean squared error

    # Save the trained model
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()  # End the MLflow run
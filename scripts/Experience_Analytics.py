import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns



#  Data Aggregation
def aggregate_data(df):
    # Replace missing values with mean for TCP DL and UL retransmission volumes
    df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
    df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean(), inplace=True)
    
    # Replace missing values for average RTT columns with the mean
    df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
    df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean(), inplace=True)

    # Aggregate data per customer (assuming MSISDN/Number is the customer identifier)
    aggregated = df.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Handset Type': 'first'  # bythe first occurrence
    }).reset_index()

    return aggregated

# aggregated_data = aggregate_data(data)

# Compute Top, Bottom, and Most Frequent Values
def compute_values(df):
    # Get top, bottom, and most frequent values for TCP DL
    tcp_dl_top = df['TCP DL Retrans. Vol (Bytes)'].nlargest(10).reset_index(drop=True)
    tcp_dl_bottom = df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10).reset_index(drop=True)
    tcp_dl_most_frequent = df['TCP DL Retrans. Vol (Bytes)'].mode().reset_index(drop=True)

    # Ensure that tcp_dl_most_frequent has 10 entries
    if len(tcp_dl_most_frequent) < 10:
        tcp_dl_most_frequent = tcp_dl_most_frequent.tolist() + [tcp_dl_most_frequent[0]] * (10 - len(tcp_dl_most_frequent))

    # Get top, bottom, and most frequent values for TCP UL
    tcp_ul_top = df['TCP UL Retrans. Vol (Bytes)'].nlargest(10).reset_index(drop=True)
    tcp_ul_bottom = df['TCP UL Retrans. Vol (Bytes)'].nsmallest(10).reset_index(drop=True)
    tcp_ul_most_frequent = df['TCP UL Retrans. Vol (Bytes)'].mode().reset_index(drop=True)

    # Ensure that tcp_ul_most_frequent has 10 entries
    if len(tcp_ul_most_frequent) < 10:
        tcp_ul_most_frequent = tcp_ul_most_frequent.tolist() + [tcp_ul_most_frequent[0]] * (10 - len(tcp_ul_most_frequent))

    # Create a summary DataFrame for better readability
    tcp_dl_summary = pd.DataFrame({
        'Top 10': tcp_dl_top,
        'Bottom 10': tcp_dl_bottom,
        'Most Frequent': tcp_dl_most_frequent
    })

    tcp_ul_summary = pd.DataFrame({
        'Top 10': tcp_ul_top,
        'Bottom 10': tcp_ul_bottom,
        'Most Frequent': tcp_ul_most_frequent
    })

    # Print summaries for better visibility
    print("TCP DL Summary:")
    print(tcp_dl_summary)
    print("\nTCP UL Summary:")
    print(tcp_ul_summary)

    return {
        'TCP DL Summary': tcp_dl_summary,
        'TCP UL Summary': tcp_ul_summary
    }
     

# values = compute_values(aggregated_data)

# Distribution of Average TCP Retransmission per Handset Type
def distribution_per_handset(df, top_n=10):
    tcp_dl_distribution = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().reset_index()
    tcp_ul_distribution = df.groupby('Handset Type')['TCP UL Retrans. Vol (Bytes)'].mean().reset_index()

    # Limit to top N handset types based on average TCP DL
    tcp_dl_distribution = tcp_dl_distribution.nlargest(top_n, 'TCP DL Retrans. Vol (Bytes)')
    tcp_ul_distribution = tcp_ul_distribution.nlargest(top_n, 'TCP UL Retrans. Vol (Bytes)')

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=tcp_dl_distribution)
    plt.title('Average TCP DL Retransmission per Handset Type')
    plt.xticks(rotation=90)

    plt.subplot(1, 2, 2)
    sns.barplot(x='Handset Type', y='TCP UL Retrans. Vol (Bytes)', data=tcp_ul_distribution)
    plt.title('Average TCP UL Retransmission per Handset Type')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

# distribution_per_handset(aggregated_data)

# K-Means Clustering
def perform_clustering(df):
    # Convert relevant columns to numeric, coercing errors
    df['TCP DL Retrans. Vol (Bytes)'] = pd.to_numeric(df['TCP DL Retrans. Vol (Bytes)'], errors='coerce')
    df['TCP UL Retrans. Vol (Bytes)'] = pd.to_numeric(df['TCP UL Retrans. Vol (Bytes)'], errors='coerce')
    df['Avg RTT DL (ms)'] = pd.to_numeric(df['Avg RTT DL (ms)'], errors='coerce')
    df['Avg RTT UL (ms)'] = pd.to_numeric(df['Avg RTT UL (ms)'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Drop the existing Cluster column if it exists
    # if 'Cluster' in df.columns:
    #     df.drop(columns=['Cluster'], inplace=True)

    # Selecting features for clustering
    features = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']]

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)

    # Count instances in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_index()

    return df,cluster_counts


# Task 4.4: Dashboard Development
# Note: Implementation would typically involve a framework like Dash or Streamlit.
# Here, we will just outline the components required.

def create_dashboard():
    # KPIs to visualize
    # 1. Average TCP DL Retransmission
    # 2. Average TCP UL Retransmission
    # 3. Average RTT DL and UL
    
    # Dashboard Usability: Ensure all visualizations are clear and labeled.
    # Interactive Elements: Use dropdowns for filtering by Handset Type.
    # Visual Appeal: Maintain a clean layout with consistent color schemes.
    
    # Deployment: Use a framework like Dash or Streamlit to deploy the dashboard online.
    pass  # Replace this with actual dashboard code using a visualization library.

# Uncomment the line below to create the dashboard
# create_dashboard()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


from scripts import load_data
# Define the SQL query to select all data from the xdr_data table
query = "SELECT * FROM xdr_data"

# Load the data from PostgreSQL using the defined query
df = load_data.load_data_from_postgres(query)

# Check if the DataFrame was successfully loaded
if df is not None:
    print("Successfully loaded the data")  # Confirmation message for successful data loading
else:
    print("Failed to load data")  # Error message if loading data failed
# Set the title for the Streamlit app
st.title("Telecom Company Analysis and Strategic Recommendations")
st.subheader("Key Figures and Insights")
# Calculate top 10 handsets

top_handsets = df['Handset Type'].value_counts().head(10)

# Streamlit code to display the results
st.title("Telecom Company Analysis")
st.subheader("Top Handsets Used by Customers")

# Display the top handsets
st.write("Top 10 Handsets:")
st.bar_chart(top_handsets)


# Calculate top 3 handset manufacturers
top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)

# Streamlit code to display the results
st.title("Telecom Company Analysis")
st.subheader("Top Handset Manufacturers")

# Display the top manufacturers
st.write("Top 3 Handset Manufacturers:")
st.bar_chart(top_manufacturers)

# Calculate top 3 handset manufacturers
top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)

# Initialize a dictionary to store top handsets for each manufacturer
top_handsets_per_manufacturer = {}

# Iterate over the top manufacturers
for manufacturer in top_manufacturers.index:
    # Filter the DataFrame for the current manufacturer and count the occurrences of each handset type
    # Store the top 5 handsets in the dictionary
    top_handsets_per_manufacturer[manufacturer] = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)

# Streamlit code to display the results
st.title("Telecom Company Analysis")
st.subheader("Top 5 Handsets per Top 3 Manufacturers")

# Display the top handsets for each manufacturer
for manufacturer, handsets in top_handsets_per_manufacturer.items():
    st.write(f"### {manufacturer}:")
    st.write(handsets)




# Aggregate user behavior
user_behavior = df.groupby('IMSI').agg(
    number_of_xDR_sessions=('Bearer Id', 'count'),
    session_duration=('Dur. (ms)', 'sum'),
    total_DL=('Total DL (Bytes)', 'sum'),
    total_UL=('Total UL (Bytes)', 'sum'),
    social_media_DL=('Social Media DL (Bytes)', 'sum'),
    google_DL=('Google DL (Bytes)', 'sum'),
    email_DL=('Email DL (Bytes)', 'sum'),
    youtube_DL=('Youtube DL (Bytes)', 'sum'),
    netflix_DL=('Netflix DL (Bytes)', 'sum'),
    gaming_DL=('Gaming DL (Bytes)', 'sum'),
    other_DL=('Other DL (Bytes)', 'sum')
).reset_index()



# Calculate mean and median for important metrics
mean_metrics = user_behavior[['total_DL', 'total_UL', 'session_duration']].mean()
median_metrics = user_behavior[['total_DL', 'total_UL', 'session_duration']].median()

# Streamlit code to display the results
st.title("User Behavior Analysis")
st.subheader("Mean and Median Metrics")

# Display mean metrics
st.write("### Mean Metrics:")
st.write(mean_metrics)

# Display median metrics
st.write("### Median Metrics:")
st.write(median_metrics)


# Compute dispersion parameters (standard deviation)
std_dev = user_behavior[['total_DL', 'total_UL', 'session_duration']].std()

# Streamlit code to display the results
st.title("User Behavior Analysis")
st.subheader("Standard Deviation of Key Metrics")

# Display standard deviation metrics
st.write("### Standard Deviation:")
st.write(std_dev)

# Streamlit code to display the distribution plot
st.title("User Behavior Analysis")
st.subheader("Session Duration Distribution")

# Create a histogram with Seaborn
plt.figure(figsize=(12, 6))
sns.histplot(user_behavior['session_duration'], bins=30, kde=True)
plt.title('Session Duration Distribution')
plt.xlabel('Session Duration (ms)')
plt.ylabel('Frequency')

# Display the plot in Streamlit
st.pyplot(plt)


# Compute a correlation matrix
correlation_matrix = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                         'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].corr()

# Streamlit code to display the correlation matrix
st.title("User Behavior Analysis")
st.subheader("Correlation Matrix")

# Display the correlation matrix
st.write("### Correlation Matrix:")
st.write(correlation_matrix)

# Optional: Use a heatmap for better visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')

# Display the heatmap in Streamlit
st.pyplot(plt)

# Compute engagement metrics
engagement_metrics = df.groupby('MSISDN/Number').agg(
    session_frequency=('Bearer Id', 'count'),  # Count sessions per customer
    total_duration=('Dur. (ms)', 'sum'),       # Sum of session durations
    total_DL=('Total DL (Bytes)', 'sum'),      # Sum of total download traffic
    total_UL=('Total UL (Bytes)', 'sum')       # Sum of total upload traffic
).reset_index()

# Calculate total traffic after aggregation
engagement_metrics['total_traffic'] = engagement_metrics['total_DL'] + engagement_metrics['total_UL']

# Streamlit code to display the engagement metrics
st.title("Customer Engagement Metrics")
st.subheader("Aggregated Metrics by MSISDN/Number")

# Display the engagement metrics DataFrame
st.write(engagement_metrics)



# Compute engagement metrics
engagement_metrics = df.groupby('MSISDN/Number').agg(
    session_frequency=('Bearer Id', 'count'),  # Count sessions per customer
    total_duration=('Dur. (ms)', 'sum'),       # Sum of session durations
    total_DL=('Total DL (Bytes)', 'sum'),      # Sum of total download traffic
    total_UL=('Total UL (Bytes)', 'sum')       # Sum of total upload traffic
).reset_index()

# Calculate total traffic after aggregation
engagement_metrics['total_traffic'] = engagement_metrics['total_DL'] + engagement_metrics['total_UL']

# Get top customers for each metric
top_customers_frequency = engagement_metrics.nlargest(10, 'session_frequency')
top_customers_duration = engagement_metrics.nlargest(10, 'total_duration')
top_customers_traffic = engagement_metrics.nlargest(10, 'total_traffic')

# Streamlit code to display the results
st.title("Top Customers Analysis")
st.subheader("Top 10 Customers by Session Frequency")
st.write(top_customers_frequency)

st.subheader("Top 10 Customers by Total Duration")
st.write(top_customers_duration)

st.subheader("Top 10 Customers by Total Traffic")
st.write(top_customers_traffic)


from sklearn.preprocessing import StandardScaler

# Normalize the engagement metrics to prepare for clustering
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(engagement_metrics[['session_frequency', 'total_duration', 'total_traffic']])
normalized_metrics


# Perform K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
engagement_metrics['cluster'] = kmeans.fit_predict(normalized_metrics)

# Streamlit code to display the results
st.title("Customer Engagement Metrics with Clustering")
st.subheader("Engagement Metrics with Cluster Labels")

# Display the engagement metrics with cluster information
st.write(engagement_metrics)

# Calculate min, max, average, and total non-normalized metrics for each cluster
cluster_summary = engagement_metrics.groupby('cluster').agg(
    min_frequency=('session_frequency', 'min'),
    max_frequency=('session_frequency', 'max'),
    avg_duration=('total_duration', 'mean'),
    total_traffic=('total_traffic', 'sum')
).reset_index()

# Streamlit code to display the cluster summary
st.title("Cluster Summary Statistics")
st.subheader("Summary of Engagement Metrics by Cluster")

# Display the cluster summary
st.write(cluster_summary)


# Create a list to hold the aggregated traffic data
app_traffic_list = []

# Define the application categories and their corresponding download and upload columns
application_categories = {
    'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
    'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
    'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
    'Youtube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
    'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
    'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
    'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
}

# Aggregate traffic for each application category
for app_name, traffic_columns in application_categories.items():
    total_DL = df[traffic_columns[0]].sum()  # Total download for the application
    total_UL = df[traffic_columns[1]].sum()  # Total upload for the application
    
    # Append the results as a dictionary to the list
    app_traffic_list.append({
        'application': app_name,
        'total_DL': total_DL,
        'total_UL': total_UL,
        'total_traffic': total_DL + total_UL
    })

# Create a DataFrame from the list of dictionaries
app_traffic = pd.DataFrame(app_traffic_list)

# Sort the DataFrame in ascending order based on total traffic
app_traffic_sorted = app_traffic.sort_values(by='total_traffic', ascending=False)

# Streamlit code to display the aggregated traffic data
st.title("Application Traffic Analysis")
st.subheader("Aggregated Traffic Data per Application")

# Display the sorted aggregated traffic data
st.write(app_traffic_sorted)
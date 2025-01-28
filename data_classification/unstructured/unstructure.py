import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def process_data(df, num_clusters):
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Date'] = df['Date'].apply(lambda x: x.toordinal())

    # Apply KMeans clustering
    X = df[['Amount', 'Date']]  # Features (Amount and Date)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Cluster labels
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = ['Low', 'Medium', 'High'][:num_clusters] 
    cluster_centers_sorted = np.argsort(cluster_centers[:, 0])  
    df['Cluster_Label'] = df['Cluster'].map(lambda x: cluster_labels[cluster_centers_sorted.tolist().index(x)])

    def categorize_transaction(row):
        group_df = df[(df['Description'] == row['Description']) & (df['Amount'] == row['Amount'])]
        group_df = group_df.sort_values(by='Date')
        if not pd.api.types.is_datetime64_any_dtype(group_df['Date']):
            group_df['Date'] = pd.to_datetime(group_df['Date'], errors='coerce')
        intervals = group_df['Date'].diff().dt.days.dropna()
        if intervals.empty:
            return 'New Transaction'
        elif intervals.mean() <= 30:
            return 'Regular'
        else:
            return 'Irregular'
    df['Transaction_Category'] = df.apply(categorize_transaction, axis=1)
    return df

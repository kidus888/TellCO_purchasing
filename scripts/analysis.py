import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Handset and Manufacturer Analysis
def handset_and_manufacturer_analysis(df):
    """
    Perform analysis to find the top handsets and manufacturers.
    """
    top_handsets = df['Handset Type'].value_counts().head(10)
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    top_5_per_manufacturer = df.groupby('Handset Manufacturer')['Handset Type'].value_counts().groupby(level=0).head(5)
    return top_handsets, top_manufacturers, top_5_per_manufacturer

# User Behavior Analysis
def user_behavior_analysis(df):
    """
    Aggregate user behavior data including xDR sessions, duration, and total data volume.
    """
    user_behavior = df.groupby('IMSI').agg({
        'Bearer Id': 'count',
        'Dur. (ms)': 'sum',
        'HTTP DL (Bytes)': 'sum',
        'HTTP UL (Bytes)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    })
    user_behavior['Total Data Volume'] = user_behavior['Total DL (Bytes)'] + user_behavior['Total UL (Bytes)']
    return user_behavior

# Handle missing values
def handle_missing_values(df):
     # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns

    # Fill missing values in numeric columns with the mean
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Fill missing values in non-numeric columns with the mode (most frequent value)
    for col in non_numeric_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Describe dataset
def describe_dataset(df):
    return df.describe()

# Decile Segmentation
def decile_segmentation(df):
    df['Duration Decile'] = pd.qcut(df['Dur. (ms)'], 10, labels=False)
    decile_data = df.groupby('Duration Decile').agg({
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    })
    decile_data['Total Data'] = decile_data['Total DL (Bytes)'] + decile_data['Total UL (Bytes)']
    return decile_data

# Bivariate Analysis
def bivariate_analysis(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Social Media DL (Bytes)', y='Total DL (Bytes)', data=df)
    plt.title('Social Media Data vs Total Download Data')
    plt.show()

# Correlation Analysis
def correlation_analysis(df):
    correlation_matrix = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                             'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].corr()
    return correlation_matrix

# Dimensionality Reduction (PCA)
def perform_pca(df):
    pca = PCA(n_components=2)
    pca_data = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].dropna()
    pca_result = pca.fit_transform(pca_data)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

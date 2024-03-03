'''
K-Means Clustering Algorithm

Overview
The K-Means algorithm is an unsupervised learning technique used for clustering similar data points into groups or clusters. It aims to partition the data into K clusters, where each data point belongs to the cluster with the nearest mean.

Algorithm Steps:
1. **Initialization**: Randomly initialize K cluster centroids.
2. **Assigning Points to Clusters**: Assign each data point to the nearest cluster centroid based on Euclidean distance.
3. **Updating Cluster Centroids**: Recalculate the centroid of each cluster by taking the mean of all data points assigned to that cluster.
4. **Repeating Steps 2 and 3**: Iterate until convergence or until a maximum number of iterations is reached.

Implementation ->
'''

import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/prasad-chavan1/MachineLearning/master/DataPreprocessing/Data.csv')

# Preprocess the data
# Assuming the missing values are handled and 'Country' column is one-hot encoded
X = data[['Age', 'Salary']].values

# Implement K-Means clustering
def k_means(X, K, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Number of clusters
K = 3

# Apply K-Means clustering
labels, centroids = k_means(X, K)

# Visualize Results (optional)
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('K-Means Clustering')
plt.show()

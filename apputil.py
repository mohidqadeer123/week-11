import seaborn as sns
from time import time
import numpy as np
from sklearn.cluster import KMeans

# Load diamonds dataset and define globally
df_diamonds = sns.load_dataset('diamonds')
diamonds_num = df_diamonds.select_dtypes(include = [np.number])


# Exercise 1
def kmeans(X, k)

    '''executes k-means clustering on numerical numpy array X'''

    model = kMeans(n_clusters = k)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels

    return centroids, labels

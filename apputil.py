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

# Exercise 2

def kmeans_diamonds(n, k)

    '''Runs k-means clustering on first n numerical rows of diamonds dataset'''

    X = diamonds_num.iloc[:n].to_numpy()
    return kmeans(X, k)


# Exercise 3
def kmeans_timer(n, k, n_iter = 5)

    '''Runs kmeans_diamonds(n, k) n_iter times and return average runtime in seconds.'''

    times = []
    for _ in range (n_iter):
        start_time = time()
        kmeans_diamonds(n, k)
        end_time = time()
        times.append(end_time - start_time)

    average_time = sum(times) / n_iter
    return average_time

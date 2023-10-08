#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List

from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

sns.set()
x, y_true = make_blobs(n_samples= 500,
                       centers = 4, cluster_std=0.80, random_state=1)
kmeans : KMeans = KMeans(n_clusters=4, random_state=0)
labels : List[int] = kmeans.fit(x).predict(x)

def plot_kmeans(kmeans : KMeans, x, n_cluster : int = 4,
                rseed : int = 0 , ax = None):
    labels = kmeans.fit_predict(x)
    plt.figure(figsize=(15,10))
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(x[:, 0], x[:, 1], c=labels,
               s=40, cmap='viridis', zorder=2)
    
    ## Plot the representations of the KMeans model
    centers = kmeans.cluster_centers_
    radiuses = [cdist(x[labels == i], [center], 'euclidean').max() for i,center in enumerate(centers)]
    for c, r in zip(centers, radiuses):
        ax.add_patch(plt.Circle(c, r, fc="#000000",
                                lw=3, alpha=0.5, zorder=1))
    
plot_kmeans(kmeans,x, )

#%%

#%%
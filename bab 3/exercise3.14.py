# Exercise 3.14 K-means Clustering (2 Tambahan Titik per Kelompok)
from sklearn.cluster import KMeans
import numpy as np

# Data X dengan penambahan 2 titik di Kelompok 1 dan 2 titik di Kelompok 2
X = np.array([
    # Kelompok 1 (Nilai awal sekitar x = 1)
    [1, 2, 3],
    [1, 4, 2],
    [1, 0, 3],
    [2, 1, 3],  # Titik tambahan 1
    [1, 3, 1],  # Titik tambahan 2
    # Kelompok 2 (Nilai awal sekitar x = 10)
    [10, 2, 4],
    [9, 4, 3],
    [11, 0, 2],
    [10, 1, 3],  # Titik tambahan 3
    [12, 2, 4],  # Titik tambahan 4
])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print("Labels:", kmeans.labels_)
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Prediction [12, 3, 1]:", kmeans.predict([[12, 3, 1]]))
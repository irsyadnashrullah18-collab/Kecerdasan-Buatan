# Exercise 3.15 K-means Clustering (3 Kelompok & 1 Label per Kelompok)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# 1. Data X dengan 3 kelompok titik data
X = np.array([
    # Kelompok 1 (sekitar x = 1)
    [1, 2, 3],
    [1, 4, 2],
    [1, 0, 3],
    # Kelompok 2 (sekitar x = 10)
    [10, 2, 4],
    [9, 4, 3],
    [11, 0, 2],
    # Kelompok 3 (Kelompok Tambahan: sekitar x = 20)
    [20, 2, 5],
    [19, 4, 4],
    [21, 1, 3],
])

# 2. Inisialisasi K-means dengan n_clusters = 3
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print("Cluster Labels:", kmeans.labels_)
print("\nCluster Centers:\n", kmeans.cluster_centers_)
print("\nPrediction [20, 3, 4]:", kmeans.predict([[20, 3, 4]]))

# 3. Plot Visualisasi Data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=100)

# Memberikan label teks HANYA pada 1 titik di tiap kelompok (Indeks 0, 3, dan 6)
labeled_indices = [0, 3, 6]
for i, idx in enumerate(labeled_indices):
    plt.annotate(
        f'Group {i+1} Sample',
        (X[idx, 0], X[idx, 1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontweight='bold',
    )

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering - 3 Groups (1 Point Labeled per Group)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
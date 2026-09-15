# Exercise 3.13 K-means Clustering using make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate sampel data buatan (300 sampel, 3 pusat klaster, 3 fitur)
X, y_true = make_blobs(
    n_samples=300, centers=3, n_features=3, cluster_std=0.60, random_state=0
)

# 2. Inisialisasi dan latih model K-means dengan 3 klaster
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# 3. Tampilkan label klaster tiap data dan koordinat pusat klaster (cluster centers)
print("Cluster Labels:\n", kmeans.labels_)
print("\nCluster Centers:\n", kmeans.cluster_centers_)

# 4. Prediksi kelompok klaster untuk data baru [0, 0, 0]
pred = kmeans.predict([[0, 0, 0]])
print("\nPrediction for [0, 0, 0]:\n", pred)
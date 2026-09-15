# Exercise 3.7 PCA on Breast Cancer Data
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer

# 1. Load Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. Plot Original Data (2 fitur pertama: mean radius & mean texture)
f = plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.xlabel("mean radius")
plt.ylabel("mean texture")
plt.title("Original Data (Breast Cancer)")
plt.colorbar(label="Target (0: Malignant, 1: Benign)")
plt.show()

# 3. Perform PCA (reduksi dari 30 fitur menjadi 3 komponen)
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

# 4. Plot PCA data (PCA1 vs PCA2)
g = plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y, cmap="viridis")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Data (Breast Cancer)")
plt.colorbar(label="Target (0: Malignant, 1: Benign)")
plt.show()
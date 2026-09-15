# Exercise 3.20 LazyClassifier dengan Wine Dataset
import lazypredict
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_wine  # Mengganti dataset menjadi load_wine
from sklearn.model_selection import train_test_split

# 1. Load dataset Wine
X, y = load_wine(return_X_y=True)

# 2. Bagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# 3. Inisialisasi dan jalankan LazyClassifier
clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

# 4. Visualisasi Akurasi Model
plt.figure(figsize=(10, 5))
plt.plot(models.index, models['Accuracy'])
plt.xticks(rotation=90)
plt.title('LazyClassifier Accuracy - Wine Dataset')
plt.tight_layout()
plt.show()
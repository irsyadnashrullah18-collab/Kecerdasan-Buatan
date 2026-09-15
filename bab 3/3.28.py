import lazypredict
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load dataset Iris
X, y = load_iris(return_X_y=True)

# 2. Bagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# 3. Inisialisasi dan jalankan LazyClassifier
clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

# 4. Visualisasi Akurasi Model (Part 3)
plt.figure(figsize=(10, 5))
plt.plot(models.index, models['Accuracy'])
plt.xticks(rotation=90)  # Memutar nama model agar mudah dibaca
plt.tight_layout()
plt.show()  # Wajib ditambahkan di Python script biasa
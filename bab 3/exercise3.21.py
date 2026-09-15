# Exercise 3.21 LazyRegressor dengan Diabetes Dataset
import lazypredict
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes  # Dataset regresi baru
from sklearn.model_selection import train_test_split

# 1. Load dataset Diabetes
X, y = load_diabetes(return_X_y=True)

# 2. Bagi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Inisialisasi dan jalankan LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

# 4. Visualisasi Performa R-Squared Model Regresi
plt.figure(figsize=(10, 5))
plt.plot(models.index, models['R-Squared'])
plt.xticks(rotation=90)
plt.title('LazyRegressor R-Squared - Diabetes Dataset')
plt.tight_layout()
plt.show()
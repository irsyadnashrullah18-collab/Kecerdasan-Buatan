# Exercise 3.9 Random Forest Classification on Diabetes Dataset
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load dataset Diabetes
X, y = load_diabetes(return_X_y=True)

# 2. Konversi nilai target kontinu menjadi kelas biner (0 dan 1)
# Karena dataset diabetes secara bawaan adalah regresi, target dikelompokkan (di atas/bawah rata-rata)
y = (y > y.mean()).astype(int)

# 3. Bagi data (50% train, 50% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

# 4. Inisialisasi dan latih model Random Forest Classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# 5. Prediksi dan tampilkan hasil
y_pred = clf.predict(X_test)
print(
    "Total points: %d Correctly labeled points : %d"
    % (y_test.shape[0], (y_test == y_pred).sum())
)
# Exercise 3.4 Python SVM Breast Cancer Classifications with Histograms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()

# --- MODIFIKASI EXERCISE 3.4: Plot Histogram 4 Fitur ---
# Buat DataFrame untuk mempermudah pemfilteran kolom berdasarkan nama fitur
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Memilih 4 fitur: radius, texture, size (diwakili oleh 'mean area'), dan smoothness
selected_features = [
    'mean radius',
    'mean texture',
    'mean area',
    'mean smoothness',
]

# Plot histogram untuk ke-4 fitur tersebut
df[selected_features].hist(figsize=(10, 8), bins=15)
plt.tight_layout()
plt.show()
# -----------------------------------------------------

# Kode SVM dari Example 3.6
X = cancer.data  # All of the features
y = cancer.target  # All of the labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)

clf = SVC()
clf.fit(X_train, y_train)

# Prediction
y_predict = clf.predict(X_test)

# Print Confusion Matrix and Classification Report
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(
    cm,
    index=['is_cancer', 'is_healthy'],
    columns=['predicted_cancer', 'predicted_healthy'],
)

print(confusion)
print(classification_report(y_test, y_predict))
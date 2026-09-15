from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score

# 1. Load dataset Iris
iris = datasets.load_iris()

# 2. Mengambil fitur ke-3 dan ke-4 (Petal Length & Petal Width)
# Dalam indeks Python: 2 adalah fitur ke-3, 3 adalah fitur ke-4
X = iris.data[:, 2:4]
y = iris.target

# 3. Inisialisasi dan latih model SVM
clf = svm.SVC(kernel="linear")
clf.fit(X, y)

# 4. Prediksi dan hitung akurasi
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Akurasi (Petal Length & Width): {acc * 100:.2f}%")
# Exercise 3.1 Python SVM Classifications (6 Samples)
from sklearn import svm

# 6 Sampel Data X [Height(cm), Weight(kg), Shoesize(UK)]
X = [
    [170, 70, 10],
    [180, 80, 12],
    [175, 75, 11],  # Sampel 3 (Male)
    [170, 65, 8],
    [160, 55, 7],
    [155, 50, 6],   # Sampel 6 (Female)
]

# 6 Label y (0: Male, 1: Female)
y = [0, 0, 0, 1, 1, 1]

# Inisialisasi dan pelatihan model SVM
clf = svm.SVC()
clf.fit(X, y)

# Prediksi data baru
p = clf.predict([[160, 60, 7]])
print(p)
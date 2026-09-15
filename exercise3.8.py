# Exercise 3.8 Decision Tree Classification on Wine Dataset
from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load dataset Wine
wine = load_wine()
X = wine.data
y = wine.target

# 2. Bagi data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Inisialisasi dan pelatihan model Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 4. Evaluasi performa model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 5. Contoh prediksi sampel data baru
p = clf.predict([X_test[0]])
print("Hasil Prediksi Sampel:", wine.target_names[p[0]])
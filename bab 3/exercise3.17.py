# Exercise 3.17 Ensemble Learning with KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Inisialisasi masing-masing classifier
clf1 = LogisticRegression(max_iter=1000, random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = SVC(probability=True, random_state=1)
clf5 = KNeighborsClassifier(n_neighbors=5)  # Penambahan KNN

# 3. Membuat Ensemble Classifier (VotingClassifier) dengan menambahkan KNN
eclf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('gnb', clf3),
        ('svc', clf4),
        ('knn', clf5),  # Menambahkan KNN ke dalam voting ensemble
    ],
    voting='hard',
)

# 4. Evaluasi performa masing-masing model dan Ensemble menggunakan Cross-Validation
classifiers = [clf1, clf2, clf3, clf4, clf5, eclf]
labels = [
    'Logistic Regression',
    'Random Forest',
    'Naive Bayes',
    'SVM',
    'KNN',
    'Ensemble',
]

for clf, label in zip(classifiers, labels):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
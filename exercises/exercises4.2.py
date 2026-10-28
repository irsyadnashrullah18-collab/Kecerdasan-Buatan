# Exercise 4.2: Ubah Example 4.2 supaya memakai tiga input (bukan dua)

from sklearn.neural_network import MLPClassifier
from itertools import product

X = [list(map(float, bits)) for bits in product([0, 1], repeat=3)]   # 8 kombinasi
y = [int(any(bits)) for bits in product([0, 1], repeat=3)]           # OR tiga input
print("X =", X)
print("y =", y)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print(clf.predict([[2., 2., 2.], [-1., -2., -3.], [0., 0., 0.], [0., 0., 1.]]))
print([coef.shape for coef in clf.coefs_])   

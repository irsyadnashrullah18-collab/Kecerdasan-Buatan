# Example 4.3 Multiple Layer Perceptron Classification (Keras)
# (di buku tertulis "Example 4.4" di komentar, itu salah ketik)
# Library: tensorflow, scikit-learn
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=7)

features = X_train.shape[1]

# MLP 3-layer Model ==============================================
model = Sequential()
model.add(Input(shape=(features,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
print(model.get_config())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

model.fit(X_train, y_train, batch_size=50, validation_split=0.2,
          epochs=100, verbose=1)

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
# Catatan: metrik yang dipakai di buku adalah 'mse', jadi results[1] sebenarnya MSE,
# bukan accuracy. Accuracy sebenarnya dihitung di bawah (output dibulatkan 0/1).
print('mse: ', results[1])
y_pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
print('accuracy: ', (y_pred == y_test).mean())

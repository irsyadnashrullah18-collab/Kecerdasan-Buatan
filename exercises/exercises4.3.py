# Exercise 4.3: Ubah Example 4.3 -> dense layer 5 neuron (bukan 10), lalu

import time
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    whole_data.data, whole_data.target, test_size=0.3, random_state=7)


def run(n_neurons):
    keras.utils.set_random_seed(42)      
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    start = time.time()
    model.fit(X_train, y_train, batch_size=50, validation_split=0.2,
              epochs=100, verbose=0)
    elapsed = time.time() - start

    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
    acc = (y_pred == y_test).mean()
    return model.count_params(), elapsed, mse, acc


print(f"{'neuron':>7} {'params':>8} {'waktu(s)':>9} {'MSE':>8} {'accuracy':>9}")
for n in (10, 5):
    params, t, mse, acc = run(n)
    print(f"{n:>7} {params:>8} {t:>9.1f} {mse:>8.4f} {acc:>9.3f}")

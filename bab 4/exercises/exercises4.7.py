# Exercise 4.7: Pada Example 4.9, ubah jumlah filter 32 -> 64, lalu ubah ukuran

import time
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical

EPOCHS = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def run(filters, kernel):
    keras.utils.set_random_seed(1)
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters, kernel, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=200, verbose=0)
    elapsed = time.time() - start
    scores = model.evaluate(X_test, y_test, verbose=0)
    return model.count_params(), elapsed, 100 - scores[1] * 100


configs = [(32, (5, 5)),   # asli (Example 4.9)
           (64, (5, 5)),   # filter 32 -> 64
           (32, (3, 3))]   # kernel (5,5) -> (3,3)

print(f"{'filters':>8} {'kernel':>8} {'params':>10} {'waktu(s)':>9} {'error(%)':>9}")
for f, k in configs:
    params, t, err = run(f, k)
    print(f"{f:>8} {str(k):>8} {params:>10,} {t:>9.1f} {err:>9.2f}")

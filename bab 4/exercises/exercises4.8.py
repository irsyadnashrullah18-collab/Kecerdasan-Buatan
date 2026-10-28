# Exercise 4.8: Pada Example 4.9, tambahkan Conv2D, MaxPooling, dan Dropout lagi

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


def build(extra_block):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    if extra_block:                                   # blok tambahan
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


print(f"{'model':>22} {'params':>10} {'waktu(s)':>9} {'error(%)':>9}")
for name, extra in [("asli (Example 4.9)", False), ("+ Conv/Pool/Dropout", True)]:
    keras.utils.set_random_seed(1)
    model = build(extra)
    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=200, verbose=0)
    elapsed = time.time() - start
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name:>22} {model.count_params():>10,} {elapsed:>9.1f} {100 - scores[1] * 100:>9.2f}")

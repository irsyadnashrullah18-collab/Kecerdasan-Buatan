# Exercise 4.4: Ubah jumlah filter di LeNet (Example 4.4) dan lihat pengaruhnya

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Input


def build_lenet(f1, f2):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(filters=f1, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=f2, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model


print(f"{'filter1':>8} {'filter2':>8} {'total params':>14}")
for f1, f2 in [(6, 16), (12, 32), (3, 8), (32, 64)]:   
    print(f"{f1:>8} {f2:>8} {build_lenet(f1, f2).count_params():>14,}")
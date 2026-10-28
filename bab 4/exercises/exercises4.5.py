# Exercise 4.5: Ubah jumlah unit dense layer di AlexNet (Example 4.5) dan lihat

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input


def build_alexnet(d1, d2, d3):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(96, (11, 11), activation='relu', strides=(4, 4), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(256, (11, 11), activation='relu', strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(d1, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(d2, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(d3, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(17, activation='softmax'))
    return model


print(f"{'dense1':>7} {'dense2':>7} {'dense3':>7} {'total params':>14}")
for d in [(4096, 4096, 1000),   
          (2048, 2048, 500),
          (1024, 1024, 256),
          (8192, 8192, 2000)]:
    print(f"{d[0]:>7} {d[1]:>7} {d[2]:>7} {build_alexnet(*d).count_params():>14,}")


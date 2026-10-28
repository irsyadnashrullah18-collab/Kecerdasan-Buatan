# Exercise 4.6: Tambahkan Conv2D, MaxPooling, dan Dropout lagi sebelum Flatten

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

# Model asli (Example 4.7) ----------------------------------------
orig = Sequential()
orig.add(Input(shape=(28, 28, 1)))
orig.add(Conv2D(32, (5, 5), activation='relu'))
orig.add(MaxPooling2D())
orig.add(Dropout(0.2))
orig.add(Flatten())
orig.add(Dense(128, activation='relu'))
orig.add(Dense(2, activation='softmax'))

# Model dengan blok Conv2D + MaxPooling + Dropout tambahan ----------
new = Sequential()
new.add(Input(shape=(28, 28, 1)))
new.add(Conv2D(32, (5, 5), activation='relu'))
new.add(MaxPooling2D())
new.add(Dropout(0.2))
new.add(Conv2D(32, (5, 5), activation='relu'))      # <-- tambahan
new.add(MaxPooling2D())                             # <-- tambahan
new.add(Dropout(0.2))                               # <-- tambahan
new.add(Flatten())
new.add(Dense(128, activation='relu'))
new.add(Dense(2, activation='softmax'))

new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
new.summary()

print("Parameter model asli :", f"{orig.count_params():,}")
print("Parameter model baru :", f"{new.count_params():,}")


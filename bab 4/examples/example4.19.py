# Example 4.19 SimpleRNN2.py
# Library: tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Activation, Input
from tensorflow.keras import optimizers

model = Sequential()
model.add(Input(shape=(49, 1)))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(46))
model.add(Activation('softmax'))

adam = optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy'])
print(model.summary())


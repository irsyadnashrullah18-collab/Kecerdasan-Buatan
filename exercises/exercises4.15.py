# Exercise 4.15: Pada Example 4.17 (Part 4), pilih layer konvolusi kedua dengan

from matplotlib import pyplot
from tensorflow.keras.applications.vgg19 import VGG19

model = VGG19()

n = 2    # block1_conv2  (layer konvolusi kedua)
print("Layer yang dipilih:", model.layers[n].name)
filters, biases = model.layers[n].get_weights()
s = filters.shape                      # (3, 3, 64, 64) = (tinggi, lebar, channel_masuk, jumlah_filter)
print("Input channels: ", s[2])
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters = 4          # tampilkan 4 filter pertama
n_channels = 3         # dan 3 channel masuk pertama dari 64 channel
ix = 1
pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(n_channels):
        ax = pyplot.subplot(n_filters, n_channels, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
pyplot.show()


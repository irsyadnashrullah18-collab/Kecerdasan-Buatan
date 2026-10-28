# Exercise 4.16: Tampilkan SEMUA filter dengan n_filters, ix = s[3], 1 (64 filter di

from matplotlib import pyplot
from tensorflow.keras.applications.vgg19 import VGG19

model = VGG19()

n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape                      # (3, 3, 3, 64)
print("Color channels: ", s[2])
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters = s[3]            # semua filter (64)
n_channels = s[2]           # 3
filters_per_row = 4
rows = (n_filters + filters_per_row - 1) // filters_per_row
cols = filters_per_row * n_channels

pyplot.figure(figsize=(12, 16))
ix = 1
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(n_channels):
        ax = pyplot.subplot(rows, cols, i * n_channels + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
pyplot.tight_layout()
pyplot.savefig('vgg19_layer1_all_filters.png', dpi=120)
pyplot.show()


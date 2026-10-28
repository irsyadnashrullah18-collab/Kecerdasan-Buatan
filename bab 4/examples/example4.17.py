# Example 4.17 CNN Visualize Filters (Part 1 - 6)
# Modified from:
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/


# %% Part 1: load VGG19 dan tampilkan summary
from tensorflow.keras.applications.vgg19 import VGG19
model = VGG19()          # unduh bobot ImageNet ~548 MB saat pertama kali
model.summary()

# %% Part 2: nomor dan nama setiap layer
n = 0
for layer in model.layers:
    print(n, layer.name)
    n += 1

# %% Part 3: bentuk filter dan bias tiap layer konvolusi
n = 0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)
    n += 1

# %% Part 4: tampilkan filter layer konvolusi pertama
from matplotlib import pyplot
# retrieve weights from the first hidden layer
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape                      # (tinggi, lebar, channel_masuk, jumlah_filter)
print("Color channels: ", s[2])
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
# n_filters, ix = s[3], 1
n_filters, ix = 4, 1
n_channels = min(s[2], 3)

pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately

    #  kernel 3x3 untuk channel ke-j)
    for j in range(n_channels):
        ax = pyplot.subplot(n_filters, n_channels, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
# show the figure
pyplot.show()

# %% Part 5: fungsi untuk menggambar feature maps
import matplotlib.pyplot as plt


def plot_feature_maps(feature_maps, max_maps=64):
    # plot feature maps (maksimal max_maps supaya window tidak terlalu besar)
    col = 8
    total = min(feature_maps.shape[3], max_maps)
    row = int(total / col)
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn off axis
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()


# %% Part 6: feature maps dari layer n untuk satu gambar
import os
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims

# load the model
model = VGG19()
# Select the hidden layer to visualize
n = 1
# redefine model to output right after the hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
model.summary()

# load the image with the required shape
img_path = 'Elephant.jpg'
if not os.path.exists(img_path):
    raise SystemExit("File 'Elephant.jpg' tidak ditemukan di folder kerja saat ini. "
                     "Taruh gambarnya di folder ini (atau ganti img_path).")
img = load_img(img_path, target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
print("Feature maps: ", feature_maps.shape)
# plot all the feature maps
plot_feature_maps(feature_maps)

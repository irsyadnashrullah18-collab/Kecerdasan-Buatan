# Exercise 4.17: Pada Example 4.17 (Part 6), pilih layer konvolusi kedua, ketiga,

import os
import matplotlib.pyplot as plt
from numpy import expand_dims
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model

LAYERS_TO_VIEW = [2, 4, 5]     # konvolusi ke-2, ke-3, ke-4
MAX_MAPS = 64                  # tampilkan maksimal 64 feature map per layer


def plot_feature_maps(feature_maps, title, max_maps=MAX_MAPS):
    col = 8
    total = min(feature_maps.shape[3], max_maps)
    row = int(total / col)
    plt.figure(figsize=(16, 16))
    plt.suptitle(title)
    ix = 1
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()


img_path = 'Elephant.jpg'
if not os.path.exists(img_path):
    raise SystemExit("File 'Elephant.jpg' tidak ditemukan di folder kerja saat ini. "
                     "Taruh gambarnya di folder ini (atau ganti img_path).")

base_model = VGG19()

img = load_img(img_path, target_size=(224, 224))
img = expand_dims(img_to_array(img), axis=0)
img = preprocess_input(img)

for n in LAYERS_TO_VIEW:
    layer = base_model.layers[n]
    model = Model(inputs=base_model.inputs, outputs=layer.output)
    feature_maps = model.predict(img, verbose=0)
    print(f"n = {n}: {layer.name}, feature maps: {feature_maps.shape}")
    plot_feature_maps(feature_maps, f"n = {n}  ({layer.name})  {feature_maps.shape[1:]}")


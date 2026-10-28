# Exercise 4.9: Ubah Example 4.10 supaya memuat VGG19 (bukan VGG16)

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

tf.keras.backend.clear_session()

# Memuat model VGG19 dengan bobot ImageNet
model = VGG19(weights='imagenet')

# Menampilkan arsitektur model VGG19
model.summary()
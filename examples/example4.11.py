# Example 4.11 Klasifikasi gambar dengan VGG16
# Library: tensorflow, numpy, pillow
# Butuh file gambar 'Elephant.jpg' di folder yang sama dengan script ini
# (atau ganti img_path ke gambar apa saja).
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
if not os.path.exists(img_path):
    raise SystemExit(f"File gambar tidak ditemukan: {img_path}\n"
                     "Taruh gambar bernama Elephant.jpg di folder ini.")

model = VGG16(weights='imagenet')

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# prediction
predictions = model.predict(x)
results = decode_predictions(predictions)

print(results)

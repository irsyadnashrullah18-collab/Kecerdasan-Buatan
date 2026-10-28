# Exercise 4.10: Klasifikasi gambar dengan VGG19 dan bandingkan dengan VGG16

import os
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
if not os.path.exists(img_path):
    raise SystemExit(f"File gambar tidak ditemukan: {img_path}\n"
                     "Taruh gambar bernama Elephant.jpg di folder ini.")

img = image.load_img(img_path, target_size=(224, 224))
x = np.expand_dims(image.img_to_array(img), axis=0)

for name, module, ctor in [("VGG16", vgg16, vgg16.VGG16),
                           ("VGG19", vgg19, vgg19.VGG19)]:
    model = ctor(weights='imagenet')
    processed = module.preprocess_input(x.copy())

    model.predict(processed, verbose=0)                 # pemanasan (tidak dihitung)
    start = time.time()
    predictions = model.predict(processed, verbose=0)
    elapsed = time.time() - start

    print(f"\n=== {name} === params: {model.count_params():,}  waktu prediksi: {elapsed:.3f} s")
    for _, label, prob in decode_predictions(predictions, top=5)[0]:
        print(f"  {label:<28} {prob:.4f}")

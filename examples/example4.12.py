# Example 4.12 Klasifikasi gambar dengan beberapa model (VGG16/ResNet50/MobileNet/InceptionV3)

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
if not os.path.exists(img_path):
    raise SystemExit(f"File gambar tidak ditemukan: {img_path}\n"
                     "Taruh gambar bernama Elephant.jpg di folder ini.")

IMG_SIZE = 224
# init the models
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet'); IMG_SIZE = 299
print(model.summary())

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# processed_image = vgg16.preprocess_input(x)
# processed_image = resnet50.preprocess_input(x)
processed_image = mobilenet.preprocess_input(x)
# processed_image = inception_v3.preprocess_input(x)

# prediction
predictions = model.predict(processed_image)
results = decode_predictions(predictions)
print(results)

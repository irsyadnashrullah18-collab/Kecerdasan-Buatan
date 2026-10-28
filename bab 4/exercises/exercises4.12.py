# Exercise 4.12: Tambahkan DenseNet ke Example 4.14 untuk klasifikasi webcam,

import time
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16, resnet50, mobilenet, inception_v3, densenet
)

# kunci tombol -> (nama, constructor, modul preprocess, ukuran gambar)
MODELS = {
    ord('1'): ("VGG16",       vgg16.VGG16,               vgg16,        224),
    ord('2'): ("ResNet50",    resnet50.ResNet50,         resnet50,     224),
    ord('3'): ("MobileNet",   mobilenet.MobileNet,       mobilenet,    224),
    ord('4'): ("InceptionV3", inception_v3.InceptionV3,  inception_v3, 299),
    ord('5'): ("DenseNet121", densenet.DenseNet121,      densenet,     224),
}
loaded = {}


def get_model(key):
    if key not in loaded:
        name, ctor, _, _ = MODELS[key]
        print(f"Memuat {name} ...")
        loaded[key] = ctor(weights='imagenet')
    return loaded[key]


current = ord('5')            # mulai dari DenseNet121
camera = cv2.VideoCapture(0)
fps = 0.0

while camera.isOpened():
    t0 = time.time()
    ok, cam_frame = camera.read()
    if not ok:
        print("Gagal membaca frame dari webcam")
        break

    name, _, module, size = MODELS[current]
    model = get_model(current)

    frame = cv2.resize(cam_frame, (size, size))
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = module.preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)

    dt = time.time() - t0
    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(name, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(cam_frame, "FPS: {:.1f}  (tekan 1-5 ganti model)".format(fps),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # ESC
        break
    if key in MODELS and key != current:
        current = key
        fps = 0.0

camera.release()
cv2.destroyAllWindows()

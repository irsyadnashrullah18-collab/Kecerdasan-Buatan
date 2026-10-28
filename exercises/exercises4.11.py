# Exercise 4.11: Klasifikasi webcam dengan VGG19 (ganti VGG16 di Example 4.13)

import time
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg19
import numpy as np

image_size = 224
model = vgg19.VGG19(weights='imagenet')
print(model.summary())

camera = cv2.VideoCapture(0)
fps = 0.0

while camera.isOpened():
    t0 = time.time()
    ok, cam_frame = camera.read()
    if not ok:
        print("Gagal membaca frame dari webcam")
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg19.preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)

    fps = 0.9 * fps + 0.1 * (1.0 / (time.time() - t0)) if fps else 1.0 / (time.time() - t0)
    cv2.putText(cam_frame, "VGG19: {}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(cam_frame, "FPS: {:.1f}".format(fps),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('video image', cam_frame)
    if cv2.waitKey(30) == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()

# Example 4.15 Klasifikasi webcam dengan library Image-Classifiers
# https://pypi.org/project/image-classifiers/

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.tfkeras import Classifiers

# Set up a model
clf, preprocess_input = Classifiers.get('vgg16')
# clf, preprocess_input = Classifiers.get('resnet50')
# clf, preprocess_input = Classifiers.get('mobilenetv2')
# clf, preprocess_input = Classifiers.get('densenet201')
sz = 224
# clf, preprocess_input = Classifiers.get('inceptionv3')
# sz = 299
model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)

model.summary()

camera = cv2.VideoCapture(0)
image_size = sz

while True:
    ret, cam_frame = camera.read()
    if not ret:
        print("Gagal membaca frame dari webcam")
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    img = np.asarray(frame)
    img = np.expand_dims(img, 0)
    img = preprocess_input(img)

    preds = model.predict(img, verbose=0)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, "{}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()

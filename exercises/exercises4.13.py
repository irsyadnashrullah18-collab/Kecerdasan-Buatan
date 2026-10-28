# Exercise 4.13: Pada Example 4.15, uncomment model yang berbeda-beda dan bandingkan

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import cv2
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.tfkeras import Classifiers

# ===== Pilih salah satu (ganti nilai MODEL_NAME) =====
MODEL_NAME = 'vgg16'
# MODEL_NAME = 'resnet50'
# MODEL_NAME = 'mobilenetv2'
# MODEL_NAME = 'densenet201'
# MODEL_NAME = 'inceptionv3'     # <- otomatis memakai ukuran 299
# =====================================================

sz = 299 if MODEL_NAME == 'inceptionv3' else 224
clf, preprocess_input = Classifiers.get(MODEL_NAME)
model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
model.summary()
print("Total parameter:", f"{model.count_params():,}")

camera = cv2.VideoCapture(0)
times = []

while True:
    t0 = time.time()
    ret, cam_frame = camera.read()
    if not ret:
        print("Gagal membaca frame dari webcam")
        break

    frame = cv2.resize(cam_frame, (sz, sz))
    img = preprocess_input(np.expand_dims(np.asarray(frame), 0))
    preds = model.predict(img, verbose=0)
    label = decode_predictions(preds)

    times.append(time.time() - t0)
    fps = 1.0 / np.mean(times[-10:])
    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(MODEL_NAME, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(cam_frame, "FPS: {:.1f}".format(fps),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Classification", cam_frame)
    if cv2.waitKey(30) == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()
if times:
    print(f"{MODEL_NAME}: rata-rata {np.mean(times) * 1000:.0f} ms/frame (~{1 / np.mean(times):.1f} FPS)")

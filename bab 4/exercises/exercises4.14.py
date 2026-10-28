# Exercise 4.14: Pada Example 4.15, pakai if-else untuk memilih model klasifikasi.

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.tfkeras import Classifiers

print("Pilih model:")
print("  1 = VGG16")
print("  2 = ResNet50")
print("  3 = MobileNetV2")
print("  4 = DenseNet201")
print("  5 = InceptionV3")
choice = input("Masukkan nomor (1-5): ").strip()

if choice == '1':
    model_name, sz = 'vgg16', 224
elif choice == '2':
    model_name, sz = 'resnet50', 224
elif choice == '3':
    model_name, sz = 'mobilenetv2', 224
elif choice == '4':
    model_name, sz = 'densenet201', 224
elif choice == '5':
    model_name, sz = 'inceptionv3', 299
else:
    print("Pilihan tidak dikenal, memakai VGG16.")
    model_name, sz = 'vgg16', 224

print("Memuat model:", model_name)
clf, preprocess_input = Classifiers.get(model_name)
model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
model.summary()

camera = cv2.VideoCapture(0)
while True:
    ret, cam_frame = camera.read()
    if not ret:
        print("Gagal membaca frame dari webcam")
        break

    frame = cv2.resize(cam_frame, (sz, sz))
    img = preprocess_input(np.expand_dims(np.asarray(frame), 0))
    preds = model.predict(img, verbose=0)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(model_name, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    if cv2.waitKey(30) == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()

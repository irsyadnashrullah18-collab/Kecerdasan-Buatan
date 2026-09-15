# Exercise 3.19 PyCaret Classification on Breast Cancer Dataset
import pandas as pd
from sklearn import datasets
from pycaret import classification

# 1. Load dataset Breast Cancer sebagai pandas DataFrame
cancer = datasets.load_breast_cancer(as_frame=True)
cancer.data['Target'] = cancer.target
df = cancer.data

# 2. Menampilkan 5 sampel data teratas
print(df.head())

# 3. Inisialisasi setup eksperimen PyCaret untuk klasifikasi
classification.setup(data=df, target='Target')

# 4. Bandingkan performa berbagai model klasifikasi secara otomatis
classification.compare_models()
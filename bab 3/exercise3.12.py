# Exercise 3.12 Multiple Linear Regression on Linnerud Dataset
from sklearn import linear_model
from sklearn.datasets import load_linnerud

# 1. Load dataset Linnerud
linnerud = load_linnerud()
x = linnerud.data    # Fitur fisik: Chins, Situps, Jumps
y = linnerud.target  # Target medis: Weight, Waist, Pulse

# 2. Inisialisasi dan latih model Multiple Linear Regression
reg = linear_model.LinearRegression()
reg.fit(x, y)

# 3. Tampilkan koefisien dan intersep
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# 4. Prediksi menggunakan sampel data pertama dari dataset
pred = reg.predict([x[0]])
print('Prediction: \n', pred)
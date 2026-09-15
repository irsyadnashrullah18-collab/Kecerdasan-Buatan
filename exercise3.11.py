# Exercise 3.11 Linear Regression with Additional Data & Plot Enhancements
import matplotlib.pyplot as plt
from scipy import stats

# 1. Menambahkan lebih banyak titik data pada x dan y (20 data points)
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6, 14, 1, 10, 15, 3, 13, 8]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86, 75, 105, 82, 70, 98, 79, 89]

# 2. Hitung regresi linear
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

# 3. Membuat Plot dengan elemen lengkap
plt.figure(figsize=(8, 6))

# Scatter plot data asli
plt.scatter(x, y, color='blue', label='Data Points')

# Garis regresi linear
plt.plot(x, mymodel, color='red', label=f'Fit Line (y = {slope:.2f}x + {intercept:.2f})')

# Menambahkan kustomisasi plot
plt.xlabel('Car Age (Years)')              # X label
plt.ylabel('Speed (mph)')                 # Y label
plt.title('Car Age vs Speed Linear Model') # Title
plt.legend(loc='upper right')             # Legend
plt.grid(True, linestyle='--', alpha=0.7) # Grid

plt.show()
# Exercise 3.18 California Housing Regression (Kompatibel dengan Python 3.11+)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load dataset California Housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Pisahkan fitur (X) dan target 'MedHouseVal' (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# 3. Bagi data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Buat pipeline regresi otomatis (Scaling + Random Forest Regressor)
model = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))

# 5. Latih model
model.fit(X_train, y_train)

# 6. Evaluasi performa model (R2 Score)
score = model.score(X_test, y_test)
print(f"Skor Evaluasi Model (R^2 Score): {score:.4f}")
# Example 3.26 (Versi Kompatibel Python 3.11+)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline otomatis (StandardScaler + Regressor)
model = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# 4. Skor R^2
score = model.score(X_test, y_test)
print(f"Skor Evaluasi (R^2): {score:.4f}")
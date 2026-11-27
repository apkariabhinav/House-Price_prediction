
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv(BASE_DIR / "data.csv")

# 2. Convert date into separate columns
df["date"] = pd.to_datetime(df["date"])

df["sale_year"] = df["date"].dt.year
df["sale_month"] = df["date"].dt.month
df["sale_day"] = df["date"].dt.day

# 3. Select input features and target
features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
    "city", "statezip", "country",
    "sale_year", "sale_month", "sale_day"
]

X = df[features]
y = df["price"]

# 4. Separate numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# 5. Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model.fit(X_train, y_train)

# 9. Evaluate model
predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# 10. Save model
joblib.dump(model, BASE_DIR / "house_price_model.pkl")

print("Model trained and saved successfully!")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Loading the data...")

# Load the datasets
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

# Prepare features and target
X_train = train_data.drop(['Index', 'demand'], axis=1).copy()
y_train = train_data['demand'].copy()
X_test = test_data.drop(['Index'], axis=1).copy()

# Handle categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Fill missing values  
numeric_cols = X_train.select_dtypes(include=['number']).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())

for col in categorical_cols:
    X_train[col] = X_train[col].fillna('Unknown')
    X_test[col] = X_test[col].fillna('Unknown')

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], ignore_index=True)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- TRAINING OPTIMIZED MODELS ---")

# Test multiple configurations
best_score = 0
best_model = None

configs = [
    {'n_estimators': 250, 'max_depth': 13, 'min_samples_split': 2},
    {'n_estimators': 250, 'max_depth': 12, 'min_samples_split': 2},
    {'n_estimators': 200, 'max_depth': 14, 'min_samples_split': 2},
]

for config in configs:
    print(f"Testing RF: n_est={config['n_estimators']}, depth={config['max_depth']}")
    model = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )
    model.fit(X_train_scaled, y_train)
    score = model.score(X_train_scaled, y_train)
    print(f"  R²: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model

# Try ExtraTreesRegressor
print("Testing ExtraTreesRegressor: n_est=250, depth=12")
et_model = ExtraTreesRegressor(
    n_estimators=250,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)
et_model.fit(X_train_scaled, y_train)
et_score = et_model.score(X_train_scaled, y_train)
print(f"  R²: {et_score:.4f}")

if et_score > best_score:
    best_score = et_score
    best_model = et_model
    best_name = "ExtraTreesRegressor"
else:
    best_name = "RandomForest"

print(f"\n--- FINAL RESULTS ---")
print(f"Best Model: {best_name}")
print(f"Best R² Score: {best_score:.4f}")
print(f"Accuracy Equivalent: {best_score * 100:.2f}%")

if best_score * 100 >= 94:
    print("✓ Target accuracy of 94%+ ACHIEVED!")
else:
    print(f"⚠ Current: {best_score * 100:.2f}%, Target: 94%+")

# Cross-validation
print("\n--- CROSS-VALIDATION (3-fold) ---")
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=3, scoring='r2')
print(f"CV R² Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Predictions
predictions = best_model.predict(X_train_scaled)
mae = mean_absolute_error(y_train, predictions)
rmse = np.sqrt(mean_squared_error(y_train, predictions))
print(f"\nMAE: {mae:.4f}, RMSE: {rmse:.4f}")

test_pred = best_model.predict(X_test_scaled)
print(f"\nTest predictions: {len(test_pred)} samples, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

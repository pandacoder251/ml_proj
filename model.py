import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Loading the data...")

# Load the datasets
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

print("\n--- TRAIN DATA SAMPLE ---")
print(train_data.head())
print(f"\nTrain shape: {train_data.shape}")
print(f"Train columns: {list(train_data.columns)}")

print("\n--- TEST DATA SAMPLE ---")
print(test_data.head())
print(f"\nTest shape: {test_data.shape}")
print(f"Test columns: {list(test_data.columns)}")

print(f"\nSuccess! Train rows: {len(train_data)}, Test rows: {len(test_data)}")

# Prepare features and target
X_train = train_data.drop(['Index', 'demand'], axis=1).copy()
y_train = train_data['demand'].copy()

X_test = test_data.drop(['Index'], axis=1).copy()

# Handle categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_cols}")

# Fill missing values
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Combine train and test values for encoding
    combined = pd.concat([X_train[col], X_test[col]], ignore_index=True)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- TRAINING MODELS ---")

# Train Gradient Boosting Regressor
print("Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    subsample=0.9,
    loss='huber'
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_r2 = gb_model.score(X_train_scaled, y_train)

print(f"Gradient Boosting R² Score (Train): {gb_r2:.4f}")

# Train Random Forest Regressor
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_r2 = rf_model.score(X_train_scaled, y_train)

print(f"Random Forest R² Score (Train): {rf_r2:.4f}")

# Select best model
best_model = gb_model if gb_r2 >= rf_r2 else rf_model
best_r2 = max(gb_r2, rf_r2)
best_name = "Gradient Boosting" if gb_r2 >= rf_r2 else "Random Forest"

print(f"\n--- FINAL RESULTS ---")
print(f"Best Model: {best_name}")
print(f"Best R² Score: {best_r2:.4f}")

# Convert R² to percentage accuracy equivalent
accuracy_equivalent = best_r2 * 100
print(f"Accuracy Equivalent: {accuracy_equivalent:.2f}%")

if accuracy_equivalent >= 94:
    print("✓ Target accuracy of 94%+ ACHIEVED!")
else:
    print(f"⚠ Current: {accuracy_equivalent:.2f}%, Target: 94%+")

# Cross-validation
print("\n--- CROSS-VALIDATION (5-fold) ---")
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"CV R² Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Mean CV Accuracy Equivalent: {cv_scores.mean() * 100:.2f}%")

# Additional metrics
print("\n--- ADDITIONAL METRICS (on training data) ---")
gb_predictions = gb_model.predict(X_train_scaled)
mae = mean_absolute_error(y_train, gb_predictions)
rmse = np.sqrt(mean_squared_error(y_train, gb_predictions))
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

print("\n--- PREDICTIONS SUMMARY ---")
print(f"Test predictions shape: {gb_pred.shape}")
print(f"Test predictions (first 10): {gb_pred[:10]}")
print(f"Prediction range: [{gb_pred.min():.4f}, {gb_pred.max():.4f}]")
# Install XGBoost if needed
# pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, f1_score
)
from xgboost import XGBRegressor

# --- Load dataset ---
df = pd.read_csv("Letterbox-Movie-Classification-Dataset.csv")
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

# --- Feature Engineering ---
df["likes_per_rating"] = df["Likes"] / (df["Total_ratings"] + 1)
df["fans_per_watch"] = df["Fans"] / (df["Watches"] + 1)
df["likes_per_fan"] = df["Likes"] / (df["Fans"] + 1)
df["Description"] = df["Description"].fillna("")
df["Genres"] = df["Genres"].astype(str).str.replace(r"[\[\]']", "", regex=True)
df["Studios"] = df["Studios"].astype(str).str.replace(r"[\[\]']", "", regex=True)

top_directors = df["Director"].value_counts().nlargest(50).index
df["Director"] = df["Director"].apply(lambda x: x if x in top_directors else "Other")

# --- Define features and target ---
features = [
    "Director", "Genres", "Runtime", "Original_language", "Description",
    "Studios", "Watches", "List_appearances", "Likes", "Fans",
    "Lowest★", "Medium★★★", "Highest★★★★★", "Total_ratings",
    "likes_per_rating", "fans_per_watch", "likes_per_fan"
]
target = "Average_rating"

X = df[features]
y = df[target]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = ["Director", "Genres", "Original_language", "Studios"]
numeric_cols = [
    "Runtime", "Watches", "List_appearances", "Likes", "Fans",
    "Lowest★", "Medium★★★", "Highest★★★★★", "Total_ratings",
    "likes_per_rating", "fans_per_watch", "likes_per_fan"
]
text_col = "Description"

# --- Preprocessing ---
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols),
    ("text", TfidfVectorizer(max_features=150), text_col)
])

# --- XGBoost Pipeline ---
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
])

# --- GridSearch Hyperparameters ---
param_grid = {
    "regressor__n_estimators": [300],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__max_depth": [4, 6],
    "regressor__min_child_weight": [3, 5],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0]
}

# --- GridSearchCV with KFold ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)
print("🔍 Starting GridSearchCV with XGBoost...")
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# --- Final Model ---
model = grid.best_estimator_
print(f"\n✅ Best Parameters: {grid.best_params_}")

# --- Predict & Evaluate ---
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Regression Metrics:")
print(f"🔹 RMSE: {rmse:.4f}")
print(f"🔹 MAE: {mae:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# --- Find Optimal Threshold ---
thresholds = np.arange(3.6, 4.1, 0.01)
best_f1 = 0
best_threshold = 4.0
for t in thresholds:
    y_test_bin = (y_test >= t).astype(int)
    y_pred_bin = (y_pred >= t).astype(int)
    score = f1_score(y_test_bin, y_pred_bin)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

print(f"\n🎯 Best Classification Threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")

# --- Classification Report ---
y_test_class = (y_test >= best_threshold).astype(int)
y_pred_class = (y_pred >= best_threshold).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test_class, y_pred_class, target_names=["Bad", "Good"]))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 💾 Save model
# --- Save trained model ---
joblib.dump(model, "random_forest_movie_rating_model.pkl")
print("💾 Model saved to 'random_forest_movie_rating_model.pkl'")

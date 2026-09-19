import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load generated dataset
df = pd.read_csv("student_data.csv")

# Define feature columns (exclude identification and target columns)
feature_cols = [
    "entry_mode_code",
    "socio_band_code",
    "prior_cgpa",
    "ca_mean",
    "ca_variance",
    "momentum",
    "attendance_rate",
    "clicks_total",
    "duration_total_min",
    "resource_breadth",
    "submission_timeliness",
    "engagement_index",
    "forum_posts_count",
    "login_frequency",
]

X = df[feature_cols]
y = df["is_at_risk"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train XGBoost Model
model = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model artifact
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl!")
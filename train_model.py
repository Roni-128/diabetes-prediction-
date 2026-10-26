import pandas as pd
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Pima Indians diabetes dataset
data = fetch_openml(
    name="diabetes",
    version=1,
    as_frame=True
)

X = data.data
y = data.target

# Convert target labels to 0 and 1
y = y.astype(str).map({"tested_negative": 0, "tested_positive": 1})

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train the model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

# Save the trained model and input feature names
joblib.dump(
    {"model": model, "features": list(X.columns)},
    "diabetes_model.joblib"
)

print("Model saved successfully!")

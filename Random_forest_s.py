import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("Running Random Forest Model...")

# Load dataset
data = pd.read_csv("pesticides.csv")

# Convert categorical columns into numbers
le = LabelEncoder()

data["Domain"] = le.fit_transform(data["Domain"])
data["Area"] = le.fit_transform(data["Area"])
data["Element"] = le.fit_transform(data["Element"])
data["Item"] = le.fit_transform(data["Item"])
data["Unit"] = le.fit_transform(data["Unit"])

# Convert Value into categories (0–3)
data["Value_category"] = pd.qcut(data["Value"], 4, labels=[0, 1, 2, 3])

# Features and target
X = data.drop(["Value", "Value_category"], axis=1)
y = data["Value_category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("First 10 Predictions:", predictions[:10])
print("Actual Values:", y_test.values[:10])

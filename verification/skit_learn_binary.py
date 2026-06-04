import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if not os.path.exists("config\\config.yaml"):
    print("Error: config.yaml not found.")
    exit()

with open("config\\config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["environment"]["data_file"]
features = config["data"]["feature_names"]
train_split = config["data"]["train_split"]
val_split = config["data"]["val_split"]

test_size = 1.0 - train_split

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Dataset not found at {data_path}")
    exit()

X = df[features].values
y = df["Target"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

models = {
    "Linear Baseline (Logistic Regression)": LogisticRegression(random_state=42),
    "Radial Basis Function SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "Random Forest Classifier (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("=" * 60)
print("         SCIKIT-LEARN MOBIUS EVALUATION REPORT")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    train_preds = model.predict(X_train_scaled)
    val_preds = model.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"\nModel: {name}")
    print(f"  Training Accuracy  : {train_acc:.6f}")
    print(f"  Validation Accuracy: {val_acc:.6f}")
    
    if val_acc > 0.95:
        print("  Status             : SUCCESS (Resolved Twist Boundary)")
    else:
        print("  Status             : FAILED (Manifold Collision)")

print("\n" + "=" * 60)
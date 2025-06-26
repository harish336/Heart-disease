"""import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

df = pd.read_csv("dataset.csv")

df.columns = df.columns.str.strip()

print("Available columns:", df.columns.tolist())

if "condition" in df.columns:
    label_column = "condition"
else:
    raise ValueError("Target column not found. Please check your dataset headers.")

X = df.drop(label_column, axis=1)
y = df[label_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_split=15,min_samples_leaf=10, random_state=2)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)

with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to model/heart_model.pkl")

from sklearn.metrics import accuracy_score

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

score = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {score}")
print(f"Mean cross-validation score: {score.mean()}%")"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle, os

df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip()

if "condition" not in df.columns:
    raise ValueError("Target column 'condition' not found.")
X = df.drop("condition", axis=1)
y = df["condition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,               
    min_samples_split=10,      
    min_samples_leaf=8,        
    random_state=42
)
model.fit(X_train, y_train)
model.fit(X_train, y_train)
os.makedirs("model", exist_ok=True)
with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"Training Accuracy: {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Accuracy: {accuracy_score(y_test, test_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, test_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv)

print(f"Cross-validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")

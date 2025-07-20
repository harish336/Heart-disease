import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip()
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

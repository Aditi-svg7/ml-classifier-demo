import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load data
df = pd.read_csv("data/dataset.csv")
X = df.drop("target", axis=1)
y = df["target"]

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Build a pipeline: preprocessing + model chained together
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Train the pipeline (fit_transform happens internally on train data only)
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
target_names = ["setosa", "versicolor", "virginica"]
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. Save the entire pipeline (scaler + model together)
joblib.dump(pipeline, "model.pkl")
print("Pipeline saved to model.pkl")
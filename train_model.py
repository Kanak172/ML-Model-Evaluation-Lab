import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load dataset
print("1. Loading dataset...")
df = pd.read_csv('data/churn.csv')
print(f"Dataset shape: {df.shape}")

# Encode target variable
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() in {'yes', '1', 'true'} else 0)

X = df.drop(columns=['Churn'])
y = df['Churn']

# Identify numeric and categorical columns dynamically
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# 2. Stratified train-test split to preserve class balance
print("2. Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build leakage-safe preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("3. Training models...")
for name, model in models.items():
    print(f"\n--- {name} ---")
    # Combine preprocessor and model into a single pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit only on training data (prevents data leakage)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\nTraining and evaluation complete!")
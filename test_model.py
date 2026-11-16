import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def test_pipeline_execution():
    # Load dataset
    df = pd.read_csv('data/churn.csv')
    
    # Clean dataset columns if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() in {'yes', '1', 'true'} else 0)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=100, random_state=42))
    ])
    
    # Fit and evaluate
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    assert accuracy_score(y_test, preds) > 0.4, "Model accuracy is below threshold."
    print("Execution verification passed successfully.")

if __name__ == "__main__":
    test_pipeline_execution()
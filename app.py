import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="ML Model Evaluation Lab", layout="wide")

st.title("🏆 ML Model Evaluation Lab")
st.markdown("Comparing Machine Learning Models for Customer Churn Prediction")

@st.cache_resource
def load_and_train_models():
    df = pd.read_csv("data/churn.csv")
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() in {'yes', '1', 'true'} else 0)
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred)
        }
    return results, df.shape

with st.spinner("Training models and loading metrics..."):
    results, shape = load_and_train_models()

st.sidebar.header("Dataset Info")
st.sidebar.write(f"Dataset Shape: {shape[0]} rows, {shape[1]} columns")

st.subheader("🏆 Model Performance Comparison")
metrics_df = pd.DataFrame({
    name: [res["accuracy"], res["precision"], res["recall"], res["f1"]]
    for name, res in results.items()
}, index=["Accuracy", "Precision", "Recall", "F1-Score"])

st.dataframe(metrics_df.style.highlight_max(axis=1, color='lightgreen'))

st.subheader("📉 Confusion Matrices")
cols = st.columns(len(results))
for i, (name, res) in enumerate(results.items()):
    with cols[i]:
        st.markdown(f"**{name}**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(res["cm"], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
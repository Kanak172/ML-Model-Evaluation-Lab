# Dataset Documentation

- **Source:** Telco Customer Churn Dataset (`data/churn.csv`)
- **Target Variable:** `Churn` (Binary: 1 for Yes, 0 for No)
- **Features:** Includes customer demographic information, account services (tenure, contract, payment method), and monthly/total charges.
- **Preprocessing:** Missing values in `TotalCharges` are imputed with median values. Categorical features are encoded using `OneHotEncoder` inside a leakage-safe scikit-learn pipeline.
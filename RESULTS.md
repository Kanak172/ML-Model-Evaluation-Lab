# Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | ~0.80 | ~0.65 | ~0.54 | ~0.59 |
| **Random Forest** | ~0.79 | ~0.63 | ~0.49 | ~0.55 |

- **Evaluation Methodology:** Trained and evaluated using an 80-20 stratified train-test split (`stratify=y`) to maintain class balance.
- **Leakage Prevention:** Preprocessing steps (scaling and encoding) are fully encapsulated within the pipeline and fitted exclusively on training folds.
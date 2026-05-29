# Defect Detection in Hot Rolling Using Machine Learning

## Project Overview

This project focuses on predicting Alpha defects in hot rolling mills using machine learning techniques. The objective is to identify defective coils based on process parameters collected during steel manufacturing.

The solution uses classification algorithms to predict whether a coil contains a defect or not, helping improve quality control and reduce production losses.

---

## Dataset Description

The dataset contains:

### Training Data
- 1352 records
- 49 process parameter features (X1 – X49)
- Target variable: Y
  - 0 = No Defect
  - 1 = Defect

### Test Data
- 339 records
- 49 process parameter features

### Additional Column
- CoilID: Unique identifier for each coil

---

## Problem Statement

Build a machine learning model capable of predicting Alpha defects in steel coils while maximizing defect detection performance.

---

## Feature Engineering

The following preprocessing steps were performed:

- Removed CoilID from model features
- Used X1–X49 as predictive variables
- Selected Y as target variable
- Handled class imbalance using class weighting
- Checked for missing values and data consistency
- Prepared data for machine learning model training

---

## Machine Learning Model

### Random Forest Classifier

Reasons for selection:

- Handles classification tasks effectively
- Works well with tabular data
- Reduces overfitting through ensemble learning
- Provides feature importance insights

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```
Defect-Detection-Hot-Rolling/
│
├── train.csv
├── test.csv
├── expected_submission.csv
├── defect_detection_code.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/defect-detection-hot-rolling.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute:

```bash
python defect_detection_code.py
```

The script will:

1. Load training and test datasets
2. Train the machine learning model
3. Generate predictions
4. Create the submission file

Output:

```text
expected_submission.csv
```

---

## Results

The model predicts defect occurrence for unseen coils and generates a submission file containing:

| CoilID | Y |
|---------|---|
| 1 | 0 |
| 2 | 1 |

---

## Future Improvements

- XGBoost implementation
- CatBoost implementation
- Hyperparameter tuning
- Ensemble learning
- Advanced feature engineering
- Model explainability using SHAP

---

## Author

Neelima Bonangi

B.Tech Graduate | Java Full Stack Developer | Data Science Enthusiast

#!/usr/bin/env python
# coding: utf-8

# In[88]:


get_ipython().system('pip install pandas numpy scikit-learn xgboost imbalanced-learn')


# In[90]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# In[92]:


train = pd.read_csv(r"Downloads\dataset.csv\dataset\train.csv")
test = pd.read_csv(r"Downloads\dataset.csv\dataset\test.csv")

print(train.shape)
print(test.shape)
train["Y"].value_counts()


# In[94]:


X = train.drop(columns=["CoilID", "Y"])
y = train["Y"]

X_test = test.drop(columns=["CoilID"])

imputer = SimpleImputer(strategy="median")

X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

scaler = StandardScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# In[105]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[107]:


model = XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=20,
    random_state=42
)

model.fit(X_resampled, y_resampled)


# In[109]:


pred_probs = model.predict_proba(X_test)[:, 1]


# In[113]:


predictions = (pred_probs > 0.001).astype(int)

submission = pd.DataFrame({
    "CoilID": test["CoilID"],
    "Y": predictions
})

submission.to_csv(r"C:\Users\USER\Downloads\expected_submission.csv", index=False)

print("Saved in Downloads")


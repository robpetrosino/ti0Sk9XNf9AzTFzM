# basics
# basics
import pandas as pd
import numpy as np

# preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# models
from xgboost import XGBClassifier

# accuracy metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# model deployment
import pickle

# the raw data was encoded, balanced, and feature-selected
data = pd.read_csv("../data/processed/dataset_final.csv")

# Prepare data
X = data.drop(columns = ['y'])
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardization
columns_to_scale = ['duration']

# make a copy of both feature datasets
X_train_stand = X_train.copy()
X_test_stand = X_test.copy()

for i in columns_to_scale:
    # standardization, i.e. (X - Xmean)/(Xsd)
    standardization = StandardScaler().fit(X_train[[i]])

    X_train_stand[i] = standardization.transform(X_train[[i]])
    X_test_stand[i] =  standardization.transform(X_test[[i]])

model_xgb = XGBClassifier(random_state=42) # all default values are the best hyperparameter settings
model_xgb.fit(X_train, y_train)

# Evaluation
y_pred = model_dt.predict(X_test)
accuracy_score(y_test, y_pred)


with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model_xgb, f)

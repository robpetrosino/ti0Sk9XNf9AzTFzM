import pickle
import pandas as pd
with open('trained_xgb_model.pkl', 'rb') as f:
    xgb = pickle.load(f)

def predict(model, new_data):
    predictions = model.predict(new_data)
    return predictions

new_data = pd.read_csv("../data/term-deposit-marketing-2020.csv")

if new_data.isnull().sum().sum() == 0: # check if the dataset has not NaNs
    # some pre-processing is required:

    ## 1. label encoding
    months = ['none', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    months_zip = dict(zip(months, range(int(len(months)))))
    education_levels = ['unknown', 'primary', 'secondary', 'tertiary']
    education_levels_zip = dict(zip(education_levels, range(int(len(education_levels)))))
    data_ = data.replace({'month': months_zip, 'education': education_levels_zip}) # rename the dataset

    ###
    binary_values = {'yes': 1, 'no': 0}
    data_ = data_.replace({'default': binary_values, 'housing': binary_values, 'loan': binary_values, 'y': binary_values})

    ###
    freq_columns = ['job', 'marital', 'contact']

    for i in freq_columns:
        freq_encoder = (data_.groupby(i).size()) / len(data)
        data_[i] = data_[i].apply(lambda x: freq_encoder[x])

    ## 2. feature selection
    corr = data_.corr()
    features_to_remove = corr.loc[:, abs(corr.loc['y']) < abs(0.1)].columns
    data_sel = data_.drop(features_to_remove, axis=1)

    data_x = data_.drop(['y'], axis=1)
    data_y = data_['y']

    ## 4. standardization
    columns_to_scale = ['duration']
    for i in columns_to_scale:
        standardization = StandardScaler().fit(X_train[[i]])
        data_x[i] = standardization.transform(X_train[[i]])

X = data_x
y = data_y

predictions = predict(xgb, X)
predictions

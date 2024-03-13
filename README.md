This repository contains the code for a machine learning model that is trained on the term deposit marketing dataset provided by Apziva. The goal of the prokject is to build a robust machine learning system that leverages information coming from call center data and improves the success rate for calls made to customers for any product that third-parties may offer. 

## Prerequisites

The following packages are required to run the code:

1. `pandas` 
2. `numpy`
3. `matplotlib` and `seaborn`
4. `scikit-learn`

You can install these packages by running the following command: `pip install -r requirements.txt`

# Train and predict

I found that the XGB classifier algorithm performs remarkably well in predicting the target label. The code for the model can be found in the `train.py` file. To train the model, run the following command: `python [file]`

The model will be trained on a pre-processed version `dataset_final.csv` of the raw dataset `term-deposit-marketing-2020.csv` dataset, and the accuracy score will be printed.

The code for using the trained model to make predictions on new data is contained in the `predict.py` file.

# Evaluation

The model uses `accuracy_score` and `f1_score` as evaluation metrics.

# Conclusion

In this project, I leveraged the power of machine learning, and employed a few pre-processing steps (label encoding, data balancing, feature-selection, and standardization), and an effective grid search cross-validation technique to identify the best parameters that allowed the XGB classifier algorithm to reach f-1 and accuracy scores of 94% -- more than 13% higher than the target value requested by the company.

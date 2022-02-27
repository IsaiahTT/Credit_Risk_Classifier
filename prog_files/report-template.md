# Credit Risk Analysis Report

## Overview of the Analysis

The purpose of this analysis was to to train separate models to be able to identify which potential borrowers of a peer-to-peer lending services company are creditworthy and which are not, and it is aimed to focus mostly on which are not, since that is the smaller (target) class. The financial data used to predict the identifications were loan size, interest rate, borrower's income, debt-to-income, number of accounts, derogatory marks, and total debt. With these series being used as features (X), the target was loan status (y), which was confirmed to have `1` (default) as the minority class out of the two with a value count of 0 - 75,036, and 1 - 2,500.
The process of the analysis using the machine learning tools, from SciKit Learn and Imbalanced Learn, began with reading in the data with Pandas and splitting the features columns (financial data) on the Dataframe from the target (loan status) column, then further splitting them into training and testing variable, officially, using `train_test_split` from SciKit Learn. After then fitting the the training variables to the model using the `.fit()` function, the prediction process is ran along with the features meant for testing, and you get the initial prediction's results, which are then used with the different SciKit Learn functions like `accuracy_score`/`balanced_accuracy_score`, `confusion_matrix`, and `classification_report`/`classification_report_imbalanced`. After, the resampled versions of the original training data is created via a RandomOverSampler model which outputs the new, resampled X and y variables by adding additional data points to the minority class based on the preexisting data of said minority class. A new model gets fitted with the resampled training data, and a prediction is ran via that new model. The same SciKit Learn analysis/report functions are then ran on the result and the two ends of insight (original reports and resampled reports) are then compared.

## Results

* Logistic Regression Model - Original Data:
  * Balanced accuracy score: 95.2%
  * Precision score: Non-Defaulting class - 100%, Defaulting class - 85%
  * Recall score: Non-Defaulting class - 99%, Defaulting class - 91%

* Logistic Regression Model - Resampled Data:
  * Balanced accuracy score: 99.4%
  * Precision score: Non-Defaulting class - 100%, Defaulting class - 84%
  * Recall score: Non-Defaulting class - 99%, Defaulting class - 99%

## Summary

The result of the original model gives a balanced accuracy score of 95.2%, which is high, though questionable given the difference in class sizes. The precision of the non-defaulting class was just about 100%, which is likely given its abundance of training data, while the precision of the defaulting class was 85%. While the recall score of the non-defaulting class stays high at 99%, the miniscule remaining percentage of the non-defaulting class's precision is the remaining 9% from the recall score of the defaulting class's recall score. This is significant because in the result of the logistic regression model based on resampled data, the defaulting class's recall score is then 99%, and so as well remains the non-defaulting class's recall score. Also, the precision scores stay about the same with only the defaulting class's score losing a single full percentage point. This means, the amount of false positives has shrunk massively, resulting in, potentially, less money lost in the company. As well as this, the balanced accuracy score of the resampled data model is 99.4%. Significantly higher than before, in this context, especially because the difference from it being 100% is 0.6%, when the difference of the classes from the loan status value count is 3.2%, and the remainder from 100% of the balanced accuracy score of the result of the original data is 4.8%.
The resampled data lead to better results, and this was because the focus was on predicting the `1`'s of the smaller class.
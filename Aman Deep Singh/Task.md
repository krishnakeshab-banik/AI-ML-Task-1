1.N500R:
# Document Summary

This Jupyter Notebook demonstrates the process of data preprocessing, model training, and evaluation using a dataset of Nifty 500 companies.

## Steps Involved:

A. **Importing the Libraries**:
    - Import necessary libraries such as pandas, matplotlib, numpy, and scikit-learn.

B. **Loading the Dataset**:
    - Load the dataset from a CSV file named `nifty_500.csv`.

C. **Conversion to Numerical**:
    - Convert specific columns to numeric data types.
    - Apply one-hot encoding to categorical columns.
    - Drop unnecessary columns.

D. **Handling Missing Values**:
    - Use `SimpleImputer` to fill missing values with the median.
    - Confirm that there are no missing values left.

E. **Splitting the Dataset and Training the Model**:
    - Split the dataset into training and testing sets.
    - Train a Linear Regression model.
    - Predict and visualize the results.
    - Calculate and print the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

2. N500C:

# Document Summary

This Jupyter Notebook demonstrates the process of data preprocessing, handling missing values, and training machine learning models on a dataset. Below is a summary of the steps involved:

A.Importing the Libraries
Necessary libraries such as pandas, matplotlib, numpy, and scikit-learn are imported.

B.Loading the Dataset
The dataset is loaded from a CSV file named `nifty_500.csv`.

C.Conversion to Numerical
Certain columns are converted to numerical data types, and categorical columns are converted to dummy variables. Unnecessary columns are dropped.

D.Handling Missing Values
Missing values in the dataset are imputed using the median strategy, and any remaining missing values are filled with the median of the respective columns.

E.Splitting the Dataset and Training the Model
The dataset is split into training and testing sets. Two machine learning models, SVM and Linear SVM, are trained on the training set. The models are evaluated using accuracy, precision, recall, and F1 score metrics.

F.Results
The performance of the SVM and Linear SVM models is evaluated and compared.

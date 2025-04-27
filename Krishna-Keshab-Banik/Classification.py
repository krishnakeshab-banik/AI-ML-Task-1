# 1. Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r'C:\Users\godre\Desktop\AI ML CN Task 1\nifty_500.csv')

columns_to_convert = [col for col in df.columns if 'Change' in col]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

le = LabelEncoder()
for col in ['Industry', 'Series']:
    df[col] = le.fit_transform(df[col])

df['Target'] = df['Percentage Change'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop(columns=['Company Name', 'Symbol', 'Last Traded Price', 'Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

lsvm_model = LinearSVC(max_iter=10000)
lsvm_model.fit(X_train, y_train)
y_pred_lsvm = lsvm_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.2f}")

evaluate_model(y_test, y_pred_svm, "SVM")
evaluate_model(y_test, y_pred_lsvm, "Linear SVM (LSVM)")

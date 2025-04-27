import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:\\Users\\godre\\Desktop\\AI ML CN Task 1\\nifty_500.csv')
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

print(df.head())

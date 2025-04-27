# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load dataset
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

# Features (X) and Target (y)
X = df.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y = df['Last Traded Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test)

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('../data/air_quality_1000_dataset.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

features = ['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity']
X = df[features].copy()
y = df['AQI']

# Same feature engineering from your notebook
X['PM_ratio']        = X['PM2.5'] / (X['PM10'] + 1e-5)
X['NO2_SO2_sum']     = X['NO2'] + X['SO2']
X['Pollution_index'] = X['PM2.5']*0.4 + X['PM10']*0.3 + X['NO2']*0.2 + X['SO2']*0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_sc, y_train)

# Save both model and scaler
joblib.dump(model, 'model/linear_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Model and scaler saved to model/")
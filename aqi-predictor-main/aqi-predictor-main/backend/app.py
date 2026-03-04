from flask import Flask, request, jsonify, render_template

from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # allows frontend to call the API

model  = joblib.load('model/linear_model.pkl')
scaler = joblib.load('model/scaler.pkl')

def categorize_aqi(aqi):
    if aqi <= 50:    return 'Good'
    elif aqi <= 100: return 'Moderate'
    elif aqi <= 150: return 'Unhealthy for Sensitive'
    elif aqi <= 200: return 'Unhealthy'
    elif aqi <= 300: return 'Very Unhealthy'
    else:            return 'Hazardous'

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([{
        'PM2.5':       data['pm25'],
        'PM10':        data['pm10'],
        'NO2':         data['no2'],
        'SO2':         data['so2'],
        'CO':          data['co'],
        'O3':          data['o3'],
        'Temperature': data['temp'],
        'Humidity':    data['humidity'],
    }])

    # Feature engineering (same as notebook)
    input_df['PM_ratio']        = input_df['PM2.5'] / (input_df['PM10'] + 1e-5)
    input_df['NO2_SO2_sum']     = input_df['NO2'] + input_df['SO2']
    input_df['Pollution_index'] = (input_df['PM2.5']*0.4 + input_df['PM10']*0.3 +
                                    input_df['NO2']*0.2  + input_df['SO2']*0.1)

    scaled     = scaler.transform(input_df)
    aqi        = float(model.predict(scaled)[0])
    category   = categorize_aqi(aqi)

    return jsonify({ 'aqi': round(aqi, 2), 'category': category })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
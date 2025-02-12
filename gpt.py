import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import joblib
import scapy.all as scapy
import threading
import time
import streamlit as st

# Load Dataset
data_path = r"C:\Users\tanis\my_folder\GEN_AI\ddos_code\application\data.csv"
ddos_data = pd.read_csv(data_path)

# Selecting Top 20 Features
top_features = [' Timestamp', ' Source Port', ' Min Packet Length', ' Fwd Packet Length Min',
                'Flow ID', ' Packet Length Mean', ' Fwd Packet Length Max', ' Average Packet Size',
                ' ACK Flag Count', ' Avg Fwd Segment Size', ' Fwd Packet Length Mean', 'Flow Bytes/s',
                ' Max Packet Length', ' Protocol', 'Fwd Packets/s', ' Flow Packets/s',
                'Total Length of Fwd Packets', ' Subflow Fwd Bytes', ' Destination Port', ' act_data_pkt_fwd']

# Filtering dataset
ddos_data = ddos_data[top_features + ['Label']]

# Encoding Labels
ddos_data['Label'] = ddos_data['Label'].apply(lambda x: 1 if x == 'DDoS' else 0)

# Handling Missing Values
ddos_data = ddos_data.dropna()

# Standardizing Features
scaler = StandardScaler()
X = scaler.fit_transform(ddos_data[top_features])
y = ddos_data['Label'].values

# Splitting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save Model & Scaler
model.save("lstm_ddos_model.h5")
joblib.dump(scaler, "scaler.pkl")

# Flask API for Real-time Predictions
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array(data['features']).reshape(1, -1)
    input_features = scaler.transform(input_features)
    prediction = model.predict(input_features)
    result = "DDoS Attack Detected" if prediction[0][0] > 0.5 else "Normal Traffic"
    return jsonify({"prediction": result})

# Function to Simulate Live Traffic
def generate_traffic():
    while True:
        packet = scapy.IP(dst="192.168.1.1") / scapy.TCP()
        scapy.send(packet, verbose=False)
        time.sleep(0.1)

# Start Traffic Generation in Background
traffic_thread = threading.Thread(target=generate_traffic, daemon=True)
traffic_thread.start()

# Streamlit Dashboard for Monitoring
st.title("DDoS Attack Detection Dashboard")
if st.button("Check Prediction"):
    response = requests.post("http://127.0.0.1:5000/predict", json={"features": list(X_test[0])})
    st.write(response.json())

if __name__ == '__main__':
    app.run(debug=True)

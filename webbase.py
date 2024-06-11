import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model('C:\\Users\\docto\\ML_BTL\\Stock Predictions Model.keras')

# Streamlit header
st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Define the time period for the stock data
start = '2012-01-01'
end = '2023-12-31'

# Download stock data
data = yf.download(stock, start, end)

# Debug print to check if st.subheader is still a function
print(f"st.subheader before: {st.subheader}")

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Debug print to check if st.subheader is still a function
print(f"st.subheader after: {st.subheader}")

# Split the data into training and testing datasets
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.8)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig2)

# Plot Price vs MA50 vs MA100 vs MA200
st.subheader('Price vs MA50 vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(ma_200_days, 'purple', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)

# Scale back the predictions
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Plot Original Price vs Prediction
st.subheader('Original Price vs Prediction')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

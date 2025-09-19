import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.title("📈 Stock Price Prediction with LSTM")

# Step 1: Load Data
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, INFY.NS)", "AAPL")
start_date = "2015-01-01"
end_date = "2023-12-31"

if st.button("Predict"):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        st.error("No data found. Check the ticker symbol.")
    else:
        st.success("Data loaded successfully!")

        # Step 2: Preprocess
        close_data = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Step 3: Build Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Step 4: Train Model (light train)
        with st.spinner("Training the model..."):
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        
        st.success("Model trained.")

        # Step 5: Predict next 30 days
        last_60_days = scaled_data[-60:]
        predicted_prices = []

        for _ in range(30):
            X_pred = last_60_days.reshape(1, 60, 1)
            pred_price = model.predict(X_pred)[0][0]
            predicted_prices.append(pred_price)
            last_60_days = np.append(last_60_days[1:], pred_price).reshape(-1, 1)

        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

        # Step 6: Plot
        st.subheader("📊 Predicted Prices (Next 30 Days)")
        plt.figure(figsize=(10, 4))
        plt.plot(predicted_prices, label='Predicted')
        plt.title(f"Predicted Closing Prices for {ticker}")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

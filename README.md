# 📈 Stock Price Prediction using LSTM

This project is a **Stock Price Prediction Web App** built with **LSTM (Long Short-Term Memory)** neural networks.  
It uses historical stock price data to forecast the next 30 days' closing prices and provides an interactive visualization.

## 🚀 Features
- Fetches real-time historical stock data using **Yahoo Finance API (yfinance)**.
- Preprocesses data with **MinMaxScaler** for LSTM training.
- Trains a **two-layer LSTM model** to capture stock market trends.
- Predicts the next **30 days of closing prices**.
- Visualizes predictions using **Matplotlib** in a **Streamlit web app**.

## 🛠️ Tech Stack
- **Python 3**
- **Streamlit**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib**
- **Scikit-learn**
- **yfinance**

## 📂 Project Structure
- app.py # Main Streamlit web app
- lstm_stock_model.h5 # Saved LSTM model
- minmax_scaler.pkl # Saved scaler for preprocessing
- Model training-3.ipynb # Jupyter Notebook for model training
- README.md # Project documentation
- LICENSE # MIT License


## ▶️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction-lstm.git
   cd stock-price-prediction-lstm

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app.py

4. Enter a stock ticker (e.g., AAPL, INFY.NS) and click Predict.
   

📊 **Sample Output**

Input stock ticker: AAPL

Predicted closing prices for the next 30 days.

Interactive line chart displaying predicted trends.

1. <img width="940" height="500" alt="image" src="https://github.com/user-attachments/assets/1b4bf65d-1796-4c5e-b403-5e2978f66282" />
2. <img width="940" height="499" alt="image" src="https://github.com/user-attachments/assets/eb7ed2dc-fee4-420b-ac05-5ded6565edc1" />


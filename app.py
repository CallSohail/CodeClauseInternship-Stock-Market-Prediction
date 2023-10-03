import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import streamlit as st
from keras.models import load_model
import plotly.graph_objects as go

# List of stock tickers
stock_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "FB", "NVDA", "JPM", "BRK-B", "JNJ",
    "V", "PG", "HD", "PYPL", "DIS",
    "INTC", "VZ", "UNH", "MA", "CRM",
    "T", "KO", "ADBE", "BAC", "CMCSA",
    "PEP", "XOM", "NFLX", "NKE", "ABT",
    "WMT", "CVX", "CSCO", "ABBV", "TMO",
    "ACN", "COST", "AMGN", "AVGO", "UNP",
    "LIN", "BMY", "HON", "PM", "LMT",
    "IBM", "MMM", "NOW", "CAT", "ORCL"
]

# Define the date range
start_date = '2008-01-01'
end_date = '2022-12-31'

# Create an attractive title with balloon text
st.markdown("""
    <h1 style='text-align: center;'>
        <span title='Welcome to Stock Trend Prediction! 😊' style='cursor: help;'>STOCK TREND PREDICTION</span>
    </h1>
""", unsafe_allow_html=True)
user_input = st.selectbox("Select Stock Ticker", stock_tickers, index=0)

# Fetch stock data using yfinance
df = yf.download(user_input, start=start_date, end=end_date)

# Describing Data
# st.subheader('Data from 2008 - 2022')
# st.write(df.describe())
# Create a Plotly table for data description
# Styling the data description with Pandas Styler
data_description = df.describe().style \
    .set_caption("Data Summary (2008 - 2022)") \
    .set_table_styles([
        {'selector': 'caption',
         'props': [('font-size', '18px'), ('color', 'navy')]}
    ]) \
    .set_table_attributes("style='font-size: 14px; border-collapse: collapse;'") \
    .highlight_max(axis=0, color='lightgreen') \
    .highlight_min(axis=0, color='lightcoral')

# Display the styled data description
st.write(data_description)


# Visualization
# st.markdown("")
st.markdown("<font color='orange' size='5'><b>Closing Price vs. Time Chart with 100MA</b></font>", unsafe_allow_html=True)
ma100 = df['Close'].rolling(100).mean()
fig = px.line(df, x=df.index, y='Close', title=f'{user_input} Stock Price with Moving Averages')
fig.add_scatter(x=df.index, y=ma100, mode='lines', name='100-day MA', line=dict(color='red'))
st.plotly_chart(fig)

st.markdown("<font color='orange' size='5'><b>Closing Price vs. Time Chart with 200MA</b></font>", unsafe_allow_html=True)
ma200 = df['Close'].rolling(200).mean()
fig = px.line(df, x=df.index, y='Close', title=f'{user_input} Stock Price with Moving Averages')
fig.add_scatter(x=df.index, y=ma200, mode='lines', name='200-day MA', line=dict(color='red'))
st.plotly_chart(fig)

# Split the data into training and testing sets
training_data = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
testing_data = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
training_data_array = scaler.fit_transform(training_data)

# Create sequences for training
x_train = []
y_train = []

for i in range(100, training_data_array.shape[0]):
    x_train.append(training_data_array[i - 100: i])
    y_train.append(training_data_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load the model
model = load_model('keras_model.h5')

# Prepare testing data
testing_data = pd.DataFrame(df['Close'][int(len(df) * 0.70) - 100:])
input_data = scaler.fit_transform(testing_data)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)

# Inverse scaling to get original price values
scaler_factor = 1 / 0.00251232
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

# Plotting results using Plotly
st.markdown("<font color='Cyan' size='6'><b>Prediction Vs. Original</b></font>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[int(len(df) * 0.70):], y=y_test, mode='lines', name='Original Price'))
fig.add_trace(go.Scatter(x=df.index[int(len(df) * 0.70):], y=y_predicted.flatten(), mode='lines', name='Predicted Price'))
fig.update_layout(title=f'{user_input} Stock Price Prediction', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig)


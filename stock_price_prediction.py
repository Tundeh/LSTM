import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import tensorflow as tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# from google.colab import drive
# drive.mount('/content/drive')

# ls drive/MyDrive/Personal/LSTM/stock_trading_data.csv

# Load the sample data
data = pd.read_csv('./stock_trading_data.csv')


#print(data.info())

data.loc[[0, 2, 3]]

# Extract the 'Close' price
close_prices = data['Close'].values.reshape(-1, 1)
# Normalize the data between 0 and 1
scaler = MinMaxScaler()
close_prices_scaled = scaler.fit_transform(close_prices)

# Prepare the data for the LSTM model
sequence_length = 10
X, y = [], []
for i in range(len(close_prices_scaled) - sequence_length):
    X.append(close_prices_scaled[i:i + sequence_length])
    y.append(close_prices_scaled[i + sequence_length])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(units=150))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")


y_pred = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

#save model
model.save('drive/MyDrive/Personal/LSTM/Stock_Price_model')

print(f"Test loss: {loss}")

# Plot the predictions against the true values
plt.plot(y_test[:], label='True Prices')
plt.plot(y_pred, label='Predicted Prices')
plt.legend()
plt.show()

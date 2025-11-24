import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Step 1: Data Load aur Preparation ---
df_ts = pd.read_csv(r'C:\Users\Lataisha\Downloads\videostatsTopRecords.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])

# Sabse zyada records wale video ko chunte hain
video_id = df_ts['ytvideoid'].value_counts().idxmax()
video_data = df_ts[df_ts['ytvideoid'] == video_id].set_index('timestamp')['views'].sort_index()

# Data ko scale karna
scaler = MinMaxScaler()
video_data_scaled = scaler.fit_transform(video_data.values.reshape(-1, 1))

# --- Step 2: Sequence Creation Function ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10 # 10 pichle steps ka data lenge

X_ts, y_ts = create_sequences(video_data_scaled, SEQ_LENGTH)
X_ts = X_ts.reshape(X_ts.shape[0], X_ts.shape[1], 1)

# Train/Test Split
train_size = int(len(X_ts) * 0.8)
X_train_ts, X_test_ts = X_ts[:train_size], X_ts[train_size:]
y_train_ts, y_test_ts = y_ts[:train_size], y_ts[train_size:]

# --- Step 3: Simple LSTM Model Design ---
model_dl = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])

# Model compile aur train karna
model_dl.compile(optimizer='adam', loss='mse')
model_dl.fit(X_train_ts, y_train_ts, epochs=10, batch_size=32, verbose=0)

# --- Step 4: Prediction aur Evaluation ---
predicted_scaled = model_dl.predict(X_test_ts)
predicted_views = scaler.inverse_transform(predicted_scaled)
actual_views = scaler.inverse_transform(y_test_ts.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual_views, predicted_views))
print(f"\n--- Deep Learning Model Evaluation ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualization (Save the plot)
plt.figure(figsize=(10, 6))
plt.plot(actual_views, label='Actual Views')
plt.plot(predicted_views, label='Predicted Views')
plt.title(f'Views Forecasting for Video ID: {video_id}')
plt.xlabel('Time Step (Test Data)')
plt.ylabel('Views')
plt.legend()
plt.savefig('lstm_forecast_plot.png')
print("Visualization saved as 'lstm_forecast_plot.png'")
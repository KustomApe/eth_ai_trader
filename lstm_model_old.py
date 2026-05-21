import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

class LSTMPredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.model = None

    def build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.lookback, 5)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # 次の終値を予測
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def prepare_data(self, df):
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        scaled = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback:i])
            y.append(scaled[i, 3])  # close price
        return np.array(X), np.array(y)

    def train(self, df, epochs=10, batch_size=32):
        X, y = self.prepare_data(df)
        if self.model is None:
            self.build_model()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, recent_df):
        features = recent_df[['open', 'high', 'low', 'close', 'volume']].values
        scaled = self.scaler.transform(features)
        X = scaled[-self.lookback:].reshape(1, self.lookback, 5)
        pred_scaled = self.model.predict(X, verbose=0)[0][0]
        # 逆変換用ダミー配列
        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred_scaled
        pred_price = self.scaler.inverse_transform(dummy)[0][3]
        return pred_price

    def get_signal(self, current_price, predicted_price, threshold=0.003):
        change = (predicted_price - current_price) / current_price
        if change > threshold:
            return "LONG", change
        elif change < -threshold:
            return "SHORT", change
        else:
            return "HOLD", change

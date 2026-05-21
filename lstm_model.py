import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import threading
import time
from datetime import datetime
import os

class LSTMPredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        self.last_trained_at = None

        # Auto-retrain用
        self._retrain_thread = None
        self._stop_retrain = threading.Event()
        self.retrain_log = []          # {"time": ..., "loss": ..., "status": ...}
        self.is_retraining = False

    # ───────────────────────────────────────────
    # モデル構築
    # ───────────────────────────────────────────
    def build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.lookback, 5)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    # ───────────────────────────────────────────
    # データ前処理
    # ───────────────────────────────────────────
    def prepare_data(self, df):
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        scaled = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback:i])
            y.append(scaled[i, 3])
        return np.array(X), np.array(y)

    # ───────────────────────────────────────────
    # 学習（手動 or auto-retrain両用）
    # ───────────────────────────────────────────
    def train(self, df, epochs=10, batch_size=32, verbose=1):
        X, y = self.prepare_data(df)
        if self.model is None:
            self.build_model()
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[es]
        )
        self.is_trained = True
        self.last_trained_at = datetime.now()
        final_loss = float(history.history['loss'][-1])
        return final_loss

    # ───────────────────────────────────────────
    # 予測
    # ───────────────────────────────────────────
    def predict(self, recent_df):
        if not self.is_trained:
            raise RuntimeError("モデルが未学習です。先にtrain()を実行してください。")
        features = recent_df[['open', 'high', 'low', 'close', 'volume']].values
        scaled = self.scaler.transform(features)
        X = scaled[-self.lookback:].reshape(1, self.lookback, 5)
        pred_scaled = self.model.predict(X, verbose=0)[0][0]
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

    # ───────────────────────────────────────────
    # Auto-retrain（バックグラウンドスレッド）
    # ───────────────────────────────────────────
    def start_auto_retrain(self, agent, interval_minutes=60):
        """
        interval_minutes ごとに最新データで自動再学習する。
        agent: TradingAgent（OHLCVデータ取得に使用）
        """
        if self._retrain_thread and self._retrain_thread.is_alive():
            return  # すでに動いている

        self._stop_retrain.clear()

        def _loop():
            while not self._stop_retrain.is_set():
                # 指定時間待機（細かく区切ってstopイベントを確認）
                for _ in range(interval_minutes * 60):
                    if self._stop_retrain.is_set():
                        return
                    time.sleep(1)

                self.is_retraining = True
                status = "success"
                loss = None
                try:
                    df = agent.get_ohlcv(limit=500)
                    loss = self.train(df, epochs=5, verbose=0)
                    self.last_trained_at = datetime.now()
                except Exception as e:
                    status = f"error: {e}"
                finally:
                    self.is_retraining = False

                self.retrain_log.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "loss": f"{loss:.6f}" if loss is not None else "—",
                    "status": status
                })
                # ログは直近50件だけ保持
                if len(self.retrain_log) > 50:
                    self.retrain_log = self.retrain_log[-50:]

        self._retrain_thread = threading.Thread(target=_loop, daemon=True)
        self._retrain_thread.start()

    def stop_auto_retrain(self):
        self._stop_retrain.set()

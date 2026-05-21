import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    """
    LSTMPredictorを使ってOHLCVデータ上でバックテストを実行する。
    """

    def __init__(self, initial_balance=1000, risk_pct=0.02,
                 tp_pct=0.010, sl_pct=0.005, leverage=1):
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.leverage = leverage

    # ───────────────────────────────────────────
    # シグナル列の生成（モデル不要のシンプル版も可）
    # ───────────────────────────────────────────
    def generate_signals_from_lstm(self, df, lstm_model, threshold=0.003):
        """
        df全体のシグナル列を生成する。
        学習済みのlstm_modelを使い、ローリング予測でシグナルを作る。
        """
        lookback = lstm_model.lookback
        signals = ["HOLD"] * len(df)
        predicted_prices = [np.nan] * len(df)

        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback:i]
            try:
                pred = lstm_model.predict(window)
                current = df.iloc[i]['close']
                change = (pred - current) / current
                predicted_prices[i] = pred
                if change > threshold:
                    signals[i] = "LONG"
                elif change < -threshold:
                    signals[i] = "SHORT"
                else:
                    signals[i] = "HOLD"
            except Exception:
                pass

        df = df.copy()
        df['signal'] = signals
        df['predicted'] = predicted_prices
        return df

    def generate_signals_simple(self, df, short_window=5, long_window=20):
        """
        LSTMなしで使えるSMA crossoverシグナル（高速バックテスト用）
        """
        df = df.copy()
        df['sma_short'] = df['close'].rolling(short_window).mean()
        df['sma_long'] = df['close'].rolling(long_window).mean()
        df['signal'] = 'HOLD'
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 'LONG'
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = 'SHORT'
        return df

    # ───────────────────────────────────────────
    # バックテスト本体
    # ───────────────────────────────────────────
    def run(self, df):
        """
        df に 'signal' 列が含まれている前提で実行。
        各バーでTP/SLに到達したか判定し、トレードを記録する。
        """
        balance = self.initial_balance
        position = None
        trades = []
        equity_curve = []

        for i, row in df.iterrows():
            price = row['close']
            signal = row.get('signal', 'HOLD')
            ts = row['timestamp']

            # ポジションのTP/SLチェック
            if position is not None:
                side = position['side']
                entry = position['entry']
                tp = position['tp']
                sl = position['sl']
                size = position['size']

                hit_tp = (side == 'LONG' and price >= tp) or \
                         (side == 'SHORT' and price <= tp)
                hit_sl = (side == 'LONG' and price <= sl) or \
                         (side == 'SHORT' and price >= sl)

                if hit_tp or hit_sl:
                    if side == 'LONG':
                        pnl = size * (price - entry) / entry * self.leverage
                    else:
                        pnl = size * (entry - price) / entry * self.leverage

                    balance += pnl
                    trades.append({
                        'open_time': position['open_time'],
                        'close_time': ts,
                        'side': side,
                        'entry': entry,
                        'exit': price,
                        'pnl': pnl,
                        'reason': 'TP' if hit_tp else 'SL',
                        'balance_after': balance
                    })
                    position = None

            # 新規エントリー
            if position is None and signal in ('LONG', 'SHORT'):
                size = balance * self.risk_pct
                if signal == 'LONG':
                    tp = price * (1 + self.tp_pct)
                    sl = price * (1 - self.sl_pct)
                else:
                    tp = price * (1 - self.tp_pct)
                    sl = price * (1 + self.sl_pct)

                position = {
                    'side': signal,
                    'entry': price,
                    'tp': tp,
                    'sl': sl,
                    'size': size,
                    'open_time': ts
                }

            equity_curve.append({'timestamp': ts, 'balance': balance})

        # 未決済ポジションを強制クローズ
        if position is not None:
            last_price = df.iloc[-1]['close']
            side = position['side']
            entry = position['entry']
            size = position['size']
            if side == 'LONG':
                pnl = size * (last_price - entry) / entry * self.leverage
            else:
                pnl = size * (entry - last_price) / entry * self.leverage
            balance += pnl
            trades.append({
                'open_time': position['open_time'],
                'close_time': df.iloc[-1]['timestamp'],
                'side': side,
                'entry': entry,
                'exit': last_price,
                'pnl': pnl,
                'reason': 'FORCE_CLOSE',
                'balance_after': balance
            })

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)

        return {
            'trades': trades_df,
            'equity_curve': equity_df,
            'stats': self._calc_stats(trades_df, equity_df)
        }

    # ───────────────────────────────────────────
    # 統計計算
    # ───────────────────────────────────────────
    def _calc_stats(self, trades_df, equity_df):
        if trades_df.empty:
            return {}

        total_trades = len(trades_df)
        win_trades = (trades_df['pnl'] > 0).sum()
        loss_trades = (trades_df['pnl'] <= 0).sum()
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if loss_trades > 0 else 0
        profit_factor = abs(avg_win * win_trades / (avg_loss * loss_trades)) \
                        if avg_loss != 0 and loss_trades > 0 else float('inf')

        # 最大ドローダウン
        eq = equity_df['balance'].values
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak * 100
        max_dd = drawdown.min()

        # シャープレシオ（簡易）
        returns = equity_df['balance'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) \
                 if returns.std() > 0 else 0

        final_balance = equity_df['balance'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100

        return {
            'total_trades': total_trades,
            'win_trades': int(win_trades),
            'loss_trades': int(loss_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 4),
            'total_return_pct': round(total_return, 2),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'profit_factor': round(profit_factor, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 3),
            'final_balance': round(final_balance, 4)
        }

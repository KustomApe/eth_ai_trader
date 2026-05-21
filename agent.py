import ccxt
import os
import time
from datetime import datetime

class TradingAgent:
    def __init__(self, initial_balance=1000, leverage=1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.position = None  # {"side": "LONG"/"SHORT", "entry": price, "size": usdt, "tp": price, "sl": price}
        self.trade_history = []
        self.pnl = 0.0

        # Binance Testnet（実運用時はTestnetをFalseに）
        self.exchange = ccxt.binance({
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_SECRET"),
            'options': {'defaultType': 'future'},
            'urls': {
                'api': {
                    'public': 'https://testnet.binancefuture.com',
                    'private': 'https://testnet.binancefuture.com',
                }
            }
        })

    def get_ohlcv(self, symbol='ETH/USDT', timeframe='5m', limit=120):
        """OHLCVデータを取得"""
        import pandas as pd
        bars = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_current_price(self, symbol='ETH/USDT'):
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    def open_position(self, side, price, risk_pct=0.02):
        """ポジションを開く（TP: 1%, SL: 0.5%）"""
        if self.position is not None:
            return False

        size = self.balance * risk_pct
        if side == "LONG":
            tp = price * 1.010
            sl = price * 0.995
        else:
            tp = price * 0.990
            sl = price * 1.005

        self.position = {
            "side": side,
            "entry": price,
            "size": size,
            "tp": tp,
            "sl": sl,
            "open_time": datetime.now().isoformat()
        }
        return True

    def check_exit(self, current_price):
        """TP/SLチェック"""
        if self.position is None:
            return None

        side = self.position["side"]
        tp = self.position["tp"]
        sl = self.position["sl"]
        entry = self.position["entry"]
        size = self.position["size"]

        hit_tp = (side == "LONG" and current_price >= tp) or \
                 (side == "SHORT" and current_price <= tp)
        hit_sl = (side == "LONG" and current_price <= sl) or \
                 (side == "SHORT" and current_price >= sl)

        if hit_tp or hit_sl:
            if side == "LONG":
                pnl = size * (current_price - entry) / entry
            else:
                pnl = size * (entry - current_price) / entry

            self.pnl += pnl
            self.balance += pnl

            result = {
                "side": side,
                "entry": entry,
                "exit": current_price,
                "pnl": pnl,
                "reason": "TP" if hit_tp else "SL",
                "time": datetime.now().isoformat()
            }
            self.trade_history.append(result)
            self.position = None
            return result

        return None

    def get_status(self):
        return {
            "balance": self.balance,
            "pnl": self.pnl,
            "pnl_pct": (self.pnl / self.initial_balance) * 100,
            "position": self.position,
            "trades": len(self.trade_history),
            "win_trades": sum(1 for t in self.trade_history if t["pnl"] > 0),
        }

from openai import OpenAI
import os

class LLMAdvisor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, current_price, predicted_price, lstm_signal, recent_prices, pnl):
        prompt = f"""
You are a crypto trading advisor. Analyze this situation briefly (2-3 sentences max):

- Asset: ETHUSDT
- Current Price: {current_price:.2f}
- LSTM Predicted Price: {predicted_price:.2f}
- LSTM Signal: {lstm_signal}
- Recent prices (last 5): {recent_prices}
- Current PnL: {pnl:.2f} USDT

Provide: 1) Market bias, 2) Key risk, 3) Recommendation (LONG/SHORT/HOLD).
Be concise and direct.
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content

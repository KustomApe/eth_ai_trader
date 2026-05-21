import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from dotenv import load_dotenv
from lstm_model import LSTMPredictor
from llm_advisor import LLMAdvisor
from agent import TradingAgent
from backtest import Backtester

load_dotenv()

st.set_page_config(page_title="AI ETH Trader", layout="wide", page_icon="🤖")

# ───────────────────────────────────────────
# セッション状態の初期化
# ───────────────────────────────────────────
if "agent"   not in st.session_state:
    st.session_state.agent   = TradingAgent(initial_balance=1000)
if "lstm"    not in st.session_state:
    st.session_state.lstm    = LSTMPredictor(lookback=60)
if "advisor" not in st.session_state:
    st.session_state.advisor = LLMAdvisor()
if "log"         not in st.session_state: st.session_state.log         = []
if "llm_comment" not in st.session_state: st.session_state.llm_comment = "—"
if "signal"      not in st.session_state: st.session_state.signal      = "HOLD"
if "predicted"   not in st.session_state: st.session_state.predicted   = 0.0
if "auto_retrain_started" not in st.session_state:
    st.session_state.auto_retrain_started = False

agent   = st.session_state.agent
lstm    = st.session_state.lstm
advisor = st.session_state.advisor

# ───────────────────────────────────────────
# タブ構成
# ───────────────────────────────────────────
tab_live, tab_bt, tab_retrain = st.tabs(["📡 ライブトレード", "📊 バックテスト", "🔄 Auto-Retrain"])

# ════════════════════════════════════════════
# サイドバー（全タブ共通）
# ════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 設定")
    auto_trade  = st.toggle("自動売買を有効化", value=False)
    refresh_sec = st.slider("更新間隔（秒）", 5, 60, 15)
    st.divider()

    if st.button("🧠 LSTMを初回学習", use_container_width=True):
        with st.spinner("データ取得・学習中..."):
            df = agent.get_ohlcv(limit=500)
            loss = lstm.train(df, epochs=5)
        st.success(f"学習完了！ Loss: {loss:.6f}")

    st.divider()
    st.markdown("**免責事項**")
    st.caption("実験的プロトタイプです。投資アドバイスではありません。先物取引は高リスクです。")

# ════════════════════════════════════════════
# TAB 1: ライブトレード
# ════════════════════════════════════════════
with tab_live:
    st.title("🤖 AI ETH/USDT Trader — 1 Agent")
    st.caption("LSTM + Claude (Anthropic) Scalping Strategy | Binance Testnet")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    chart_area  = st.empty()
    advice_area = st.empty()
    pos_area    = st.empty()
    log_area    = st.empty()

    def run_cycle():
        try:
            df_live = agent.get_ohlcv(limit=120)
            current_price = agent.get_current_price()

            if lstm.is_trained and not lstm.is_retraining:
                predicted = lstm.predict(df_live)
                signal, _ = lstm.get_signal(current_price, predicted)
            else:
                predicted = current_price
                signal = "HOLD"

            st.session_state.predicted = predicted
            st.session_state.signal    = signal

            exit_r = agent.check_exit(current_price)
            if exit_r:
                st.session_state.log.append(
                    f"[EXIT] {exit_r['side']} | PnL: {exit_r['pnl']:.2f} USDT | {exit_r['reason']}"
                )

            if auto_trade and signal != "HOLD" and agent.position is None:
                if agent.open_position(signal, current_price):
                    st.session_state.log.append(f"[OPEN] {signal} @ {current_price:.2f}")

            # LLMアドバイス（5サイクルに1回）
            if len(st.session_state.log) % 5 == 0:
                status = agent.get_status()
                recent = df_live['close'].tail(5).tolist()
                st.session_state.llm_comment = advisor.analyze(
                    current_price, predicted, signal, recent, status['pnl']
                )

            return df_live, current_price
        except Exception as e:
            st.error(f"エラー: {e}")
            return None, None

    df_live, current_price = run_cycle()
    status = agent.get_status()

    if current_price:
        kpi1.metric("💰 残高",      f"{status['balance']:.2f} USDT", f"{status['pnl_pct']:+.2f}%")
        kpi2.metric("📈 現在価格",  f"{current_price:.2f}", f"予測: {st.session_state.predicted:.2f}")
        kpi3.metric("🎯 シグナル",  st.session_state.signal)
        kpi4.metric("📊 取引 / 勝", f"{status['trades']}回",
                    f"勝: {status['win_trades']}回" if status['trades'] > 0 else "—")

        with chart_area.container():
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                                subplot_titles=("ETH/USDT 5m Candlestick", "Volume"))
            fig.add_trace(go.Candlestick(
                x=df_live['timestamp'], open=df_live['open'], high=df_live['high'],
                low=df_live['low'],  close=df_live['close'], name="OHLCV"
            ), row=1, col=1)
            if agent.position:
                p = agent.position
                fig.add_hline(y=p['entry'], line_dash="dash",  line_color="yellow", annotation_text="Entry", row=1, col=1)
                fig.add_hline(y=p['tp'],    line_dash="dot",   line_color="green",  annotation_text="TP",    row=1, col=1)
                fig.add_hline(y=p['sl'],    line_dash="dot",   line_color="red",    annotation_text="SL",    row=1, col=1)
            fig.add_trace(go.Bar(x=df_live['timestamp'], y=df_live['volume'],
                                 marker_color='rgba(0,150,255,0.4)', name="Volume"), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        advice_area.info(f"🧠 **Claude AI Advisor:** {st.session_state.llm_comment}")

        with pos_area.container():
            if agent.position:
                p = agent.position
                st.success(f"📌 ポジション中: **{p['side']}** | エントリー: {p['entry']:.2f} | TP: {p['tp']:.2f} | SL: {p['sl']:.2f}")
            else:
                st.warning("📌 ポジションなし")

        with log_area.container():
            st.subheader("📋 トレードログ")
            for entry in reversed(st.session_state.log[-20:]):
                st.text(entry)

    if auto_trade:
        time.sleep(refresh_sec)
        st.rerun()

# ════════════════════════════════════════════
# TAB 2: バックテスト
# ════════════════════════════════════════════
with tab_bt:
    st.header("📊 バックテスト")

    c1, c2, c3 = st.columns(3)
    with c1:
        bt_bars   = st.number_input("取得バー数", 100, 2000, 500, step=100)
        bt_tf     = st.selectbox("時間足", ["1m", "5m", "15m", "1h"], index=1)
    with c2:
        bt_tp     = st.slider("TP (%)", 0.5, 5.0, 1.0, 0.1)
        bt_sl     = st.slider("SL (%)", 0.2, 3.0, 0.5, 0.1)
    with c3:
        bt_risk   = st.slider("リスク / トレード (%)", 1, 10, 2)
        bt_mode   = st.radio("シグナルモード", ["SMA Crossover（高速）", "LSTM（要学習済み）"])

    run_bt = st.button("▶ バックテスト実行", use_container_width=True, type="primary")

    if run_bt:
        with st.spinner("データ取得・バックテスト実行中..."):
            try:
                df_bt = agent.get_ohlcv(symbol='ETH/USDT', timeframe=bt_tf, limit=bt_bars)
                bt = Backtester(
                    initial_balance=1000,
                    risk_pct=bt_risk / 100,
                    tp_pct=bt_tp / 100,
                    sl_pct=bt_sl / 100
                )

                if "LSTM" in bt_mode:
                    if not lstm.is_trained:
                        st.error("LSTMが未学習です。サイドバーから先に学習してください。")
                        st.stop()
                    df_bt = bt.generate_signals_from_lstm(df_bt, lstm)
                else:
                    df_bt = bt.generate_signals_simple(df_bt)

                result = bt.run(df_bt)
                stats  = result['stats']
                trades = result['trades']
                equity = result['equity_curve']

                # KPI
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("総リターン",    f"{stats.get('total_return_pct', 0):.2f}%")
                s2.metric("勝率",          f"{stats.get('win_rate', 0):.1f}%")
                s3.metric("最大DD",        f"{stats.get('max_drawdown_pct', 0):.2f}%")
                s4.metric("プロフィットF", f"{stats.get('profit_factor', 0):.3f}")

                s5, s6, s7, s8 = st.columns(4)
                s5.metric("総トレード数",  stats.get('total_trades', 0))
                s6.metric("勝トレード",    stats.get('win_trades', 0))
                s7.metric("負トレード",    stats.get('loss_trades', 0))
                s8.metric("シャープレシオ",f"{stats.get('sharpe_ratio', 0):.3f}")

                # エクイティカーブ
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=equity['timestamp'], y=equity['balance'],
                    fill='tozeroy', line=dict(color='cyan'), name="残高"
                ))
                fig_eq.update_layout(title="エクイティカーブ", template="plotly_dark",
                                     height=350, yaxis_title="USDT")
                st.plotly_chart(fig_eq, use_container_width=True)

                # PnL分布
                if not trades.empty:
                    fig_pnl = go.Figure()
                    colors = ['green' if p > 0 else 'red' for p in trades['pnl']]
                    fig_pnl.add_trace(go.Bar(
                        x=list(range(len(trades))), y=trades['pnl'],
                        marker_color=colors, name="PnL per trade"
                    ))
                    fig_pnl.update_layout(title="トレード別PnL", template="plotly_dark",
                                          height=300, yaxis_title="USDT")
                    st.plotly_chart(fig_pnl, use_container_width=True)

                    # トレード一覧
                    st.subheader("トレード一覧")
                    st.dataframe(
                        trades[['open_time','close_time','side','entry','exit','pnl','reason','balance_after']]
                        .style.applymap(lambda v: 'color: lightgreen' if isinstance(v, float) and v > 0
                                        else ('color: salmon' if isinstance(v, float) and v < 0 else ''),
                                        subset=['pnl']),
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"バックテストエラー: {e}")

# ════════════════════════════════════════════
# TAB 3: Auto-Retrain
# ════════════════════════════════════════════
with tab_retrain:
    st.header("🔄 Auto-Retrain 管理")

    ar1, ar2 = st.columns(2)
    with ar1:
        retrain_interval = st.number_input("再学習間隔（分）", 10, 1440, 60, step=10)

    with ar2:
        if lstm.last_trained_at:
            st.metric("最終学習時刻", lstm.last_trained_at.strftime("%H:%M:%S"))
        else:
            st.metric("最終学習時刻", "未学習")

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button("▶ Auto-Retrain 開始", use_container_width=True, type="primary",
                     disabled=st.session_state.auto_retrain_started):
            if not lstm.is_trained:
                st.warning("先にサイドバーから初回学習を行ってください。")
            else:
                lstm.start_auto_retrain(agent, interval_minutes=retrain_interval)
                st.session_state.auto_retrain_started = True
                st.success(f"Auto-Retrain 開始しました（{retrain_interval}分ごと）")

    with col_stop:
        if st.button("⏹ Auto-Retrain 停止", use_container_width=True,
                     disabled=not st.session_state.auto_retrain_started):
            lstm.stop_auto_retrain()
            st.session_state.auto_retrain_started = False
            st.info("Auto-Retrain を停止しました。")

    # 現在のステータス
    st.divider()
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if lstm.is_retraining:
            st.warning("⏳ 現在再学習中...")
        elif st.session_state.auto_retrain_started:
            st.success("✅ Auto-Retrain 稼働中")
        else:
            st.info("⏸ Auto-Retrain 停止中")

    with status_col2:
        if st.button("🔄 画面を更新", use_container_width=True):
            st.rerun()

    # 再学習ログ
    st.subheader("再学習ログ")
    if lstm.retrain_log:
        log_df = pd.DataFrame(lstm.retrain_log[::-1])  # 新しい順
        st.dataframe(
            log_df,
            column_config={
                "time":   st.column_config.TextColumn("実行時刻"),
                "loss":   st.column_config.TextColumn("Loss"),
                "status": st.column_config.TextColumn("ステータス"),
            },
            use_container_width=True
        )
    else:
        st.info("再学習はまだ実行されていません。")

    # 手動で今すぐ再学習
    st.divider()
    st.subheader("手動再学習")
    if st.button("⚡ 今すぐ再学習する", use_container_width=True):
        with st.spinner("再学習中..."):
            df_now = agent.get_ohlcv(limit=500)
            loss = lstm.train(df_now, epochs=5)
            lstm.retrain_log.append({
                "time": lstm.last_trained_at.strftime("%Y-%m-%d %H:%M:%S"),
                "loss": f"{loss:.6f}",
                "status": "manual"
            })
        st.success(f"再学習完了！ Loss: {loss:.6f}")
        st.rerun()

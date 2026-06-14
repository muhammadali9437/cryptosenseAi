# ============================================================
#  predict_multi.py
#  Run: python predict_multi.py
#  Kaam: Sab intervals ka live signal ek saath dikhata hai
# ============================================================

import pandas as pd
import numpy as np
import requests
import joblib
import time
import warnings
warnings.filterwarnings("ignore")

from dataset import add_features   # tumhara dataset.py same folder mein ho

ALL_INTERVALS = ["15m", "30m", "2h", "4h", "6h", "8h", "12h"]

# ── User interval selection ───────────────────────────────────
def select_intervals():
    print("\n" + "═"*70)
    print("  BTC PRICE PREDICTION — TIMEFRAME SELECTOR")
    print("═"*70)
    print("\n  Available Timeframes:")
    for i, tf in enumerate(ALL_INTERVALS, 1):
        print(f"    {i}. {tf}")
    print(f"    {len(ALL_INTERVALS)+1}. Select ALL timeframes")
    print(f"    0. Exit")
    
    print("\n  Apni marzi se intervals select karo (comma se alag karo)")
    print("  Example: 1,2,4  ya  8 (sab k liye)")
    choice = input("\n  Enter: ").strip()
    
    if choice == "0":
        print("\n  Goodbye!")
        exit()
    
    if choice == str(len(ALL_INTERVALS)+1) or choice.lower() == "8" or choice.lower() == "all":
        return ALL_INTERVALS
    
    try:
        indices = [int(x.strip())-1 for x in choice.split(",")]
        selected = [ALL_INTERVALS[i] for i in indices if 0 <= i < len(ALL_INTERVALS)]
        if not selected:
            print("\n  ❌ Invalid selection! Sab intervals use honge.")
            return ALL_INTERVALS
        return selected
    except:
        print("\n  ❌ Invalid input! Sab intervals use honge.")
        return ALL_INTERVALS

# ── Live candles fetch ────────────────────────────────────────
def get_live_candles(interval, limit=300):
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    r      = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades",
        "taker_buy_base","taker_buy_quote","_"
    ]
    df = pd.DataFrame(r.json(), columns=cols).drop(columns=["_"])
    for col in ["open","high","low","close","volume",
                "quote_volume","taker_buy_base","taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

# ── OBV/VWAP per interval fresh calculate ─────────────────────
def fix_cumulative(df):
    df  = df.copy()
    c   = df["close"]
    v   = df["volume"]
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"]           = obv
    df["obv_sma_14"]    = obv.rolling(14).mean()
    df["obv_trend"]     = np.sign(obv - obv.rolling(14).mean())
    df["vwap"]          = df["quote_volume"].cumsum() / (v.cumsum() + 1e-9)
    df["close_vs_vwap"] = (c - df["vwap"]) / df["vwap"] * 100
    return df

# ── Main prediction ───────────────────────────────────────────
def predict_all(selected_intervals):
    print("\nFetching live BTC data...\n")

    signals   = {}
    btc_price = None

    for interval in selected_intervals:
        try:
            # Limit — zyada data chahiye features k liye
            limit = 300 if interval in ["2h","4h","6h","8h","12h"] else 500

            df_live = get_live_candles(interval, limit=limit)
            df_feat = add_features(df_live, interval)
            df_feat = fix_cumulative(df_feat)
            df_feat.dropna(subset=["ema_200","rsi_14","atr_14"], inplace=True)
            df_feat.reset_index(drop=True, inplace=True)

            if len(df_feat) == 0:
                raise ValueError("No rows after dropna")

            # Models load
            model    = joblib.load(f"model_{interval}.pkl")
            scaler   = joblib.load(f"scaler_{interval}.pkl")
            FEATURES = joblib.load(f"features_{interval}.pkl")

            # Last candle prediction
            last_row = df_feat[FEATURES].iloc[[-1]]
            last_sc  = scaler.transform(last_row)

            prob   = model.predict_proba(last_sc)[0][1]
            pred   = int(model.predict(last_sc)[0])
            signal = "BULLISH" if pred == 1 else "BEARISH"

            # Extra indicators
            last = df_feat.iloc[-1]
            signals[interval] = {
                "signal"    : signal,
                "prob"      : prob,
                "rsi"       : last["rsi_14"],
                "macd_hist" : last["macd_hist"],
                "adx"       : last["adx_14"],
                "bb_pct"    : last["bb_pct_20"],
                "vol_ratio" : last["volume_ratio_14"],
                "trend_dir" : last["trend_direction"],
            }

            if btc_price is None:
                btc_price = last["close"]

            time.sleep(0.3)

        except Exception as e:
            signals[interval] = {"signal": "ERROR", "prob": 0.5, "error": str(e)}

    # ── Print results ─────────────────────────────────────
    print(f"  BTC Price : ${btc_price:,.2f}")
    print(f"\n{'═'*70}")
    print(f"  {'TF':<6} {'Signal':<10} {'Conf':>7} "
          f"{'RSI':>6} {'ADX':>6} {'BB%':>6} {'VolR':>6}  Result")
    print(f"  {'-'*68}")

    bull_votes  = 0
    bear_votes  = 0
    total_prob  = 0
    valid_count = 0

    for interval, s in signals.items():
        if s["signal"] == "ERROR":
            print(f"  {interval:<6} ERROR — {s.get('error','')[:40]}")
            continue

        marker = "✅" if s["signal"] == "BULLISH" else "❌"
        conf   = s["prob"]
        rsi    = s["rsi"]
        adx    = s["adx"]
        bb     = s["bb_pct"]
        volr   = s["vol_ratio"]

        print(f"  {interval:<6} {s['signal']:<10} {conf:>6.1%} "
              f"{rsi:>6.1f} {adx:>6.1f} {bb:>6.2f} {volr:>6.2f}  {marker}")

        if s["signal"] == "BULLISH":
            bull_votes += 1
        else:
            bear_votes += 1

        total_prob  += conf
        valid_count += 1

    # ── Combined signal ───────────────────────────────────
    avg_conf = total_prob / valid_count if valid_count > 0 else 0.5

    print(f"\n{'═'*70}")
    print(f"  Bullish votes : {bull_votes}/{valid_count}")
    print(f"  Bearish votes : {bear_votes}/{valid_count}")
    print(f"  Avg Confidence: {avg_conf:.1%}")
    print(f"{'═'*70}")

    # Signal strength
    if bull_votes == valid_count:
        strength = "🔥 VERY STRONG BULLISH"
        advice   = "Sab 7 timeframes agree — maximum confidence"
    elif bull_votes >= 5:
        strength = "💚 STRONG BULLISH"
        advice   = "5-6 timeframes bullish — good signal"
    elif bull_votes == 4:
        strength = "🟡 MODERATE BULLISH"
        advice   = "4 timeframes bullish — carefully enter"
    elif bear_votes >= 5:
        strength = "🔴 STRONG BEARISH"
        advice   = "5-6 timeframes bearish — downside likely"
    elif bear_votes == valid_count:
        strength = "💀 VERY STRONG BEARISH"
        advice   = "Sab 7 timeframes bearish — avoid long"
    else:
        strength = "⚪ NEUTRAL — NO TRADE"
        advice   = "Mixed signals — wait karo"

    print(f"\n  FINAL SIGNAL  : {strength}")
    print(f"  ADVICE        : {advice}")
    print(f"{'═'*70}\n")

    # ── Indicator summary ─────────────────────────────────
    print("  INDICATOR GUIDE:")
    print("  RSI < 30  = Oversold (buy zone)")
    print("  RSI > 70  = Overbought (sell zone)")
    print("  ADX > 25  = Strong trend active")
    print("  BB% > 1.0 = Price above upper band (overbought)")
    print("  BB% < 0.0 = Price below lower band (oversold)")
    print("  VolR > 2  = Volume spike — strong move likely")


if __name__ == "__main__":
    selected_intervals = select_intervals()
    predict_all(selected_intervals)
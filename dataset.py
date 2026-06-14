"""
╔══════════════════════════════════════════════════════════════════╗
║         BTC Dataset Fetcher — VS Code Ready                     ║
║         Binance Public API  |  No API Key Needed                ║
║                                                                  ║
║  SETUP (run once in VS Code terminal):                          ║
║      pip install pandas numpy requests                          ║
║                                                                  ║
║  RUN:                                                            ║
║      python dataset.py                                          ║
║                                                                  ║
║  OUTPUT:                                                         ║
║      BTCUSDT_ALL_INTERVALS_4yr.csv                             ║
║                                                                  ║
║  INTERVALS:                                                      ║
║      15m (~140,000 rows) — 2-3 ghante download time            ║
║      30m (~70,000 rows)  — 1-2 ghante download time            ║
║      2h, 4h, 6h, 8h, 12h — 10-15 minute                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

# ════════════════════════════════════════════════════════
#  CONFIG — sirf yahan changes karo
# ════════════════════════════════════════════════════════
SYMBOL      = "BTCUSDT"
INTERVALS   = ["15m", "30m", "2h", "4h", "6h", "8h", "12h"]  # 15m aur 30m add
YEARS_BACK  = 4
BASE_URL    = "https://api.binance.com"
OUTPUT_FILE = "BTCUSDT_ALL_INTERVALS_4yr.csv"
LIMIT       = 1000
SLEEP_SEC   = 0.25  # thoda fast — Binance rate limit safe hai


# ════════════════════════════════════════════════════════
#  STEP 1 — BINANCE SE DATA LENA
# ════════════════════════════════════════════════════════

def fetch_klines(symbol, interval, start_ms, end_ms):
    url     = f"{BASE_URL}/api/v3/klines"
    candles = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current,
            "endTime":   end_ms,
            "limit":     LIMIT,
        }
        # Retry logic — network error pe 5 second wait karke dobara try karega
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"\n  Attempt {attempt+1} failed: {e} — retrying in 5s...")
                time.sleep(5)
        else:
            print(f"\n  3 attempts failed for [{interval}] — skipping batch")
            break

        if not data:
            break

        candles.extend(data)
        current = data[-1][6] + 1
        time.sleep(SLEEP_SEC)
        print(f"   [{interval}]  {len(candles):>8,} candles  —  "
              f"upto {pd.to_datetime(data[-1][0], unit='ms').date()}", end="\r")

    print()
    return candles


def raw_to_df(candles):
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades",
        "taker_buy_base","taker_buy_quote","_"
    ]
    df = pd.DataFrame(candles, columns=cols).drop(columns=["_"])
    for col in ["open","high","low","close","volume",
                "quote_volume","taker_buy_base","taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.drop_duplicates("open_time", inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ════════════════════════════════════════════════════════
#  STEP 2 — FEATURES BANANA
# ════════════════════════════════════════════════════════

def add_features(df, interval):
    # df.copy() se fragmentation warning band hogi
    df   = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # Sab features pehle dict mein banao — phir ek saath concat karo
    feat = {}

    feat["coin"]     = "BTC"
    feat["interval"] = interval

    # — Candle shape —
    feat["price_change"]     = c - o
    feat["price_change_pct"] = (c - o) / o * 100
    feat["hl_range"]         = h - l
    feat["hl_range_pct"]     = (h - l) / o * 100
    body                     = (c - o).abs()
    uw                       = h - pd.concat([o,c], axis=1).max(axis=1)
    lw                       = pd.concat([o,c], axis=1).min(axis=1) - l
    feat["body_size"]        = body
    feat["body_pct"]         = body / o * 100
    feat["upper_wick"]       = uw
    feat["lower_wick"]       = lw
    feat["upper_wick_pct"]   = uw / o * 100
    feat["lower_wick_pct"]   = lw / o * 100
    feat["is_bullish"]       = (c > o).astype(int)
    feat["is_doji"]          = (feat["body_pct"] < 0.1).astype(int)
    feat["is_hammer"]        = ((lw > 2*body) & (uw < body)).astype(int)
    feat["is_shooting_star"] = ((uw > 2*body) & (lw < body)).astype(int)

    # — Moving averages —
    for n in [7,14,21,50,100,200]:
        feat[f"sma_{n}"] = c.rolling(n).mean()
        feat[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()
    w14 = np.arange(1, 15)
    feat["wma_14"] = c.rolling(14).apply(
        lambda x: np.dot(x, w14)/w14.sum(), raw=True)
    for n in [7,21,50,200]:
        sma = c.rolling(n).mean()
        feat[f"close_vs_sma{n}"] = (c - sma) / sma * 100
    sma7   = c.rolling(7).mean()
    sma21  = c.rolling(21).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    feat["sma7_vs_sma21"]  = (sma7  - sma21)  / sma21  * 100
    feat["sma21_vs_sma50"] = (sma21 - sma50)  / sma50  * 100
    feat["golden_cross"]   = ((sma50 > sma200) &
                               (sma50.shift(1) <= sma200.shift(1))).astype(int)
    feat["death_cross"]    = ((sma50 < sma200) &
                               (sma50.shift(1) >= sma200.shift(1))).astype(int)

    # — RSI —
    def rsi(s, p):
        d  = s.diff()
        g  = d.clip(lower=0).rolling(p).mean()
        ls = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / ls.replace(0, np.nan))
    for n in [7, 14, 21]:
        feat[f"rsi_{n}"] = rsi(c, n)
    feat["rsi_14_overbought"] = (feat["rsi_14"] > 70).astype(int)
    feat["rsi_14_oversold"]   = (feat["rsi_14"] < 30).astype(int)

    # — MACD —
    e12      = c.ewm(span=12, adjust=False).mean()
    e26      = c.ewm(span=26, adjust=False).mean()
    macd     = e12 - e26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    feat["macd"]            = macd
    feat["macd_signal"]     = macd_sig
    feat["macd_hist"]       = macd - macd_sig
    feat["macd_cross_up"]   = ((macd > macd_sig) &
                                (macd.shift(1) <= macd_sig.shift(1))).astype(int)
    feat["macd_cross_down"] = ((macd < macd_sig) &
                                (macd.shift(1) >= macd_sig.shift(1))).astype(int)

    # — Stochastic —
    for kp, dp in [(9,3),(14,3)]:
        lo = l.rolling(kp).min()
        hi = h.rolling(kp).max()
        k  = 100*(c-lo)/(hi-lo+1e-9)
        feat[f"stoch_k_{kp}"] = k
        feat[f"stoch_d_{kp}"] = k.rolling(dp).mean()
    feat["williams_r_14"] = -100*(h.rolling(14).max()-c) / (
        h.rolling(14).max()-l.rolling(14).min()+1e-9)
    tp = (h+l+c)/3
    feat["cci_14"] = (tp - tp.rolling(14).mean()) / (
        0.015 * tp.rolling(14).apply(
            lambda x: np.mean(np.abs(x-x.mean())), raw=True) + 1e-9)
    for n in [5,7,14,21]:
        feat[f"roc_{n}"] = (c - c.shift(n)) / c.shift(n) * 100

    # — ATR —
    def atr(p):
        tr = pd.concat([
            h-l,
            (h-c.shift()).abs(),
            (l-c.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(p).mean()
    for n in [7,14]:
        feat[f"atr_{n}"]     = atr(n)
        feat[f"atr_pct_{n}"] = feat[f"atr_{n}"] / c * 100

    # — Bollinger Bands —
    for n in [14,20]:
        mid = c.rolling(n).mean()
        std = c.rolling(n).std()
        feat[f"bb_upper_{n}"] = mid + 2*std
        feat[f"bb_lower_{n}"] = mid - 2*std
        feat[f"bb_mid_{n}"]   = mid
        feat[f"bb_width_{n}"] = (4*std) / mid * 100
        feat[f"bb_pct_{n}"]   = (c-(mid-2*std)) / (4*std+1e-9)
    feat["bb_squeeze"] = (
        feat["bb_width_20"] < feat["bb_width_20"].rolling(20).mean()*0.8
    ).astype(int)
    feat["garman_klass_vol_14"] = np.sqrt(
        (0.5*np.log(h/l)**2 -
         (2*np.log(2)-1)*np.log(c/o)**2).rolling(14).mean())
    lr = np.log(c / c.shift(1))
    for n in [7,14,21,30]:
        feat[f"rolling_vol_{n}"] = lr.rolling(n).std() * np.sqrt(n)
        feat[f"rolling_ret_{n}"] = c.pct_change(n) * 100

    # — ADX —
    p   = 14
    tr2 = pd.concat([
        h-l,
        (h-c.shift()).abs(),
        (l-c.shift()).abs()
    ], axis=1).max(axis=1)
    pdm = (h-h.shift()).clip(lower=0)
    mdm = (l.shift()-l).clip(lower=0)
    pdm[pdm < mdm] = 0
    mdm[mdm < pdm] = 0
    ae  = tr2.ewm(alpha=1/p, adjust=False).mean()
    pdi = 100*pdm.ewm(alpha=1/p, adjust=False).mean()/(ae+1e-9)
    mdi = 100*mdm.ewm(alpha=1/p, adjust=False).mean()/(ae+1e-9)
    dx  = 100*(pdi-mdi).abs()/(pdi+mdi+1e-9)
    feat["adx_14"]           = dx.ewm(alpha=1/p, adjust=False).mean()
    feat["plus_di_14"]       = pdi
    feat["minus_di_14"]      = mdi
    feat["adx_strong_trend"] = (feat["adx_14"] > 25).astype(int)
    feat["trend_direction"]  = np.where(pdi > mdi, 1, -1)

    # — Volume / Order Flow —
    feat["taker_buy_ratio"]   = df["taker_buy_base"] / (v+1e-9)
    feat["taker_sell_ratio"]  = 1 - feat["taker_buy_ratio"]
    feat["buy_sell_pressure"] = feat["taker_buy_ratio"] - feat["taker_sell_ratio"]
    feat["trade_intensity"]   = df["num_trades"] / (v+1e-9)
    feat["avg_trade_size"]    = v / (df["num_trades"]+1e-9)
    for n in [7,14,21]:
        feat[f"volume_sma_{n}"] = v.rolling(n).mean()
    feat["volume_ratio_7"]  = v / (feat["volume_sma_7"]+1e-9)
    feat["volume_ratio_14"] = v / (feat["volume_sma_14"]+1e-9)
    feat["volume_spike"]    = (feat["volume_ratio_14"] > 2.0).astype(int)

    # OBV — fresh per interval (cumulative fix)
    obv              = (np.sign(c.diff())*v).fillna(0).cumsum()
    feat["obv"]      = obv
    feat["obv_sma_14"] = obv.rolling(14).mean()
    feat["obv_trend"]  = np.sign(obv - obv.rolling(14).mean())

    # VWAP — fresh per interval
    feat["vwap"]          = df["quote_volume"].cumsum() / (v.cumsum()+1e-9)
    feat["close_vs_vwap"] = (c - feat["vwap"]) / feat["vwap"] * 100

    # — Support / Resistance —
    pp = (h.shift(1)+l.shift(1)+c.shift(1)) / 3
    feat["pivot_point"]    = pp
    feat["resistance_1"]   = 2*pp - l.shift(1)
    feat["support_1"]      = 2*pp - h.shift(1)
    feat["resistance_2"]   = pp + (h.shift(1)-l.shift(1))
    feat["support_2"]      = pp - (h.shift(1)-l.shift(1))
    feat["close_vs_pivot"] = (c - pp) / pp * 100
    feat["close_vs_r1"]    = (c - feat["resistance_1"]) / feat["resistance_1"] * 100
    feat["close_vs_s1"]    = (c - feat["support_1"])    / feat["support_1"]    * 100

    # — Lag features —
    for lag in [1,2,3,5,7,14]:
        feat[f"close_lag_{lag}"]        = c.shift(lag)
        feat[f"close_return_lag_{lag}"] = c.pct_change(lag)*100
        feat[f"volume_lag_{lag}"]       = v.shift(lag)
    for lag in [1,2,3]:
        feat[f"rsi_14_lag_{lag}"] = feat["rsi_14"].shift(lag)
        feat[f"atr_14_lag_{lag}"] = feat[f"atr_14"].shift(lag)

    # — Time features —
    dt = df["open_time"]
    feat["hour"]           = dt.dt.hour
    feat["day_of_week"]    = dt.dt.dayofweek
    feat["day_of_month"]   = dt.dt.day
    feat["week_of_year"]   = dt.dt.isocalendar().week.astype(int).values
    feat["month"]          = dt.dt.month
    feat["quarter"]        = dt.dt.quarter
    feat["year"]           = dt.dt.year
    feat["is_weekend"]     = (feat["day_of_week"] >= 5).astype(int)
    feat["is_month_end"]   = dt.dt.is_month_end.astype(int)
    feat["is_month_start"] = dt.dt.is_month_start.astype(int)
    feat["hour_sin"]       = np.sin(2*np.pi*feat["hour"]/24)
    feat["hour_cos"]       = np.cos(2*np.pi*feat["hour"]/24)
    feat["dow_sin"]        = np.sin(2*np.pi*feat["day_of_week"]/7)
    feat["dow_cos"]        = np.cos(2*np.pi*feat["day_of_week"]/7)
    feat["month_sin"]      = np.sin(2*np.pi*feat["month"]/12)
    feat["month_cos"]      = np.cos(2*np.pi*feat["month"]/12)

    # — Targets —
    for n in [1,3,7,14]:
        fut = c.shift(-n)/c - 1
        feat[f"target_{n}"]     = (fut > 0).astype(float)
        feat[f"target_pct_{n}"] = fut * 100
    r1 = c.shift(-1)/c - 1
    feat["market_state"] = pd.cut(
        r1, bins=[-np.inf,-0.01,0.01,np.inf], labels=[0,1,2]
    ).astype(float)

    # Sab features ek saath concat — PerformanceWarning nahi aayegi
    result = pd.concat([
        df[["open_time","open","high","low","close","volume",
            "close_time","quote_volume","num_trades",
            "taker_buy_base","taker_buy_quote"]],
        pd.DataFrame(feat, index=df.index)
    ], axis=1)

    return result


# ════════════════════════════════════════════════════════
#  STEP 3 — MAIN
# ════════════════════════════════════════════════════════

def main():
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((pd.Timestamp.now(tz="UTC") -
                    pd.DateOffset(years=YEARS_BACK)).timestamp() * 1000)

    # Expected rows per interval — progress guide
    expected = {
        "15m": 140_000,
        "30m":  70_000,
        "2h":   17_500,
        "4h":    8_750,
        "6h":    5_800,
        "8h":    4_380,
        "12h":   2_920,
    }

    all_dfs = []

    for interval in INTERVALS:
        print(f"\n{'─'*55}")
        print(f"  Downloading  BTCUSDT  [{interval}]  —  past {YEARS_BACK} years")
        print(f"  Expected ~{expected.get(interval, '?'):,} candles")
        print(f"{'─'*55}")

        raw    = fetch_klines(SYMBOL, interval, start_ms, now_ms)
        df_raw = raw_to_df(raw)
        print(f"  Raw candles  : {len(df_raw):,}")

        df_feat = add_features(df_raw, interval)
        df_feat.dropna(subset=["ema_200","rsi_14","atr_14"], inplace=True)
        df_feat.reset_index(drop=True, inplace=True)

        print(f"  After cleanup: {len(df_feat):,} rows  x  {len(df_feat.columns)} cols")
        all_dfs.append(df_feat)

    # Sab intervals ek CSV mein merge karo
    print(f"\n{'═'*55}")
    print("  Merging all intervals into one CSV ...")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values(["interval","open_time"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\n  ✅  File saved  : {OUTPUT_FILE}")
    print(f"  Total rows     : {len(combined):,}")
    print(f"  Total columns  : {len(combined.columns)}")
    print(f"\n  Interval breakdown:")
    print(f"  {'Interval':<10} {'Rows':>8}  {'Date start':<12}  {'Date end'}")
    print(f"  {'─'*50}")
    for iv, grp in combined.groupby("interval"):
        d0 = grp["open_time"].min().date()
        d1 = grp["open_time"].max().date()
        print(f"  {iv:<10} {len(grp):>8,}  {str(d0):<12}  {d1}")

    print(f"\n  Total download complete!")
    print(f"  Ab train_all.py chalao model train karne k liye.")


if __name__ == "__main__":
    main()
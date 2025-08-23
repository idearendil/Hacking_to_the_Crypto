import os
import pandas as pd
from tqdm import tqdm
import ta
from sklearn.preprocessing import StandardScaler
import numpy as np

INPUT_DIR = "data_min15_350000"
OUTPUT_DIR = "G:/preprocessed_data_min15_350000"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDLES_PER_DAY = 96
SMA_WINDOWS = [20, 50, 100, 200]

# -------------------------
# 1단계: 모든 코인의 summary feature 모으기
# -------------------------
all_summaries = []
file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for file_name in tqdm(file_list, desc="Collecting summaries"):
    file_path = os.path.join(INPUT_DIR, file_name)
    df = pd.read_csv(file_path)
    summary = {
        "coin": file_name,
        "mean_close": df["close"].mean(),
        "std_close": df["close"].std(),
        "min_close": df["close"].min(),
        "max_close": df["close"].max(),
        "mean_volume": df["volume"].mean(),
        "std_volume": df["volume"].std(),
        "min_volume": df["volume"].min(),
        "max_volume": df["volume"].max(),
    }
    all_summaries.append(summary)

summary_df = pd.DataFrame(all_summaries)

# -------------------------
# summary feature 정규화 (z-score)
# -------------------------
scaler = StandardScaler()
summary_features = ["mean_close","std_close","min_close","max_close",
                    "mean_volume","std_volume","min_volume","max_volume"]
scaled_values = scaler.fit_transform(summary_df[summary_features])
summary_scaled = pd.DataFrame(scaled_values, columns=[f"norm_{c}" for c in summary_features])
summary_df = pd.concat([summary_df, summary_scaled], axis=1)

# coin -> 정규화된 summary dict 매핑
summary_map = summary_df.set_index("coin").to_dict(orient="index")

# -------------------------
# 2단계: 각 코인별 전처리
# -------------------------
for file_name in tqdm(file_list, desc="Processing coins"):
    file_path = os.path.join(INPUT_DIR, file_name)
    df = pd.read_csv(file_path)
    if len(df) < 96 * 200:
        continue

    coin_summary = summary_map[file_name]

    # -------------------------
    # 1) 원본 가격 기반 feature 계산
    # -------------------------
    df['close_open_pct'] = ((df['close'] - df['open']) / df['open'])
    df['high_open_pct'] = ((df['high'] - df['open']) / df['open'])
    df['low_open_pct'] = ((df['low'] - df['open']) / df['open'])
    df['close_change_pct'] = df['close'].pct_change().fillna(0)

    # -------------------------
    # 2) 보조지표 계산 (원본 가격 사용)
    # -------------------------
    df_ind = ta.add_all_ta_features(
        df,
        open="open", high="high", low="low", close="close", volume="volume",
        fillna=True
    )

    # 이동평균선 (SMA, EMA)
    for days in SMA_WINDOWS:
        window = days * CANDLES_PER_DAY
        df_ind[f"SMA_{days}D"] = df_ind["close"].rolling(window=window).mean()
        df_ind[f"EMA_{days}D"] = df_ind["close"].ewm(span=window, adjust=False).mean()

    # RSI
    df_ind["RSI_14D"] = ta.momentum.rsi(df_ind["close"], window=14 * CANDLES_PER_DAY)

    # MACD
    short_window = 12 * CANDLES_PER_DAY
    long_window = 26 * CANDLES_PER_DAY
    signal_window = 9 * CANDLES_PER_DAY
    ema_short = df_ind["close"].ewm(span=short_window, adjust=False).mean()
    ema_long = df_ind["close"].ewm(span=long_window, adjust=False).mean()
    df_ind["MACD_12D_26D"] = ema_short - ema_long
    df_ind["MACD_signal_9D"] = df_ind["MACD_12D_26D"].ewm(span=signal_window, adjust=False).mean()
    df_ind["MACD_hist_9D"] = df_ind["MACD_12D_26D"] - df_ind["MACD_signal_9D"]

    # Bollinger Bands
    for days in [20, 50]:
        window = days * CANDLES_PER_DAY
        sma = df_ind["close"].rolling(window=window).mean()
        std = df_ind["close"].rolling(window=window).std()
        df_ind[f"BB_{days}D_high"] = sma + (2 * std)
        df_ind[f"BB_{days}D_low"] = sma - (2 * std)
        df_ind[f"BB_{days}D_width"] = (df_ind[f"BB_{days}D_high"] - df_ind[f"BB_{days}D_low"]) / sma

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high=df_ind["high"], low=df_ind["low"], close=df_ind["close"],
        window=14 * CANDLES_PER_DAY, smooth_window=3
    )
    df_ind["Stoch_%K"] = stoch.stoch()
    df_ind["Stoch_%D"] = stoch.stoch_signal()

    # ADX
    adx = ta.trend.ADXIndicator(
        high=df_ind["high"], low=df_ind["low"], close=df_ind["close"],
        window=14 * CANDLES_PER_DAY
    )
    df_ind["ADX"] = adx.adx()
    df_ind["ADX_pos"] = adx.adx_pos()
    df_ind["ADX_neg"] = adx.adx_neg()

    # CCI
    cci = ta.trend.CCIIndicator(
        high=df_ind["high"], low=df_ind["low"], close=df_ind["close"],
        window=20 * CANDLES_PER_DAY
    )
    df_ind["CCI_20D"] = cci.cci()

    # MFI
    mfi = ta.volume.MFIIndicator(
        high=df_ind["high"], low=df_ind["low"], close=df_ind["close"],
        volume=df_ind["volume"], window=14 * CANDLES_PER_DAY
    )
    df_ind["MFI_14D"] = mfi.money_flow_index()

    # ATR
    atr = ta.volatility.AverageTrueRange(
        high=df_ind["high"], low=df_ind["low"], close=df_ind["close"],
        window=14 * CANDLES_PER_DAY
    )
    df_ind["ATR_14D"] = atr.average_true_range()

    # Williams %R
    wr = ta.momentum.WilliamsRIndicator(
        high=df_ind["high"], low=df_ind["low"], close=df_ind["close"],
        lbp=14 * CANDLES_PER_DAY
    )
    df_ind["Williams_%R"] = wr.williams_r()

    # -------------------------
    # 3) 정규화 (per-coin z-score)
    # -------------------------
    feature_cols = ["open", "high", "low", "close", "volume"] + \
                   [c for c in df_ind.columns if c not in ["coin", "label"] and not c.startswith("norm_")]
    for col in feature_cols:
        mean_val = df_ind[col].mean()
        std_val = df_ind[col].std()
        df_ind[col] = (df_ind[col] - mean_val) / (std_val + 1e-8)

    # -------------------------
    # 4) 글로벌 정규화 summary feature 붙이기
    # -------------------------
    for k, v in coin_summary.items():
        if k.startswith("norm_"):
            df_ind[k] = v

    # -------------------------
    # 5) 시간 feature
    # -------------------------
    END_TIME = pd.Timestamp("2025-08-21 23:45:00")
    INTERVAL_MIN = 15
    n_rows = len(df)
    df['timestamp'] = [END_TIME - pd.Timedelta(minutes=INTERVAL_MIN * i) for i in range(n_rows-1, -1, -1)]

    seconds_in_day = 24*60*60
    time_in_day = df['timestamp'].dt.hour*3600 + df['timestamp'].dt.minute*60 + df['timestamp'].dt.second
    df_ind['hour_sin'] = np.sin(2 * np.pi * time_in_day / seconds_in_day)
    df_ind['hour_cos'] = np.cos(2 * np.pi * time_in_day / seconds_in_day)

    day_of_week = df['timestamp'].dt.weekday
    df_ind['weekday_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df_ind['weekday_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    day_of_month = df['timestamp'].dt.day
    df_ind['monthday_sin'] = np.sin(2 * np.pi * (day_of_month-1) / 31)
    df_ind['monthday_cos'] = np.cos(2 * np.pi * (day_of_month-1) / 31)

    day_of_year = df['timestamp'].dt.dayofyear
    df_ind['yearday_sin'] = np.sin(2 * np.pi * (day_of_year-1) / 365)
    df_ind['yearday_cos'] = np.cos(2 * np.pi * (day_of_year-1) / 365)

    df_ind.drop(columns=['timestamp'], inplace=True)

    # -------------------------
    # 6) label 계산 (binary)
    # -------------------------
    LOOKAHEAD = 3 * CANDLES_PER_DAY
    UP_THRESHOLD = 0.02
    DOWN_THRESHOLD = 0.01
    close_values = df_ind['close'].values
    n = len(close_values)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        end_idx = min(i + LOOKAHEAD, n)
        future = close_values[i+1:end_idx]
        if len(future) == 0:
            labels[i] = 0
            continue
        pct_changes = (future - close_values[i]) / close_values[i]
        up_idx = np.where(pct_changes >= UP_THRESHOLD)[0]
        down_idx = np.where(pct_changes <= -DOWN_THRESHOLD)[0]
        if len(up_idx) > 0:
            first_up = up_idx[0]
            if len(down_idx) == 0 or np.all(down_idx > first_up):
                labels[i] = 1
            else:
                labels[i] = 0
        else:
            labels[i] = 0
    df_ind['label'] = labels

    df_ind.interpolate(method='linear', inplace=True)
    df_ind.fillna(0, inplace=True)

    # 저장
    output_path = os.path.join(OUTPUT_DIR, file_name)
    df_ind.to_csv(output_path, index=False)

print("✅ 모든 코인의 전처리가 완료되었습니다! (정규화 + 보조지표 + summary feature 포함)")
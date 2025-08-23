import os
import pandas as pd
from tqdm import tqdm
import ta
from sklearn.preprocessing import StandardScaler
import numpy as np

INPUT_DIR = "data_min15_350000"
OUTPUT_DIR = "preprocessed_data_min15_350000"
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

    # coin summary 가져오기
    coin_summary = summary_map[file_name]

    df['close_open_pct'] = ((df['close'] - df['open']) / df['open'])
    df['high_open_pct'] = ((df['high'] - df['open']) / df['open'])
    df['low_open_pct'] = ((df['low'] - df['open']) / df['open'])
    df['close_change_pct'] = df['close'].pct_change().fillna(0)

    # -------------------------
    # 정규화 (z-score, per-coin)
    # -------------------------
    for col in ["open", "high", "low", "close", "volume"]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - mean_val) / (std_val + 1e-8)

    # -------------------------
    # 보조지표 계산
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
    # 정규화된 summary feature 붙이기
    # -------------------------
    for k, v in coin_summary.items():
        if k.startswith("norm_"):  # 정규화된 값만 추가
            df_ind[k] = v

    # CSV를 불러온 df 기준
    # 1행씩 과거로 갈수록 시간 감소, 마지막 행이 2025-08-21 24:00
    END_TIME = pd.Timestamp("2025-08-21 24:00:00")  # 맨 마지막 행 기준
    INTERVAL_MIN = 15  # 15분 간격

    n_rows = len(df)
    # 각 행의 timestamp 계산
    df['timestamp'] = [END_TIME - pd.Timedelta(minutes=INTERVAL_MIN * i) for i in range(n_rows-1, -1, -1)]

    # 시간 feature (하루 주기)
    seconds_in_day = 24*60*60
    time_in_day = df['timestamp'].dt.hour*3600 + df['timestamp'].dt.minute*60 + df['timestamp'].dt.second
    df['hour_sin'] = np.sin(2 * np.pi * time_in_day / seconds_in_day)
    df['hour_cos'] = np.cos(2 * np.pi * time_in_day / seconds_in_day)

    # 요일 feature (0=월요일, 6=일요일)
    day_of_week = df['timestamp'].dt.weekday
    df['weekday_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # 한 달 feature (1~31)
    day_of_month = df['timestamp'].dt.day
    df['monthday_sin'] = np.sin(2 * np.pi * (day_of_month-1) / 31)
    df['monthday_cos'] = np.cos(2 * np.pi * (day_of_month-1) / 31)

    # 1년 feature (1~365)
    day_of_year = df['timestamp'].dt.dayofyear
    df['yearday_sin'] = np.sin(2 * np.pi * (day_of_year-1) / 365)
    df['yearday_cos'] = np.cos(2 * np.pi * (day_of_year-1) / 365)

    # timestamp 칼럼 제거
    df.drop(columns=['timestamp'], inplace=True)

    # -------------------------
    # label 계산 (binary)
    # -------------------------
    LOOKAHEAD = 3 * CANDLES_PER_DAY  # 3일 = 288캔들
    UP_THRESHOLD = 0.02   # 2% 상승
    DOWN_THRESHOLD = 0.01 # 1% 하락

    close_values = df_ind['close'].values  # 이미 정규화되었지만 pct 계산에는 원래 close가 필요하면 df['close'] 사용
    n = len(close_values)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        # 남은 캔들이 LOOKAHEAD보다 적으면 마지막까지 확인
        end_idx = min(i + LOOKAHEAD, n)
        future = close_values[i+1:end_idx]

        if len(future) == 0:
            labels[i] = 0
            continue

        # 상승/하락 pct 계산
        pct_changes = (future - close_values[i]) / close_values[i]

        # 2% 이상 상승하는 인덱스
        up_idx = np.where(pct_changes >= UP_THRESHOLD)[0]
        down_idx = np.where(pct_changes <= -DOWN_THRESHOLD)[0]

        # label 조건
        if len(up_idx) > 0:
            first_up = up_idx[0]
            if len(down_idx) == 0 or np.all(down_idx > first_up):
                labels[i] = 1
            else:
                labels[i] = 0
        else:
            labels[i] = 0

    df_ind['label'] = labels

    # 저장
    output_path = os.path.join(OUTPUT_DIR, file_name)
    df_ind.to_csv(output_path, index=False)

print("✅ 모든 코인의 전처리가 완료되었습니다! (정규화 + 전통 지표 + 글로벌 정규화 summary feature 포함)")
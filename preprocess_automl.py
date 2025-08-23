import os
import pandas as pd
from tqdm import tqdm
import ta  # technical analysis library
import numpy as np

INPUT_DIR = "data_day_3500"
OUTPUT_DIR = "G:/preprocessed_data_day_3500"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_time_features_infer_dates(df: pd.DataFrame, last_date_str="2025-08-21") -> pd.DataFrame:
    """
    CSV 마지막 날짜를 기준으로 날짜 역추적 후 정수형 시간 feature 추가
    - last_date_str: CSV 마지막 행 날짜 (YYYY-MM-DD)
    """
    n = len(df)
    last_date = pd.to_datetime(last_date_str)
    
    # 마지막 날짜를 기준으로 일 단위로 역순 날짜 생성
    dates = pd.date_range(end=last_date, periods=n, freq='D')
    df["date"] = dates
    
    # 요일 / 월일 / 연중일 추가
    df["day_of_week"] = df["date"].dt.weekday
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 이동평균
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma100"] = df["close"].rolling(100).mean()
    
    # RSI
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # ATR (변동성)
    df["atr14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # 결측치 보간
    df.interpolate(method='linear', inplace=True)
    df = df.ffill().bfill()

    return df

def create_label(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    n = len(df)
    for i in range(n):
        if i + 3 < n:
            base_open = df.loc[i, "open"]
            label = 0
            for j in range(i, i+3):
                if df.loc[j, "low"] <= base_open * 0.99:
                    break
                if df.loc[j, "high"] >= base_open * 1.02:
                    label = 1
                    break
            labels.append(label)
        else:
            labels.append(np.nan)
    df["label"] = labels
    return df

# 전체 파일 처리
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for file in tqdm(files, desc="Processing CSV files"):
    path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(path)

    if len(df) < 200:
        continue
    
    # 컬럼명 통일 (소문자 강제)
    df.columns = [c.lower() for c in df.columns]
    
    # 시간 feature 추가
    df = add_time_features_infer_dates(df)
    
    # 보조지표 추가
    df = add_indicators(df)
    
    # 라벨 생성
    df = create_label(df)
    
    # 저장
    save_path = os.path.join(OUTPUT_DIR, file)
    df.to_csv(save_path, index=False)
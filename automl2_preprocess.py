import os
import pandas as pd
from tqdm import tqdm
import ta  # technical analysis library
import numpy as np

INPUT_DIR = "data_hour4_21900"
OUTPUT_DIR = "G:/preprocessed_data_hour4_21900"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_time_features_infer_dates(df: pd.DataFrame, last_date_str="2025-08-21 20:00") -> pd.DataFrame:
    """
    CSV 마지막 날짜를 기준으로 날짜 역추적 후 정수형 시간 feature 추가
    - last_date_str: CSV 마지막 행 날짜 (YYYY-MM-DD HH:MM)
    - 데이터 간격: 4시간
    """
    n = len(df)
    last_date = pd.to_datetime(last_date_str)
    
    # 마지막 행부터 4시간 간격으로 역순 날짜 생성
    dates = pd.date_range(end=last_date, periods=n, freq="4h")
    df["date"] = dates

    # 요일 / 월일 / 연중일 / 하루 시간(시)
    df["day_of_week"] = df["date"].dt.weekday
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["hour_of_day"] = df["date"].dt.hour
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 이동평균 (윈도우 ×6)
    df["sma120"] = df["close"].rolling(120).mean()
    df["sma300"] = df["close"].rolling(300).mean()
    df["sma600"] = df["close"].rolling(600).mean()
    
    # RSI (window 14 → 84)
    df["rsi84"] = ta.momentum.RSIIndicator(df["close"], window=84).rsi()
    
    # MACD (EMA 12/26/9 → 72/156/54)
    macd = ta.trend.MACD(df["close"], window_slow=156, window_fast=72, window_sign=54)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands (window 20 → 120)
    bb = ta.volatility.BollingerBands(df["close"], window=120, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    
    # Stochastic Oscillator (window 14 → 84, smooth_window 3 → 18)
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=84, smooth_window=18
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # ATR (window 14 → 84)
    df["atr84"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=84).average_true_range()

    # 결측치 보간
    df.interpolate(method="linear", inplace=True)
    df = df.ffill().bfill()
    return df


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    labels1 = []
    labels2 = []
    labels3 = []
    n = len(df)
    for i in range(n):
        if i + 7 < n:
            base_open = df.loc[i+1, "open"]
            label1 = 0
            label2 = 0
            label3 = 0
            for j in range(i+1, i+7):
                if df.loc[j, "low"] <= base_open * 0.99:
                    break
                if df.loc[j, "high"] >= base_open * 1.02:
                    label1 = 1
                if df.loc[j, "high"] >= base_open * 1.03:
                    label2 = 1
                if df.loc[j, "high"] >= base_open * 1.04:
                    label3 = 1
            labels1.append(label1)
            labels2.append(label2)
            labels3.append(label3)
        else:
            labels1.append(np.nan)
            labels2.append(np.nan)
            labels3.append(np.nan)
    df["label1"] = labels1
    df["label2"] = labels2
    df["label3"] = labels3
    return df


# 전체 파일 처리
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
for file in tqdm(files, desc="Processing CSV files"):
    path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(path)
    if len(df) < 600:  # 최소 window 길이에 맞게 조건 강화
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

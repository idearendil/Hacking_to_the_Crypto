import os
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from tqdm import tqdm

# -----------------------------
# 기본 설정
# -----------------------------
INPUT_DIR = "G:/preprocessed_data_hour4_21900"
SEED_MONEY = 1000000
LABEL = "label"
MODEL_PATH = "hour4_models"  # predictor 저장된 경로로 바꿔주세요

predictor = TabularPredictor.load(MODEL_PATH)

# -----------------------------
# CSV 불러오기
# -----------------------------
coin_data = {}
for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv"):
        coin = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(INPUT_DIR, file))
        if len(df) < 600:
            continue
        df = df.tail(361).reset_index(drop=True)  # 마지막 60일 + 이전날 입력용 1개
        df['coin'] = coin
        coin_data[coin] = df

# -----------------------------
# 백테스트 로직
# -----------------------------
capital = SEED_MONEY
holding_coin = None
buy_price = None

records = []  # 거래 기록 저장

for i in tqdm(range(1, 361)):
    today_data = {c: df.iloc[i] for c, df in coin_data.items()}
    prev_data = {c: df.iloc[i - 1] for c, df in coin_data.items()}  # 모델 입력용 (전날)

    X_batch = []
    coins = []

    for c, row in prev_data.items():
        features = row.drop([LABEL], errors="ignore")
        X_batch.append(features)
        coins.append(c)

    X_batch = pd.DataFrame(X_batch)

    # 전체 코인 한번에 예측
    probs = predictor.predict_proba(X_batch)[1].values  # class=1 확률들

    # 코인별 매핑
    preds = {coin: prob for coin, prob in zip(coins, probs)}

    if holding_coin is not None and preds[holding_coin] < 0.5:
        row = today_data[holding_coin]
        sell_price = row["open"] - buy_price * 0.00139

        capital = capital * (sell_price / buy_price)
        records.append({
            "date": i,
            "action": "SELL",
            "coin": holding_coin,
            "price": sell_price,
            "capital": capital
        })
        holding_coin = None
        buy_price = None

    if holding_coin is None:
        # 가장 확률이 높으면서 거래량이 충분한 코인 선택
        while True:
            best_coin = max(preds, key=preds.get)
            if preds[best_coin] < 0.65:
                break
            if prev_data[best_coin]["volume"] * prev_data[best_coin]["close"] < capital * 200:
                preds[best_coin] = 0.0
            else:
                break
        if preds[best_coin] < 0.65:
            continue
        buy_price = today_data[best_coin]["open"]
        holding_coin = best_coin
        holding_days = 1

        records.append({
            "date": i,
            "action": "BUY",
            "coin": best_coin,
            "price": buy_price,
            "capital": capital
        })

    if holding_coin is not None:
        row = today_data[holding_coin]
        low, high, open_price = row["low"], row["high"], row["open"]

        sell = False
        sell_price = None

        if low <= buy_price * 0.99:
            sell = True
            sell_price = buy_price * 0.99 - buy_price * 0.00139
        elif high >= buy_price * 1.02:
            sell = True
            sell_price = buy_price * 1.02 - buy_price * 0.00139

        if sell:
            capital = capital * (sell_price / buy_price)
            records.append({
                "date": i,
                "action": "SELL",
                "coin": holding_coin,
                "price": sell_price,
                "capital": capital
            })
            holding_coin = None
            buy_price = None
            holding_days = 0
            

# -----------------------------
# 결과 저장 및 그래프
# -----------------------------
result_df = pd.DataFrame(records)
print(result_df)
print(f"최종 자본: {capital:.2f} 원")

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(result_df["date"], result_df["capital"], marker="o", linestyle="-", label="Capital")
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Capital (KRW)")
plt.title("Backtest Capital Curve")
plt.legend()
plt.tight_layout()
plt.savefig("automl2_backtest_result.png", dpi=300)
plt.close()

print("그래프 저장 완료: automl2_backtest_result.png")
import os
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from tqdm import tqdm
from agent1 import Agent1

# -----------------------------
# 기본 설정
# -----------------------------
INPUT_DIR = "G:/preprocessed_data_hour4_21900"
SEED_MONEY = 10000000
LABEL = "label"
MODEL_PATH = "hour4_models"  # predictor 저장된 경로로 바꿔주세요

predictor = TabularPredictor.load(MODEL_PATH)
agent = Agent1(predictor)


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
        df = df.tail(201).reset_index(drop=True)  # 마지막 60일 + 이전날 입력용 1개
        df['coin'] = coin
        coin_data[coin] = df

# -----------------------------
# 백테스트 로직
# -----------------------------
capital = SEED_MONEY
holding_coin = None
buy_price = None

records = []  # 거래 기록 저장

for i in tqdm(range(1, 201)):
    today_data = {c: df.iloc[i] for c, df in coin_data.items()}
    prev_data = {c: df.iloc[i - 1] for c, df in coin_data.items()}  # 모델 입력용 (전날)

    order_lst = agent.act(prev_data, capital, holding_coin, buy_price)
    if order_lst is None:
        continue

    for an_order in order_lst:
        if an_order['type'] == 'MARKET_SELL':
            sell_price = today_data[an_order['coin']]['open']
            capital = capital * (sell_price / buy_price) * 0.9995
            records.append({
                "date": i,
                "action": "SELL",
                "coin": holding_coin,
                "price": sell_price,
                "capital": capital
            })
            holding_coin = None
            buy_price = None
        elif an_order['type'] == 'MARKET_BUY':
            holding_coin = an_order['coin']
            buy_price = today_data[an_order['coin']]['open'] * 1.0005
            records.append({
                "date": i,
                "action": "BUY",
                "coin": holding_coin,
                "price": buy_price,
                "capital": capital
            })
        elif an_order['type'] == 'LIMIT_SELL_MIN':
            if today_data[an_order['coin']]['low'] <= an_order['price']:
                sell_price = an_order['price']
                capital = capital * (sell_price / buy_price) * 0.99861
                records.append({
                    "date": i,
                    "action": "SELL",
                    "coin": holding_coin,
                    "price": sell_price,
                    "capital": capital
                })
                holding_coin = None
                buy_price = None
        elif an_order['type'] == 'LIMIT_SELL_MAX':
            if holding_coin is not None and today_data[an_order['coin']]['high'] >= an_order['price']:
                sell_price = an_order['price']
                capital = capital * (sell_price / buy_price) * 0.99861
                records.append({
                    "date": i,
                    "action": "SELL",
                    "coin": holding_coin,
                    "price": sell_price,
                    "capital": capital
                })
                holding_coin = None
                buy_price = None

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
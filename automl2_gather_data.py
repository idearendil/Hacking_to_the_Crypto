import pyupbit
import pandas as pd
from tqdm import tqdm
import os

folder_name = "data_hour4_21900"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

print(f"'{folder_name}' 폴더 준비 완료!")

tickers_lst = pyupbit.get_tickers(fiat="KRW")
print(tickers_lst)
print(len(tickers_lst))

for ticker in tqdm(tickers_lst):
    df = pyupbit.get_ohlcv(ticker, count=21900, interval="minute240", to="20250822")
    if df is not None:
        df.to_csv(f"{folder_name}/{ticker}.csv", index=False, encoding="utf-8-sig")

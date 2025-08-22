import pyupbit

tickers_lst = pyupbit.get_tickers(fiat="KRW")
print(tickers_lst)
print(len(tickers_lst))
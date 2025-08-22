import pyupbit

tickers_lst = pyupbit.get_tickers()
print(tickers_lst)
print(len(tickers_lst))
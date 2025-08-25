import pyupbit

print(pyupbit.get_ohlcv("KRW-BTC", interval="minute240", count=1))
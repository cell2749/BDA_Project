import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

sys.path.append('../')

import pystan
import stan_utility

data = pd.read_csv('/home/jonatan/projects/school/BDA_Project/data/selected/combined.csv', delimiter=",")
data.columns = ['date', 'LTC mCap', 'LTC price', 'LTC  exVol',
                'LTC fees', 'LTC  txVol', 'ETH mCap',
                'ETH price', 'ETH exVol', 'ETH fees',
                'ETH txVol', 'ETC mCap', 'ETC price',
                'ETC exVol', 'ETC fees', 'ETC txVol',
                'DASH mCap', 'DASH price', 'DASH exVol',
                'DASH fees', 'DASH txVol', 'BTC mCap',
                'BTC price', 'BTC exVol', 'BTC fees',
                'BTC txVol']
data = data[['date', 'BTC price', 'LTC price', 'DASH price', 'ETH price', 'ETC price']]
# data = (data - data.mean()) / (data.max() - data.min())
print(data.head(5))
data.drop(columns=['date'], inplace=True)


def normalize(values):
    return (values - values.mean()) / (values.max() - values.min())


m = 5

y1 = pd.read_csv('../data/small/dgb.csv')[['price']].values.flatten()
y2 = pd.read_csv('../data/small/gas.csv')[['price']].values.flatten()
y3 = pd.read_csv('../data/small/vtc.csv')[['price']].values.flatten()
y4 = pd.read_csv('../data/small/xvg.csv')[['price']].values.flatten()
n = y1.shape[0]
x = data[['BTC price']].values.flatten()[-n:]
x = normalize(x)
y1 = normalize(y1)
y2 = normalize(y2)
y3 = normalize(y3)
y4 = normalize(y4)
plt.plot(y1)
plt.plot(y2)
plt.plot(y3)
plt.plot(y4)
plt.plot(x)
plt.show()
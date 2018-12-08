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
    return (values - values.mean())

m = 5

y1 = pd.read_csv('../data/small/dgb.csv')[['price']].values.flatten()
y2 = pd.read_csv('../data/small/gas.csv')[['price']].values.flatten()
y3 = pd.read_csv('../data/small/vtc.csv')[['price']].values.flatten()
y4 = pd.read_csv('../data/small/xvg.csv')[['price']].values.flatten()
n = y1.shape[0]
x = data[['BTC price']].values.flatten()[-n:]
x = normalize(x)
p = np.linspace(x.min(), x.max(), m)
y1 = normalize(y1)
y2 = normalize(y2)
y3 = normalize(y3)
y4 = normalize(y4)
print(p)

model = stan_utility.compile_model('../prediction/lin_ex2.stan')

data1 = dict(N=n, M=m, x=x, y=y1, xpreds=p)
fit1 = model.sampling(data=data1, seed=74749)
samples1 = fit1.extract(permuted=True)

data2 = dict(N=n, M=m, x=x, y=y2, xpreds=p)
fit2 = model.sampling(data=data2, seed=74749)
samples2 = fit2.extract(permuted=True)

data3 = dict(N=n, M=m, x=x, y=y3, xpreds=p)
fit3 = model.sampling(data=data3, seed=74749)
samples3 = fit3.extract(permuted=True)

data4 = dict(N=n, M=m, x=x, y=y4, xpreds=p)
fit4 = model.sampling(data=data4, seed=74749)
samples4 = fit4.extract(permuted=True)

f, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
preds = samples1['ypreds'].T
ax = axes[0, 0]
ax.scatter(x, y1, label='Price DGB/BTC')
ax.set_ylabel('DGB price')
ax.set_xlabel('BTC price')
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.01, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

preds = samples2['ypreds'].T
ax = axes[0, 1]
ax.scatter(x, y2, label='Price GAS/BTC')
ax.set_ylabel('GAS price')
ax.set_xlabel('BTC price')
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.01, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

preds = samples3['ypreds'].T
ax = axes[1, 0]
ax.scatter(x, y3, label='Price VTC/BTC')
ax.set_ylabel('VTC price')
ax.set_xlabel('BTC price')
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.01, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

preds = samples4['ypreds'].T
ax = axes[1, 1]
ax.scatter(x, y4, label='Price XVG/BTC')
ax.set_ylabel('XVG price')
ax.set_xlabel('BTC price')
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.01, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

plt.legend(loc='best')
plt.savefig('predicting_small_coins_2.png')
plt.show()
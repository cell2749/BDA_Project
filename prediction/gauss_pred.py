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


def print_loo_and_ks(samples):
    from psis import psisloo

    loglik = samples['log_lik']
    loo, loos, ks = psisloo(loglik)
    print("Loo: %.2f" % loo)

    ks_sum = [[
        (ks <= 0.5).sum(),
        sum([1 for k in ks if k > 0.5 and k <= 0.7]),
        (ks > 0.7).sum()
    ]]

    ks_df = pd.DataFrame(ks_sum, columns=["k<=0.5", "0.5<k<=0.7", "0.7<k"])
    print(ks_df)


print(data.head(5))
data.drop(columns=['date'], inplace=True)
data = (data - data.mean()) / (data.max() - data.min())

n = data.shape[0]
m = 5
x = data[['BTC price']].values.flatten()[0:n]
y1 = data[['DASH price']].values.flatten()[0:n]
y2 = data[['LTC price']].values.flatten()[0:n]
y3 = data[['ETH price']].values.flatten()[0:n]
y4 = data[['ETC price']].values.flatten()[0:n]
p = np.linspace(data[['BTC price']].min(), data[['BTC price']].max(), m)
print(p)

model = stan_utility.compile_model('prediction/lin_ex2.stan')

data1 = dict(N=n, M=m, x=x, y=y1, xpreds=p)
fit1 = model.sampling(data=data1, seed=74749)
samples1 = fit1.extract(permuted=True)
print_loo_and_ks(samples1)

data2 = dict(N=n, M=m, x=x, y=y2, xpreds=p)
fit2 = model.sampling(data=data2, seed=74749)
samples2 = fit2.extract(permuted=True)
print_loo_and_ks(samples2)

data3 = dict(N=n, M=m, x=x, y=y3, xpreds=p)
fit3 = model.sampling(data=data3, seed=74749)
samples3 = fit3.extract(permuted=True)
print_loo_and_ks(samples3)

data4 = dict(N=n, M=m, x=x, y=y4, xpreds=p)
fit4 = model.sampling(data=data4, seed=74749)
samples4 = fit4.extract(permuted=True)
print_loo_and_ks(samples4)

f, axes = plt.subplots(2, 2, figsize=(14, 10))
preds = samples1['ypreds'].T
ax = axes[0, 0]
ax.scatter(data['BTC price'], data['DASH price'], alpha=0.5)
ax.set_ylabel('DASH price')
ax.set_xlabel('BTC price')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.1, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

preds = samples2['ypreds'].T
ax = axes[0, 1]
ax.scatter(data['BTC price'], data['LTC price'], alpha=0.5)
ax.set_ylabel('LTC price')
ax.set_xlabel('BTC price')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.1, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

preds = samples3['ypreds'].T
ax = axes[1, 0]
ax.scatter(data['BTC price'], data['ETH price'], alpha=0.5)
ax.set_ylabel('ETH price')
ax.set_xlabel('BTC price')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.1, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

preds = samples4['ypreds'].T
ax = axes[1, 1]
ax.scatter(data['BTC price'], data['ETC price'], alpha=0.5)
ax.set_ylabel('ETC price')
ax.set_xlabel('BTC price')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for i in range(m):
    ax.scatter([p[i]] * len(preds[i]), preds[i], alpha=0.1, c='g')
    ax.scatter(p[i], np.mean(preds[i]), c='r')

import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red',)
green_patch = mpatches.Patch(color='green')
f.legend([red_patch, green_patch], ['mean', 'spread'], loc='upper right', ncol=1)
f.suptitle("Gaussian linear prediction", fontsize=14)
plt.savefig('predicting_big_coins_gauss.png')
plt.show()

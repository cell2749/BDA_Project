import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("../data/selected/combined.csv")
features = df.columns
COINS = ['LTC', 'ETH', 'ETC', 'DASH']
PRICE_FEATURES = [coin + ' price(USD)' for coin in COINS] #['LTC price(USD)', 'BTC price(USD)', 'ETH price(USD)', 'ETC price(USD)', 'DASH price(USD)']
print(features)
df = df[PRICE_FEATURES]
print(df)
df = (df)/(df.std())
df = df
# df.plot()
plt.figure()
for feature in PRICE_FEATURES:
    print(feature)
    ax = df[feature].plot.hist(bins=50, alpha=0.5)

ax.legend(COINS)
plt.show()
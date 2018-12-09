import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/selected/combined.csv', delimiter=",")
f, axes = plt.subplots(2,2,figsize=(14,7))
ax = axes[0,0]
ax.scatter(data['BTC price(USD)'], data['LTC price(USD)'], alpha=0.5)
ax.set_title('BTC price vs LTC price')
ax = axes[0,1]
ax.scatter(data['BTC price(USD)'], data['DASH price(USD)'], alpha=0.5)
ax.set_title('BTC price vs DASH price')
ax = axes[1,0]
ax.scatter(data['BTC price(USD)'], data['ETH price(USD)'], alpha=0.5)
ax.set_title('BTC price vs ETH price')
ax = axes[1,1]
ax.scatter(data['BTC price(USD)'], data['ETC price(USD)'], alpha=0.5)
ax.set_title('BTC price vs ETC price')
plt.tight_layout()
plt.savefig('prices.png')

plt.show()
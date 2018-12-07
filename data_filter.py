import pandas as pd
import glob
import os

small_data_path = "./data/small/"
big_data_path = "./data/big/"
all_data_path = "./data/all/"
allFiles = glob.glob(all_data_path + "*.csv")

selected_features = [
    "date",
    "marketcap(USD)",
    "price(USD)",
    "txVolume(USD)",
    "txCount",
    "fees"
]
rename_columns = {
    "date": "date",
    "marketcap(USD)": "marketcap",
    "price(USD)": "price",
    "txVolume(USD)": "txVol",
    "txCount": "txCount",
    "fees": "fees"
}

marketcaps = {}
for file in allFiles:
    df = pd.read_csv(file)
    missing_feature = False
    for feature in selected_features:
        if feature not in df.columns:
            missing_feature = True
    if missing_feature:
        # Check next data set
        continue
    else:
        crypto_name = file.split(os.sep)[-1].split(".")[0]
        # Ignore stellar as the data is outdated
        if crypto_name != "xlm":
            # Record market cap for big/small selection
            marketcaps[crypto_name] = df["marketcap(USD)"].tail(1).item()

sorted_mc = sorted(marketcaps.items(), key=lambda x: x[1])
small_coins = []
big_coins = []
#############
# FILTERING #
#############
for coin in sorted_mc:
    # coin -> index 1 is market cap, index 0 is name of the coin
    M = (10 ** 6)
    # Select small and big coins
    if coin[1] < 160 * M:  # and coin[1] > 100 * M:
        small_coins.append(coin[0])
    if coin[1] > 1000 * M:
        big_coins.append(coin[0])

cryptos = {}
# Variables for trimming by least data available
small_min_value = float("+inf")
small_min_name = ""
big_min_value = float("+inf")
big_min_name = ""
for name in small_coins + big_coins:
    df = pd.read_csv(all_data_path + name + ".csv")[
        selected_features
    ] \
        .rename(index=str, columns=rename_columns) \
        .dropna()
    # Find mean fee value for each coin
    df["meanFee"] = df["fees"] / df["txCount"]
    cryptos[name] = df[["date", "marketcap", "price", "txVol", "meanFee"]]
    # Identify trimming coins
    _min = df.count()[0]
    if name in small_coins and small_min_value > _min:
        small_min_value = _min
        small_min_name = name
    if name in big_coins and big_min_value > _min:
        big_min_value = _min
        big_min_name = name

print("Small trimmed by:", small_min_name, small_min_value)
print("Big trimmed by:", big_min_name, big_min_value)
# Trim and save
for name in small_coins:
    if name != small_min_name:
        trimmed = cryptos[name].loc[
            cryptos[name]["date"].isin(cryptos[small_min_name]["date"])
        ]
    else:
        trimmed = cryptos[name]
    trimmed.to_csv(small_data_path + name + ".csv", sep=',', encoding='utf-8')

for name in big_coins:
    if name != big_min_name:
        trimmed = cryptos[name].loc[
            cryptos[name]["date"].isin(cryptos[big_min_name]["date"])
        ]
    else:
        trimmed = cryptos[name]
    trimmed.to_csv(big_data_path + name + ".csv", sep=',', encoding='utf-8')

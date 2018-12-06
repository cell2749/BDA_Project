import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

small_data_path = "../data/small/"
smallFiles = glob.glob(small_data_path + "*.csv")

# Features:
# "date",
# "marketcap",
# "price",
# "txVol",
# "txCount",
# "fees"

GRAPH_ADJ = 3
marketcaps = {}
GLOB_COUNTER = 0
for file in smallFiles:
    f, axes = plt.subplots(2, GRAPH_ADJ, figsize=(12, 7), sharex=True)
    df = pd.read_csv(file, index_col=0)
    features = df.columns
    date = features[0]
    name = file.split(os.sep)[-1].split(".")[0]
    df.drop(columns=[date], inplace=True)

    # Normalize data frame
    df = (df - df.mean()) / (df.max() - df.min())

    for _i in range(1, len(features)):
        for __i in range(_i + 1, len(features)):
            normalized_feature_1 = df[features[_i]]
            normalized_feature_2 = df[features[__i]]

            ax = axes[GLOB_COUNTER // GRAPH_ADJ, GLOB_COUNTER % GRAPH_ADJ]

            ax.set_title(features[__i] + " - " + features[_i])
            ax.scatter(normalized_feature_2, normalized_feature_1)
            ax.set_xlabel(features[__i])
            ax.set_ylabel(features[_i])

            GLOB_COUNTER += 1

    plt.savefig("small_coin_pics/" + name + ".png")
    plt.clf()
    GLOB_COUNTER = 0

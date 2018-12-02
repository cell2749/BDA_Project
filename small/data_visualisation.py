import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

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
    name = file.split("\\")[1].split(".")[0]
    for _i in range(1, len(features)):
        for __i in range(_i + 1, len(features)):
            df_norm_i = (df[features[_i]] - df[features[_i]].mean()) / (
                df[features[_i]].max() - df[features[_i]].min())
            df_norm__i = (df[features[__i]] - df[features[__i]].mean()) / (
                df[features[__i]].max() - df[features[__i]].min())

            ax = axes[GLOB_COUNTER // GRAPH_ADJ, GLOB_COUNTER % GRAPH_ADJ]

            ax.set_title(features[__i] + " - " + features[_i])
            ax.scatter(df_norm__i, df_norm_i)
            ax.set_xlabel(features[__i])
            ax.set_ylabel(features[_i])

            GLOB_COUNTER += 1

    plt.savefig("small_coin_pics/" + name + ".png")
    plt.clf()
    GLOB_COUNTER = 0

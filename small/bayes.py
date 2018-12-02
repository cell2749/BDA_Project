import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pystan
import pandas as pd
import glob
from psis import psisloo

small_data_path = "../data/small/"
smallFiles = glob.glob(small_data_path + "*.csv")

# Features:
# "date",
# "marketcap",
# "price",
# "txVol",
# "fees"

def psispeffk(log_lik):
    loo, loos, kw = psisloo(log_lik)
    print("\n")
    print("psis-loo:", loo)
    err_05 = 0
    err_07 = 0
    err_1 = 0
    err_inf = 0
    for i in range(0, len(kw)):
        if kw[i] <= 0.5:
            err_05 += 1
        elif kw[i] <= 0.7:
            err_07 += 1
        elif kw[i] <= 1.0:
            err_1 += 1
        else:
            err_inf += 1
    print("\n")
    print("K-VALUES:\n")
    print("(-inf;0.5] &", err_05, "&", 100 * err_05 / len(kw), "\%\\\\\\hline")
    print("(0.5;0.7] &", err_07, "&", 100 * err_07 / len(kw), "\%\\\\\\hline")
    print("(0.7;1.0] &", err_1, "&", 100 * err_1 / len(kw), "\%\\\\\\hline")
    print("(1.0;inf) &", err_inf, "&", 100 * err_inf / len(kw), "\%\\\\\\hline")
    _sum = 0
    for i in range(0, 30):
        _sum += np.log(np.mean(np.exp(log_lik[:, i])))
    print("\n")
    print("p_eff:", _sum - loo)

    return 0


def Bayesian_Procedure_hier(hier_data):
    # hier_data should be in the form:
    #
    # 1 Conceptual Analysis tx/fee = 1/x, price - non-normal
    # 2 Define Observations. Range of values - from -1 to 1 if normalised.
    CAUSE_FEATURE = "meanFee"
    EFFECT_FEATURE = "txVol"
    SINGLE_FEATURE = "price"
    k_groups_cause = []
    k_groups_effect = []
    k_groups_single = []
    for coin in hier_data:
        print(coin)
        k_groups_single.append(hier_data[coin][SINGLE_FEATURE].tolist())
        k_groups_cause.append(hier_data[coin][CAUSE_FEATURE].tolist())
        k_groups_effect.append(hier_data[coin][EFFECT_FEATURE].tolist())
    N = len(k_groups_single[0])
    K = len(k_groups_single)
    print("Number of observations per group/coin:", N)
    print("Number of coins/groups:", K)
    print("Total number of observations:", N * K)
    # 3 Summary statistic - scatter plot or histogram
    # 4 Model
    with open('single_bayes.stan', 'r') as stan_file:
        stan_code = stan_file.read()
    hier_model = pystan.StanModel(model_code=stan_code)
    h_flat = np.array(np.transpose(k_groups_single)).flatten()
    h_x = np.tile(np.arange(1, K + 1), N)
    stan_data = dict(
        N=len(h_flat),
        K=K,
        x=h_x,
        y=h_flat
    )
    # Stan results
    fit = hier_model.sampling(data=stan_data)
    print(fit)
    samples = fit.extract(permuted=True)
    psispeffk(samples["log_lik"])
    return 0

if __name__ == '__main__':
    coins = {}
    for file in smallFiles:
        name = file.split("\\")[1].split(".")[0]
        coins[name] = pd.read_csv(file, index_col=0)

    Bayesian_Procedure_hier(coins)

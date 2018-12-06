import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pystan
import pandas as pd
import glob
from psis import psisloo
import platform
import os

if platform.system() == 'Windows':
    import winsound

# Sound for code finishing
duration = 2000  # millisecond
freq = 440  # Hz


small_data_path = "../data/small/"
smallFiles = glob.glob(small_data_path + "*.csv")

# Features:
# "date",
# "marketcap",
# "price",
# "txVol",
# "fees"
psi_comparisons = {}


def psispeffk(log_lik, name):
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
    _sum = 0
    for i in range(0, 30):
        _sum += np.log(np.mean(np.exp(log_lik[:, i])))

    psi_comparisons[name] = "psis-loo: " + str(
        loo) + " " + "K-VALUES:\n" + "\n" + " " + "(-inf;0.5] &" + " " + str(
        err_05) + " " + "&" + " " + str(
        100 * err_05 / len(kw)) + "\n" + " " + "(0.5;0.7] &" + " " + str(
        err_07) + " " + "&" + " " + str(
        100 * err_07 / len(kw)) + "\n" + " " + "(0.7;1.0] &" + " " + str(
        err_1) + " " + "&" + " " + str(
        100 * err_1 / len(kw)) + "\n" + " " + "(1.0;inf) &" + " " + str(
        err_inf) + " " + "&" + " " + str(100 * err_inf / len(
        kw)) + "\n" + " " + "\n" + "\n" + " " + "p_eff:" + str(_sum - loo)
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
    # 4 Models
    stan_names = [
        "norm_single.stan",
        "lognormal.stan",
        "chi.stan",
        "inv_chi.stan",
        "weibull.stan"
        # "t_single.stan",
        # "laplace_single.stan"
    ]
    h_flat = np.array(np.transpose(k_groups_single)).flatten()
    h_x = np.tile(np.arange(1, K + 1), N)
    plt.hist(np.log(h_flat), bins=50, rwidth=1)
    plt.grid(False)
    plt.title("log prices")
    plt.savefig("log_prices_hist.png")
    plt.clf()
    plt.hist(h_flat, bins=50, rwidth=1)
    plt.grid(False)
    plt.title("pure prices")
    plt.savefig("prices_hist.png")
    plt.clf()
    for stan_name in stan_names:
        print(stan_name)
        with open("stan/" + stan_name, 'r') as stan_file:
            stan_code = stan_file.read()
        hier_model = pystan.StanModel(model_code=stan_code)

        stan_data = dict(
            N=len(h_flat),
            K=K,
            x=h_x,
            y=h_flat,
            low=np.nextafter(0, 1)
        )
        # Stan results
        fit = hier_model.sampling(data=stan_data)
        print(fit)
        samples = fit.extract(permuted=True)
        psispeffk(samples["log_lik"], stan_name)
    return 0


if __name__ == '__main__':
    coins = {}
    for file in smallFiles:
        name = file.split(os.sep)[-1].split(".")[0]
        coins[name] = pd.read_csv(file, index_col=0)

    Bayesian_Procedure_hier(coins)
    for name in psi_comparisons:
        print(name)
        print(psi_comparisons[name])

    if platform.system() == 'Windows':
        winsound.Beep(freq, duration)

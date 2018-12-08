import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pystan
import pandas as pd
import glob
from psis import psisloo
import winsound

# Sound for code finishing
duration = 2000  # millisecond
freq = 440  # Hz

small_data_path = "../data/small/"
smallFiles = glob.glob(small_data_path + "*.csv")
big_data_path = "../data/big/"
bigFiles = glob.glob(big_data_path + "*.csv")
# Features:
# "date",
# "marketcap",
# "price",
# "txVol",
# "fees"
PSIS_COMPARISONS = {}


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

    PSIS_COMPARISONS[name] = "psis-loo: " + str(
        loo) + "\nK-VALUES:\n" + "\n" + " " + "(-inf;0.5] &" + " " + str(
        err_05) + " " + "&" + " " + str(
        100 * err_05 / len(kw)) + "\n" + " " + "(0.5;0.7] &" + " " + str(
        err_07) + " " + "&" + " " + str(
        100 * err_07 / len(kw)) + "\n" + " " + "(0.7;1.0] &" + " " + str(
        err_1) + " " + "&" + " " + str(
        100 * err_1 / len(kw)) + "\n" + " " + "(1.0;inf) &" + " " + str(
        err_inf) + " " + "&" + " " + str(100 * err_inf / len(
        kw)) + "\n" + " " + "\n" + "\n" + " " + "p_eff:" + str(_sum - loo)
    return 0


# One can pass all coins in cheap_coins set
# if there is no need to split the coins to cheap and expensive
def Bayesian_Procedure_hier(hier_data, id_name, _cheap_coins):
    # Distributions
    stan_names = [
        # "norm_mix.stan"
        "norm.stan",
        # Positive
        # "lognormal.stan"
        # "chi.stan",
        # "inv_chi.stan",
        # "weibull.stan"
        # Continuous
        "laplace.stan",
        "logistic.stan",
        "cauchy.stan"
    ]
    # g for graphs, k groups for bayes
    k_cheap = []
    g_cheap = {}
    k_exp = []
    g_exp = {}
    for coin in hier_data:
        # xrp is bugged
        if coin != "xrp":
            mc = hier_data[coin]["marketcap"].tolist()[-1] // 1000000
            data = hier_data[coin]["price"].tolist()
            print(coin, len(data))
            log_data = np.log(data)
            if coin in _cheap_coins:
                k_cheap.append(log_data)
                g_cheap[coin] = (log_data, mc)
            else:
                k_exp.append(log_data)
                g_exp[coin] = (log_data, mc)

    g_set = {
        "cheap": g_cheap,
        "expensive": g_exp
    }
    for g_sub_name in g_set:
        g = g_set[g_sub_name]
        for coin in g:
            # coin_tuple 0 - data, 1 - marketcap
            c_t = g[coin]
            plt.hist(c_t[0], bins=50,
                     rwidth=1, alpha=0.7, label=coin + " " + str(c_t[1]))
        plt.grid(False)
        plt.title(id_name + " " + g_sub_name + " coins")
        plt.xlabel("log price (USD)")
        plt.ylabel("count")
        plt.legend()
        plt.savefig(id_name + "_" + g_sub_name + "_prices.png")
        plt.clf()
    k_set = {
        "cheap": k_cheap
        #"expensive": k_exp
    }
    for k_sub_name in k_set:
        k_groups = k_set[k_sub_name]
        N = len(k_groups[0])  # Observations
        K = len(k_groups)  # Groups/coins
        h_flat = np.transpose(k_groups).flatten()
        h_x = np.tile(np.arange(1, K + 1), N)
        for stan_name in stan_names:
            print(stan_name)
            with open("stan/cont/" + stan_name, 'r') as stan_file:
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
            fit = hier_model.sampling(data=stan_data, iter=4000)
            with open("Fits.txt", "a") as text_file:
                print(id_name + "_" + k_sub_name + "_" + stan_name + "\n" +
                      str(fit), file=text_file)
            samples = fit.extract(permuted=True)
            posterior_data = samples["ypred"]
            post_g_set = {}
            i = 0
            # Complex for loop for plotting the posterior. Don't bother to read.
            # Need labels. Assuming they go in same order.
            no_error = False
            for g_sub_name in g_set:
                # if cheap==cheap
                if g_sub_name == k_sub_name:
                    # Get dictionary of cheap or exp coins
                    g = g_set[g_sub_name]
                    # Get coin name to put posterior samples into dictionary
                    for coin in g:
                        # g tuple 0 - data, 1 - marketcap, coin - name
                        if len(posterior_data.shape) == 1:
                            post_g_set[coin] = posterior_data
                        else:
                            post_g_set[coin] = posterior_data[:, i]
                        try:
                            plt.hist(post_g_set[coin], bins=50,
                                     rwidth=1, alpha=0.7,
                                     label=coin)
                            no_error = True
                        except:
                            print("NaN")
                        i += 1
            if no_error:
                plt.grid(False)
                plt.title("posterior " + id_name + " " + k_sub_name + " " +
                          stan_name.split(".")[0])
                plt.xlabel("log price (USD)")
                plt.ylabel("count")
                plt.legend()
                plt.savefig(
                    "post_" + id_name + "_" + k_sub_name + "_" +
                    stan_name.split(".")[
                        0] + ".png")
                plt.clf()
            psispeffk(samples["log_lik"], k_sub_name + stan_name)
    return 0


if __name__ == '__main__':
    #coins = {}
    #for file in smallFiles:
    #    name = file.split("\\")[1].split(".")[0]
    #    coins[name] = pd.read_csv(file, index_col=0)
    #cheap_coins = ["dgb", "xvg"]
    #Bayesian_Procedure_hier(coins, "small", cheap_coins)
    #for name in PSIS_COMPARISONS:
    #    with open("psis_loo_vals.txt", "a") as text_file:
    #        print(
    #            "small\n" + name + "\n" + PSIS_COMPARISONS[name],
    #            file=text_file
    #        )
    #PSIS_COMPARISONS={}
    coins = {}
    for file in bigFiles:
        name = file.split("\\")[1].split(".")[0]
        if name=="ada":
            coins[name] = pd.read_csv(file, index_col=0)
    cheap_coins = ["ada"]
    Bayesian_Procedure_hier(coins, "big", cheap_coins)
    for name in PSIS_COMPARISONS:
        with open("psis_loo_vals.txt", "a") as text_file:
            print(
                "big\n" + name + "\n" + PSIS_COMPARISONS[name],
                file=text_file
            )
    winsound.Beep(freq, duration)

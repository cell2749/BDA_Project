# Already modified a bit, just to keep the code away from main file:
import matplotlib
import matplotlib.pyplot as plot
import numpy
import scipy.stats as stats
import multiprocessing
import math
import pystan
import stan_utility
import pandas as pd
import glob

small_data_path = "../data/small/"
smallFiles = glob.glob(small_data_path + "*.csv")

light = "#DCBCBC"
light_highlight = "#C79999"
mid = "#B97C7C"
mid_highlight = "#A25050"
dark = "#8F2727"
dark_highlight = "#7C0000"
green = "#00FF00"

# Features:
# "date",
# "marketcap",
# "price",
# "txVol",
# "fees"
def Bayesian_Procedure_hier(hier_data):
    # 1 Conceptual Analysis tx/fee = 1/x, price - non-normal
    # 2 Define Observations. Range of values - from -1 to 1 if normalised.
    print("Number of observations:", len(hier_data[0]))
    print("Number of coins/groups:", len(hier_data))
    # 3 Summary statistic - scatter plot or histogram
    # 4 Model
    # Simulation:

    # Simulated data points
    sim_R = 1000  # 1000 draws from the Bayesian joint distribution
    # Simulated coins
    sim_K = 100

    sim_data = dict(N=sim_N)

    # model = stan_utility.compile_model('sample_joint_ensemble.stan')
    with open('sample_joint_ensemble.stan', 'r') as stan_file:
        stan_code = stan_file.read()
    simulated_model = pystan.StanModel(model_code=stan_code)
    # Change model simulation to beta/hier?
    # Fixed_param -> Evalueates only generated quantities
    fit = simulated_model.sampling(data=sim_data,
                         iter=sim_R, warmup=0, chains=1, refresh=sim_R,
                         seed=4838282, algorithm="Fixed_param")
    simu_lambdas = fit.extract()['lambda']
    simu_ys = fit.extract()['y'].astype(numpy.int64)
    with open('bayes.stan', 'r') as stan_file:
        stan_code = stan_file.read()
        print(stan_code)
    hier_model = pystan.StanModel(model_code=stan_code)
    return 0

# Number of groups
_K = 0
# Number of data points
_N = 0
coins = {}
for file in smallFiles:
    _K += 1
    name = file.split("\\")[1].split(".")[0]
    coins[name] = pd.read_csv(file, index_col=0)
    _N = coins[name].count()

print(_N)
# Simulated data points
sim_N = 1000  # 1000 draws from the Bayesian joint distribution
# Simulated coins
sim_K = 100

sim_data = dict(N=sim_N)

#model = stan_utility.compile_model('sample_joint_ensemble.stan')
with open('sample_joint_ensemble.stan', 'r') as stan_file:
    stan_code = stan_file.read()
model = pystan.StanModel(model_code=stan_code)
# Change model simulation to beta/hier?
# Fixed_param -> Evalueates only generated quantities
fit = model.sampling(data=sim_data,
                     iter=sim_K, warmup=0, chains=1, refresh=sim_K,
                     seed=4838282, algorithm="Fixed_param")

# Change to sim_alpha and sim_beta
#simu_lambdas = fit.extract()['lambda']
sim_ys = fit.extract()['y'].astype(numpy.int64)

#max_y = 40
#B = max_y + 1

#bins = [b - 0.5 for b in range(B + 1)]
# FIX!
idxs = [idx for idx in range(B) for r in range(2)]
# FIX!
xs = [idx + delta for idx in range(B) for delta in [-0.5, 0.5]]

counts = [numpy.histogram(sim_ys[r])[0] for r in range(_N)]

for n in range(10):
    plot.plot(xs, [counts[n][idx] for idx in idxs], linewidth=2.5, color=dark,
              alpha=0.5)
plot.gca().set_xlim(-0.5, B + 0.5)
plot.gca().set_xlabel("y")
plot.gca().set_ylim(0, 50)
plot.show()

probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
creds = [numpy.percentile([count[b] for count in counts], probs)
         for b in range(B)]
pad_creds = [creds[idx] for idx in idxs]

plot.fill_between(xs, [c[0] for c in pad_creds], [c[8] for c in pad_creds],
                  facecolor=light, color=light)
plot.fill_between(xs, [c[1] for c in pad_creds], [c[7] for c in pad_creds],
                  facecolor=light_highlight, color=light_highlight)
plot.fill_between(xs, [c[2] for c in pad_creds], [c[6] for c in pad_creds],
                  facecolor=mid, color=mid)
plot.fill_between(xs, [c[3] for c in pad_creds], [c[5] for c in pad_creds],
                  facecolor=mid_highlight, color=mid_highlight)
plot.plot(xs, [c[4] for c in pad_creds], color=dark)

plot.gca().set_xlim([min(bins), max(bins)])
plot.gca().set_xlabel("y")
plot.gca().set_ylim([0, max([c[8] for c in creds])])
plot.gca().set_ylabel("Prior Predictive Distribution")

plot.axvline(x=25, linewidth=2.5, color="white")
plot.axvline(x=25, linewidth=2, color="black")

plot.show()
float(len([y for y in simu_ys.flatten() if y > 25])) / len(simu_ys.flatten())
simus = zip(simu_lambdas, simu_ys)
fit_model = stan_utility.compile_model('fit_data.stan')


def analyze_simu(simu):
    simu_l = simu[0]
    simu_y = simu[1]

    # Fit the simulated observation
    input_data = dict(N=N, y=simu_y)

    fit = fit_model.sampling(data=input_data, seed=4938483, n_jobs=1)

    # Compute diagnostics
    warning_code = stan_utility.check_all_diagnostics(fit, quiet=True)

    # Compute rank of prior draw with respect to thinned posterior draws
    thinned_l = fit.extract()['lambda'][numpy.arange(0, 4000 - 7, 8)]
    sbc_rank = len(list(filter(lambda x: x > simu_l, thinned_l)))

    # Compute posterior sensitivities
    summary = fit.summary(probs=[0.5])
    post_mean_l = [x[0] for x in summary['summary']][0]
    post_sd_l = [x[2] for x in summary['summary']][0]

    prior_sd_l = 6.44787

    z_score = (post_mean_l - simu_l) / post_sd_l
    shrinkage = 1 - (post_sd_l / prior_sd_l) ** 2

    return [warning_code, sbc_rank, z_score, shrinkage]


pool = multiprocessing.Pool(4)
ensemble_output = pool.map(analyze_simu, simus)

warning_codes = [x[0] for x in ensemble_output]
if sum(warning_codes) is not 0:
    print("Some posterior fits in the generative " +
          "ensemble encountered problems!")
    for r in range(R):
        if warning_codes[r] is not 0:
            print('Replication {} of {}'.format(r, R))
            print('Simulated lambda = {}'.format(simu_lambdas[r]))
            stan_utility.parse_warning_code(warning_codes[r])
            print("")
else:
    print("No posterior fits in the generative " +
          "ensemble encountered problems!")
sbc_low = stats.binom.ppf(0.005, R, 25.0 / 500)
sbc_mid = stats.binom.ppf(0.5, R, 25.0 / 500)
sbc_high = stats.binom.ppf(0.995, R, 25.0 / 500)

bar_x = [-10, 510, 500, 510, -10, 0, -10]
bar_y = [sbc_high, sbc_high, sbc_mid, sbc_low, sbc_low, sbc_mid, sbc_high]

plot.fill(bar_x, bar_y, color="#DDDDDD", ec="#DDDDDD")
plot.plot([0, 500], [sbc_mid, sbc_mid], color="#999999", linewidth=2)

sbc_ranks = [x[1] for x in ensemble_output]

plot.hist(sbc_ranks, bins=[25 * x - 0.5 for x in range(21)],
          color=dark, ec=dark_highlight, zorder=3)

plot.gca().set_xlabel("Prior Rank")
plot.gca().set_xlim(-10, 510)
plot.gca().axes.get_yaxis().set_visible(False)

plot.show()

z_scores = [x[2] for x in ensemble_output]
shrinkages = [x[3] for x in ensemble_output]

plot.scatter(shrinkages, z_scores, color=dark, alpha=0.2)
plot.gca().set_xlabel("Posterior Shrinkage")
plot.gca().set_xlim(0, 1)
plot.gca().set_ylabel("Posterior z-Score")
plot.gca().set_ylim(-5, 5)

plot.show()

data = pystan.read_rdump('workflow.data.R')

model = stan_utility.compile_model('fit_data_ppc.stan')
fit = model.sampling(data=data, seed=4838282)
stan_utility.check_all_diagnostics(fit)

params = fit.extract()

plot.hist(params['lambda'], bins=25, color=dark, ec=dark_highlight)
plot.gca().set_xlabel("lambda")
plot.gca().axes.get_yaxis().set_visible(False)
plot.show()
max_y = 40
B = max_y + 1

bins = [b - 0.5 for b in range(B + 1)]

idxs = [idx for idx in range(B) for r in range(2)]
xs = [idx + delta for idx in range(B) for delta in [-0.5, 0.5]]

obs_counts = numpy.histogram(data['y'], bins=bins)[0]
pad_obs_counts = [obs_counts[idx] for idx in idxs]

counts = [numpy.histogram(params['y_ppc'][n], bins=bins)[0] for n in
          range(4000)]
probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
creds = [numpy.percentile([count[b] for count in counts], probs)
         for b in range(B)]
pad_creds = [creds[idx] for idx in idxs]

plot.fill_between(xs, [c[0] for c in pad_creds], [c[8] for c in pad_creds],
                  facecolor=light, color=light)
plot.fill_between(xs, [c[1] for c in pad_creds], [c[7] for c in pad_creds],
                  facecolor=light_highlight, color=light_highlight)
plot.fill_between(xs, [c[2] for c in pad_creds], [c[6] for c in pad_creds],
                  facecolor=mid, color=mid)
plot.fill_between(xs, [c[3] for c in pad_creds], [c[5] for c in pad_creds],
                  facecolor=mid_highlight, color=mid_highlight)
plot.plot(xs, [c[4] for c in pad_creds], color=dark)

plot.plot(xs, pad_obs_counts, linewidth=2.5, color="white")
plot.plot(xs, pad_obs_counts, linewidth=2.0, color="black")

plot.gca().set_xlim([min(bins), max(bins)])
plot.gca().set_xlabel("y")
plot.gca().set_ylim([0, max(max(obs_counts), max([c[8] for c in creds]))])
plot.gca().set_ylabel("Posterior Predictive Distribution")

plot.show()

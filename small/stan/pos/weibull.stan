// Hierarchical model
data {
    int<lower=0> N; // number of data points
    int<lower=0> K; // number of groups
    int<lower=1,upper=K> x[N]; // group indicator
    vector[N] y; //
    real low;
}
parameters {
    real<lower=low> mu0;             // prior mean
    real<lower=low> sigma0; // prior std
    vector<lower=low>[K] sigma;         // group means
    real<lower=low> mu;  // common std
}
model {
// Change to multivariate? With alpha and beta from hier prior
  sigma ~ lognormal(mu0, sigma0); // population prior with unknown parameters
  y ~ weibull(mu, sigma[x]);
}
generated quantities {
  vector[K] ypred;
  real<lower=low> sigpred;
  real sypred;
  vector[N] log_lik;
  // Samples for seventh machine
  sigpred = lognormal_rng(mu0,sigma0);
  sypred = weibull_rng(mu,sigpred);
  // Samples for six machines
  for (i in 1:K)
    ypred[i] = weibull_rng(mu, sigma[i]);
  // Log_likelihood for psis_loo values
  for (i in 1:N)
    log_lik[i] = weibull_lpdf(y[i] | mu, sigma[x[i]]);
}

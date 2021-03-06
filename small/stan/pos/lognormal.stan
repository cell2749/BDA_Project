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
    vector<lower=low>[K] mu;         // group means
    real<lower=low> sigma;  // common std
}
model {
// Change to multivariate? With alpha and beta from hier prior
  mu ~ lognormal(mu0, sigma0); // population prior with unknown parameters
  y ~ lognormal(mu[x], sigma);
}
generated quantities {
  vector<lower=low>[K] ypred;
  real<lower=low> mpred;
  real<lower=low> mypred;
  vector[N] log_lik;
  // Samples for seventh machine
  mpred = lognormal_rng(mu0,sigma0);
  mypred = lognormal_rng(mpred,sigma);
  // Samples for six machines
  for (i in 1:K)
    ypred[i] = lognormal_rng(mu[i], sigma);
  // Log_likelihood for psis_loo values
  for (i in 1:N)
    log_lik[i] = lognormal_lpdf(y[i] | mu[x[i]], sigma);
}

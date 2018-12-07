// Hierarchical model
data {
    int<lower=0> N; // number of data points
    int<lower=0> K; // number of groups
    int<lower=1,upper=K> x[N]; // group indicator
    vector[N] y; //
    real low;
}
parameters {
    real mu0;             // prior mean
    real<lower=low> sigma0; // prior std
    vector[K] mu;         // group means
    real<lower=low> sigma;  // common std
}
model {
// Change to multivariate? With alpha and beta from hier prior
  mu ~ normal(mu0, sigma0); // population prior with unknown parameters
  y ~ cauchy(mu[x], sigma);
}
generated quantities {
  vector[K] ypred;
  real mpred;
  real mypred;
  vector[N] log_lik;
  // Samples for seventh machine
  mpred = normal_rng(mu0,sigma0);
  mypred = cauchy_rng(mpred,sigma);
  // Samples for six machines
  for (i in 1:K)
    ypred[i] = cauchy_rng(mu[i], sigma);
  // Log_likelihood for psis_loo values
  for (i in 1:N)
    log_lik[i] = cauchy_lpdf(y[i] | mu[x[i]], sigma);
}

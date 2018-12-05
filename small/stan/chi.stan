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
}
model {
// Change to multivariate? With alpha and beta from hier prior
  mu ~ lognormal(mu0, sigma0); // population prior with unknown parameters
  y ~ chi_square(mu[x]);
}
generated quantities {
  vector[K] ypred;
  real mpred;
  real mypred;
  vector[N] log_lik;
  // Samples for seventh machine
  mpred = lognormal_rng(mu0,sigma0);
  mypred = chi_square_rng(mpred);
  // Samples for six machines
  for (i in 1:K)
    ypred[i] = chi_square_rng(mu[i]);
  // Log_likelihood for psis_loo values
  for (i in 1:N)
    log_lik[i] = chi_square_lpdf(y[i] | mu[x[i]]);
}

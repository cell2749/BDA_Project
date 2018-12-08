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
  mu ~ normal(mu0, sigma0); // population prior with unknown parameters
  y ~ normal(mu[x], sigma);
}
generated quantities {
  vector[K] ypred;
  vector[N] log_lik;
  // Posterior
  for (i in 1:K)
    ypred[i] = normal_rng(mu[i], sigma);
  // Log_likelihood for psis_loo values
  for (i in 1:N)
    log_lik[i] = normal_lpdf(y[i] | mu[x[i]], sigma);
}

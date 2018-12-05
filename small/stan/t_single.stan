// Hierarchical model
data {
    int<lower=0> N; // number of data points
    int<lower=0> K; // number of groups
    int<lower=1,upper=K> x[N]; // group indicator
    vector[N] y; //
}
parameters {
    real mu0;             // prior mean
    real<lower=0.001> sigma0; // prior std
    //real mu1;             // prior mean
    //real<lower=0> sigma1; // prior std
    vector[K] mu;         // group means
    real<lower=0.001,upper=100> sigma;  // common std
    real<lower=0.001,upper=100> gamma;  // common std
}
model {
// Change to multivariate? With alpha and beta from hier prior
  mu ~ normal(mu0, sigma0); // population prior with unknown parameters
  //gamma ~ normal(mu1, sigma1); // population prior with unknown parameters
  y ~ student_t(gamma,mu[x], sigma);
}
generated quantities {
  vector[K] ypred;
  real mpred;
  //real gammapred;
  real mypred;
  vector[N] log_lik;
  // Samples for seventh machine
  mpred = normal_rng(mu0, sigma0);
  //gammapred = normal_rng(mu1, sigma1);
  mypred = student_t_rng(gamma, mpred, sigma);
  // Samples for six machines
  for (i in 1:K)
    ypred[i] = student_t_rng(gamma, mu[i], sigma);
  // Log_likelihood for psis_loo values
  for (i in 1:N)
    log_lik[i] = student_t_lpdf(y[i] | gamma, mu[x[i]], sigma);
}

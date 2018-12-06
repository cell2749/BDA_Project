// Hierarchical model
data {
    int<lower=0> N; // number of data points
    int<lower=0> K; // number of groups
    int<lower=1,upper=K> x[N]; // group indicator
    vector[N] y; //
    real low;
}
parameters {
    real<lower=0,upper=1> lambda;
    real mu01;
    real<lower=low> sigma01;
    real mu02;
    real<lower=low> sigma02;
    real mu1;
    real<lower=low> sigma1;
    real mu2;
    real<lower=low> sigma2;
}
model {
  mu1 ~ normal(mu01, sigma01);
  mu2 ~ normal(mu02, sigma02);
  y ~ log_mix(lambda, normal(mu1, sigma1), normal(mu2, sigma2));
}
generated quantities {
  vector[K] ypred;
  real mpred;
  real mypred;
  vector[N] log_lik;
  // Samples for seventh machine
  //mpred = normal_rng(mu0,sigma0);
  //mypred = normal_rng(mpred,sigma);
  // Samples for six machines
  //for (i in 1:K)
    //ypred[i] = log_mix(lambda,normal_rng(mu1[i], sigma1),);
  // Log_likelihood for psis_loo values
  for (i in 1:N)
    log_lik[i] = log_mix(lambda, normal_lpdf(y[i] | mu1, sigma1), normal_lpdf(y[i] | mu2, sigma2));
}

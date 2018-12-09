data {
    int<lower=0> N;	// number of data points
    int<lower=0> M;	// number of prediction points
    vector[N] x;	// Bitcoin prices
    vector[N] y;	// Coin x prices
    vector[M] xpreds;	// Coin 1 hypothetical future prices
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = alpha + beta*x;
}
model {
    alpha ~ normal(0, 1);
    beta ~ normal(0, 1);
    y ~ normal(mu, sigma);
}
generated quantities {
    vector[M] ypreds; // Predictions based on hypothetical future prices of coin 1
    vector[N] log_lik;
    for(i in 1:M) {
        ypreds[i] = normal_rng(alpha + beta*xpreds[i], sigma);
    }
    for (i in 1:N){
        log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
    }
}

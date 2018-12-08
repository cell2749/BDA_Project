// Linear student-t model
data {
    int<lower=0> N; // number of data points
    int<lower=0> M;	// number of prediction points
    vector[N] x; //
    vector[N] y; //
    vector[M] xpreds; // input location for prediction
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
    real<lower=1, upper=80> nu;
}
transformed parameters {
    vector[N] mu;
    mu = alpha + beta*x;
}
model {
    nu ~ gamma(2, 0.1); // Juarez and Steel(2010)
    y ~ student_t(nu, mu, sigma);
}
generated quantities {
    vector[M] ypreds;
    vector[N] log_lik;
    for(i in 1:M) {
        ypreds[i] = normal_rng(alpha + beta*xpreds[i], sigma);
    }
    for (i in 1:N){
        log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);
    }
}

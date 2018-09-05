"""
    ExpLik <: Likelihood

Exponential likelihood
```math
p(y | f) = θ\\exp(-θy),
```
where ``θ = \\exp(-f)`` and ``f`` is the latent Gaussian process.
"""
struct ExpLik <: Likelihood end

#log of probability density
function log_dens(exponential::ExpLik, f::VecF64, y::VecF64)
    #where we exponentiate for positivity f = exp(fi)
    return [-fi - exp(-fi)*yi for (fi,yi) in zip(f,y)]
end

#derivative of pdf wrt latent function
function dlog_dens_df(exponential::ExpLik, f::VecF64, y::VecF64)
    return [(yi*exp(-fi)-1) for (fi,yi) in zip(f,y)]
end

#mean and variance under likelihood
mean_lik(exponential::ExpLik, f::VecF64) = exp.(f)
var_lik(exponential::ExpLik, f::VecF64) = exp.(f).^2

get_params(exponential::ExpLik) = []
num_params(exponential::ExpLik) = 0





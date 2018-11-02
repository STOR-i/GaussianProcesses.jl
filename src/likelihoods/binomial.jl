"""
    BinLik <: Likelihood

Binomial likelihood
```math
p(y = k | f) = k!/(n!(n-k)!) θᵏ(1 - θ)^{n-k}
```
for number of successes ``k ∈ \\{0, 1, …, n\\}`` out of ``n`` Bernoulli trials, where
``θ = \\exp(f)/(1 + \\exp(f))`` and ``f`` is the latent Gaussian process.
"""
struct BinLik <: Likelihood
    "Fixed number of trials"
    n::Int
end

function log_dens(binomial::BinLik, f::AbstractVector, y::Vector{Int})
    θ = @. exp(f) / (1 + exp(f))
    return Float64[lgamma(binomial.n+1.0) - lgamma(yi+1.0) - lgamma(binomial.n-yi+1.0) + yi*log(θi) + (binomial.n-yi)*log(1-θi) for (θi,yi) in zip(θ,y)]
end

function dlog_dens_df(binomial::BinLik, f::AbstractVector, y::Vector{Int})
    return Float64[yi/(1.0+exp(fi)) - (binomial.n-yi)*exp(fi)/(1+exp(fi)) for (fi,yi) in zip(f,y)]
end

#mean and variance under the likelihood
mean_lik(binomial::BinLik, f::AbstractVector) = Float64[binomial.n*exp(fi)/(1.0+exp(fi)) for fi in f]
var_lik(binomial::BinLik, f::AbstractVector) = Float64[binomial.n*(exp(fi)/(1.0+exp(fi)))*(1.0/(1.0+exp(fi))) for fi in f]

get_params(binomial::BinLik) = []
num_params(binomial::BinLik) = 0

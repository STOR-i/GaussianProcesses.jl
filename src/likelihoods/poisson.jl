"""
    PoisLik <: Likelihood

Poisson likelihood
```math
p(yᵢ = k | fᵢ) = θᵏ\\exp(-θ)/k!
```
for ``k ∈ N₀``, where ``θ = \\exp(f)`` and ``f`` is the latent Gaussian process.
"""
struct PoisLik <: Likelihood end

#log of probability density
function log_dens(poisson::PoisLik, f::VecF64, y::Vector{Int64})
    #where we exponentiate for positivity f = exp(fi)
    return y.*f - exp.(f) - lgamma.(1.0 .+ y)
end

#derivative of pdf wrt latent function
function dlog_dens_df(poisson::PoisLik, f::VecF64, y::Vector{Int64})
    return y - exp.(f)
end

#mean and variance under likelihood
mean_lik(poisson::PoisLik, f::VecF64) = exp.(f)
var_lik(poisson::PoisLik, f::VecF64) = exp.(f)

get_params(poisson::PoisLik) = []
num_params(poisson::PoisLik) = 0



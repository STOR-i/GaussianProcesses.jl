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
function log_dens(poisson::PoisLik, f::AbstractVector, y::Vector{Int})
    #where we exponentiate for positivity f = exp(fi)
    return y.*f - exp.(f) - lgamma.(1.0 .+ y)
end

#derivative of pdf wrt latent function
function dlog_dens_df(poisson::PoisLik, f::AbstractVector, y::Vector{Int})
    return y - exp.(f)
end

#mean and variance under likelihood
mean_lik(poisson::PoisLik, f::AbstractVector) = exp.(f)
var_lik(poisson::PoisLik, f::AbstractVector) = exp.(f)

get_params(poisson::PoisLik) = []
num_params(poisson::PoisLik) = 0

function var_exp(ll::PoisLik, y::AbstractArray, m::AbstractArray, V::AbstractMatrix)
    tot = 0
    V_diag = diag(V)
    for (a, b, c) in zip(y, m, V_diag)
        tot +=  a*b - exp(b + c/2) - log(factorial(convert(Int64, a))) # convert to lgamma(y+1)
    end
    return tot
end

function var_exp(ll::PoisLik, y::AbstractArray, m::AbstractArray, V::AbstractArray)
    tot = 0
    for (a, b, c) in zip(y, m, V)
        tot +=  a*b - exp(b + c/2) - log(factorial(convert(Int64, a))) # convert to lgamma(y+1)
    end
    return tot
end


function var_exp(ll::PoisLik, y::Number, m::Number, V::Number)
    return y*m - exp(m + V/2) - log(factorial(convert(Int64, y))) # convert to lgamma(y+1)
end

function dv_var_exp(ll::PoisLik, y::Number, m::Number, V::Number)
    return gradient(x -> var_exp(ll, y, m, x), V)[1]
end

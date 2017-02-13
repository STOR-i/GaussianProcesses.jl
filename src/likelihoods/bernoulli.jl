"""
    # Description
    Constructor for the Bernoulli likelihood, where the link function is the probit function

    p(y=k|p) = pᵏ(1-p)¹⁻ᵏ, for k=0,1
    # Arguments:
    * `p::Float64`: probability of a success
    """
type BernLik <: Likelihood
    BernLik() = new()
end

function log_dens(bernoulli::BernLik, f::Vector{Float64}, y::Vector{Bool})
    return Float64[yi? log(Φ(fi)) : log(1.0 - Φ(fi)) for (fi,yi) in zip(f,y)]
end

function dlog_dens_df(bernoulli::BernLik, f::Vector{Float64}, y::Vector{Bool})
    return Float64[yi? φ(fi)/Φ(fi) : -φ(fi)/(1.0 - Φ(fi)) for (fi,yi) in zip(f,y)]
end                   

#mean and variance under the likelihood
mean_lik(bernoulli::BernLik, f::Vector{Float64}) = Float64[Φ(fi) for fi in f]
var_lik(bernoulli::BernLik, f::Vector{Float64}) = Float64[Φ(fi)*(1-Φ(fi)) for fi in f]

get_params(bernoulli::BernLik) = []
num_params(bernoulli::BernLik) = 0

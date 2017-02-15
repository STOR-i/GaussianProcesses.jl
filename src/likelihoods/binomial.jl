"""
    # Description
    Constructor for the Binomial likelihood

    p(y=k|f,n) = (n!/k!(n-k)!) × pᵏ(1-p)ⁿ⁻ᵏ, for k = 0,1,...,n
    # Arguments:
    * `θ::Float64`: probability of a success, where θ = exp(f)/(1+exp(f))
    * `n::Int64`: number of trials
    * `k::Int64`: number of successes

    """
type BinLik <: Likelihood
    n::Int64    #number of trials
    BinLik(n::Int64) = new(n)
end

function log_dens(binomial::BinLik, f::Vector{Float64}, y::Vector{Int64})
    θ = exp(f)./(1.0+exp(f))
    return Float64[lgamma(binomial.n+1.0) - lgamma(yi+1.0) - lgamma(binomial.n-yi+1.0) + yi*log(θi) + (binomial.n-yi)*log(1-θi) for (θi,yi) in zip(θ,y)]
end

function dlog_dens_df(binomial::BinLik, f::Vector{Float64}, y::Vector{Int64})
    return Float64[yi/(1.0+exp(fi)) - (binomial.n-yi)*exp(fi)/(1+exp(fi)) for (fi,yi) in zip(f,y)]
end                   

#mean and variance under the likelihood
mean_lik(binomial::BinLik, f::Vector{Float64}) = Float64[binomial.n*exp(fi)/(1.0+exp(fi)) for fi in f]
var_lik(binomial::BinLik, f::Vector{Float64}) = Float64[binomial.n*(exp(fi)/(1.0+exp(fi)))*(1.0/(1.0+exp(fi))) for fi in f]

get_params(binomial::BinLik) = []
num_params(binomial::BinLik) = 0

#This file contains a list of the currently implemented likelihood function

import Base.show

@compat abstract type Likelihood end

function show(io::IO, lik::Likelihood, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(lik)), Params: ")
    show(io, get_params(lik))
    print(io, "\n")
end

include("bernoulli.jl")
include("exponential.jl")
include("gaussian.jl")
include("studentT.jl")
include("poisson.jl")
include("binomial.jl")

#———————————————————————————————————————————————————————————————-
#Priors

get_priors(lik::Likelihood) = lik.priors

function set_priors!(lik::Likelihood, priors::Array)
    length(priors) == num_params(lik) || throw(ArgumentError("$(typeof(lik)) has exactly $(num_params(lik)) parameters"))
    lik.priors = priors
end

function prior_logpdf(lik::Likelihood)
    priors == get_priors(lik)
    if priors==[]
        return 0.0
    else
        return sum(logpdf(prior,param) for (prior, param) in zip(priors,get_params(lik)))
    end    
end

function prior_gradlogpdf(lik::Likelihood)
    priors == get_priors(lik)
    if priors == []
        return zeros(num_params(lik))
    else
        return [gradlogpdf(prior,param) for (prior, param) in zip(priors,get_params(lik))]
    end    
end

#————————————————————————————————————————————
#Predict observations at test locations


""" Computes the predictive mean and variance given a Gaussian distribution for f using quadrature"""
function predict_obs(lik::Likelihood, fmean::Vector{Float64}, fvar::Vector{Float64}) 
    n_gaussHermite = 20
    nodes, weights = gausshermite(n_gaussHermite)
    weights /= sqrt(pi)
    f = fmean .+ sqrt(2.0*fvar)*nodes'
    
    mLik = Array{Float64}(size(f)); vLik = Array{Float64}(size(f));
    for i in 1:n_gaussHermite
        mLik[:,i] = mean_lik(lik, f[:,i]) 
        vLik[:,i] = var_lik(lik, f[:,i])
    end    
    μ = mLik*weights
    σ² = (vLik + mLik.^2)*weights - μ.^2
    return μ, σ²
end



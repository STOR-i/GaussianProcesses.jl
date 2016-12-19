#This file contains a list of the currently implemented likelihood function

import Base.show

abstract Likelihood

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

################
#Priors
################

function set_priors!(lik::Likelihood, priors::Array)
    length(priors) == num_params(lik) || throw(ArgumentError("$(typeof(lik)) has exactly $(num_params(lik)) parameters"))
    lik.priors = priors
end

function prior_logpdf(lik::Likelihood)
    if lik.priors==[]
        return 0.0
    else
        return sum(Distributions.logpdf(prior,param) for (prior, param) in zip(lik.priors,get_params(lik)))
    end    
end

function prior_gradlogpdf(lik::Likelihood)
    if lik.priors==[]
        return zeros(num_params(lik))
    else
        return [Distributions.gradlogpdf(prior,param) for (prior, param) in zip(lik.priors,get_params(lik))]
    end    
end
################
#Predict observations at test locations
###############

#computes the mean and variance of p(y|f) using quadrature
function predict_obs(lik::Likelihood, fmean::Vector{Float64},fvar::Vector{Float64}) 
    n_gaussHermite = 20
    nodes, weights = gausshermite(n_gaussHermite)
    weights /= sqrt(pi)
    f = fmean + nodes*sqrt(2.0*fvar) 
    μ = weights*mean_lik(lik, f)
    σ² = weights*(var_lik(lik, f) +mean_lik(lik, f).^2)  - μ.^2
    return μ, σ²
end



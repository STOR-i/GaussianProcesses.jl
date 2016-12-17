#This file contains a list of the currently implemented likelihood function

import Base.show

abstract Likelihood

function show(io::IO, lik::Likelihood, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(lik)), Params: ")
    show(io, get_params(lik))
    print(io, "\n")
end

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

include("bernoulli.jl")
include("exponential.jl")
include("gaussian.jl")
include("studentT.jl")
include("poisson.jl")

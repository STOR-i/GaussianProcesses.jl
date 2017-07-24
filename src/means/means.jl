#This file contains a list of the currently available mean functions

import Base.show

abstract type Mean end

function show(io::IO, m::Mean, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(m)), Params: ")
    show(io, get_params(m))
    print(io, "\n")
end

# Calculates the stack [dm / dθᵢ] of mean matrix gradients
function grad_stack(m::Mean, X::Matrix{Float64})
    n = num_params(m)
    d, nobsv = size(X)
    mat = Array{Float64}( nobsv, n)
    for i in 1:nobsv
        @inbounds mat[i,:] = grad_mean(m, X[:,i])
    end
    return mat
end

#————————————————————————————————————————————————————————————————
#Priors

function set_priors!(m::Mean, priors::Array)
    length(priors) == num_params(m) || throw(ArgumentError("$(typeof(m)) has exactly $(num_params(m)) parameters"))
    m.priors = priors
end

function prior_logpdf(m::Mean)
    if num_params(m)==0
        return 0.0
    elseif m.priors==[]
        return 0.0
    else
        return sum(logpdf(prior,param) for (prior, param) in zip(m.priors,get_params(m)))
    end    
end

function prior_gradlogpdf(m::Mean)
    if num_params(m)==0
        return zeros(num_params(m))
    elseif m.priors==[]
        return zeros(num_params(m))
    else
        return [gradlogpdf(prior,param) for (prior, param) in zip(m.priors,get_params(m))]
    end    
end

#————————————————————————————————————————————

include("mZero.jl")          # Zero mean function
include("mConst.jl")         # Constant mean function
include("mLin.jl")           # Linear mean function
include("mPoly.jl")          # Polynomial mean function
include("sum_mean.jl")       # Sum mean functions
include("prod_mean.jl")      # Product of mean functions


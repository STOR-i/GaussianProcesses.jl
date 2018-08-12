#This file contains a list of the currently available mean functions

abstract type Mean end

function Base.show(io::IO, m::Mean, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(m)), Params: ")
    show(io, get_params(m))
    print(io, "\n")
end

# Calculate mean for matrix of observations
function Statistics.mean(m::Mean, X::MatF64)
    mu = Array{Float64}(undef, size(X, 2))
    for i in 1:size(X,2)
        @inbounds mu[i] = mean(m, X[:, i])
    end
    return mu
end


# Calculates the stack [dm / dθᵢ] of mean matrix gradients
function grad_stack(m::Mean, X::MatF64)
    n = num_params(m)
    d, nobsv = size(X)
    mat = Array{Float64}(undef, nobsv, n)
    @inbounds for i in 1:nobsv
        mat[i,:] = grad_mean(m, view(X,:,i))
    end
    return mat
end

##########
# Priors #
##########

get_priors(m::Mean) = m.priors

function set_priors!(m::Mean, priors::Array)
    length(priors) == num_params(m) || throw(ArgumentError("$(typeof(m)) has exactly $(num_params(m)) parameters"))
    m.priors = priors
end

function prior_logpdf(m::Mean)
    priors = get_priors(m)
    if isempty(priors)
        return 0.0
    else
        return sum(logpdf(prior,param) for (prior, param) in zip(m.priors,get_params(m)))
    end
end

function prior_gradlogpdf(m::Mean)
    priors = get_priors(m)
    if isempty(priors)
        return zeros(num_params(m))
    else
        return [gradlogpdf(prior,param) for (prior, param) in zip(priors, get_params(m))]
    end
end


#————————————————————————————————————————————

include("mZero.jl")          # Zero mean function
include("mConst.jl")         # Constant mean function
include("mLin.jl")           # Linear mean function
include("mPoly.jl")          # Polynomial mean function
include("composite_mean.jl") # Composite means

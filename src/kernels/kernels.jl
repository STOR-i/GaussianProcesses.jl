# This file contains a list of the currently available covariance functions

import Base.show

abstract Kernel

"""
Data to be used with a kernel object to
calculate a covariance matrix, which is independent of kernel hyperparameters.

# See also
`EmptyData`
"""
abstract KernelData

"""
Default KernelData type which is empty.
"""
type EmptyData <: KernelData
end

KernelData{M<:MatF64}(k::Kernel, X::M) = EmptyData()

"""
# Description
Constructs covariance matrix from kernel and input observations

# Arguments
* `k::Kernel`: kernel for calculating covariance between pairs of points
* `X₁::Matrix{Float64}`: matrix of observations (each column is an observation)
* `X₂::Matrix{Float64}`: another matrix of observations

# Return
`Σ::Matrix{Float64}`: covariance matrix where `Σ[i,j]` is the covariance of the Gaussian process between points `X[:,i]` and `Y[:,j]`.
"""
function cov{M1<:MatF64,M2<:MatF64}(k::Kernel, X₁::M1, X₂::M2)
    d(x1,x2) = cov(k, x1, x2)
    return map_column_pairs(d, X₁, X₂)
end
function cov!{M1<:MatF64,M2<:MatF64}(cK::MatF64, k::Kernel, X₁::M1, X₂::M2)
    d(x1,x2) = cov(k, x1, x2)
    return map_column_pairs!(cK, d, X₁, X₂)
end

"""
# Description
Constructs covariance matrix from kernel and kernel data

# Arguments
* `k::Kernel`: kernel for calculating covariance between pairs of points
* `X::Matrix{Float64}`: matrix of input observations (each column is an observation)    
* `data::KernelData`: data, constructed from input observations, used for calculating covariance matrix

# Return
`Σ::Matrix{Float64}`: covariance matrix where `Σ[i,j]` is the covariance of the Gaussian process between points `X[:,i]` and `X[:,j]`.

# See also
Kernel, KernelData
"""
function cov{M<:MatF64}(k::Kernel, X::M, data::EmptyData)
    d(x,y) = cov(k, x, y)
    return map_column_pairs(d, X)
end
function cov!{M<:MatF64}(cK::MatF64, k::Kernel, X::M, data::EmptyData)
    d(x,y) = cov(k, x, y)
    return map_column_pairs!(cK, d, X)
end

cov{M<:MatF64}(k::Kernel, X::M) = cov(k, X, KernelData(k, X))
cov!{M<:MatF64}(cK:: MatF64, k::Kernel, X::M) = cov!(cK, k, X, KernelData(k, X))
#=function cov!(cK::Matrix{Float64}, k::Kernel, X::Matrix{Float64}, data::KernelData)=#
#=    cK[:,:] = cov(k,X,data)=#
#=end=#
function addcov!{M<:MatF64}(cK::MatF64, k::Kernel, X::M)
    cK[:,:] .+= cov(k, X, KernelData(k, X))
    return cK
end
function addcov!{M<:MatF64}(cK::MatF64, k::Kernel, X::M, data::KernelData)
    cK[:,:] .+= cov(k, X, data)
    return cK
end
function multcov!{M<:MatF64}(cK::MatF64, k::Kernel, X::M)
    cK[:,:] .*= cov(k, X, KernelData(k, X))
    return cK
end
function multcov!{M<:MatF64}(cK::MatF64, k::Kernel, X::M, data::KernelData)
    cK[:,:] .*= cov(k, X, data)
    return cK
end

function grad_slice!{M1<:MatF64,M2<:MatF64}(dK::M1, k::Kernel, X::M2, data::KernelData, p::Int)
    dim = size(X,1)
    nobsv = size(X,2)
    @inbounds for j in 1:nobsv
        @simd for i in 1:j
            dK[i,j] = dKij_dθp(k,X,data,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end
function grad_slice!{M1<:MatF64,M2<:MatF64}(dK::M1, k::Kernel, X::M2, data::EmptyData, p::Int)
    dim = size(X,1)
    nobsv = size(X,2)
    @inbounds for j in 1:nobsv
        @simd for i in 1:j
            dK[i,j] = dKij_dθp(k,X,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end
# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!{M<:MatF64}(stack::AbstractArray, k::Kernel, X::M, data::KernelData)
    npars = num_params(k)
    for p in 1:npars
        grad_slice!(view(stack,:,:,p),k,X,data,p)
    end
    return stack
end
function grad_stack!{M<:MatF64}(stack::AbstractArray, k::Kernel, X::M)
    grad_stack!(stack, k, X, KernelData(k, X))
end
grad_stack{M<:MatF64}(k::Kernel, X::M) = grad_stack(k, X, KernelData(k, X))
function grad_stack{M<:MatF64}(k::Kernel, X::M, data::KernelData)
    n = num_params(k)
    n_obsv = size(X, 2)
    stack = Array(Float64, n_obsv, n_obsv, n)
    grad_stack!(stack, k, X, data)
    return stack
end


##############################
# Parameter name definitions #
##############################

# This generates names like [:ll_1, :ll_2, ...] for parameter vectors
get_param_names(n::Int, prefix::Symbol) = [Symbol(prefix, :_, i) for i in 1:n]
get_param_names(v::Vector, prefix::Symbol) = get_param_names(length(v), prefix)

# Fallback. Yields names like :Matl2Iso_param_1 => 0.5
# Ideally this is never used, because the names are uninformative.
get_param_names(obj::Union{Kernel, Mean}) =
    get_param_names(num_params(obj),
                    Symbol(typeof(obj).name.name, :_param_))

""" `composite_param_names(objects, prefix)`, where `objects` is a
vector of kernels/means, calls `get_param_names` on each object and prefixes the
name returned with `prefix` + object #. Eg.

    get_param_names(ProdKernel(Mat(1/2, 1/2, 1/2), SEArd([0.0, 1.0],0.0)))

yields

    :pk1_ll  
    :pk1_lσ  
    :pk2_ll_1
    :pk2_ll_2
    :pk2_lσ  
"""
function composite_param_names(objects, prefix)
    p = Symbol[]
    for (i, obj) in enumerate(objects)
        append!(p, [Symbol(prefix, i, :_, sym) for sym in get_param_names(obj)])
    end
    p
end

function show(io::IO, k::Kernel, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(k)), Params: ")
    # params_dict = zip(get_param_names(k), get_params(k))
    # for (k, val) in params_dict
    #     print(io, "$(k)=$(val) ")
    # end
    show(io, get_params(k))
    print(io, "\n")
end

num_params(k::Kernel)=throw(ArgumentError("Undefined number of parameters"))

################
#Priors
################

function set_priors!(k::Kernel, priors::Array)
    length(priors) == num_params(k) || throw(ArgumentError("$(typeof(k)) has exactly $(num_params(k)) parameters"))
    k.priors = priors
end

function prior_logpdf(k::Kernel)
    if k.priors==[]
        return 0.0
    else
        return sum(Distributions.logpdf(prior,param) for (prior, param) in zip(k.priors,get_params(k)))
    end    
end

function prior_gradlogpdf(k::Kernel)
    if k.priors==[]
        return zeros(num_params(k))
    else
        return [Distributions.gradlogpdf(prior,param) for (prior, param) in zip(k.priors,get_params(k))]
    end    
end


include("stationary.jl")

include("lin.jl")               # Linear covariance function
include("se.jl")                # Squared exponential covariance function
include("rq.jl")                # Rational quadratic covariance function
include("mat.jl")               # Matern covariance function
include("periodic.jl")          # Periodic covariance function
include("poly.jl")              # Polnomial covariance function
include("noise.jl")             # White noise covariance function
include("const.jl")             # Constant (bias) covariance function

# Composite kernels
include("sum_kernel.jl")        # Sum of kernels
include("prod_kernel.jl")       # Product of kernels
include("masked_kernel.jl")     # Masked kernels (apply to subset of X dims)
include("fixed_kernel.jl")      # Fixed kernels (fix some hyperparameters)

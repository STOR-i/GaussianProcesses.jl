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

KernelData(k::Kernel, X::Matrix{Float64}) = EmptyData()

"""
# Description
Constructs covariance matrix from kernel and input observations

# Arguments
# `k::Kernel`: kernel for calculating covariance between pairs of points
* `X::Matrix{Float64}`: matrix of observations (each column is an observation)
# `Y::Matrix{Float64}`: another matrix of observations

# Return
`Σ::Matrix{Float64}`: covariance matrix where `Σ[i,j]` is the covariance of the Gaussian process between points `X[:,i]` and `Y[:,j]`.
"""
function cov(k::Kernel, X::Matrix{Float64}, Y::Matrix{Float64})
    d(x,y) = cov(k, x, y)
    return map_column_pairs(d, X, Y)
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
function cov(k::Kernel, X::Matrix{Float64}, data::EmptyData)
    d(x,y) = cov(k, x, y)
    return map_column_pairs(d, X)
end

cov(k::Kernel, X::Matrix{Float64}) = cov(k, X, KernelData(k, X))

# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!(stack::AbstractArray, k::Kernel, X::Matrix{Float64}, data::EmptyData)
    d, nobsv = size(X)
    for j in 1:nobsv, i in 1:nobsv
        @inbounds stack[i,j,:] = grad_kern(k, X[:,i], X[:,j])
    end
    return stack
end

function grad_stack(k::Kernel, X::Matrix{Float64}, data::KernelData)
    n = num_params(k)
    n_obsv = size(X, 2)
    stack = Array(Float64, n_obsv, n_obsv, n)
    grad_stack!(stack, k, X, data)
    return stack
end

function grad_stack!(stack::AbstractArray, k::Kernel, X::Matrix{Float64})
    grad_stack!(stack, k, X, KernelData(k, X))
end

grad_stack(k::Kernel, X::Matrix{Float64}) = grad_stack(k, X, KernelData(k, X))

##############################
# Parameter name definitions #
##############################

# This generates names like [:ll_1, :ll_2, ...] for parameter vectors
get_param_names(n::Int, prefix::Symbol) = [symbol(prefix, :_, i) for i in 1:n]
get_param_names(v::Vector, prefix::Symbol) = get_param_names(length(v), prefix)

# Fallback. Yields names like :Matl2Iso_param_1 => 0.5
# Ideally this is never used, because the names are uninformative.
get_param_names(obj::Union{Kernel, Mean}) =
    get_param_names(num_params(obj),
                    symbol(typeof(obj).name.name, :_param_))

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
        append!(p, [symbol(prefix, i, :_, sym) for sym in get_param_names(obj)])
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

include("stationary.jl")

include("lin.jl")               # Linear covariance function
include("se.jl")                # Squared exponential covariance function
include("rq.jl")                # Rational quadratic covariance function
include("mat.jl")               # Matern covariance function
include("periodic.jl")          # Periodic covariance function
include("poly.jl")              # Polnomial covariance function
include("noise.jl")             # White noise covariance function

# Composite kernels
include("sum_kernel.jl")        # Sum of kernels
include("prod_kernel.jl")       # Product of kernels

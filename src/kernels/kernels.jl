# This file contains a list of the currently available covariance functions

import Base.show

abstract Kernel

function show(io::IO, k::Kernel, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(k)), Params: ")
    show(io, get_params(k))
    print(io, "\n")
end

# Returns matrix where D[i,j] = kernel(x1[i], x2[j])
#
# Arguments:
#  x1 matrix of observations (each column is an observation)
#  x2 matrix of observations (each column is an observation)
#  k kernel object
function crossKern(x1::Matrix{Float64}, x2::Matrix{Float64}, k::Kernel)
    d(x,y) = kern(k, x, y)
    return crossKern(x1, x2, d)
end

# Returns matrix of distances D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
#  k kernel object
function crossKern(x::Matrix{Float64}, k::Kernel)
    d(x,y) = kern(k, x, y)
    return crossKern(x, d)
end

# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!(stack::AbstractArray, x::Matrix{Float64}, k::Kernel)
    n = num_params(k)
    d, nobsv = size(x)
    for j in 1:nobsv, i in 1:nobsv
        @inbounds stack[i,j,:] = grad_kern(k, x[:,i], x[:,j])
    end
    return stack
end

function grad_stack(x::Matrix{Float64}, k::Kernel)
    n = num_params(k)
    d, nobsv = size(x)
    stack = Array(Float64, nobsv, nobsv, n)
    grad_stack!(stack, x, k)
    return stack
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

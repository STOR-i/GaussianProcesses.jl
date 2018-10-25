# This file contains a list of the currently available covariance functions

abstract type Kernel end

"""
    KernelData

Data to be used with a kernel object to calculate a covariance matrix,
which is independent of kernel hyperparameters.

See also [`EmptyData`](@ref).
"""
abstract type KernelData end

"""
    EmptyData <: KernelData

Default empty `KernelData`.
"""
struct EmptyData <: KernelData end

KernelData(k::Kernel, X::MatF64) = EmptyData()
kernel_data_key(k::Kernel, X::MatF64) = "EmptyData"

"""
    cov(k::Kernel, X₁::Matrix{Float64}, X₂::Matrix{Float64})

Create covariance matrix from kernel `k` and matrices of observations `X₁` and `X₂`, where
each column is an observation.
"""
function Statistics.cov(k::Kernel, X₁::MatF64, X₂::MatF64)
    d(x1,x2) = cov(k, x1, x2)
    return map_column_pairs(d, X₁, X₂)
end

"""
    cov!(cK::Matrix{Float64}, k::Kernel, X₁::Matrix{Float64}, X₂::Matrix{Float64})

Like [`cov(k, X₁, X₂)`](@ref), but stores the result in `cK` rather than a new matrix.
"""
function cov!(cK::MatF64, k::Kernel, X₁::MatF64, X₂::MatF64)
    d(x1,x2) = cov(k, x1, x2)
    return map_column_pairs!(cK, d, X₁, X₂)
end

"""
    cov(k::Kernel, X::Matrix{Float64}[, data::KernelData = KernelData(k, X)])

Create covariance function from kernel `k`, matrix of observations `X`, where each column is
an observation, and kernel data `data` constructed from input observations.
"""
Statistics.cov(k::Kernel, X::MatF64, data::EmptyData) = cov(k, X)
cov!(k::Kernel, X::MatF64, data::EmptyData) = cov!(cK, k, X)

function cov!(cK::MatF64, k::Kernel, X::MatF64)
    dim, nobsv = size(X)
    @inbounds for j in 1:nobsv
        cK[j,j] = cov_ij(k, X, j, j, dim)
        for i in 1:j-1
            cK[i,j] = cov_ij(k, X, i, j, dim)
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function Statistics.cov(k::Kernel, X::AbstractArray{T, 2}) where T
    dim, nobsv = size(X)
    cK = Array{T}(undef, nobsv, nobsv)
    cov!(cK, k, X)
end
function cov!(cK::MatF64, k::Kernel, X::MatF64, data::KernelData)
    dim, nobsv = size(X)
    @inbounds for j in 1:nobsv
        cK[j,j] = cov_ij(k, X, data, j, j, dim)
        for i in 1:j-1
            cK[i,j] = cov_ij(k, X, data, i, j, dim)
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function Statistics.cov(k::Kernel, X::MatF64, data::KernelData)
    dim, nobsv = size(X)
    cK = Array{Float64}(undef, nobsv, nobsv)
    cov!(cK, k, X, data)
end

@inline cov_ij(k::Kernel, X::MatF64, i::Int, j::Int, dim::Int) = cov(k, @view(X[:,i]), @view(X[:,j]))
@inline cov_ij(k::Kernel, X::MatF64, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(k, X, i, j, dim)

############################
##### Kernel Gradients #####
############################
@inline @inbounds function dKij_dθ!(dK::VecF64, kern::Kernel, X::MatF64, 
                                    i::Int, j::Int, dim::Int, npars::Int)
    for p in 1:npars
        dK[p] = dKij_dθp(kern, X, i, j, p, dim)
    end
end
@inline @inbounds function dKij_dθ!(dK::VecF64, kern::Kernel, X::MatF64, data::KernelData, 
                                    i::Int, j::Int, dim::Int, npars::Int)
    for iparam in 1:npars
        dK[iparam] = dKij_dθp(kern, X, data, i, j, iparam, dim)
    end
end

function grad_slice!(dK::MatF64, k::Kernel, X::MatF64, data::KernelData, p::Int)
    dim = size(X,1)
    @inbounds for j in 1:size(X, 2)
        @simd for i in 1:j
            dK[i,j] = dKij_dθp(k,X,data,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end

function grad_slice!(dK::MatF64, k::Kernel, X::MatF64, data::EmptyData, p::Int)
    dim = size(X, 1)
    @inbounds for j in 1:size(X, 2)
        @simd for i in 1:j
            dK[i,j] = dKij_dθp(k,X,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end

# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!(stack::AbstractArray, k::Kernel, X::MatF64, data::KernelData)
    @inbounds for p in 1:num_params(k)
        grad_slice!(view(stack, :, :, p), k, X, data, p)
    end
    stack
end

grad_stack!(stack::AbstractArray, k::Kernel, X::MatF64) =
    grad_stack!(stack, k, X, KernelData(k, X))

grad_stack(k::Kernel, X::MatF64) = grad_stack(k, X, KernelData(k, X))

function grad_stack(k::Kernel, X::MatF64, data::KernelData)
    nobs = size(X, 2)
    stack = Array{Float64}(undef, nobs, nobs, num_params(k))
    grad_stack!(stack, k, X, data)
end

include("stationary.jl")
include("distance.jl")
include("lin.jl")               # Linear covariance function
include("se.jl")                # Squared exponential covariance function
include("rq.jl")                # Rational quadratic covariance function
include("mat.jl")               # Matern covariance function
include("periodic.jl")          # Periodic covariance function
include("poly.jl")              # Polnomial covariance function
include("noise.jl")             # White noise covariance function
include("const.jl")             # Constant (bias) covariance function

# Composite kernels
include("composite_kernel.jl")
include("pair_kernel.jl")
include("sum_kernel.jl")
include("prod_kernel.jl")

# Wrapped kernels
include("masked_kernel.jl")     # Masked kernels (apply to subset of X dims)
include("fixed_kernel.jl")      # Fixed kernels (fix some hyperparameters)

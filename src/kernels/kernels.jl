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

KernelData(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) = EmptyData()
kernel_data_key(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) = "EmptyData"

"""
    cov(k::Kernel, X₁::AbstractMatrix, X₂::AbstractMatrix)

Create covariance matrix from kernel `k` and matrices of observations `X₁` and `X₂`, where
each column is an observation.
"""
function cov(k::Kernel, X₁::AbstractMatrix, X₂::AbstractMatrix, kerneldata::KernelData=EmptyData())
    n1, n2 = size(X₁, 2), size(X₂, 2)
    cK = Array{promote_type(eltype(X₁), eltype(X₂))}(undef, n1, n2)
    cov!(cK, k, X₁, X₂, kerneldata)
end

"""
    cov!(cK::AbstractMatrix, k::Kernel, X₁::AbstractMatrix, X₂::AbstractMatrix)

Like [`cov(k, X₁, X₂)`](@ref), but stores the result in `cK` rather than a new matrix.
"""
function cov!(cK::AbstractMatrix, k::Kernel, X₁::AbstractMatrix, X₂::AbstractMatrix, kerneldata::KernelData=EmptyData())
    n1, n2 = size(X₁, 2), size(X₂, 2)
    dim = size(X₁, 1)
    @inbounds for i in 1:n1
        for j in 1:n2
            cK[i,j] = cov_ij(k, X₁, X₂, kerneldata, i, j, dim)
        end
    end
    return cK
end

"""
    cov(k::Kernel, X::AbstractMatrix[, data::KernelData = KernelData(k, X, X)])

Create covariance function from kernel `k`, matrix of observations `X`, where each column is
an observation, and kernel data `data` constructed from input observations.
"""
cov(k::Kernel, X::AbstractMatrix, data::EmptyData) = cov(k, X)
cov!(k::Kernel, X::AbstractMatrix, data::EmptyData) = cov!(cK, k, X)

function cov!(cK::AbstractMatrix, k::Kernel, X::AbstractMatrix)
    dim, nobsv = size(X)
    @inbounds for j in 1:nobsv
        cK[j,j] = cov_ij(k, X, X, j, j, dim)
        for i in 1:j-1
            cK[i,j] = cov_ij(k, X, X, i, j, dim)
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function cov(k::Kernel, X::AbstractMatrix)
    dim, nobsv = size(X)
    cK = Array{eltype(X)}(undef, nobsv, nobsv)
    cov!(cK, k, X)
end
function cov!(cK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::KernelData)
    dim, nobsv = size(X)
    @inbounds for j in 1:nobsv
        cK[j,j] = cov_ij(k, X, X, data, j, j, dim)
        for i in 1:j-1
            cK[i,j] = cov_ij(k, X, X, data, i, j, dim)
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function cov(k::Kernel, X::AbstractMatrix, data::KernelData)
    dim, nobsv = size(X)
    cK = Array{eltype(X)}(undef, nobsv, nobsv)
    cov!(cK, k, X, data)
end

@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = cov(k, @view(X1[:,i]), @view(X2[:,j]))
@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(k, X1, X2, i, j, dim)

############################
##### Kernel Gradients #####
############################
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Kernel, X::AbstractMatrix, 
                                    i::Int, j::Int, dim::Int, npars::Int)
    for p in 1:npars
        dK[p] = dKij_dθp(kern, X, i, j, p, dim)
    end
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Kernel, X::AbstractMatrix, data::KernelData, 
                                    i::Int, j::Int, dim::Int, npars::Int)
    for iparam in 1:npars
        dK[iparam] = dKij_dθp(kern, X, data, i, j, iparam, dim)
    end
end

function grad_slice!(dK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::KernelData, p::Int)
    dim = size(X,1)
    @inbounds for j in 1:size(X, 2)
        @simd for i in 1:j
            dK[i,j] = dKij_dθp(k,X,data,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end

function grad_slice!(dK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::EmptyData, p::Int)
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
function grad_stack!(stack::AbstractArray, k::Kernel, X::AbstractMatrix, data::KernelData)
    @inbounds for p in 1:num_params(k)
        grad_slice!(view(stack, :, :, p), k, X, data, p)
    end
    stack
end

grad_stack!(stack::AbstractArray, k::Kernel, X::AbstractMatrix) =
    grad_stack!(stack, k, X, KernelData(k, X, X))

grad_stack(k::Kernel, X::AbstractMatrix) = grad_stack(k, X, KernelData(k, X, X))

function grad_stack(k::Kernel, X::AbstractMatrix, data::KernelData)
    nobs = size(X, 2)
    stack = Array{eltype(X)}(undef, nobs, nobs, num_params(k))
    grad_stack!(stack, k, X, data)
end

@inline dKij_dθp(k::Kernel, X::AbstractMatrix, data::EmptyData, i::Int, j::Int, p::Int, dim::Int) = dKij_dθp(k, X, i, j, p, dim)

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

# Wrapped kernels
include("masked_kernel.jl")     # Masked kernels (apply to subset of X dims)
include("fixed_kernel.jl")      # Fixed kernels (fix some hyperparameters)

# Composite kernels
include("composite_kernel.jl")
include("pair_kernel.jl")
include("sum_kernel.jl")
include("prod_kernel.jl")

# Autodifferentiation
include("autodiff.jl")

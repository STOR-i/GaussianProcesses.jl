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
    cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix)

Create covariance matrix from kernel `k` and matrices of observations `X1` and `X2`, where
each column is an observation.
"""
function cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    cK = Array{promote_type(eltype(X1), eltype(X2))}(undef, nobs1, nobs2)
    cov!(cK, k, X1, X2, data)
end

function cov!(cK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::KernelData=EmptyData())
    dim, nobs = size(X)
    (nobs,nobs) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) and X has size $(size(X))"))
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()] # in case k is not threadsafe (e.g. ADkernel)
    @inbounds Threads.@threads for j in 1:nobs
        kthread = kcopies[Threads.threadid()]
        cK[j,j] = cov_ij(kthread, X, X, data, j, j, dim)
        for i in 1:j-1
            cK[i,j] = cov_ij(kthread, X, X, data, i, j, dim)
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
"""
    cov!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())

Like [`cov(k, X1, X2)`](@ref), but stores the result in `cK` rather than a new matrix.
"""
function cov!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())
    if X1 === X2
        return cov!(cK, k, X1, data)
    end
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    dim = size(X1, 1)
    (nobs1,nobs2) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) X1 $(size(X1)) and X2 $(size(X2))"))
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for i in 1:nobs1
        kthread = kcopies[Threads.threadid()]
        for j in 1:nobs2
            cK[i,j] = cov_ij(kthread, X1, X2, data, i, j, dim)
        end
    end
    return cK
end

"""
    cov(k::Kernel, X::AbstractMatrix[, data::KernelData = EmptyData()])

Create covariance matrix from kernel `k`, matrix of observations `X`, where each column is
an observation, and kernel data `data` constructed from input observations.
"""
cov(k::Kernel, X::AbstractMatrix, data::KernelData=EmptyData()) = cov(k, X, X, data)


@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = cov(k, @view(X1[:,i]), @view(X2[:,j]))
# the default is to drop the KernelData
@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(k, X1, X2, i, j, dim)

############################
##### Kernel Gradients #####
############################
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Kernel, X1::AbstractMatrix, X2::AbstractMatrix,
                                    data::KernelData, i::Int, j::Int, dim::Int, npars::Int)
    for iparam in 1:npars
        dK[iparam] = dKij_dθp(kern, X1, X2, data, i, j, iparam, dim)
    end
end

function grad_slice!(dK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::KernelData, p::Int)
    dim, nobs = size(X)
    (nobs,nobs) == size(dK) || throw(ArgumentError("dK has size $(size(dK)) and X has size $(size(X))"))
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for j in 1:nobs
        kthread = kcopies[Threads.threadid()]
        dK[j,j] = dKij_dθp(kthread,X,X,data,j,j,p,dim)
        @simd for i in 1:(j-1)
            dK[i,j] = dKij_dθp(kthread,X,X,data,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end
function grad_slice!(dK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, p::Int)
    if X1 === X2
        return grad_slice!(dK, k, X1, data, p)
    end
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    (nobs1,nobs2) == size(dK) || throw(ArgumentError("dK has size $(size(dK)) X1 $(size(X1)) and X2 $(size(X2))"))
    dim=dim1
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for i in 1:nobs1
        kthread = kcopies[Threads.threadid()]
        @simd for j in 1:nobs2
            dK[i,j] = dKij_dθp(kthread,X1,X2,data,i,j,p,dim)
        end
    end
    return dK
end


# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!(stack::AbstractArray, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData)
    @inbounds for p in 1:num_params(k)
        grad_slice!(view(stack, :, :, p), k, X1, X2, data, p)
    end
    stack
end

grad_stack!(stack::AbstractArray, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) =
    grad_stack!(stack, k, X1, X2, KernelData(k, X1, X2))

grad_stack(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) = grad_stack(k, X1, X2, KernelData(k, X1, X2))

function grad_stack(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData)
    nobs1 = size(X1, 2)
    nobs2 = size(X2, 2)
    stack = Array{eltype(X)}(undef, nobs1, nobs2, num_params(k))
    grad_stack!(stack, k, X1, X2, data)
end

@inline function dKij_dθp(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int) 
    return dKij_dθp(k, X1, X2, i, j, p, dim)
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

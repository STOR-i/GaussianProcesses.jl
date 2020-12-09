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

@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Kernel, X1::AbstractMatrix, X2::AbstractMatrix,
                                    data::KernelData, i::Int, j::Int, dim::Int, npars::Int)
    for iparam in 1:npars
        dK[iparam] = dKij_dθp(kern, X1, X2, data, i, j, iparam, dim)
    end
end

@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = cov(k, @view(X1[:,i]), @view(X2[:,j]))
# the default is to drop the KernelData
@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(k, X1, X2, i, j, dim)


include("stationary.jl")
include("distance.jl")

include("kernels/lin.jl")               # Linear covariance function
include("kernels/se.jl")                # Squared exponential covariance function
include("kernels/rq.jl")                # Rational quadratic covariance function
include("kernels/mat.jl")               # Matern covariance function
include("kernels/periodic.jl")          # Periodic covariance function
include("kernels/poly.jl")              # Polnomial covariance function
include("kernels/noise.jl")             # White noise covariance function
include("kernels/const.jl")             # Constant (bias) covariance function

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

# Implements interface for covariance and gradients:
include("covariance.jl")

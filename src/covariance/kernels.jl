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

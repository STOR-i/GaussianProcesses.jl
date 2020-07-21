abstract type Kernel end 
abstract type Stationary <: Kernel end


mutable struct SquaredExponential{T<:KernelParameter{Float64, Bool}} <:Stationary
    lengthscale::T
    variance::T
end

SquaredExponential{T}(lengthscale, variance) where {T} = SquaredExponential(lengthscale, variance)

function SquaredExponential(lengthscale::Float64, variance::Float64)
    ℓ = KernelParameter(lengthscale, true)
    σ = KernelParameter(variance, true)
    return SquaredExponential(ℓ, σ)
end

function k(kernel::SquaredExponential)
    f(τ) = kernel.variance.value*exp.(-(τ.^2)/(2*kernel.lengthscale.value^2))
    return f
end

function cov(kernel::Stationary, x::AbstractArray, y::AbstractArray)
    @assert ndims(x) > 1 && ndims(y) > 1 
    kern_compute = k(kernel)
    nobs1 = size(x, 1)
    nobs2 = size(y, 1)
    distance = pairwise(Euclidean(), x, dims =1)
    Gram = kern_compute(distance)
    return Gram
end

function cov(kernel::Stationary, x::AbstractArray)
    return cov(kernel, x, x)
end

# TODO: Create a union type to capture all kernels with just lengthscale and variance i.e. RBF, Matern.etc
function return_params(kernel::SquaredExponential)
    return [kernel.lengthscale.value, kernel.variance.value]
end

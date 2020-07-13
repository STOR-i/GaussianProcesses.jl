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
    f(τ) = kernel.variance.value*exp(-0.5*τ^2/kernel.lengthscale.value^2)
    return f
end

function cov(kernel::Stationary, x::AbstractArray, y::AbstractArray)
    @assert ndims(x) > 1 && ndims(y) > 1 
    kern_compute = k(kernel)
    dim1, nobs1 = size(x)
    dim2, nobs2 = size(y)
    Gram = Array{Float64}(undef, nobs1, nobs2)
    for (idx, xval) in enumerate(x)
        for (jdx, yval) in enumerate(y)
            τ = abs(xval-yval)
            Gram[idx, jdx] = kern_compute(τ)
        end
    end
    return Gram
end

function cov(kernel::Stationary, x::AbstractArray)
    return cov(kernel, x, x)
end

# TODO: Create a union type to capture all kernels with just lengthscale and variance i.e. RBF, Matern.etc
function return_params(kernel::SquaredExponential)
    return [kernel.lengthscale.value, kernel.variance.value]
end

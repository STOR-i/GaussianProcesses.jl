# # Transformations
# abstract type Transform end

# struct LogTransform<:Transform
#     forward::Function
#     backward::Function
# end
# LogTransform()=LogTransform(log, exp)
# TODO: Move to transformations.jl 
#############################################

using Distributions, Plots

abstract type Kernel end 
abstract type Parameter end

abstract type Stationary <: Kernel end

mutable struct KernelParameter<:Parameter
    value::Float64
    trainable::Bool
end
loadK(value::Float64, trainable::Bool) = KernelParameter(value, trainable)


mutable struct SquaredExponential<:Stationary
    lengthscale::KernelParameter
    variance::KernelParameter
end

function SquaredExponential(lengthscale::Float64, variance::Float64)
    ℓ = loadK(lengthscale, true)
    σ = loadK(variance, true)
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

abstract type GP end
abstract type MeanFunction end
struct Zero <: MeanFunction end


mean(mZero::Zero, x::AbstractMatrix) =  zeros(Float64, size(x))


struct GPR <: GP
    MeanFunc::MeanFunction
    Kernel::Kernel
end
function Construct_GPR(mean::MeanFunction, covariance::Kernel)
    return GPR(mean, covariance)
end

function Base.rand(gp::GPR, x::AbstractArray; n_samples::Int=1)
    n = length(vec(x))
    μ = mean(gp.MeanFunc, x)
    Σ = cov(gp.Kernel, x) + 1e-6*I(n)
    dist = MultivariateNormal(vec(μ), Σ) # TODO: Return a vector from mean function
    return rand(dist, n_samples)
end


mean_func = Zero()
kernel = SquaredExponential(1.0, 1.0)
f = Construct_GPR(mean_func, kernel)
x = cat(collect(-1:0.01:1), dims=2)'

samples = rand(f, x; n_samples=10)
p = plot(vec(x), samples, color=:blue, alpha=0.5)



μ = mean(mean_func, x)
Kxx = cov(kernel, x)
Kxx2 = cov(kernel,x, x)
@assert Kxx==Kxx2




# Matern 3/2 ARD covariance function

"""
    Mat32Ard <: MaternARD

ARD Matern 3/2 kernel (covariance)
```math
k(x,x') = σ²(1 + √3|x-x'|/L)\\exp(- √3|x-x'|/L)
```
with length scale ``ℓ = (ℓ₁, ℓ₂, …)`` and signal standard deviation ``σ`` where
``L = diag(ℓ₁, ℓ₂, …)``.
"""
mutable struct Mat32Ard{T} <: MaternARD where {T<:Real}
    "Inverse squared length scale"
    iℓ2::Vector{T}
    "Signal variance"
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

"""
    Mat32Ard(ll::Vector{T}, lσ::T)

Create `Mat32Ard` with length scale `exp.(ll)` and signal standard deviation `exp(lσ)`.
"""
Mat32Ard(ll::Vector{T}, lσ::T) where T = Mat32Ard{T}(exp.(-2 .* ll), exp(2 * lσ), [])

function set_params!(mat::Mat32Ard, hyp::AbstractVector)
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat32 kernel has $(num_params(mat)) parameters, received $(length(hyp))."))
    @views @. mat.iℓ2 = exp(-2 * hyp[1:(end-1)])
    mat.σ2 = exp(2 * hyp[end])
end

get_params(mat::Mat32Ard{T}) where T = T[-log.(mat.iℓ2) / 2; log(mat.σ2) / 2]
get_param_names(mat::Mat32Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat32Ard) = length(mat.iℓ2) + 1

cov(mat::Mat32Ard, r::Number) =
    (s = √3 * r; mat.σ2 * (1 + s) * exp(-s))

dk_dll(mat::Mat32Ard, r::Real, wdiffp::Real) = 3 * mat.σ2 * wdiffp * exp(-√3 * r)

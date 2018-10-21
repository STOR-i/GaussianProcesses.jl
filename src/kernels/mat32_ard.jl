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
mutable struct Mat32Ard <: MaternARD
    "Inverse squared length scale"
    iℓ2::Vector{Float64}
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat32Ard(ll::Vector{Float64}, lσ::Float64)

    Create `Mat32Ard` with length scale `exp.(ll)` and signal standard deviation `exp(lσ)`.
    """
    Mat32Ard(ll::Vector{Float64}, lσ::Float64) = new(exp.(-2 .* ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat32Ard, hyp::VecF64)
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat32 kernel only has $(num_params(mat)) parameters"))
    @views @. mat.iℓ2 = exp(-2 * hyp[1:(end-1)])
    mat.σ2 = exp(2 * hyp[end])
end

get_params(mat::Mat32Ard) = [-log.(mat.iℓ2) / 2; log(mat.σ2) / 2]
get_param_names(mat::Mat32Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat32Ard) = length(mat.iℓ2) + 1

Statistics.cov(mat::Mat32Ard, r::Number) =
    (s = √3 * r; mat.σ2 * (1 + s) * exp(-s))

dk_dll(mat::Mat32Ard, r::Float64, wdiffp::Float64) = 3 * mat.σ2 * wdiffp * exp(-√3 * r)

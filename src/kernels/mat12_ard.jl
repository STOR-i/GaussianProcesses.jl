# Matern 1/2 ARD covariance Function

"""
    Mat12Ard <: MaternARD

ARD Matern 1/2 kernel (covariance)
```math
k(x,x') = σ² \\exp(-|x-x'|/L)
```
with length scale ``ℓ = (ℓ₁, ℓ₂, …)`` and signal standard deviation ``σ`` where
``L = diag(ℓ₁, ℓ₂, …)``.
"""
mutable struct Mat12Ard <: MaternARD
    "Inverse squared length scale"
    iℓ2::Vector{Float64}
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat12Ard(ll::Vector{Float64}, lσ::Float64)

    Create `Mat12Ard` with length scale `exp.(ll)` and signal standard deviation `exp(σ)`.
    """
    Mat12Ard(ll::Vector{Float64}, lσ::Float64) = new(exp.(-2 .* ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat12Ard, hyp::VecF64)
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat12 kernel only has $(num_params(mat)) parameters"))
    @views @. mat.iℓ2  = exp(-2 * hyp[1:(end-1)])
    mat.σ2 = exp(2 * hyp[end])
end

get_params(mat::Mat12Ard) = [-log.(mat.iℓ2) / 2; log(mat.σ2) / 2]
get_param_names(mat::Mat12Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat12Ard) = length(mat.iℓ2) + 1

Statistics.cov(mat::Mat12Ard, r::Number) = mat.σ2 * exp(-r)

dk_dll(mat::Mat12Ard, r::Float64, wdiffp::Float64) = wdiffp / r * cov(mat,r)

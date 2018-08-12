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
    iℓ2::VecF64
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat12Ard(ll::Vector{Float64}, lσ::Float64)

    Create `Mat12Ard` with length scale `exp.(ll)` and signal standard deviation `exp(σ)`.
    """
    Mat12Ard(ll::VecF64, lσ::Float64) = new(exp.(-2 * ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat12Ard, hyp::VecF64)
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat12 kernel only has $(num_params(mat)) parameters"))
    d = length(mat.iℓ2)
    mat.iℓ2  = exp.(-2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat12Ard) = [-log.(mat.iℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat12Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat12Ard) = length(mat.iℓ2) + 1

Statistics.cov(mat::Mat12Ard, r::Float64) = mat.σ2*exp(-r)

dk_dll(mat::Mat12Ard, r::Float64, wdiffp::Float64) = wdiffp/r*cov(mat,r)

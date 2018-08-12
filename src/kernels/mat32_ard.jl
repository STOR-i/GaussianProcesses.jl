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
    iℓ2::VecF64
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat32Ard(ll::Vector{Float64}, lσ::Float64)

    Create `Mat32Ard` with length scale `exp.(ll)` and signal standard deviation `exp(lσ)`.
    """
    Mat32Ard(ll::VecF64, lσ::Float64) = new(exp.(-2 * ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat32Ard, hyp::VecF64)
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat32 kernel only has $(num_params(mat)) parameters"))
    d=length(mat.iℓ2)
    mat.iℓ2 = exp.(-2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat32Ard) = [-log.(mat.iℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat32Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat32Ard) = length(mat.iℓ2) + 1

Statistics.cov(mat::Mat32Ard, r::Float64) = mat.σ2*(1+sqrt(3)*r)*exp(-sqrt(3)*r)

dk_dll(mat::Mat32Ard, r::Float64, wdiffp::Float64) = 3.0*mat.σ2*wdiffp*exp(-sqrt(3)*r)

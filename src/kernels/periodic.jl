# Periodic Function

"""
    Periodic <: Isotropic{Euclidean}

Periodic kernel (covariance)
```math
k(x,x') = σ²\\exp(-2\\sin²(π|x-x'|/p)/ℓ²)
```
with length scale ``ℓ``, signal standard deviation ``σ``, and period ``p``.
"""
mutable struct Periodic <: Isotropic{Euclidean}
    "Squared length scale"
    ℓ2::Float64
    "Signal variance"
    σ2::Float64
    "Period"
    p::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Periodic(ll::Float64, lσ::Float64, lp::Float64)

    Create `Periodic` with length scale `exp(ll)`, signal standard deviation `exp(lσ)`, and
    period `exp(lp)`.
    """
    Periodic(ll::Float64, lσ::Float64, lp::Float64) =
        new(exp(2 * ll), exp(2 * lσ), exp(lp), [])
end

get_params(pe::Periodic) = Float64[log(pe.ℓ2)/2.0, log(pe.σ2)/2.0, log(pe.p)]
get_param_names(pe::Periodic) = [:ll, :lσ, :lp]
num_params(pe::Periodic) = 3

function set_params!(pe::Periodic, hyp::VecF64)
    length(hyp) == 3 || throw(ArgumentError("Periodic function has only three parameters"))
    pe.ℓ2, pe.σ2 = exp.(2.0*hyp[1:2])
    pe.p = exp(hyp[3])
end

Statistics.cov(pe::Periodic, r::Float64) = pe.σ2*exp(-2.0/pe.ℓ2*sin(π*r/pe.p)^2)


@inline dk_dll(pe::Periodic, r::Float64) = 4.0*pe.σ2*(sin(π*r/pe.p)^2/pe.ℓ2)*exp(-2.0/pe.ℓ2*sin(π*r/pe.p)^2)  # dK_dlogℓ
@inline dk_dlp(pe::Periodic, r::Float64) = 4.0/pe.ℓ2*pe.σ2*(π*r/pe.p)*sin(π*r/pe.p)*cos(π*r/pe.p)*exp(-2/pe.ℓ2*sin(π*r/pe.p)^2)    # dK_dlogp
@inline function dk_dθp(pe::Periodic, r::Float64, p::Int)
    if p==1
        dk_dll(pe, r)
    elseif p==2
        dk_dlσ(pe, r)
    elseif p==3
        dk_dlp(pe, r)
    else
        return NaN
    end
end

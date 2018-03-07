"""
# Description
Constructor for the isotropic Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀ(x-x')/2ℓ²)
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
"""
type ExpIso <: Isotropic{Euclidean}
    ℓ::Float64      # Length scale
    σ2::Float64      # Signal std
    priors::Array          # Array of priors for kernel parameters
    ExpIso(ll::Float64, lσ::Float64) = new(exp(ll), exp(2*lσ),[])
end

function set_params!(expk::ExpIso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Exponential kernel only has two parameters"))
    expk.ℓ = exp(hyp[1])
    expk.σ2 = exp(2*hyp[2])
end

get_params(expk::ExpIso) = Float64[log(expk.ℓ), log(expk.σ2)/2.0]
get_param_names(expk::ExpIso) = [:ll, :lσ]
num_params(expk::ExpIso) = 2

cov(expk::ExpIso, r::Float64) = expk.σ2*exp(-r/expk.ℓ)

@inline dk_dll(expk::ExpIso, r::Float64) = r/expk.ℓ*cov(expk,r)
@inline function dk_dθp(expk::ExpIso, r::Float64, p::Int)
    if p==1
        return dk_dll(expk, r)
    elseif p==2
        return dk_dlσ(expk, r)
    else
        return NaN
    end
end

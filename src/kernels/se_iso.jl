"""
# Description
Constructor for the isotropic Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀ(x-x')/2ℓ²)
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
"""
type SEIso <: Isotropic
    ℓ2::Float64      # Length scale
    σ2::Float64      # Signal std
    priors::Array          # Array of priors for kernel parameters
    SEIso(ll::Float64, lσ::Float64) = new(exp(2*ll), exp(2*lσ),[])
end

function set_params!(se::SEIso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Squared exponential only has two parameters"))
    se.ℓ2, se.σ2 = exp(2.0*hyp)
end

get_params(se::SEIso) = Float64[log(se.ℓ2)/2.0, log(se.σ2)/2.0]
get_param_names(::SEIso) = [:ll, :lσ]
num_params(se::SEIso) = 2

metric(se::SEIso) = SqEuclidean()
cov(se::SEIso, r::Float64) = se.σ2*exp(-0.5*r/se.ℓ2)

@inline dk_dll(se::SEIso, r::Float64) = r/se.ℓ2*cov(se,r)
@inline function dk_dθp(se::SEIso, r::Float64, p::Int)
    if p==1
        return dk_dll(se, r)
    elseif p==2
        return dk_dlσ(se, r)
    else
        return NaN
    end
end

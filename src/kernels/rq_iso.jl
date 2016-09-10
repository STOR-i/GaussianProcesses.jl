# Rational Quadratic Isotropic Covariance Function 

@doc """
# Description
Constructor for the isotropic Rational Quadratic kernel (covariance)

k(x,x') = σ²(1+(x-x')ᵀ(x-x')/2αℓ²)^{-α}
# Arguments:
* `ll::Float64`: Log of length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
* `lα::Float64`: Log of shape parameter α
""" ->
type RQIso <: Isotropic
    ℓ2::Float64      # Length scale 
    σ2::Float64      # Signal std
    α::Float64       # shape parameter
    RQIso(ll::Float64, lσ::Float64, lα::Float64) = new(exp(2*ll), exp(2*lσ), exp(lα))
end

function set_params!(rq::RQIso, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Rational Quadratic function has three parameters"))
    rq.ℓ2, rq.σ2, rq.α = exp(2.0*hyp[1]), exp(2.0*hyp[2]), exp(hyp[3])
end

get_params(rq::RQIso) = Float64[log(rq.ℓ2)/2.0, log(rq.σ2)/2.0, log(rq.α)]
get_param_names(rq::RQIso) = [:ll, :lσ, :lα]
num_params(rq::RQIso) = 3

metric(rq::RQIso) = SqEuclidean()
cov(rq::RQIso, r::Float64) = rq.σ2*(1.0+r/(2.0*rq.α*rq.ℓ2))^(-rq.α)

function addcov!(s::AbstractMatrix{Float64}, rq::RQIso, X::Matrix{Float64}, data::IsotropicData)
    nobsv = size(X, 2)
    R = distance(rq, X, data)
    for j in 1:nobsv
        @inbounds @simd for i in 1:j
            s[i,j] += cov(rq, R[i,j])
            s[j,i] = s[i,j]
        end
    end
    return R
end

@inline dk_dll(rq::RQIso, r::Float64) = rq.σ2*(r/rq.ℓ2)*(1.0+r/(2.0*rq.α*rq.ℓ2))^(-rq.α-1.0) # dK_d(log ℓ)dK_dℓ
@inline function dk_dlα(rq::RQIso, r::Float64)
    part = (1.0+r/(2.0*rq.α*rq.ℓ2))
    return rq.σ2*part^(-rq.α)*((r)/(2.0*rq.ℓ2*part)-rq.α*log(part))  # dK_d(log α)
end
@inline function dk_dθp(rq::RQIso, r::Float64, p::Int)
    if p==1
        return dk_dll(rq, r)
    elseif p==2
        return dk_dlσ(rq, r)
    elseif p==3
        return dk_dlα(rq, r)
    else
        return NaN
    end
end

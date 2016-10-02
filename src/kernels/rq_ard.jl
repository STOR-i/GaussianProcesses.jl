# Rational Quadratic ARD Covariance Function 

@doc """
# Description
Constructor for the ARD Rational Quadratic kernel (covariance)

k(x,x') = σ²(1+(x-x')ᵀL⁻²(x-x')/2α)^{-α}, where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of length scale ℓ
* `lσ::Float64`        : Log of the signal standard deviation σ
* `lα::Float64`        : Log of shape parameter α
""" ->
type RQArd <: StationaryARD
    iℓ2::Vector{Float64}      # Inverse squared Length scale
    σ2::Float64              # Signal std
    α::Float64               # Shape parameter
    RQArd(ll::Vector{Float64}, lσ::Float64, lα::Float64) = new(exp(-2.0*ll), exp(2.0*lσ), exp(lα))
end

function set_params!(rq::RQArd, hyp::Vector{Float64})
    length(hyp) == num_params(rq) || throw(ArgumentError("RQArd kernel has $(num_params(rq_ard)) parameters"))
    d = length(rq.iℓ2)
    rq.iℓ2 = exp(-2.0*hyp[1:d])
    rq.σ2 = exp(2.0*hyp[d+1])
    rq.α = exp(hyp[d+2])
end

get_params(rq::RQArd) = [-log(rq.iℓ2)/2.0; log(rq.σ2)/2.0; log(rq.α)]
get_param_names(rq::RQArd) = [get_param_names(rq.iℓ2, :ll); :lσ; :lα]
num_params(rq::RQArd) = length(rq.iℓ2) + 2

metric(rq::RQArd) = WeightedSqEuclidean(rq.iℓ2)
cov(rq::RQArd,r::Float64) = rq.σ2*(1+0.5*r/rq.α)^(-rq.α)

@inline dk_dll(rq::RQArd, r::Float64, wdiffp::Float64) = rq.σ2*wdiffp*(1.0+0.5*r/rq.α)^(-rq.α-1.0)
@inline function dk_dlα(rq::RQArd, r::Float64)
    part  = (1+0.5*r/rq.α)
    return rq.σ2*part^(-rq.α)*(0.5*r/part-rq.α*log(part))
end
@inline function dKij_dθp{M<:MatF64}(rq::RQArd, X::M, i::Int, j::Int, p::Int, dim::Int)
    if p <= dim
        return dk_dll(rq, distij(metric(rq),X,i,j,dim), distijk(metric(rq),X,i,j,p))
    elseif p==dim+1
        return dk_dlσ(rq, distij(metric(rq),X,i,j,dim))
    else
        return dk_dlα(rq, distij(metric(rq),X,i,j,dim))
    end
end
@inline function dKij_dθp{M<:MatF64}(rq::RQArd, X::M, data::StationaryARDData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(rq,X,i,j,p,dim)
end

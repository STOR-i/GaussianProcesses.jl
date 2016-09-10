# Squared Exponential Function with ARD

@doc """
# Description
Constructor for the ARD Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀL⁻²(x-x')/2), where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the inverse length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type SEArd <: StationaryARD
    iℓ2::Vector{Float64}      # Inverse squared Length scale
    σ2::Float64              # Signal variance
    SEArd(ll::Vector{Float64}, lσ::Float64) = new(exp(-2.0*ll),exp(2.0*lσ))
end

function set_params!(se::SEArd, hyp::Vector{Float64})
    length(hyp) == num_params(se) || throw(ArgumentError("SEArd only has $(num_params(se)) parameters"))
    d = length(se.iℓ2)
    se.iℓ2 = exp(-2.0*hyp[1:d])
    se.σ2 = exp(2.0*hyp[d+1])
end

get_params(se::SEArd) = [-log(se.iℓ2)/2.0; log(se.σ2)/2.0]
get_param_names(k::SEArd) = [get_param_names(k.iℓ2, :ll); :lσ]
num_params(se::SEArd) = length(se.iℓ2) + 1

metric(se::SEArd) = WeightedSqEuclidean(se.iℓ2)
cov(se::SEArd, r::Float64) = se.σ2*exp(-0.5*r)

function grad_kern(se::SEArd, x::Vector{Float64}, y::Vector{Float64})
    r = distance(se, x, y)
    exp_r = exp(-0.5*r)
    wdiff = ((x-y).^2).*se.iℓ2
    
    g1   = se.σ2.*wdiff*exp_r   #dK_d(log ℓ)
    g2 = 2.0*se.σ2*exp_r        #dK_d(log σ)
    
    return [g1; g2]
end

@inline dk_dll(se::SEArd, r::Float64, wdiffp::Float64) = wdiffp*cov(se,r)
@inline function dKij_dθp(se::SEArd, X::Matrix{Float64}, i::Int, j::Int, p::Int, dim::Int)
    if p <= dim
        return dk_dll(se, distij(metric(se),X,i,j,dim), distijk(metric(se),X,i,j,p))
    elseif p==dim+1
        return dk_dlσ(se, distij(metric(se),X,i,j,dim))
    else
        return NaN
    end
end
@inline function dKij_dθp(se::SEArd, X::Matrix{Float64}, data::StationaryARDData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(se,X,i,j,p,dim)
end
